#!/usr/bin/env python
"""Run CheckerboardSMC on 8x8 patches from the M71 image."""

import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.append("/home/twhit/smcdet/")

from smcdet.checkerboard import CheckerboardSMC
from smcdet.images import M71ImageModel
from smcdet.kernel import SingleComponentMH
from smcdet.prior import M71Prior
from utils.misc import select_cuda_device

# --- GPU setup ---
device = select_cuda_device()
torch.cuda.set_device(device)
torch.set_default_device(device)

# --- Change to script directory ---
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# --- Load data ---
full_image = torch.load("data/full_image.pt").to(device)
patch_indices = torch.load("data/patch_indices.pt")
num_patches = patch_indices.shape[0]
print(f"Full image shape: {full_image.shape}")
print(f"Loaded {num_patches} valid patches")

with open("data/params.pkl", "rb") as f:
    params = pickle.load(f)

# --- Parameters ---
tile_dim = 4
pad = 4
image_pad = 4
patch_dim = 8
max_objects = 8
num_catalogs = 10000
batch_size = 10
num_batches = (num_patches + batch_size - 1) // batch_size

num_tiles_per_patch = (patch_dim // tile_dim) ** 2  # 4
max_combined_sources = num_tiles_per_patch * max_objects  # 32

print(f"tile_dim={tile_dim}, pad={pad}, image_pad={image_pad}")
print(f"max_objects={max_objects}, num_catalogs={num_catalogs}")
print(f"num_batches={num_batches}, batch_size={batch_size}")


def extract_padded_patch(full_image, row, col, patch_dim, image_pad, background):
    """Extract a patch with surrounding context from the full image.

    Uses real neighboring pixels where available, background for out-of-bounds.
    Returns a (patch_dim + 2*image_pad) x (patch_dim + 2*image_pad) tensor.
    """
    H, W = full_image.shape
    h_start = row * patch_dim - image_pad
    h_end = (row + 1) * patch_dim + image_pad
    w_start = col * patch_dim - image_pad
    w_end = (col + 1) * patch_dim + image_pad

    padded = torch.full(
        (patch_dim + 2 * image_pad, patch_dim + 2 * image_pad),
        background,
        device=full_image.device,
    )

    # Compute overlap between requested region and actual image
    src_h_start = max(h_start, 0)
    src_h_end = min(h_end, H)
    src_w_start = max(w_start, 0)
    src_w_end = min(w_end, W)

    dst_h_start = src_h_start - h_start
    dst_h_end = dst_h_start + (src_h_end - src_h_start)
    dst_w_start = src_w_start - w_start
    dst_w_end = dst_w_start + (src_w_end - src_w_start)

    padded[dst_h_start:dst_h_end, dst_w_start:dst_w_end] = full_image[
        src_h_start:src_h_end, src_w_start:src_w_end
    ]

    return padded


# --- Run CheckerboardSMC on each patch ---
os.makedirs("results/checkerboard", exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

for b in range(num_batches):
    batch_start = b * batch_size
    batch_end = min(batch_start + batch_size, num_patches)
    current_batch_size = batch_end - batch_start

    runtime = torch.zeros([current_batch_size])
    counts = torch.zeros([current_batch_size, num_catalogs])
    locs = torch.zeros([current_batch_size, num_catalogs, max_combined_sources, 2])
    fluxes = torch.zeros([current_batch_size, num_catalogs, max_combined_sources])

    for i in range(current_batch_size):
        patch_index = batch_start + i
        row, col = patch_indices[patch_index]
        print(
            f"\n--- Batch {b}, Patch {i} (global index {patch_index}, "
            f"grid position ({row}, {col})) ---"
        )

        padded_patch = extract_padded_patch(
            full_image,
            row.item(),
            col.item(),
            patch_dim,
            image_pad,
            params["background"],
        )

        patch = full_image[
            row * patch_dim : (row + 1) * patch_dim,
            col * patch_dim : (col + 1) * patch_dim,
        ]

        checkerboard = CheckerboardSMC(
            image=patch,
            tile_dim=tile_dim,
            pad=pad,
            image_pad=image_pad,
            PriorClass=M71Prior,
            prior_kwargs=dict(
                num_objects=max_objects,
                counts_rate=params["counts_rate"],
                h_lower=0,
                h_upper=tile_dim,
                w_lower=0,
                w_upper=tile_dim,
                flux_alpha=params["flux_alpha"],
                flux_lower=params["flux_lower"],
                flux_upper=params["flux_upper"],
            ),
            ImageModelClass=M71ImageModel,
            image_model_kwargs=dict(
                background=params["background"],
                adu_per_nmgy=params["adu_per_nmgy"],
                psf_params=params["psf_params"],
                psf_radius=params["psf_radius"],
                noise_additive=params["noise_additive"],
                noise_multiplicative=params["noise_multiplicative"],
            ),
            MutationKernelClass=SingleComponentMH,
            kernel_kwargs=dict(
                num_iters=100,
                locs_stdev=0.1,
                fluxes_stdev=2.5,
                fluxes_min=params["flux_lower"],
                fluxes_max=params["flux_upper"],
            ),
            num_catalogs=num_catalogs,
            ess_threshold_prop=0.5,
            resample_method="multinomial",
            max_smc_iters=100,
            prune_flux_lower=params["flux_detection_threshold"],
            print_every=5,
        )

        # Override padded_image with real surrounding pixels
        checkerboard.padded_image = padded_patch

        start = time.perf_counter()
        checkerboard.run()
        end = time.perf_counter()

        # Extract results
        cb_locs = checkerboard.combined_locs.squeeze(0).squeeze(0)
        cb_fluxes = checkerboard.combined_fluxes.squeeze(0).squeeze(0)
        cb_counts = checkerboard.combined_counts.squeeze(0).squeeze(0)

        runtime[i] = end - start
        counts[i] = cb_counts
        n_sources = cb_locs.shape[-2]
        locs[i, :, :n_sources, :] = cb_locs
        fluxes[i, :, :n_sources] = cb_fluxes

        print(f"  Runtime: {end - start:.1f}s")
        print(f"  Mean count: {cb_counts.float().mean():.1f}")

    # Save batch results
    torch.save(runtime.cpu(), f"results/checkerboard/runtime_{b}.pt")
    torch.save(counts.cpu(), f"results/checkerboard/counts_{b}.pt")
    torch.save(locs.cpu(), f"results/checkerboard/locs_{b}.pt")
    torch.save(fluxes.cpu(), f"results/checkerboard/fluxes_{b}.pt")

    print(f"\nSaved batch {b} results to results/checkerboard/")

print("\nDone!")
