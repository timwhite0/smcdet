#!/usr/bin/env python

##############################################
# SETUP

import sys

sys.path.append("/home/twhit/smc_object_detection/")

import time

import numpy as np
import torch

from smc.aggregate import Aggregate
from smc.images import M71ImageModel
from smc.kernel import SingleComponentMH
from smc.prior import M71Prior
from smc.sampler import SMCsampler
from utils.misc import select_cuda_device

device = select_cuda_device()
torch.cuda.set_device(device)
torch.set_default_device(device)
##############################################

##############################################
# LOAD IN IMAGES AND CATALOGS

tiles = torch.load("data/tiles.pt").to(device)
true_counts = torch.load("data/counts_magcut.pt").to(device)
true_fluxes = torch.load("data/fluxes_magcut.pt").to(device)

num_images = tiles.shape[0]
image_height = tiles.shape[1]
image_width = tiles.shape[2]
##############################################

##############################################
# SPECIFY TILE-LEVEL IMAGE MODEL, PRIOR, AND MUTATION KERNEL

tile_dim = 8

prior = M71Prior(
    max_objects=6,
    counts_rate=2.085,
    image_height=tile_dim,
    image_width=tile_dim,
    flux_alpha=0.27274554633062026,
    flux_lower=0.6313902139663695,
    flux_upper=16546.183593750004,
    pad=2.0,
)

imagemodel = M71ImageModel(
    image_height=tile_dim,
    image_width=tile_dim,
    background=491.5867919921875,
    flux_calibration=966.0794677734375,
    psf_params=torch.tensor([1.36, 4.8475, 8.3333, 3.0000, 0.144, 0.0068779]),
    noise_scale=2.0,
)

mh = SingleComponentMH(
    num_iters=100,
    locs_stdev=0.1,
    fluxes_stdev=2.5,
    fluxes_min=prior.flux_lower,
    fluxes_max=prior.flux_upper,
)

aggmh = SingleComponentMH(
    num_iters=100,
    locs_stdev=0.1,
    fluxes_stdev=2.5,
    fluxes_min=prior.flux_lower,
    fluxes_max=prior.flux_upper,
)
##############################################

##############################################
# SPECIFY NUMBER OF CATALOGS AND BATCH SIZE FOR SAVING RESULTS

num_catalogs_per_count = 10000
num_catalogs = (prior.max_objects + 1) * num_catalogs_per_count

batch_size = 10
num_batches = num_images // batch_size
##############################################

##############################################
# RUN SMC

torch.manual_seed(0)
np.random.seed(0)

for b in range(num_batches):
    runtime = torch.zeros([batch_size])
    num_iters = torch.zeros([batch_size])
    counts = torch.zeros([batch_size, num_catalogs])
    locs = torch.zeros([batch_size, num_catalogs, prior.max_objects, 2])
    fluxes = torch.zeros([batch_size, num_catalogs, prior.max_objects])
    posterior_predictive_total_flux = torch.zeros([batch_size, num_catalogs])

    for i in range(batch_size):
        image_index = b * batch_size + i

        print(f"image {image_index + 1} of {num_images}")
        print(f"Number of stars within image boundary: {true_counts[image_index]}")
        print(
            "Total intrinsic flux of stars within image boundary: ",
            f"{true_fluxes[image_index].sum(-1).round()}",
        )
        print(f"Total observed flux: {tiles[image_index].sum().round()}\n")

        sampler = SMCsampler(
            image=tiles[image_index],
            tile_dim=tile_dim,
            Prior=prior,
            ImageModel=imagemodel,
            MutationKernel=mh,
            num_catalogs_per_count=num_catalogs_per_count,
            ess_threshold_prop=0.5,
            resample_method="multinomial",
            max_smc_iters=100,
            print_every=2,
        )

        start = time.perf_counter()

        sampler.run()

        agg = Aggregate(
            sampler.Prior,
            sampler.ImageModel,
            aggmh,
            sampler.tiled_image,
            sampler.counts,
            sampler.locs,
            sampler.fluxes,
            sampler.weights_intercount,
            sampler.log_normalizing_constant,
            ess_threshold_prop=0.5,
            resample_method="multinomial",
        )

        agg.run()

        end = time.perf_counter()

        runtime[i] = end - start
        num_iters[i] = sampler.iter
        counts[i] = agg.counts.squeeze([0, 1])
        index = agg.locs.shape[-2]
        locs[i, :, :index, :] = agg.locs.squeeze([0, 1])
        fluxes[i, :, :index] = agg.fluxes.squeeze([0, 1])
        posterior_predictive_total_flux[i] = (
            agg.posterior_predictive_total_observed_flux
        )

        agg.summarize()
        print(f"\nruntime = {runtime[i]}\n\n\n")

    torch.save(runtime.cpu(), f"results/smc/runtime_{b}.pt")
    torch.save(num_iters.cpu(), f"results/smc/num_iters_{b}.pt")
    torch.save(counts.cpu(), f"results/smc/counts_{b}.pt")
    torch.save(locs.cpu(), f"results/smc/locs_{b}.pt")
    torch.save(fluxes.cpu(), f"results/smc/fluxes_{b}.pt")
    torch.save(
        posterior_predictive_total_flux.cpu(),
        f"results/smc/posterior_predictive_total_flux_{b}.pt",
    )

print("Done!")
##############################################
