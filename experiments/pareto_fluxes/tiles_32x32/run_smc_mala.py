#!/usr/bin/env python

##############################################
# SETUP

import sys

sys.path.append("/home/twhit/smc_object_detection/")

import time

import torch

from smc.aggregate import Aggregate
from smc.images import ImageModel
from smc.kernel import MetropolisAdjustedLangevin
from smc.prior import ParetoStarPrior
from smc.sampler import SMCsampler
from utils.misc import select_cuda_device

device = select_cuda_device()
torch.cuda.set_device(device)
torch.set_default_device(device)
##############################################

##############################################
# LOAD IN IMAGES AND CATALOGS

images = torch.load("data/images.pt").to(device)
true_counts = torch.load("data/true_counts.pt").to(device)
true_locs = torch.load("data/true_locs.pt").to(device)
true_fluxes = torch.load("data/true_fluxes.pt").to(device)

num_images = images.shape[0]
image_height = images.shape[1]
image_width = images.shape[2]
##############################################

##############################################
# SPECIFY TILE-LEVEL IMAGE MODEL, PRIOR, AND MUTATION KERNEL

tile_dim = 8

prior = ParetoStarPrior(
    max_objects=5,
    image_height=tile_dim,
    image_width=tile_dim,
    flux_scale=600,
    flux_alpha=1.25,
    pad=2,
)

imagemodel = ImageModel(
    image_height=tile_dim, image_width=tile_dim, psf_stdev=1.0, background=300
)

mala = MetropolisAdjustedLangevin(
    num_iters=100,
    locs_step=0.1,
    fluxes_step=100,
    fluxes_min=prior.flux_scale,
    fluxes_max=1e6,
)

aggmala = MetropolisAdjustedLangevin(
    num_iters=10,
    locs_step=0.01,
    fluxes_step=50,
    fluxes_min=prior.flux_scale,
    fluxes_max=1e6,
)
##############################################

##############################################
# SPECIFY NUMBER OF CATALOGS AND BATCH SIZE FOR SAVING RESULTS

num_catalogs_per_count = 200
num_catalogs = (prior.max_objects + 1) * num_catalogs_per_count

batch_size = 10
num_batches = num_images // batch_size
##############################################

##############################################
# RUN SMC

torch.manual_seed(1)

for b in range(num_batches):
    runtime = torch.zeros([batch_size])
    num_iters = torch.zeros([batch_size])
    counts = torch.zeros([batch_size, num_catalogs])
    locs = torch.zeros([batch_size, num_catalogs, 10 * prior.max_objects, 2])
    fluxes = torch.zeros([batch_size, num_catalogs, 10 * prior.max_objects])

    for i in range(batch_size):
        image_index = b * batch_size + i

        print(f"image {image_index + 1} of {num_images}")
        print(f"true count = {true_counts[image_index]}")
        print(f"true total flux = {true_fluxes[image_index].sum()}\n")

        sampler = SMCsampler(
            image=images[image_index],
            tile_dim=tile_dim,
            Prior=prior,
            ImageModel=imagemodel,
            MutationKernel=mala,
            num_catalogs_per_count=num_catalogs_per_count,
            ess_threshold=0.75 * num_catalogs_per_count,
            resample_method="multinomial",
            max_smc_iters=100,
        )

        start = time.perf_counter()

        sampler.run()

        agg = Aggregate(
            sampler.Prior,
            sampler.ImageModel,
            aggmala,
            sampler.tiled_image,
            sampler.counts,
            sampler.locs,
            sampler.fluxes,
            sampler.weights_intercount,
            resample_method="multinomial",
            merge_method="naive",
            merge_multiplier=1,
            ess_threshold=(sampler.Prior.max_objects + 1) * sampler.ess_threshold,
        )

        agg.run()

        end = time.perf_counter()

        runtime[i] = end - start
        num_iters[i] = sampler.iter
        counts[i] = agg.counts.squeeze([0, 1])
        index = agg.locs.shape[-2]
        locs[i, :, :index, :] = agg.locs.squeeze([0, 1])
        fluxes[i, :, :index] = agg.fluxes.squeeze([0, 1])

        agg.summarize()
        print(f"\nruntime = {runtime[i]}\n\n\n")

    torch.save(runtime.cpu(), f"results/smc_mala/runtime_{b}.pt")
    torch.save(num_iters.cpu(), f"results/smc_mala/num_iters_{b}.pt")
    torch.save(counts.cpu(), f"results/smc_mala/counts_{b}.pt")
    torch.save(locs.cpu(), f"results/smc_mala/locs_{b}.pt")
    torch.save(fluxes.cpu(), f"results/smc_mala/fluxes_{b}.pt")

print("Done!")
##############################################
