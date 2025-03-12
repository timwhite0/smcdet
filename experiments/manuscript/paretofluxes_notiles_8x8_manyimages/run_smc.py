#!/usr/bin/env python

##############################################
# SETUP

import sys

sys.path.append("/home/twhit/smc_object_detection/")

import time

import numpy as np
import torch

from smc.aggregate import Aggregate
from smc.images import ImageModel
from smc.kernel import SingleComponentMH
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

psf_stdev = 0.93
psf_max = 1 / (2 * np.pi * (psf_stdev**2))
background = 200
max_objects = 12
flux_scale = 5 * np.sqrt(background) / psf_max
flux_alpha = (-np.log(1 - 0.99)) / (
    np.log(50 * np.sqrt(background) / psf_max) - np.log(flux_scale)
)
quantile01_flux = flux_scale * ((1 - 0.1) ** (-1 / flux_alpha))
pad = np.sqrt(-2 * (psf_stdev**2) * np.log(flux_scale / quantile01_flux))

prior = ParetoStarPrior(
    max_objects=max_objects,
    image_height=tile_dim,
    image_width=tile_dim,
    flux_scale=flux_scale,
    flux_alpha=flux_alpha,
    pad=pad,
)

imagemodel = ImageModel(
    image_height=tile_dim,
    image_width=tile_dim,
    psf_stdev=psf_stdev,
    background=background,
)

mh = SingleComponentMH(
    num_iters=50,
    locs_stdev=0.25,
    fluxes_stdev=250,
    fluxes_min=prior.flux_scale,
    fluxes_max=1e6,
)

aggmh = SingleComponentMH(
    num_iters=50,
    locs_stdev=0.1,
    fluxes_stdev=100,
    fluxes_min=prior.flux_scale,
    fluxes_max=1e6,
)
##############################################

##############################################
# SPECIFY NUMBER OF CATALOGS AND BATCH SIZE FOR SAVING RESULTS

num_catalogs_per_count = 5000
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
    locs = torch.zeros([batch_size, num_catalogs, prior.max_objects, 2])
    fluxes = torch.zeros([batch_size, num_catalogs, prior.max_objects])
    estimated_total_flux = torch.zeros([batch_size, num_catalogs])

    for i in range(batch_size):
        image_index = b * batch_size + i

        print(f"image {image_index + 1} of {num_images}")
        print(f"true count = {true_counts[image_index]}")
        print(f"true total flux = {images[image_index].sum()}\n")

        sampler = SMCsampler(
            image=images[image_index],
            tile_dim=tile_dim,
            Prior=prior,
            ImageModel=imagemodel,
            MutationKernel=mh,
            num_catalogs_per_count=num_catalogs_per_count,
            ess_threshold_prop=0.5,
            resample_method="multinomial",
            max_smc_iters=100,
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
        estimated_total_flux[i] = agg.estimated_total_flux

        agg.summarize()
        print(f"\nruntime = {runtime[i]}\n\n\n")

    torch.save(runtime.cpu(), f"results/smc/runtime_{b}.pt")
    torch.save(num_iters.cpu(), f"results/smc/num_iters_{b}.pt")
    torch.save(counts.cpu(), f"results/smc/counts_{b}.pt")
    torch.save(locs.cpu(), f"results/smc/locs_{b}.pt")
    torch.save(fluxes.cpu(), f"results/smc/fluxes_{b}.pt")
    torch.save(estimated_total_flux.cpu(), f"results/smc/total_flux_{b}.pt")

print("Done!")
##############################################
