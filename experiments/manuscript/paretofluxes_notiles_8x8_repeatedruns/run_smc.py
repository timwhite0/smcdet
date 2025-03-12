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
# SPECIFY NUMBER OF CATALOGS AND NUMBER OF RUNS

num_catalogs_per_count = [500, 2000, 5000]
num_catalogs = (prior.max_objects + 1) * num_catalogs_per_count

num_runs = 100
##############################################

##############################################
# RUN SMC

torch.manual_seed(2)

for i in range(num_images):
    print(f"image {i + 1} of {num_images}")
    print(f"true count = {true_counts[i]}")
    print(f"true total flux = {true_fluxes[i].sum()}\n")

    for c in range(len(num_catalogs_per_count)):
        num_catalogs = (prior.max_objects + 1) * num_catalogs_per_count[c]
        print(f"{num_catalogs_per_count[c]} catalogs per count\n")

        runtime = torch.zeros([num_runs])
        num_iters = torch.zeros([num_runs])
        pruned_counts = torch.zeros([num_runs, num_catalogs])
        log_normalizing_constants = torch.zeros([num_runs, prior.max_objects + 1])
        estimated_total_flux = torch.zeros([num_runs, num_catalogs])

        for r in range(num_runs):
            print(f"run {r + 1} of {num_runs}\n")

            sampler = SMCsampler(
                image=images[i],
                tile_dim=tile_dim,
                Prior=prior,
                ImageModel=imagemodel,
                MutationKernel=mh,
                num_catalogs_per_count=num_catalogs_per_count[c],
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

            runtime[r] = end - start
            num_iters[r] = sampler.iter
            pruned_counts[r] = agg.pruned_counts.squeeze([0, 1])
            log_normalizing_constants[r] = sampler.log_normalizing_constant.squeeze(
                [0, 1]
            )
            estimated_total_flux[r] = agg.estimated_total_flux

            agg.summarize()
            print(f"\nruntime = {runtime[r]}\n\n\n")

        torch.save(
            runtime.cpu(),
            f"results/smc/runtime_image{i+1}_catalogs{num_catalogs_per_count[c]}.pt",
        )
        torch.save(
            num_iters.cpu(),
            f"results/smc/num_iters_image{i+1}_catalogs{num_catalogs_per_count[c]}.pt",
        )
        torch.save(
            pruned_counts.cpu(),
            f"results/smc/pruned_counts_image{i+1}_catalogs{num_catalogs_per_count[c]}.pt",
        )
        torch.save(
            log_normalizing_constants.cpu(),
            f"results/smc/log_norm_const_image{i+1}_catalogs{num_catalogs_per_count[c]}.pt",
        )
        torch.save(
            estimated_total_flux.cpu(),
            f"results/smc/total_flux_image{i+1}_catalogs{num_catalogs_per_count[c]}.pt",
        )

print("Done!")
##############################################
