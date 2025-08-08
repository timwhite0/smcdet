#!/usr/bin/env python

##############################################
# SETUP

import sys

sys.path.append("/home/twhit/smc_object_detection/")

import pickle
import time

import numpy as np
import torch

from smcdet.aggregate import Aggregate
from smcdet.images import M71ImageModel
from smcdet.kernel import SingleComponentMH
from smcdet.prior import M71Prior
from smcdet.sampler import SMCsampler
from utils.misc import select_cuda_device

device = select_cuda_device()
torch.cuda.set_device(device)
torch.set_default_device(device)
##############################################

##############################################
# LOAD IN IMAGES AND CATALOGS

tiles = torch.load("data/tiles.pt").to(device)
true_counts = torch.load("data/pruned_counts_magcut.pt").to(device)
true_fluxes = torch.load("data/pruned_fluxes_magcut.pt").to(device)

num_images = tiles.shape[0]
image_height = tiles.shape[1]
image_width = tiles.shape[2]
##############################################

##############################################
# SPECIFY TILE-LEVEL IMAGE MODEL, PRIOR, AND MUTATION KERNEL

tile_dim = 8
pad = 4

with open("data/params.pkl", "rb") as f:
    params = pickle.load(f)

prior = M71Prior(
    min_objects=0,
    max_objects=10,
    counts_rate=params["counts_rate"],
    image_height=tile_dim,
    image_width=tile_dim,
    flux_alpha=params["flux_alpha"],
    flux_lower=params["flux_lower"],
    flux_upper=params["flux_upper"],
    pad=pad,
)

imagemodel = M71ImageModel(
    image_height=tile_dim,
    image_width=tile_dim,
    background=params["background"],
    adu_per_nmgy=params["adu_per_nmgy"],
    psf_params=params["psf_params"],
    noise_additive=params["noise_additive"],
    noise_multiplicative=params["noise_multiplicative"],
)

mh = SingleComponentMH(
    num_iters=100,
    locs_stdev=0.1,
    fluxes_stdev=5,
    fluxes_min=prior.flux_lower,
    fluxes_max=prior.flux_upper,
)

aggmh = SingleComponentMH(
    num_iters=100,
    locs_stdev=0.1,
    fluxes_stdev=5,
    fluxes_min=prior.flux_lower,
    fluxes_max=prior.flux_upper,
)
##############################################

##############################################
# SPECIFY NUMBER OF CATALOGS AND BATCH SIZE FOR SAVING RESULTS

num_catalogs_per_count = 10000
num_catalogs = (prior.max_objects - prior.min_objects + 1) * num_catalogs_per_count

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
