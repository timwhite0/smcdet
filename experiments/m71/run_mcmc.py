#!/usr/bin/env python

##############################################
# SETUP

import sys

sys.path.append("/home/twhit/smcdet/")

import os
import pickle
import time

import numpy as np
import torch

from smcdet.images import M71ImageModel
from smcdet.prior import M71Prior
from smcdet.sampler import MHsampler

device = "cpu"
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
    min_objects=10,
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
    psf_radius=params["psf_radius"],
    noise_additive=params["noise_additive"],
    noise_multiplicative=params["noise_multiplicative"],
)
##############################################

##############################################
#

num_samples_total = 50000
num_samples_burnin = 30000
keep_every_k = 2
num_samples_stored = (num_samples_total - num_samples_burnin) // keep_every_k

batch_size = 10
num_batches = num_images // batch_size
##############################################

##############################################
# RUN MH

batch_index = int(os.getenv("BATCH_INDEX", 0))
batch_min = batch_index
batch_max = batch_index + 1

torch.manual_seed(batch_index)
np.random.seed(batch_index)

for b in range(batch_min, batch_max):
    runtime = torch.zeros([batch_size])
    counts = torch.zeros([batch_size, num_samples_stored])
    locs = torch.zeros([batch_size, num_samples_stored, prior.max_objects, 2])
    fluxes = torch.zeros([batch_size, num_samples_stored, prior.max_objects])
    posterior_predictive_total_flux = torch.zeros([batch_size, num_samples_stored])

    for i in range(batch_size):
        image_index = b * batch_size + i

        print(f"image {image_index + 1} of {num_images}")
        print(f"Number of stars within image boundary: {true_counts[image_index]}")
        print(
            "Total intrinsic flux of stars within image boundary: ",
            f"{true_fluxes[image_index].sum(-1).round()}",
        )
        print(f"Total observed flux: {tiles[image_index].sum().round()}\n")

        sampler = MHsampler(
            image=tiles[image_index],
            tile_dim=tile_dim,
            Prior=prior,
            ImageModel=imagemodel,
            locs_stdev=0.1,
            fluxes_stdev=2.5,
            flux_detection_threshold=params["flux_detection_threshold"],
            num_samples_total=num_samples_total,
            num_samples_burnin=num_samples_burnin,
            keep_every_k=keep_every_k,
            print_every=1000,
        )

        start = time.perf_counter()

        sampler.run()

        end = time.perf_counter()

        sampler.summarize()

        runtime[i] = end - start
        counts[i] = sampler.counts.squeeze([0, 1])
        index = sampler.locs.shape[-2]
        locs[i, :, :index, :] = sampler.locs.squeeze([0, 1])
        fluxes[i, :, :index] = sampler.fluxes.squeeze([0, 1])
        posterior_predictive_total_flux[i] = (
            sampler.posterior_predictive_total_observed_flux
        )
        print(f"\nruntime = {runtime[i]}\n\n\n")

    torch.save(runtime.cpu(), f"/data/scratch/twhit/m71_results/mcmc/runtime_{b}.pt")
    torch.save(counts.cpu(), f"/data/scratch/twhit/m71_results/mcmc/counts_{b}.pt")
    torch.save(locs.cpu(), f"/data/scratch/twhit/m71_results/mcmc/locs_{b}.pt")
    torch.save(fluxes.cpu(), f"/data/scratch/twhit/m71_results/mcmc/fluxes_{b}.pt")
    torch.save(
        posterior_predictive_total_flux.cpu(),
        f"/data/scratch/twhit/m71_results/mcmc/posterior_predictive_total_flux_{b}.pt",
    )

print("Done!")
##############################################
