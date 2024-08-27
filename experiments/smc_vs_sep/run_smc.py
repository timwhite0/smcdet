#!/usr/bin/env python

##############################################
# SETUP

import sys
import time

import torch

from smc.aggregate import Aggregate
from smc.images import ImageModel
from smc.kernel import MetropolisHastings
from smc.prior import StarPrior
from smc.sampler import SMCsampler

sys.path.append("/home/twhit/smc_object_detection/")
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
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

prior = StarPrior(
    max_objects=10,
    image_height=image_height,
    image_width=image_width,
    flux_mean=80000,
    flux_stdev=15000,
    pad=2,
)

imagemodel = ImageModel(
    image_height=image_height, image_width=image_width, psf_stdev=1.5, background=100000
)

mh = MetropolisHastings(
    num_iters=75,
    locs_stdev=0.1,
    features_stdev=1000,
    features_min=50000,
    features_max=110000,
)
##############################################

##############################################
# CREATE EMPTY TENSORS TO STORE RESULTS

num_catalogs_per_count = 2500
num_catalogs = (prior.max_objects + 1) * num_catalogs_per_count

runtime = torch.zeros([num_images])
num_iters = torch.zeros([num_images])
counts = torch.zeros([num_images, num_catalogs])
locs = torch.zeros([num_images, num_catalogs, prior.max_objects, 2])
fluxes = torch.zeros([num_images, num_catalogs, prior.max_objects])
##############################################

##############################################
# RUN SMC

torch.manual_seed(1)

for i in range(num_images):
    print(f"image {i+1} of {num_images}")
    print(f"true count = {true_counts[i]}")
    print(f"true total flux = {true_fluxes[i].sum()}\n")

    sampler = SMCsampler(
        image=images[i],
        tile_dim=image_height,
        Prior=prior,
        ImageModel=imagemodel,
        MutationKernel=mh,
        num_catalogs_per_count=num_catalogs_per_count,
        max_smc_iters=200,
    )

    start = time.perf_counter()

    sampler.run(print_progress=True)

    agg = Aggregate(
        sampler.Prior,
        sampler.ImageModel,
        sampler.tiled_image,
        sampler.counts,
        sampler.locs,
        sampler.features,
        sampler.weights_intercount,
    )

    agg.run()

    end = time.perf_counter()

    runtime[i] = end - start
    num_iters[i] = sampler.iter
    counts[i] = agg.counts.squeeze([0, 1])
    locs[i] = agg.locs.squeeze([0, 1])
    fluxes[i] = agg.features.squeeze([0, 1])

    print(f"runtime = {runtime[i]}")
    print(f"num iters = {num_iters[i]}")
    print(f"posterior mean count = {agg.posterior_mean_counts.item()}")
    print(f"posterior mean total flux = {agg.posterior_mean_total_flux.item()}\n\n\n")

    torch.save(runtime[: (i + 1)].cpu(), "results/smc/runtime.pt")
    torch.save(num_iters[: (i + 1)].cpu(), "results/smc/num_iters.pt")
    torch.save(counts[: (i + 1)].cpu(), "results/smc/counts.pt")
    torch.save(locs[: (i + 1)].cpu(), "results/smc/locs.pt")
    torch.save(fluxes[: (i + 1)].cpu(), "results/smc/fluxes.pt")

print("Done!")
##############################################
