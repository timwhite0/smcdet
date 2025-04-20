#!/usr/bin/env python

##############################################
# SETUP

import sys

sys.path.append("/home/twhit/smc_object_detection/")

import pickle
import time

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

images = torch.load("data/images.pt").to(device)
unpruned_counts = torch.load("data/unpruned_counts.pt").to(device)
pruned_counts = torch.load("data/pruned_counts.pt").to(device)
unpruned_fluxes = torch.load("data/unpruned_fluxes.pt").to(device)
pruned_fluxes = torch.load("data/pruned_fluxes.pt").to(device)

num_images = images.shape[0]
image_height = images.shape[1]
image_width = images.shape[2]
##############################################

##############################################
# SPECIFY TILE-LEVEL IMAGE MODEL AND PRIOR

with open("../m71_manyimages/data/params.pkl", "rb") as f:
    params = pickle.load(f)

tile_dim = 8
pad = 1
noise_scale = 1.0

prior = M71Prior(
    max_objects=6,
    counts_rate=params["counts_rate"] * ((tile_dim + 2 * pad) ** 2) / (tile_dim**2),
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
    flux_calibration=params["flux_calibration"],
    psf_params=params["psf_params"],
    noise_scale=noise_scale,
)
##############################################

##############################################
# SPECIFY NUMBER OF CATALOGS AND NUMBER OF RUNS

num_catalogs_per_count = [2500, 5000, 10000]
num_catalogs = (prior.max_objects + 1) * num_catalogs_per_count

num_mh_iters = [25, 50, 100]

num_runs = 100
##############################################

##############################################
# RUN SMC

torch.manual_seed(2)

for i in range(num_images):
    print(f"image {i + 1} of {num_images}")
    print(f"Number of stars including padding: {unpruned_counts[i]}")
    print(f"Number of stars within image boundary: {pruned_counts[i]}")
    print(
        "Total intrinsic flux of all stars (including padding): ",
        f"{unpruned_fluxes[i].sum(-1).round()}",
    )
    print(
        "Total intrinsic flux of stars within image boundary: ",
        f"{pruned_fluxes[i].sum(-1).round()}",
    )
    print(f"Total observed flux: {images[i].sum().round()}\n")

    for c in range(len(num_catalogs_per_count)):
        num_catalogs = (prior.max_objects + 1) * num_catalogs_per_count[c]

        for m in range(len(num_mh_iters)):
            print(f"{num_catalogs_per_count[c]} catalogs per count")
            print(f"{num_mh_iters[m]} MH iterations\n")

            runtime = torch.zeros([num_runs])
            num_iters = torch.zeros([num_runs])
            log_normalizing_constants = torch.zeros([num_runs, prior.max_objects + 1])
            total_intrinsic_flux = torch.zeros([num_runs, num_catalogs])

            mh = SingleComponentMH(
                num_iters=num_mh_iters[m],
                locs_stdev=0.1,
                fluxes_stdev=5,
                fluxes_min=prior.flux_lower,
                fluxes_max=prior.flux_upper,
            )

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
                    mh,
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
                log_normalizing_constants[r] = sampler.log_normalizing_constant.squeeze(
                    [0, 1]
                )
                total_intrinsic_flux[r] = agg.fluxes.sum(-1).squeeze([0, 1])

                agg.summarize()
                print(f"\nruntime = {runtime[r]}\n\n\n")

            torch.save(
                runtime.cpu(),
                f"results/smc/runtime_image{i+1}_cats{num_catalogs_per_count[c]}_mh{num_mh_iters[m]}.pt",  # noqa: E501
            )
            torch.save(
                num_iters.cpu(),
                f"results/smc/num_iters_image{i+1}_cats{num_catalogs_per_count[c]}_mh{num_mh_iters[m]}.pt",  # noqa: E501
            )
            torch.save(
                log_normalizing_constants.cpu(),
                f"results/smc/log_norm_const_image{i+1}_cats{num_catalogs_per_count[c]}_mh{num_mh_iters[m]}.pt",  # noqa: E501
            )
            torch.save(
                total_intrinsic_flux.cpu(),
                f"results/smc/total_intrinsic_flux_image{i+1}_cats{num_catalogs_per_count[c]}_mh{num_mh_iters[m]}.pt",  # noqa: E501
            )

print("Done!")
##############################################
