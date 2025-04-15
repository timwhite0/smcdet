#!/usr/bin/env python

##############################################
# SETUP

import sys

sys.path.append("/home/twhit/smc_object_detection/")

import time

import sep
import torch
from torch.nn.functional import pad

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
background = 491.5867919921875
padding = (
    0  # if > 0, image is reflectively padded to allow detections outside the boundary
)
padded_images = pad(tiles, (padding, padding, padding, padding), mode="reflect")
max_detections = 50
##############################################

##############################################
# SELECT SEP HYPERPARAMETERS VIA GRID SEARCH

print("Starting grid search...\n")

thresh = torch.linspace(start=20, end=65, steps=10)
minarea = torch.linspace(start=1, end=3, steps=3)
deblend_cont = torch.logspace(start=-5, end=-3, steps=3)
clean_param = torch.logspace(start=1, end=3, steps=3)

pruned_count = torch.zeros(
    thresh.shape[0],
    minarea.shape[0],
    deblend_cont.shape[0],
    clean_param.shape[0],
    num_images,
)
sep_mse = torch.zeros(
    thresh.shape[0], minarea.shape[0], deblend_cont.shape[0], clean_param.shape[0]
)

for t in range(thresh.shape[0]):
    for m in range(minarea.shape[0]):
        for d in range(deblend_cont.shape[0]):
            for c in range(clean_param.shape[0]):
                print(f"thresh = {thresh[t]}")
                print(f"minarea = {minarea[m]}")
                print(f"deblend_cont = {deblend_cont[d]}")
                print(f"clean_param = {clean_param[c]}\n")

                for i in range(num_images):
                    sep_results = sep.extract(
                        (padded_images[i] - background).cpu().numpy(),
                        thresh=thresh[t],
                        minarea=minarea[m],
                        deblend_cont=deblend_cont[d],
                        deblend_nthresh=64,
                        filter_kernel=None,
                        clean=True,
                        clean_param=clean_param[c],
                    )

                    unpruned_count = len(sep_results)
                    locs = torch.zeros(max_detections, 2)
                    locs[:unpruned_count, 0] = (
                        torch.from_numpy(sep_results["y"]) - padding + 0.5
                    )  # match SMC locs convention
                    locs[:unpruned_count, 1] = (
                        torch.from_numpy(sep_results["x"]) - padding + 0.5
                    )  # match SMC locs convention

                    in_bounds = torch.all(
                        torch.logical_and(
                            locs > 0, locs < torch.tensor((image_height, image_width))
                        ),
                        dim=-1,
                    )
                    pruned_count[t, m, d, c, i] = in_bounds.sum(-1).float()

                sep_mse[t, m, d, c] = (
                    (pruned_count[t, m, d, c, :] - true_counts) ** 2
                ).mean()
                print(f"mse = {sep_mse[t,m,d,c]}\n\n")

for t in range(thresh.shape[0]):
    for m in range(minarea.shape[0]):
        for d in range(deblend_cont.shape[0]):
            for c in range(clean_param.shape[0]):
                if sep_mse[t, m, d, c] == sep_mse.min():
                    thresh_best = thresh[t]
                    minarea_best = minarea[m]
                    deblend_cont_best = deblend_cont[d]
                    clean_param_best = clean_param[c]

print("Hyperparameters selected:")
print(f"thresh = {thresh_best.item()}")
print(f"minarea = {minarea_best.item()}")
print(f"deblend_cont = {deblend_cont_best}")
print(f"clean_param = {clean_param_best}\n")
##############################################

##############################################
# RUN SEP

print("Running SEP...\n")

runtime = torch.zeros(num_images)
unpruned_counts = torch.zeros(num_images)
unpruned_locs = torch.zeros(num_images, max_detections, 2)
unpruned_fluxes = torch.zeros(num_images, max_detections)

for i in range(num_images):
    start = time.perf_counter()

    sep_results = sep.extract(
        (padded_images[i] - background).cpu().numpy(),
        thresh=thresh_best,
        minarea=minarea_best,
        deblend_cont=deblend_cont_best,
        deblend_nthresh=2000,
        filter_kernel=None,
        clean=True,
        clean_param=clean_param_best,
    )

    end = time.perf_counter()

    runtime[i] = end - start
    unpruned_counts[i] = len(sep_results)
    unpruned_count = unpruned_counts[i].int().item()
    unpruned_locs[i, :unpruned_count, 0] = (
        torch.from_numpy(sep_results["y"]) - padding + 0.5
    )  # match SMC locs convention
    unpruned_locs[i, :unpruned_count, 1] = (
        torch.from_numpy(sep_results["x"]) - padding + 0.5
    )  # match SMC locs convention
    unpruned_fluxes[i, :unpruned_count] = torch.from_numpy(sep_results["flux"])

# remove unnecessary trailing zeros
unpruned_locs = unpruned_locs[:, : unpruned_counts.max().int().item(), :]
unpruned_fluxes = unpruned_fluxes[:, : unpruned_counts.max().int().item()]
##############################################

##############################################
# SAVE RESULTS

print("Saving results...\n")

torch.save(runtime.cpu(), "results/sep/runtime.pt")
torch.save(unpruned_counts.cpu(), "results/sep/counts.pt")
torch.save(unpruned_locs.cpu(), "results/sep/locs.pt")
torch.save(unpruned_fluxes.cpu(), "results/sep/fluxes.pt")

print("Done!")
##############################################
