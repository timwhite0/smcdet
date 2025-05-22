#!/usr/bin/env python

##############################################
# SETUP

import sys

sys.path.append("/home/twhit/smc_object_detection/")

import pickle
import time

import numpy as np
import sep
import torch
from hydra import compose, initialize
from hydra.utils import instantiate

from smc.images import M71ImageModel
from smc.metrics import compute_precision_recall_f1, match_catalogs
from smc.prior import M71Prior

##############################################

##############################################
# TUNE SEP HYPERPARAMETERS USING F1 ON A NEW BATCH OF 200 TILES

with open("../../m71/manyimages/data/params.pkl", "rb") as f:
    params = pickle.load(f)

image_dim = 8
pad = 1

prior = M71Prior(
    max_objects=100,
    counts_rate=params["counts_rate"],
    image_height=image_dim,
    image_width=image_dim,
    flux_alpha=params["flux_alpha"],
    flux_lower=params["flux_lower"],
    flux_upper=params["flux_upper"],
    pad=pad,
)

imagemodel = M71ImageModel(
    image_height=image_dim,
    image_width=image_dim,
    background=params["background"],
    adu_per_nmgy=params["adu_per_nmgy"],
    psf_params=params["psf_params"],
    noise_additive=params["noise_additive"],
    noise_multiplicative=params["noise_multiplicative"],
)

torch.manual_seed(42)

num_images = 200

(
    _,
    _,
    _,
    true_counts,
    true_locs,
    true_fluxes,
    tiles,
) = imagemodel.generate(Prior=prior, num_images=num_images)

with open("../m71_manyimages/data/params.pkl", "rb") as f:
    params = pickle.load(f)

num_images = tiles.shape[0]
image_height = tiles.shape[1]
image_width = tiles.shape[2]
background = params["background"]
adu_per_nmgy = params["adu_per_nmgy"]
psf_params = params["psf_params"]
noise_additive = params["noise_additive"]
noise_multiplicative = params["noise_multiplicative"]
max_detections = 50

with initialize(config_path="../m71_manyimages/", version_base=None):
    cfg = compose(config_name="config")

sdss = instantiate(cfg.surveys.sdss)
sdss.prepare_data()
# trim SDSS PSF to 5x5
sdss_psf = sdss.psf.psf_galsim[sdss.image_id(0)][2].original.image.array[10:15, 10:15]

mag_bins = torch.arange(14.0, 22.5, 8)  # we'll compute F1 for the bin [14.0, 22.5)


print("Starting grid search...\n")

thresh = torch.arange(start=1.0, end=8.5, step=0.5)
minarea = torch.linspace(start=1, end=7, steps=7)
deblend_cont = torch.logspace(start=-10, end=-2, steps=5)
clean_param = torch.logspace(start=-1, end=2, steps=4)

sep_f1 = torch.zeros(
    thresh.shape[0], minarea.shape[0], deblend_cont.shape[0], clean_param.shape[0]
)

for t in range(thresh.shape[0]):
    for m in range(minarea.shape[0]):
        for d in range(deblend_cont.shape[0]):
            for c in range(clean_param.shape[0]):
                print(f"thresh = {thresh[t]}")
                print(f"minarea = {minarea[m]}")
                print(f"deblend_cont = {deblend_cont[d]}")
                print(f"clean_param = {clean_param[c]}")

                counts = torch.zeros(num_images)
                locs = torch.zeros(num_images, max_detections, 2)
                fluxes = torch.zeros(num_images, max_detections)

                for i in range(num_images):
                    sep_results = sep.extract(
                        (tiles[i] - background).cpu().numpy(),
                        thresh=thresh[t],
                        minarea=minarea[m],
                        deblend_cont=deblend_cont[d],
                        deblend_nthresh=64,
                        filter_kernel=None,  # no filter works better than using the SDSS PSF
                        clean=True,
                        var=np.sqrt(background),
                        clean_param=clean_param[c],
                    )

                    counts[i] = len(sep_results)
                    locs[i, : counts[i].int(), 0] = (
                        torch.from_numpy(sep_results["y"]) + 0.5
                    )  # match SMC locs convention
                    locs[i, : counts[i].int(), 1] = (
                        torch.from_numpy(sep_results["x"]) + 0.5
                    )  # match SMC locs convention
                    fluxes[i, : counts[i].int()] = (
                        torch.from_numpy(sep_results["flux"]) / adu_per_nmgy
                    )

                matching_results = match_catalogs(
                    true_counts,
                    true_locs,
                    true_fluxes,
                    counts.unsqueeze(-1),
                    locs.unsqueeze(-2),
                    fluxes.unsqueeze(-1),
                    1,
                    0.5,
                    0.5,
                    mag_bins,
                )
                true_total, true_matches, est_total, est_matches = matching_results
                precision, recall, f1 = compute_precision_recall_f1(
                    true_total, true_matches, est_total, est_matches
                )
                sep_f1[t, m, d, c] = f1[0][-1]
                print(f"f1 = {sep_f1[t,m,d,c].item()}\n")

for t in range(thresh.shape[0]):
    for m in range(minarea.shape[0]):
        for d in range(deblend_cont.shape[0]):
            for c in range(clean_param.shape[0]):
                if sep_f1[t, m, d, c] == sep_f1.max():
                    thresh_best = thresh[t]
                    minarea_best = minarea[m]
                    deblend_cont_best = deblend_cont[d]
                    clean_param_best = clean_param[c]
                    break

print("Hyperparameters selected:")
print(f"thresh = {thresh_best.item()}")
print(f"minarea = {minarea_best.item()}")
print(f"deblend_cont = {deblend_cont_best}")
print(f"clean_param = {clean_param_best}\n")
##############################################

##############################################
# RUN SEP WITH OPTIMAL HYPERPARAMETERS ON THE SAME TILES AS SMC

print("Running SEP...\n")

tiles_test = torch.load("data/images.pt")  # transform to raw tiles
num_images = tiles_test.shape[0]
runtime = torch.zeros(num_images)
sep_counts = torch.zeros(num_images)
sep_locs = torch.zeros(num_images, max_detections, 2)
sep_fluxes = torch.zeros(num_images, max_detections)

for i in range(num_images):
    start = time.perf_counter()

    sep_results = sep.extract(
        (tiles_test[i] - background).cpu().numpy(),
        thresh=thresh_best,
        minarea=minarea_best,
        deblend_cont=deblend_cont_best,
        deblend_nthresh=64,
        filter_kernel=None,  # no filter works better than using the SDSS PSF
        clean=True,
        var=np.sqrt(background),
        clean_param=clean_param_best,
    )

    end = time.perf_counter()

    runtime[i] = end - start
    sep_counts[i] = len(sep_results)
    count = sep_counts[i].int().item()
    sep_locs[i, :count, 0] = (
        torch.from_numpy(sep_results["y"]) + 0.5
    )  # match SMC locs convention
    sep_locs[i, :count, 1] = (
        torch.from_numpy(sep_results["x"]) + 0.5
    )  # match SMC locs convention
    sep_fluxes[i, :count] = torch.from_numpy(sep_results["flux"]) / adu_per_nmgy

# remove unnecessary trailing zeros
sep_locs = sep_locs[:, : sep_counts.max().int().item(), :]
sep_fluxes = sep_fluxes[:, : sep_counts.max().int().item()]
##############################################

##############################################
# SAVE RESULTS

print("Saving results...\n")

torch.save(runtime.cpu(), "results/sep/runtime.pt")
torch.save(sep_counts.cpu(), "results/sep/counts.pt")
torch.save(sep_locs.cpu(), "results/sep/locs.pt")
torch.save(sep_fluxes.cpu(), "results/sep/fluxes.pt")

print("Done!")
##############################################
