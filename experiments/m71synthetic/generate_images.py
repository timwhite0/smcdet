#!/usr/bin/env python

##############################################
# SETUP

import sys

sys.path.append("/home/twhit/smcdet/")

import pickle

import torch

from smcdet.images import M71ImageModel, generate_images
from smcdet.prior import M71Prior
from utils.misc import select_cuda_device

device = select_cuda_device()
torch.cuda.set_device(device)
torch.set_default_device(device)
##############################################

##############################################
with open("../m71/data/params.pkl", "rb") as f:
    params = pickle.load(f)

image_dim = 8
pad = 4

prior = M71Prior(
    min_objects=0,
    max_objects=100,
    counts_rate=params["counts_rate"],
    image_height=image_dim,
    image_width=image_dim,
    flux_alpha=params["flux_alpha"],
    flux_lower=params[
        "flux_detection_threshold"
    ],  # counts_rate was fit using only detectable stars
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
##############################################

##############################################
torch.manual_seed(0)

num_images = 1000

res = generate_images(
    Prior=prior,
    ImageModel=imagemodel,
    flux_threshold=params["flux_detection_threshold"],
    loc_threshold_lower=0,
    loc_threshold_upper=image_dim,
    num_images=1000,
)
(
    unpruned_counts,
    unpruned_locs,
    unpruned_fluxes,
    pruned_counts,
    pruned_locs,
    pruned_fluxes,
    images,
) = res

torch.save(pruned_counts.cpu(), "data/pruned_counts.pt")
torch.save(pruned_locs.cpu(), "data/pruned_locs.pt")
torch.save(pruned_fluxes.cpu(), "data/pruned_fluxes.pt")
torch.save(unpruned_counts.cpu(), "data/unpruned_counts.pt")
torch.save(unpruned_locs.cpu(), "data/unpruned_locs.pt")
torch.save(unpruned_fluxes.cpu(), "data/unpruned_fluxes.pt")
torch.save(images.cpu(), "data/images.pt")
##############################################
