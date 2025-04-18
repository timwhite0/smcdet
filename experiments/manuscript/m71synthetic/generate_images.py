#!/usr/bin/env python

##############################################
# SETUP

import sys

sys.path.append("/home/twhit/smc_object_detection/")

import pickle

import torch

from smc.images import M71ImageModel
from smc.prior import M71Prior
from utils.misc import select_cuda_device

device = select_cuda_device()
torch.cuda.set_device(device)
torch.set_default_device(device)
##############################################

##############################################
with open("../m71/data/params.pkl", "rb") as f:
    params = pickle.load(f)

image_dim = 8
pad = 1

prior = M71Prior(
    max_objects=20,
    counts_rate=params["counts_rate"] * ((image_dim + 2 * pad) ** 2) / (image_dim**2),
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
    flux_calibration=params["flux_calibration"],
    psf_params=params["psf_params"],
    noise_scale=1.5,
)
##############################################

##############################################
torch.manual_seed(0)

num_images = 1000

(
    unpruned_counts,
    unpruned_locs,
    unpruned_fluxes,
    pruned_counts,
    pruned_locs,
    pruned_fluxes,
    images,
) = imagemodel.generate(Prior=prior, num_images=num_images)

torch.save(pruned_counts.cpu(), "data/pruned_counts.pt")
torch.save(pruned_locs.cpu(), "data/pruned_locs.pt")
torch.save(pruned_fluxes.cpu(), "data/pruned_fluxes.pt")
torch.save(unpruned_counts.cpu(), "data/unpruned_counts.pt")
torch.save(unpruned_locs.cpu(), "data/unpruned_locs.pt")
torch.save(unpruned_fluxes.cpu(), "data/unpruned_fluxes.pt")
torch.save(images.cpu(), "data/images.pt")
##############################################
