#!/usr/bin/env python

##############################################
# SETUP

import sys

sys.path.append("/home/twhit/smcdet/")

import pickle

import torch

from smcdet.images import M71ImageModel
from smcdet.prior import M71Prior
from utils.misc import select_cuda_device

device = select_cuda_device()
torch.cuda.set_device(device)
torch.set_default_device(device)
##############################################

##############################################
with open("../../m71/manyimages/data/params.pkl", "rb") as f:
    params = pickle.load(f)

image_dim = 8
pad = 1

prior = M71Prior(
    max_objects=20,
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
##############################################

##############################################
# GENERATE IMAGES

torch.manual_seed(3)

num_images = 100

(
    unpruned_counts,
    unpruned_locs,
    unpruned_fluxes,
    pruned_counts,
    pruned_locs,
    pruned_fluxes,
    images,
) = imagemodel.generate(Prior=prior, num_images=num_images)

# select one image each with count (including padding) = 1, 3
indexes = [
    torch.arange(images.shape[0])[unpruned_counts == 1][0].item(),
    torch.arange(images.shape[0])[unpruned_counts == 3][0].item(),
]

torch.save(pruned_counts[indexes].cpu(), "data/pruned_counts.pt")
torch.save(pruned_locs[indexes].cpu(), "data/pruned_locs.pt")
torch.save(pruned_fluxes[indexes].cpu(), "data/pruned_fluxes.pt")
torch.save(unpruned_counts[indexes].cpu(), "data/unpruned_counts.pt")
torch.save(unpruned_locs[indexes].cpu(), "data/unpruned_locs.pt")
torch.save(unpruned_fluxes[indexes].cpu(), "data/unpruned_fluxes.pt")
torch.save(images[indexes].cpu(), "data/images.pt")
##############################################
