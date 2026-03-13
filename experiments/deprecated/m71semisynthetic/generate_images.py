#!/usr/bin/env python

##############################################
# SETUP

import sys

sys.path.append("/home/twhit/smcdet/")

import pickle

import torch
from einops import rearrange

from smcdet.images import M71ImageModel
from utils.misc import select_cuda_device

device = select_cuda_device()
torch.cuda.set_device(device)
torch.set_default_device(device)
##############################################

##############################################
# LOAD M71 CATALOGS

m71_unpruned_locs = torch.load("../m71/data/unpruned_locs_magcut.pt").to(device)
m71_unpruned_fluxes = torch.load("../m71/data/unpruned_fluxes_magcut.pt").to(device)

with open("../m71/data/params.pkl", "rb") as f:
    params = pickle.load(f)

image_dim = 8
pad = 4

imagemodel = M71ImageModel(
    image_height=image_dim,
    image_width=image_dim,
    background=params["background"],
    adu_per_nmgy=params["adu_per_nmgy"],
    psf_params=params["psf_params"],
    psf_radius=params["psf_radius"],
    noise_additive=params["noise_additive"],
    noise_multiplicative=params["noise_multiplicative"],
)
##############################################

##############################################
# GENERATE SEMI-SYNTHETIC IMAGES
torch.manual_seed(42)

images = imagemodel.sample(
    rearrange(m71_unpruned_locs, "n d t -> 1 1 n d t"),
    rearrange(m71_unpruned_fluxes, "n d -> 1 1 n d"),
)
images = rearrange(images.squeeze([0, 1]), "dimH dimW n -> n dimH dimW")

torch.save(images.cpu(), "data/images.pt")
##############################################
