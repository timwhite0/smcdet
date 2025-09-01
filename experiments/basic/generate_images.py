#!/usr/bin/env python

##############################################
# SETUP

import sys

sys.path.append("/home/twhit/smcdet/")

import numpy as np
import torch

from smcdet.images import ImageModel, generate_images
from smcdet.prior import ParetoStarPrior
from utils.misc import select_cuda_device

device = select_cuda_device()
torch.cuda.set_device(device)
torch.set_default_device(device)
##############################################

##############################################
# DEFINE AN IMAGE MODEL AND A PRIOR

# image attributes
image_dim = 8
psf_stdev = 0.93  # FWHM of SDSS PSF is 2.2
psf_max = 1 / (2 * np.pi * (psf_stdev**2))
background = 200  # arbitrary fixed background

imagemodel = ImageModel(
    image_height=image_dim,
    image_width=image_dim,
    psf_stdev=psf_stdev,
    background=background,
)

# prior
max_objects = 8
# make min flux an approximately 5sigma detection
flux_scale = 5 * np.sqrt(background) / psf_max
# choose alpha s.t. 0.99 quantile is an approximately 50sigma detection
flux_alpha = (-np.log(1 - 0.99)) / (
    np.log(50 * np.sqrt(background) / psf_max) - np.log(flux_scale)
)

# set padding width to 2 pixels
pad = 2

prior = ParetoStarPrior(
    min_objects=0,
    max_objects=max_objects,
    image_height=image_dim,
    image_width=image_dim,
    flux_scale=flux_scale * 0.9,  # generate stars fainter than detection threshold
    flux_alpha=flux_alpha,
    pad=pad,
)
##############################################

##############################################
# GENERATE IMAGES

torch.manual_seed(1)

num_images = 2000

res = generate_images(
    Prior=prior,
    ImageModel=imagemodel,
    flux_threshold=flux_scale,
    loc_threshold_lower=0,
    loc_threshold_upper=image_dim,
    num_images=num_images,
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
