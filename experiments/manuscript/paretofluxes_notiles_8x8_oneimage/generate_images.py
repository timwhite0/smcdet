#!/usr/bin/env python

##############################################
# SETUP

import sys

sys.path.append("/home/twhit/smc_object_detection/")

import numpy as np
import torch

from smc.images import ImageModel
from smc.prior import ParetoStarPrior
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
pad = 1
prior = ParetoStarPrior(
    max_objects=max_objects + 8,
    image_height=image_dim,
    image_width=image_dim,
    flux_scale=flux_scale,
    flux_alpha=flux_alpha,
    pad=pad,
)
##############################################

##############################################
# GENERATE IMAGES

torch.manual_seed(121)

num_images = 100

true_counts, true_locs, true_fluxes, images = imagemodel.generate(
    Prior=prior, num_images=10 * num_images
)

index = true_counts <= max_objects
true_counts = true_counts[index]
true_locs = true_locs[index]
true_fluxes = true_fluxes[index]
images = images[index]

probs = 1 / true_counts.unique(return_counts=True)[1]
index = torch.multinomial(probs[true_counts], num_samples=num_images, replacement=False)
true_counts = true_counts[index]
true_locs = true_locs[index]
true_fluxes = true_fluxes[index]
images = images[index]

# select one image each with count = 2, 4, 6, 8
indexes = [
    torch.arange(images.shape[0])[true_counts == 2][3].item(),
    torch.arange(images.shape[0])[true_counts == 4][3].item(),
    torch.arange(images.shape[0])[true_counts == 6][3].item(),
    torch.arange(images.shape[0])[true_counts == 8][3].item(),
]

torch.save(true_counts[indexes].cpu(), "data/true_counts.pt")
torch.save(true_locs[indexes].cpu(), "data/true_locs.pt")
torch.save(true_fluxes[indexes].cpu(), "data/true_fluxes.pt")
torch.save(images[indexes].cpu(), "data/images.pt")
##############################################
