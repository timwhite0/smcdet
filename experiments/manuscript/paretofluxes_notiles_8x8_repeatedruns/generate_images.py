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

# choose padding s.t. 0.1-quantile-flux star at a distance of pad pixels outside
# the boundary contributes the same as a min-flux star at the boundary
quantile01_flux = flux_scale * ((1 - 0.1) ** (-1 / flux_alpha))
pad = np.sqrt(-2 * (psf_stdev**2) * np.log(flux_scale / quantile01_flux))

prior = ParetoStarPrior(
    max_objects=max_objects,
    image_height=image_dim,
    image_width=image_dim,
    flux_scale=flux_scale,
    flux_alpha=flux_alpha,
    pad=pad,
)
##############################################

##############################################
# GENERATE IMAGES

torch.manual_seed(2)

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

# select one image each with count (including padding) = 3, 6
indexes = [
    torch.arange(images.shape[0])[unpruned_counts == 3][0].item(),
    torch.arange(images.shape[0])[unpruned_counts == 6][0].item(),
]

torch.save(pruned_counts[indexes].cpu(), "data/pruned_counts.pt")
torch.save(pruned_locs[indexes].cpu(), "data/pruned_locs.pt")
torch.save(pruned_fluxes[indexes].cpu(), "data/pruned_fluxes.pt")
torch.save(unpruned_counts[indexes].cpu(), "data/unpruned_counts.pt")
torch.save(unpruned_locs[indexes].cpu(), "data/unpruned_locs.pt")
torch.save(unpruned_fluxes[indexes].cpu(), "data/unpruned_fluxes.pt")
torch.save(images[indexes].cpu(), "data/images.pt")
##############################################
