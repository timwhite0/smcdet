#!/usr/bin/env python

##############################################
# SETUP

import sys

sys.path.append("/home/twhit/smc_object_detection/")

import torch

from smc.images import ImageModel
from smc.prior import StarPrior
from utils.misc import select_cuda_device

device = select_cuda_device()
torch.cuda.set_device(device)
torch.set_default_device(device)
##############################################

##############################################
# DEFINE AN IMAGE MODEL AND A PRIOR

# image attributes
image_dim = 32
psf_stdev = 1.0
background = 300

imagemodel = ImageModel(
    image_height=image_dim,
    image_width=image_dim,
    psf_stdev=psf_stdev,
    background=background,
)

# prior
max_objects = 20
flux_mean = 1300
flux_stdev = 250
pad = 2
prior = StarPrior(
    max_objects=max_objects + 8,
    image_height=image_dim,
    image_width=image_dim,
    flux_mean=flux_mean,
    flux_stdev=flux_stdev,
    pad=pad,
)
##############################################

##############################################
# GENERATE IMAGES

torch.manual_seed(1)

num_images = 1000

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

torch.save(true_counts.cpu(), "data/true_counts.pt")
torch.save(true_locs.cpu(), "data/true_locs.pt")
torch.save(true_fluxes.cpu(), "data/true_fluxes.pt")
torch.save(images.cpu(), "data/images.pt")
##############################################
