#!/usr/bin/env python

#################################
### LIBRARIES

import sys
sys.path.append("../..")

from smc.prior import CatalogPrior
from smc.images import ImageAttributes
import torch

# torch.cuda.is_available()
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
torch.set_default_device(device)
#################################

#################################
### IMAGE ATTRIBUTES FOR BOTH SETTINGS

img_dim = 16
img_width = img_dim
img_height = img_dim
max_objects = 8
psf_stdev = 3.25
background_intensity = 19200

prior = CatalogPrior(max_objects = max_objects,
                     img_height = img_height,
                     img_width = img_width,
                     min_flux = background_intensity/3.)

image_attributes = ImageAttributes(img_height = img_height,
                                   img_width = img_width,
                                   max_objects = max_objects,
                                   psf_stdev = psf_stdev,
                                   background_intensity = background_intensity)
#################################

#################################
### SETTING 1

torch.manual_seed(608)

# Generate 50 images, we'll select one each with s = 2, 4, 6, 8
num_images = 50

setting1_true_counts, setting1_true_fluxes, setting1_true_locs, setting1_true_total_intensities, setting1_images = image_attributes.generate(prior, num_images)

indexes = [torch.arange(num_images)[setting1_true_counts==2][0].item(),
           torch.arange(num_images)[setting1_true_counts==4][0].item(),
           torch.arange(num_images)[setting1_true_counts==6][0].item(),
           torch.arange(num_images)[setting1_true_counts==8][0].item()]

setting1_true_counts = setting1_true_counts[indexes]
setting1_true_fluxes = setting1_true_fluxes[indexes]
setting1_true_locs = setting1_true_locs[indexes]
setting1_true_total_intensities = setting1_true_total_intensities[indexes]
setting1_images = setting1_images[indexes]

sys.path.append("experiments/wastefree")
torch.save(setting1_true_counts, "images/setting1_true_counts.pt")
torch.save(setting1_true_fluxes, "images/setting1_true_fluxes.pt")
torch.save(setting1_true_locs, "images/setting1_true_locs.pt")
torch.save(setting1_true_total_intensities, "images/setting1_true_total_intensities.pt")
torch.save(setting1_images, "images/setting1_images.pt")
#################################

#################################
### SETTING 2

torch.manual_seed(608)

# Generate 1000
num_images = 1000

setting2_true_counts, setting2_true_fluxes, setting2_true_locs, setting2_true_total_intensities, setting2_images = image_attributes.generate(prior, num_images)

torch.save(setting2_true_counts, "images/setting2_true_counts.pt")
torch.save(setting2_true_fluxes, "images/setting2_true_fluxes.pt")
torch.save(setting2_true_locs, "images/setting2_true_locs.pt")
torch.save(setting2_true_total_intensities, "images/setting2_true_total_intensities.pt")
torch.save(setting2_images, "images/setting2_images.pt")
#################################


