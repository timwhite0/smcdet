#!/usr/bin/env python

#################################
### LIBRARIES

import sys
sys.path.append("../..")

from smc.prior import CellPrior
from smc.images import ImageAttributes
import torch

# torch.cuda.is_available()
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
torch.set_default_device(device)
#################################

#################################
### IMAGE ATTRIBUTES

img_dim = 32
img_height = img_dim
img_width = img_dim
max_objects = 4
psf_size = 5
psf_stdev = 1
background = 10

prior = CellPrior(max_objects, img_height, img_width, min_fluor = background)

image_attributes = ImageAttributes(img_height, img_width, max_objects,
                                   psf_size, psf_stdev, background)
#################################

#################################
### GENERATE IMAGES

torch.manual_seed(882)

num_images = 1000

true_counts, true_fluors, true_locs, true_axes, true_angles, true_total_intensities, images = image_attributes.generate(prior, num_images)

sys.path.append("experiments/wastefree")
torch.save(true_counts.cpu(), "images/true_counts.pt")
torch.save(true_fluors.cpu(), "images/true_fluors.pt")
torch.save(true_locs.cpu(), "images/true_locs.pt")
torch.save(true_axes.cpu(), "images/true_axes.pt")
torch.save(true_angles.cpu(), "images/true_angles.pt")
torch.save(true_total_intensities.cpu(), "images/true_total_intensities.pt")
torch.save(images.cpu(), "images/images.pt")
#################################