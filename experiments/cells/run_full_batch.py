#!/usr/bin/env python

#################################
### LIBRARIES

import sys
sys.path.append("../..")

from smc.sampler import SMCsampler
from smc.prior import CellPrior
from smc.images import ImageAttributes

import torch
# torch.cuda.is_available()
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
torch.set_default_device(device)

import time
#################################

#################################
### LOAD IMAGES

sys.path.append("experiments/cells")
true_counts = torch.load("images/true_counts.pt").to(device)
true_fluors = torch.load("images/true_fluors.pt").to(device)
images = torch.load("images/images.pt").to(device)

num_images = images.shape[0]
img_height = images.shape[1]
img_width = images.shape[2]
psf_size = 5
psf_stdev = 1
background = 10

image_attributes = ImageAttributes(img_height = img_height,
                                   img_width = img_width,
                                   max_objects = true_counts.max().item(),
                                   psf_size = psf_size,
                                   psf_stdev = psf_stdev,
                                   background = background)
#################################

#################################
### SAMPLER SETTINGS AND PRIOR

tile_side_length = img_height
kernel_num_iters = 100
catalogs_per_block = 2000
max_objects_smc = 4

prior = CellPrior(max_objects_smc,
                  tile_side_length, tile_side_length,
                  min_fluor = 2 * background,
                  pad = 0)

posterior_mean_count = torch.zeros(num_images)
posterior_mean_total_fluor = torch.zeros(num_images)
reconstructed_image = torch.zeros(num_images, img_height, img_width)
runtime = torch.zeros(num_images)
num_iters = torch.zeros(num_images)
#################################

#################################
### RUN SAMPLER

torch.manual_seed(882)

for i in range(num_images):
    print(f"image {i+1} of {num_images}")
    print(f"True count: {true_counts[i]}")
    print(f"True total fluorescence: {true_fluors[i].sum()}\n")
    
    smc = SMCsampler(images[i], image_attributes, tile_side_length, prior,
                     max_objects = max_objects_smc,
                     catalogs_per_block = catalogs_per_block,
                     kernel_num_iters = kernel_num_iters,
                     product_form_multiplier = 1,
                     max_smc_iters = 500)
    
    start = time.time()
    smc.run(print_progress = True)
    end = time.time()
    
    posterior_mean_count[i] = smc.posterior_mean_count
    posterior_mean_total_fluor[i] = smc.posterior_mean_total_fluor
    reconstructed_image[i] = smc.reconstructed_image
    runtime[i] = end - start
    num_iters[i] = smc.iter
    
    print(f"image {i+1} took {(runtime[i]).round()} seconds\n")
    
    smc.summarize(display_images = False)
#################################

#################################
### WRITE TO RESULTS DIRECTORY

torch.save(posterior_mean_count, "results/posterior_mean_count.pt")
torch.save(posterior_mean_total_fluor, "results/posterior_mean_total_fluor.pt")
torch.save(reconstructed_image, "results/reconstructed_image.pt")
torch.save(runtime, "results/runtime.pt")
torch.save(num_iters, "results/num_iters.pt")
#################################