#!/usr/bin/env python

#################################
### LIBRARIES

import sys
sys.path.append("../..")

from smc.sampler import SMCsampler
from smc.prior import CatalogPrior
from smc.images import ImageAttributes

import torch
# torch.cuda.is_available()
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
torch.set_default_device(device)

import matplotlib.pyplot as plt
import time
#################################

#################################
### LOAD IMAGES

sys.path.append("experiments/wastefree")
true_counts = torch.load("images/setting2_true_counts.pt").to(device)
true_fluxes = torch.load("images/setting2_true_fluxes.pt").to(device)
images = torch.load("images/setting2_images.pt").to(device)

num_images = images.shape[0]
img_height = images.shape[1]
img_width = images.shape[2]
psf_stdev = 3.25
background_intensity = 19200
pad = 0

image_attributes = ImageAttributes(img_height = img_height,
                                   img_width = img_width,
                                   max_objects = true_counts.max().item(),
                                   psf_stdev = psf_stdev,
                                   background_intensity = background_intensity)
#################################

#################################
### TUNING PARAMETERS AND SAMPLER SETTINGS

N0 = 20000
kernel_num_iters = 100
catalogs_per_block = 300

max_objects = 10
pad = 0
prior = CatalogPrior(max_objects, img_width, img_height, background_intensity/3., pad)

setting2_std_posterior_mean_count = torch.zeros(num_images)
setting2_std_posterior_mean_total_flux = torch.zeros(num_images)
setting2_std_log_normalizing_constant = torch.zeros(num_images)
setting2_std_reconstructed_image = torch.zeros(num_images, img_height, img_width)
setting2_std_runtime = torch.zeros(num_images)
setting2_std_num_iters = torch.zeros(num_images)
#################################

#################################
### RUN SAMPLER

torch.manual_seed(608)

for i in range(num_images):
    print(f"image {i+1} of {num_images}")
    
    smc = SMCsampler(images[i], image_attributes, prior,
                     max_objects = max_objects,
                     catalogs_per_block = catalogs_per_block,
                     kernel_num_iters = kernel_num_iters,
                     max_smc_iters = 500,
                     wastefree = False, wastefree_M = 1)
    
    start = time.time()
    smc.run(print_progress = True)
    end = time.time()
    
    setting2_std_posterior_mean_count[i] = smc.posterior_mean_count
    setting2_std_posterior_mean_total_flux[i] = smc.posterior_mean_total_flux
    setting2_std_log_normalizing_constant[i] = smc.log_normalizing_constant
    setting2_std_reconstructed_image[i] = smc.reconstructed_image
    setting2_std_runtime[i] = end - start
    setting2_std_num_iters[i] = smc.iter
    
    print(f"image {i+1} took {(setting2_std_runtime[i]).round()} seconds\n")
    
    smc.summarize(display_images = False)
#################################

#################################
### SAVE RESULTS

torch.save(setting2_std_posterior_mean_count, "setting2_results/setting2_std_posterior_mean_count.pt")
torch.save(setting2_std_posterior_mean_total_flux, "setting2_results/setting2_std_posterior_mean_total_flux.pt")
torch.save(setting2_std_log_normalizing_constant, "setting2_results/setting2_std_log_normalizing_constant.pt")
torch.save(setting2_std_reconstructed_image, "setting2_results/setting2_std_reconstructed_image.pt")
torch.save(setting2_std_runtime, "setting2_results/setting2_std_runtime.pt")
torch.save(setting2_std_num_iters, "setting2_results/setting2_std_num_iters.pt")
#################################
