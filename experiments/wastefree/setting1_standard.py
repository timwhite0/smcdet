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
true_counts = torch.load("images/setting1_true_counts.pt").to(device)
true_fluxes = torch.load("images/setting1_true_fluxes.pt").to(device)
images = torch.load("images/setting1_images.pt").to(device)

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
kernel_num_iters = [10, 25, 50, 100, 200]
catalogs_per_block = [(3*N0) // (2*k) for k in kernel_num_iters]

num_runs = 50
max_objects = 10
pad = 0
prior = CatalogPrior(max_objects, img_width, img_height, background_intensity/3., pad)

setting1_std_posterior_mean_count = torch.zeros(num_images, len(kernel_num_iters), num_runs)
setting1_std_posterior_mean_total_flux = torch.zeros(num_images, len(kernel_num_iters), num_runs)
setting1_std_log_normalizing_constant = torch.zeros(num_images, len(kernel_num_iters), num_runs)
setting1_std_runtime = torch.zeros(num_images, len(kernel_num_iters), num_runs)
setting1_std_num_iters = torch.zeros(num_images, len(kernel_num_iters), num_runs)
#################################

#################################
### RUN SAMPLER

torch.manual_seed(608)

for i in range(num_images):
    for k in range(len(kernel_num_iters)):
        for r in range(num_runs):
            print(f"image {i+1} of {num_images}")
            print(f"run {r+1} of {num_runs}")
            print(f"k = {kernel_num_iters[k]}, N = {catalogs_per_block[k]}\n")
            
            smc = SMCsampler(images[i], image_attributes, prior,
                             max_objects = max_objects,
                             catalogs_per_block = catalogs_per_block[k],
                             kernel_num_iters = kernel_num_iters[k],
                             max_smc_iters = 500,
                             wastefree = False, wastefree_M = 1)
            
            start = time.time()
            smc.run(print_progress = True)
            end = time.time()
            
            setting1_std_posterior_mean_count[i,k,r] = smc.posterior_mean_count
            setting1_std_posterior_mean_total_flux[i,k,r] = smc.posterior_mean_total_flux
            setting1_std_log_normalizing_constant[i,k,r] = smc.log_normalizing_constant
            setting1_std_runtime[i,k,r] = end - start
            setting1_std_num_iters[i,k,r] = smc.iter
            
            print(f"image {i+1}, run {r+1} took {(setting1_std_runtime[i,k,r]).round()} seconds\n")
            
            smc.summarize(display_images = False)
#################################

#################################
### SAVE RESULTS

torch.save(setting1_std_posterior_mean_count, "setting1_results/setting1_std_posterior_mean_count.pt")
torch.save(setting1_std_posterior_mean_total_flux, "setting1_results/setting1_std_posterior_mean_total_flux.pt")
torch.save(setting1_std_log_normalizing_constant, "setting1_results/setting1_std_log_normalizing_constant.pt")
torch.save(setting1_std_runtime, "setting1_results/setting1_std_runtime.pt")
torch.save(setting1_std_num_iters, "setting1_results/setting1_std_num_iters.pt")
#################################
