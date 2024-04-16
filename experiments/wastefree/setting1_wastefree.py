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
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
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

N0 = 10000
wastefree_M = [25, 50, 80, 125, 200]

num_runs = 100
max_objects = 10
pad = 0
prior = CatalogPrior(max_objects, img_width, img_height, background_intensity/3., pad)

setting1_wf_posterior_mean_count = torch.zeros(num_images, len(wastefree_M), num_runs)
setting1_wf_posterior_mean_total_flux = torch.zeros(num_images, len(wastefree_M), num_runs)
setting1_wf_log_normalizing_constant = torch.zeros(num_images, len(wastefree_M), num_runs)
setting1_wf_runtime = torch.zeros(num_images, len(wastefree_M), num_runs)
setting1_wf_num_iters = torch.zeros(num_images, len(wastefree_M), num_runs)
#################################

#################################
### RUN SAMPLER

torch.manual_seed(608)

for i in range(num_images):
    for m in range(len(wastefree_M)):
        for r in range(num_runs):
            print(f"image {i+1} of {num_images}")
            print(f"run {r+1} of {num_runs}")
            print(f"wastefree_M = {wastefree_M[m]}, wastefree_P = {N0 // wastefree_M[m]}\n")
            
            smc = SMCsampler(images[i], image_attributes, prior,
                             max_objects = max_objects,
                             catalogs_per_block = N0,
                             kernel_num_iters = 1,
                             max_smc_iters = 500,
                             wastefree = True, wastefree_M = wastefree_M[m])
            
            start = time.time()
            smc.run(print_progress = True)
            end = time.time()
            
            setting1_wf_posterior_mean_count[i,m,r] = smc.posterior_mean_count
            setting1_wf_posterior_mean_total_flux[i,m,r] = smc.posterior_mean_total_flux
            setting1_wf_log_normalizing_constant[i,m,r] = smc.log_normalizing_constant[true_counts[i]]
            setting1_wf_runtime[i,m,r] = end - start
            setting1_wf_num_iters[i,m,r] = smc.iter
            
            print(f"image {i+1}, run {r+1} took {(setting1_wf_runtime[i,m,r]).round()} seconds\n")
            
            smc.summarize(display_images = False)
#################################

#################################
### SAVE RESULTS

torch.save(setting1_wf_posterior_mean_count, "setting1_results/setting1_wf_posterior_mean_count.pt")
torch.save(setting1_wf_posterior_mean_total_flux, "setting1_results/setting1_wf_posterior_mean_total_flux.pt")
torch.save(setting1_wf_log_normalizing_constant, "setting1_results/setting1_wf_log_normalizing_constant.pt")
torch.save(setting1_wf_runtime, "setting1_results/setting1_wf_runtime.pt")
torch.save(setting1_wf_num_iters, "setting1_results/setting1_wf_num_iters.pt")
#################################
