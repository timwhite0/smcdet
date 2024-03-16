#!/usr/bin/env python

#################################
### LIBRARIES

import sys
sys.path.append("../..")

from smc.sampler import SMCsampler
from smc.prior import CatalogPrior
from smc.images import ImageAttributes, PSF
import time
import sep
import torch

# torch.cuda.is_available()
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
torch.set_default_device(device)
#################################

#################################
### GENERATE IMAGES
print("Generating images.")

num_images = 5
img_dim = 15 # Needs to be square for now
img_width = img_dim
img_height = img_dim
max_objects = 10
psf_stdev = 3.25
background_intensity = 19200

image_attributes = ImageAttributes(img_width = img_width,
                                   img_height = img_height,
                                   max_objects = max_objects,
                                   psf_stdev = psf_stdev,
                                   background_intensity = background_intensity)

torch.manual_seed(0)
true_counts, true_fluxes, true_locs, true_total_intensities, images = image_attributes.generate(num_images)

print("Done generating.")
#################################

#################################
### SEP
print("Starting SEP.")

# Grid search over several SEP parameters to obtain optimal performance
num_detection_thresholds_to_try = 20
detection_thresholds = torch.linspace(start = 0, end = 1000, steps = num_detection_thresholds_to_try)

num_minarea_to_try = 3
minarea = torch.linspace(start = 1, end = 5, steps = num_minarea_to_try)

num_deblend_cont_to_try = 5
deblend_cont = torch.linspace(start = 1e-4, end = 1e-2, steps = num_deblend_cont_to_try)

num_dblend_nthresh_to_try = 5
deblend_nthresh = torch.linspace(start = 16, end = 80, steps = num_dblend_nthresh_to_try)

sep_estimated_count = torch.zeros(num_detection_thresholds_to_try, num_minarea_to_try,
                              num_deblend_cont_to_try, num_dblend_nthresh_to_try, num_images)
sep_prop_correct = torch.zeros(num_detection_thresholds_to_try, num_minarea_to_try,
                               num_deblend_cont_to_try, num_dblend_nthresh_to_try)
sep_mse = torch.zeros(num_detection_thresholds_to_try, num_minarea_to_try,
                      num_deblend_cont_to_try, num_dblend_nthresh_to_try)
sep_mae = torch.zeros(num_detection_thresholds_to_try, num_minarea_to_try,
                      num_deblend_cont_to_try, num_dblend_nthresh_to_try)

for t in range(num_detection_thresholds_to_try):
    for m in range(num_minarea_to_try):
        for c in range(num_deblend_cont_to_try):
            for h in range(num_dblend_nthresh_to_try):
                    for img in range(num_images):
                        detected_sources = sep.extract((images[img] - background_intensity).cpu().numpy(),
                                                       thresh = detection_thresholds[t], minarea = minarea[m],
                                                       deblend_cont = deblend_cont[c],
                                                       deblend_nthresh = deblend_nthresh[h], clean = False)
                        sep_estimated_count[t, m, c, h, img] = len(detected_sources)

                    sep_prop_correct[t,m,c,h] = ((sep_estimated_count[t, m, c, h,:] == true_counts).sum()/num_images)
                    sep_mse[t,m,c,h] = ((sep_estimated_count[t, m, c, h,:] - true_counts)**2).mean()
                    sep_mae[t,m,c,h] = (sep_estimated_count[t, m, c, h,:] - true_counts).abs().mean()

for t in range(num_detection_thresholds_to_try):
    for m in range(num_minarea_to_try):
        for c in range(num_deblend_cont_to_try):
            for h in range(num_dblend_nthresh_to_try):
                    if sep_mse[t,m,c,h] == sep_mse.min():
                        detection_threshold_optim = detection_thresholds[t]
                        minarea_optim = minarea[m]
                        deblend_cont_optim = deblend_cont[c]
                        deblend_nthresh_optim = deblend_nthresh[h]

# Run SEP with optimal parameters
sep_estimated_count = torch.zeros(num_images)
sep_loc_x = torch.zeros(num_images)
sep_loc_y = torch.zeros(num_images)
sep_flux = torch.zeros(num_images)
sep_reconstruction = torch.zeros(num_images, image_attributes.img_height, image_attributes.img_width)

for img in range(num_images):
    detected_sources = sep.extract((images[img] - background_intensity).cpu().numpy(),
                                    thresh = detection_threshold_optim, minarea = minarea_optim,
                                    deblend_cont = deblend_cont_optim,
                                    deblend_nthresh = deblend_nthresh_optim, clean = False)
        
    sep_loc_x = (torch.from_numpy(detected_sources['x'])).to(device)
    sep_loc_y = (torch.from_numpy(detected_sources['y'])).to(device)
    sep_flux = (torch.from_numpy(detected_sources['flux'])).to(device)
    sep_estimated_count[img] = len(detected_sources)
    
    if sep_estimated_count[img] > 1:
        sep_reconstruction[img] = (
            PSF(image_attributes.PSF_marginal_W, image_attributes.PSF_marginal_H,
                sep_estimated_count[img].int().item(), sep_loc_x, sep_loc_y, image_attributes.psf_stdev
                ) * sep_flux.view(1, 1, sep_estimated_count[img].int().item())).squeeze().sum(2) + background_intensity
    elif sep_estimated_count[img] == 1:
        sep_reconstruction[img] = (
            PSF(image_attributes.PSF_marginal_W, image_attributes.PSF_marginal_H,
                sep_estimated_count[img].int().item(), sep_loc_x, sep_loc_y, image_attributes.psf_stdev
                ) * sep_flux.view(1, 1, sep_estimated_count[img].int().item())).squeeze() + background_intensity
    else:
        sep_reconstruction[img] = (
            PSF(image_attributes.PSF_marginal_W, image_attributes.PSF_marginal_H,
                1, torch.zeros(1), torch.zeros(1), image_attributes.psf_stdev
                ) * torch.zeros(1).view(1, 1, 1)).squeeze() + background_intensity
        
print("Done with SEP.")
#################################

#################################
### SMC
print("Starting SMC.\n")
torch.manual_seed(0)

max_objects_smc = max_objects + 2
prior = CatalogPrior(max_objects_smc, img_width, img_height, background_intensity/3.)

smc_posterior_mean_count = torch.zeros(num_images)
smc_posterior_total_flux = torch.zeros(num_images)
smc_reconstruction = torch.zeros(num_images, img_width, img_height)
smc_runtime = torch.zeros(num_images)
smc_num_iters = torch.zeros(num_images)

for img in range(num_images):
    print(f"image {img+1} of {num_images}\n")
    
    smc = SMCsampler(images[img], image_attributes, prior,
                     num_blocks = max_objects_smc + 1, catalogs_per_block = 500, max_smc_iters = 500)

    print(f"True count: {true_counts[img]}")
    print(f"True total flux: {true_fluxes[img].sum()}\n")

    start = time.time()
    smc.run(print_progress = True)
    end = time.time()
    
    smc_runtime[img] = end - start
    print(f"image {img+1} took {(smc_runtime[img]).round()} seconds\n")
    
    smc_posterior_mean_count[img] = smc.posterior_mean_count
    smc_posterior_total_flux[img] = smc.posterior_mean_total_flux
    smc_reconstruction[img] = smc.reconstructed_image
    smc_num_iters[img] = smc.iter
    smc.summarize(display_images = False)

print("Done with SMC.")
#################################

#################################
### SAVE RESULTS
print("Writing results.")

sys.path.append("experiments/smc_vs_sep")

# Synthetic images
torch.save(true_counts, "results/true_counts.pt")
torch.save(true_fluxes, "results/true_fluxes.pt")
torch.save(true_locs, "results/true_locs.pt")
torch.save(true_total_intensities, "results/true_total_intensities.pt")
torch.save(images, "results/images.pt")

# SEP results
torch.save(sep_estimated_count, "results/sep_estimated_count.pt")
torch.save(sep_reconstruction, "results/sep_reconstruction.pt")

# SMC results
torch.save(smc_posterior_mean_count, "results/smc_posterior_mean_count.pt")
torch.save(smc_reconstruction, "results/smc_reconstruction.pt")
torch.save(smc_runtime, "results/smc_runtime.pt")
torch.save(smc_num_iters, "results/smc_num_iters.pt")

print("Done writing results.")
#################################