#!/usr/bin/env python

##############################################
### SETUP

import torch
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
torch.set_default_device(device)

import time
import sep
##############################################

##############################################
### LOAD IN IMAGES AND CATALOGS

images = torch.load("data/images.pt").to(device)
true_counts = torch.load("data/true_counts.pt").to(device)
true_locs = torch.load("data/true_locs.pt").to(device)
true_fluxes = torch.load("data/true_fluxes.pt").to(device)

num_images = images.shape[0]
image_height = images.shape[1]
image_width = images.shape[2]
background = 100000
##############################################

##############################################
### SELECT SEP HYPERPARAMETERS VIA GRID SEARCH

print('Starting grid search...\n')

thresh = torch.linspace(start = 500, end = 2000, steps = 4)
minarea = torch.linspace(start = 1, end = 5, steps = 3)
deblend_cont = torch.logspace(start = -4, end = -2, steps = 3)
clean_param = torch.logspace(start = -3, end = 1, steps = 3)

sep_estimated_count = torch.zeros(thresh.shape[0], minarea.shape[0],
                                  deblend_cont.shape[0],
                                  clean_param.shape[0], num_images)
sep_mse = torch.zeros(thresh.shape[0], minarea.shape[0],
                      deblend_cont.shape[0], clean_param.shape[0])

for t in range(thresh.shape[0]):
    for m in range(minarea.shape[0]):
        for d in range(deblend_cont.shape[0]):
            for c in range(clean_param.shape[0]):
                print(f'thresh = {thresh[t]}')
                print(f'minarea = {minarea[m]}')
                print(f'deblend_cont = {deblend_cont[d]}')
                print(f'clean_param = {clean_param[c]}\n')
                
                for i in range(num_images):
                    detected_sources = sep.extract((images[i] - background).cpu().numpy(),
                                                    thresh = thresh[t], minarea = minarea[m],
                                                    deblend_cont = deblend_cont[d],
                                                    deblend_nthresh = 2000,
                                                    filter_kernel = None,
                                                    clean = True, clean_param = clean_param[c])
                    sep_estimated_count[t, m, d, c, i] = len(detected_sources)

                sep_mse[t,m,d,c] = ((sep_estimated_count[t, m, d, c,:] - true_counts)**2).mean()

for t in range(thresh.shape[0]):
    for m in range(minarea.shape[0]):
        for d in range(deblend_cont.shape[0]):
            for c in range(clean_param.shape[0]):
                if sep_mse[t,m,d,c] == sep_mse.min():
                    thresh_best = thresh[t]
                    minarea_best = minarea[m]
                    deblend_cont_best = deblend_cont[d]
                    clean_param_best = clean_param[c]

print('Hyperparameters selected:')
print(f'thresh = {thresh_best.item()}')
print(f'minarea = {minarea_best.item()}')
print(f'deblend_cont = {deblend_cont_best}')
print(f'clean_param = {clean_param_best}\n')
##############################################

##############################################
### RUN SEP

print('Running SEP...\n')

max_detections = 100
runtime = torch.zeros(num_images)
counts = torch.zeros(num_images)
locs = torch.zeros(num_images, max_detections, 2)
fluxes = torch.zeros(num_images, max_detections)

for i in range(num_images):
    start = time.perf_counter()
    
    sep_results = sep.extract((images[i] - background).cpu().numpy(),
                              thresh = thresh_best, minarea = minarea_best,
                              deblend_cont = deblend_cont_best,
                              deblend_nthresh = 2000, filter_kernel = None,
                              clean = True, clean_param = clean_param_best)
    
    end = time.perf_counter()
    
    runtime[i] = end - start
    counts[i] = len(sep_results)
    count = counts[i].int().item()
    locs[i,:count,0] = torch.from_numpy(sep_results['x'])
    locs[i,:count,1] = torch.from_numpy(sep_results['y'])
    fluxes[i,:count] = torch.from_numpy(sep_results['flux'])

# remove unnecessary trailing zeros
locs = locs[:,:counts.max().int().item(),:]
fluxes = fluxes[:,:counts.max().int().item()]
##############################################

##############################################
### SAVE RESULTS

print('Saving results...\n')

torch.save(runtime.cpu(), "results/sep/runtime.pt")
torch.save(counts.cpu(), "results/sep/counts.pt")
torch.save(locs.cpu(), "results/sep/locs.pt")
torch.save(fluxes.cpu(), "results/sep/fluxes.pt")

print('Done!')
##############################################
