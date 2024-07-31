#!/usr/bin/env python

##############################################
### SETUP

import sys
sys.path.append("/home/twhit/smc_object_detection/")
from smc.sampler import SMCsampler
from smc.prior import StarPrior
from smc.images import ImageModel
from smc.kernel import MetropolisHastings
from smc.aggregate import Aggregate

import torch
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
torch.set_default_device(device)

import time
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
##############################################

##############################################
### SPECIFY TILE-LEVEL IMAGE MODEL, PRIOR, AND MUTATION KERNEL

tile_dim = 8

prior = StarPrior(max_objects = 5,
                  image_height = tile_dim,
                  image_width = tile_dim,
                  flux_mean = 80000,
                  flux_stdev = 15000,
                  pad = 2)

imagemodel = ImageModel(image_height = tile_dim,
                        image_width = tile_dim,
                        psf_stdev = 1.5,
                        background = 100000)

mh = MetropolisHastings(num_iters = 75,
                        locs_stdev = 0.1,
                        features_stdev = 1000,
                        features_min = 50000,
                        features_max = 110000)
##############################################

##############################################
### SPECIFY NUMBER OF CATALOGS AND BATCH SIZE FOR SAVING RESULTS

num_catalogs_per_count = 2500
num_catalogs = (prior.max_objects + 1) * num_catalogs_per_count

batch_size = 10
num_batches = num_images // batch_size
##############################################

##############################################
### RUN SMC

torch.manual_seed(1)

for b in range(num_batches):
    runtime = torch.zeros([batch_size])
    num_iters = torch.zeros([batch_size])
    counts = torch.zeros([batch_size, num_catalogs])
    locs = torch.zeros([batch_size, num_catalogs, 8 * prior.max_objects, 2])
    fluxes = torch.zeros([batch_size, num_catalogs, 8 * prior.max_objects])
    
    for i in range(batch_size):
        image_index = b * batch_size + i
        
        print(f'image {image_index + 1} of {num_images}')
        print(f"true count = {true_counts[image_index]}")
        print(f"true total flux = {true_fluxes[image_index].sum()}\n")
        
        sampler = SMCsampler(image = images[image_index],
                             tile_dim = tile_dim,
                             Prior = prior,
                             ImageModel = imagemodel,
                             MutationKernel = mh,
                             num_catalogs_per_count = num_catalogs_per_count,
                             max_smc_iters = 200)

        start = time.perf_counter()
        
        sampler.run(print_progress = True)
        
        agg = Aggregate(sampler.Prior,
                        sampler.ImageModel,
                        sampler.tiled_image,
                        sampler.counts,
                        sampler.locs,
                        sampler.features,
                        sampler.weights_intercount)

        agg.run()
        
        end = time.perf_counter()
        
        runtime[i] = end - start
        num_iters[i] = sampler.iter
        counts[i] = agg.counts.squeeze([0,1])
        index = agg.locs.shape[-2]
        locs[i,:,:index,:] = agg.locs.squeeze([0,1])
        fluxes[i,:,:index] = agg.features.squeeze([0,1])
        
        print(f'runtime = {runtime[i]}')
        print(f'num iters = {num_iters[i]}')
        print(f'posterior mean count = {agg.posterior_mean_counts.item()}')
        print(f'posterior mean total flux = {agg.posterior_mean_total_flux.item()}\n\n\n')
        
    torch.save(runtime.cpu(), f"results/runtime_{b}.pt")
    torch.save(num_iters.cpu(), f"results/num_iters_{b}.pt")
    torch.save(counts.cpu(), f"results/counts_{b}.pt")
    torch.save(locs.cpu(), f"results/locs_{b}.pt")
    torch.save(fluxes.cpu(), f"results/fluxes_{b}.pt")

print('Done!')
##############################################
