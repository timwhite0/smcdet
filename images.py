import torch
from torch.distributions import Poisson, Normal, Uniform, Distribution, Categorical
import numpy as np
from prior import CatalogPrior

def PSF(img_width, img_height, dim, loc_W, loc_H, psf_stdev):
    psf_marginal_W = 1 + torch.arange(img_width, dtype=torch.float32)
    psf_marginal_H = 1 + torch.arange(img_height, dtype=torch.float32)
    
    psf = ((-(psf_marginal_W.view(1, 1, img_width, 1) - loc_W.view(-1, 1, 1, dim))**2 - (psf_marginal_H.view(1, img_height, 1, 1) - loc_H.view(-1, 1, 1, dim))**2)/(2*psf_stdev**2)).exp()
    psf = psf/psf.sum([1,2]).view(-1, 1, 1, dim)
    
    return psf

class ImageCollection(object):
    def __init__(self,
                 img_width: int,
                 img_height: int,
                 max_objects_generated: int,
                 psf_stdev: float,
                 background_intensity: float):
        
        self.img_width = img_width
        self.img_height = img_height
        
        self.max_objects_generated = max_objects_generated
        self.max_count = self.max_objects_generated + 1
        
        self.psf_stdev = psf_stdev
        self.background_intensity = background_intensity
        self.min_flux = self.background_intensity/3
    
    def generate(self,
                 num = 1):
        
        prior = CatalogPrior(max_objects_generated = self.max_objects_generated,
                             img_width = self.img_width,
                             img_height = self.img_height,
                             min_flux = self.min_flux)
        
        catalogs = prior.sample(num)
        counts, fluxes, locs = catalogs
        
        source_intensities = (fluxes.view(num, 1, 1, self.max_count) * PSF(self.img_width, self.img_height,
                                                                           self.max_count, locs[:,:,0],
                                                                           locs[:,:,1], self.psf_stdev)).sum(3)
        total_intensities = source_intensities + self.background_intensity
        
        images = Poisson(total_intensities).sample()
        
        return [counts, fluxes, locs, total_intensities, images]