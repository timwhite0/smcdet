import torch
from torch.distributions import Poisson

class ImageAttributes(object):
    def __init__(self,
                 img_height: int,
                 img_width: int,
                 max_objects: int,
                 psf_stdev: float,
                 background_intensity: float):
        
        self.img_height = img_height
        self.img_width = img_width
        self.PSF_marginal_H = (1 + torch.arange(self.img_height, dtype=torch.float32)).view(1, self.img_height, 1, 1)
        self.PSF_marginal_W = (1 + torch.arange(self.img_width, dtype=torch.float32)).view(1, 1, self.img_width, 1)
        
        self.max_objects = max_objects
        self.max_count = self.max_objects + 1
        
        self.psf_stdev = psf_stdev
        self.background_intensity = background_intensity
    
    def PSF(self, num_layers, loc_H, loc_W):
        psf = ((-(self.PSF_marginal_H - loc_H.view(-1, 1, 1, num_layers))**2 -
                 (self.PSF_marginal_W - loc_W.view(-1, 1, 1, num_layers))**2)/(2*self.psf_stdev**2)).exp()
        psf = psf/psf.sum([1,2]).view(-1, 1, 1, num_layers)
        
        return psf
    
    def generate(self,
                 prior,
                 num = 1):
        
        catalogs = prior.sample(num)
        counts, fluxes, locs = catalogs
        
        source_intensities = (fluxes.view(num, 1, 1, self.max_objects) * self.PSF(self.max_objects,
                                                                                  locs[:,:,0],
                                                                                  locs[:,:,1])).sum(3)
        total_intensities = source_intensities + self.background_intensity
        
        images = Poisson(total_intensities).sample()
        
        return [counts, fluxes, locs, total_intensities, images]