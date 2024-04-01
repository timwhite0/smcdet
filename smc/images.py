import torch
from torch.distributions import Poisson
from smc.prior import CatalogPrior

def PSF(marginal_W, marginal_H, dim, loc_W, loc_H, psf_stdev):    
    psf = ((-(marginal_W - loc_W.view(-1, 1, 1, dim))**2 -
            (marginal_H - loc_H.view(-1, 1, 1, dim))**2)/(2*psf_stdev**2)).exp()
    psf = psf/psf.sum([1,2]).view(-1, 1, 1, dim)
    
    return psf

class ImageAttributes(object):
    def __init__(self,
                 img_width: int,
                 img_height: int,
                 max_objects: int,
                 psf_stdev: float,
                 background_intensity: float):
        
        self.img_width = img_width
        self.img_height = img_height
        self.PSF_marginal_W = (1 + torch.arange(self.img_width, dtype=torch.float32)).view(1, 1, self.img_width, 1)
        self.PSF_marginal_H = (1 + torch.arange(self.img_height, dtype=torch.float32)).view(1, self.img_height, 1, 1)
        
        self.max_objects = max_objects
        self.max_count = self.max_objects + 1
        
        self.psf_stdev = psf_stdev
        self.background_intensity = background_intensity
        self.min_flux = self.background_intensity/3.
    
    def generate(self,
                 num = 1):
        
        prior = CatalogPrior(max_objects = self.max_objects,
                             img_width = self.img_width,
                             img_height = self.img_height,
                             pad = 0,
                             min_flux = self.min_flux)
        
        catalogs = prior.sample(num)
        counts, fluxes, locs = catalogs
        
        source_intensities = (fluxes.view(num, 1, 1, self.max_objects) * PSF(self.PSF_marginal_W, self.PSF_marginal_H,
                                                                             self.max_objects, locs[:,:,0],
                                                                             locs[:,:,1], self.psf_stdev)).sum(3)
        total_intensities = source_intensities + self.background_intensity
        
        images = Poisson(total_intensities).sample()
        
        return [counts, fluxes, locs, total_intensities, images]