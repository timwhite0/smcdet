import torch
from torch.distributions import Poisson

class ImageAttributes(object):
    def __init__(self,
                 img_dim,
                 psf_stdev,
                 background):
        self.img_dim = img_dim
        
        psf_marginal = 1 + torch.arange(self.img_dim, dtype = torch.float32)
        self.psf_marginal_h = psf_marginal.view(1, self.img_dim, 1, 1)
        self.psf_marginal_w = psf_marginal.view(1, 1, self.img_dim, 1)
        
        self.psf_stdev = psf_stdev
        
        self.background = background
    
    
    def psf(self, locs):
        loc_h = locs[...,0].unsqueeze(-2).unsqueeze(-3)
        loc_w = locs[...,1].unsqueeze(-2).unsqueeze(-3)
        logpsf = (- (self.psf_marginal_h - loc_h)**2 - (self.psf_marginal_w - loc_w)**2) / (2 * self.psf_stdev**2)
        psf = (logpsf - logpsf.logsumexp([-2,-3]).unsqueeze(-2).unsqueeze(-3)).exp()
        
        return psf
    
    
    def generate(self, prior, num = 1):
        catalogs = prior.sample(num_catalogs = num)
        counts, locs, fluxes = catalogs
        
        source_intensities = (fluxes.unsqueeze(1).unsqueeze(2) * self.psf(locs)).sum(3)
        total_intensities = source_intensities + self.background
        
        images = Poisson(total_intensities).sample()
        
        return [counts, locs, fluxes, total_intensities, images]
