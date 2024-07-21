import torch
import math
from torch.distributions import Independent, Normal, Poisson
from einops import rearrange

class ImageModel(object):
    def __init__(self,
                 image_height,
                 image_width,
                 psf_stdev,
                 background):
        self.image_height = image_height
        self.image_width = image_width
        self.psf_stdev = psf_stdev
        self.background = background
        
        self.psf_pad = math.ceil(4 * psf_stdev)  # pad PSF by 4 stdevs
        self.psf_density = Independent(Normal(torch.zeros(1),
                                              self.psf_stdev * torch.ones(1)), 1)
        self.update_psf_grid()
    
    
    def update_psf_grid(self):
        psf_marginal_h = torch.arange(-self.psf_pad, self.image_height + self.psf_pad)
        psf_marginal_w = torch.arange(-self.psf_pad, self.image_width + self.psf_pad)
        grid_h, grid_w = torch.meshgrid(psf_marginal_h, psf_marginal_w, indexing = 'ij')
        self.psf_grid = torch.stack([grid_h, grid_w], dim = -1)
    
    
    def psf(self, locs):
        psf_grid_adjusted = self.psf_grid - (locs.unsqueeze(-2).unsqueeze(-3) - 0.5)
        logpsf_padded = self.psf_density.log_prob(psf_grid_adjusted)
        logpsf = logpsf_padded[...,
                               self.psf_pad:(self.image_height + self.psf_pad),
                               self.psf_pad:(self.image_width + self.psf_pad)]
        psf = logpsf.exp()
        psf = rearrange(psf, 'numH numW n d dimH dimW -> numH numW dimH dimW n d')
        
        return psf
    
    
    def generate(self, Prior, num_images = 1):
        catalogs = Prior.sample(num_catalogs = num_images)
        counts, locs, features = catalogs
        
        psf = self.psf(locs)
        rate = (psf * features.unsqueeze(-3).unsqueeze(-4)).sum(-1) + self.background
        images = Poisson(rate).sample()
        
        counts = counts.squeeze([0,1])
        locs = locs.squeeze([0,1])
        features = features.squeeze([0,1])
        images = rearrange(images.squeeze([0,1]), 'dimH dimW n -> n dimH dimW')
        
        return [counts, locs, features, images]


    def loglikelihood(self, tiled_image, locs, features):
        psf = self.psf(locs)
        rate = (psf * features.unsqueeze(-3).unsqueeze(-4)).sum(-1) + self.background
        loglik = Poisson(rate).log_prob(tiled_image.unsqueeze(-1)).sum([-2,-3])
        
        return loglik
