import torch
from torch.distributions import Poisson
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
        self.update_attrs()
    
    
    def update_attrs(self):
        marginal_h = 0.5 + torch.arange(self.image_height, dtype = torch.float32)
        marginal_w = 0.5 + torch.arange(self.image_width, dtype = torch.float32)
        self.psf_marginal_h = marginal_h.view(1, self.image_height, 1, 1)
        self.psf_marginal_w = marginal_w.view(1, 1, self.image_width, 1)
    
    
    def psf(self, locs):
        loc_h = locs[...,0].unsqueeze(-2).unsqueeze(-3)
        loc_w = locs[...,1].unsqueeze(-2).unsqueeze(-3)
        logpsf = (- (self.psf_marginal_h - loc_h)**2 - (self.psf_marginal_w - loc_w)**2) / (2 * self.psf_stdev**2)
        psf = (logpsf - logpsf.logsumexp([-2,-3]).unsqueeze(-2).unsqueeze(-3)).exp()
        
        return psf
    
    
    def generate(self, Prior, num_images = 1):
        catalogs = Prior.sample(num_catalogs = num_images)
        counts, locs, features = catalogs
        
        counts = counts.squeeze([0,1])
        locs = locs.squeeze([0,1])
        features = features.squeeze([0,1])
        
        psf = self.psf(locs)
        rate = (psf * features.unsqueeze(-2).unsqueeze(-3)).sum(-1) + self.background
        images = Poisson(rate).sample()
        
        return [counts, locs, features, images]


    def loglikelihood(self, tiled_image, locs, features):
        psf = self.psf(locs)
        rate = (psf * features.unsqueeze(-2).unsqueeze(-3)).sum(-1) + self.background
        rate = rearrange(rate, '... n h w -> ... h w n')
        loglik = Poisson(rate).log_prob(tiled_image.unsqueeze(-1)).sum([-2,-3])

        return loglik
