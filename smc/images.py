import torch
from torch.distributions import Poisson

class ImageModel(object):
    def __init__(self,
                 image_dim,
                 psf_stdev,
                 background):
        self.image_dim = image_dim
        
        psf_marginal = 1 + torch.arange(self.image_dim, dtype = torch.float32)
        self.psf_marginal_h = psf_marginal.view(1, self.image_dim, 1, 1)
        self.psf_marginal_w = psf_marginal.view(1, 1, self.image_dim, 1)
        
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
        counts, locs, features = catalogs
        
        psf = self.psf(locs)
        rate = (psf * features.unsqueeze(-2).unsqueeze(-3)).sum(3) + self.background
        images = Poisson(rate).sample()
        
        return [counts, locs, features, images]


    # first two dims of tiled_image should be num_tiles_per_side x num_tiles_per_side
    def loglikelihood(self, tiled_image, locs, features):
        psf = self.psf(locs)
        rate = (psf * features.unsqueeze(-2).unsqueeze(-3)).sum(5) + self.background
        rate = rate.permute((0, 1, 3, 4, 2))
        loglik = Poisson(rate).log_prob(tiled_image.unsqueeze(4)).sum([2,3])

        return loglik
