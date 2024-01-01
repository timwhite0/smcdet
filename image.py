import torch
from torch.distributions import Poisson, Normal, Uniform, Distribution, Categorical

class Images(object):
    def __init__(self, num_images = 100, image_dim = 15, max_objects_generated = 10,
                 psf_stdev = 3.25, min_flux = 6400.):
        self.num_images = num_images
        
        self.image_dim = image_dim
        self.H = image_dim
        self.W = image_dim
        
        self.max_objects_generated = max_objects_generated
        self.D = max_objects_generated + 1
        
        self.psf_stdev = psf_stdev
        self.min_flux = torch.tensor(min_flux)
        self.background_intensity = 3 * min_flux

    # Move out of class?
    def psf(self, loc_h, loc_w):
        psf_marginal_H = 1 + torch.arange(self.H, dtype=torch.float32)
        psf_marginal_W = 1 + torch.arange(self.W, dtype=torch.float32)
        
        psf = ((-(psf_marginal_H.view(1, self.H, 1, 1) - loc_h.view(1, 1, self.D, -1))**2 - (psf_marginal_W.view(self.W, 1, 1, 1) - loc_w.view(1, 1, self.D, -1))**2)/(2*self.psf_stdev**2)).exp()
        psf = psf/psf.sum([0,1]).view(1, 1, self.D, -1)
        
        return psf.squeeze()