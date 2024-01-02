import torch
from torch.distributions import Normal, Uniform, Categorical
import numpy as np

class CatalogPrior(object):
    def __init__(self,
                 max_objects_generated: int,
                 img_width: int,
                 img_height: int,
                 min_flux: float):
        
        self.max_objects_generated = max_objects_generated
        self.D = self.max_objects_generated + 1
        
        self.img_width = img_width
        self.img_height = img_height
        
        self.min_flux = torch.tensor(min_flux)
        
        self.count_prior = Categorical((1/self.D) * torch.ones(self.D))
        self.flux_prior = Normal(10 * self.min_flux, 2 * self.min_flux)
        self.loc_prior = Uniform(torch.zeros(2), torch.tensor((self.img_width, self.img_height)))
    
    def sample(self,
               num_catalogs = 1,
               in_blocks = False,
               num_blocks = None,
               particles_per_block = None):
        
        if in_blocks is True and (num_blocks is None or particles_per_block is None):
            raise ValueError("If in_blocks is True, need to specify num_blocks and particles_per_block.")
        elif in_blocks is False and (num_blocks is not None or particles_per_block is not None):
            raise ValueError("If in_blocks is False, do not specify num_blocks or particles_per_block.")
        elif in_blocks is False:
            dim = self.D
            count = self.count_prior.sample([num_catalogs])
        elif in_blocks is True:
            dim = num_blocks
            num_catalogs = num_blocks * particles_per_block
            count = torch.ones(num_blocks * particles_per_block) * torch.arange(num_blocks).repeat_interleave(particles_per_block)
        
        count_indicator = torch.logical_and(torch.arange(dim).unsqueeze(0) <= count.unsqueeze(1),
                                            torch.arange(dim).unsqueeze(0) > torch.zeros(num_catalogs).unsqueeze(1))
        
        flux = self.flux_prior.sample([num_catalogs, dim]) * count_indicator
        loc = self.loc_prior.sample([num_catalogs, dim]) * count_indicator.unsqueeze(2)
        
        return [count, flux, loc]
    
    def log_prob(self,
                 count, flux, loc):
        ...