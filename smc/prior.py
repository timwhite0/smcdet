import torch
from torch.distributions import Normal, Uniform, Categorical

class CatalogPrior(object):
    def __init__(self,
                 max_objects: int,
                 img_height: int,
                 img_width: int,
                 min_flux: float,
                 pad = 0):
        
        self.max_objects = max_objects
        self.D = self.max_objects + 1
        
        self.img_height = img_height
        self.img_width = img_width
        self.pad = pad
        
        self.min_flux = torch.tensor(min_flux)
        
        self.count_prior = Categorical((1/self.D) * torch.ones(self.D))
        self.flux_prior = Normal(10 * self.min_flux, 2 * self.min_flux)
        self.loc_prior = Uniform(torch.zeros(2) - self.pad*torch.ones(2),
                                 torch.tensor((self.img_height, self.img_width)) + self.pad*torch.ones(2))
    
    def sample(self,
               num_catalogs = 1,
               num_tiles_h = 1,
               num_tiles_w = 1,
               in_blocks = False,
               num_blocks = None,
               catalogs_per_block = None):
        
        if in_blocks is True and (num_blocks is None or catalogs_per_block is None):
            raise ValueError("If in_blocks is True, need to specify num_blocks and catalogs_per_block.")
        elif in_blocks is False and (num_blocks is not None or catalogs_per_block is not None):
            raise ValueError("If in_blocks is False, do not specify num_blocks or catalogs_per_block.")
        elif in_blocks is False:
            dim = self.D
            
            counts = self.count_prior.sample([num_catalogs])
            count_indicator = torch.arange(1, dim).unsqueeze(0) <= counts.unsqueeze(1)
            
            fluxes = self.flux_prior.sample([num_catalogs, self.max_objects]) * count_indicator
            locs = self.loc_prior.sample([num_catalogs, self.max_objects]) * count_indicator.unsqueeze(2)
        elif in_blocks is True:
            dim = num_blocks
            num_catalogs = num_blocks * catalogs_per_block
            
            counts = torch.ones(num_tiles_h, num_tiles_w, num_blocks * catalogs_per_block) * torch.arange(num_blocks).repeat_interleave(catalogs_per_block)
            count_indicator = torch.arange(1, dim).unsqueeze(0) <= counts.unsqueeze(3)
                   
            fluxes = self.flux_prior.sample([num_tiles_h, num_tiles_w, num_catalogs, self.max_objects]) * count_indicator
            locs = self.loc_prior.sample([num_tiles_h, num_tiles_w, num_catalogs, self.max_objects]) * count_indicator.unsqueeze(4)
        
        return [counts, fluxes, locs]
    
    # log_prob is defined for the in_blocks case within a SMCsampler
    def log_prob(self,
                 counts, fluxes, locs):

        dim = fluxes.shape[3]
        
        count_indicator = 1 + torch.arange(dim).unsqueeze(0) <= counts.unsqueeze(3)

        log_prior = self.count_prior.log_prob(counts)
        log_prior += (self.flux_prior.log_prob(fluxes) * count_indicator).sum(3)
        log_prior += (self.loc_prior.log_prob(locs) * count_indicator.unsqueeze(4)).sum([3,4])
        
        return log_prior