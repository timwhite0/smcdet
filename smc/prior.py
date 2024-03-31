import torch
from torch.distributions import Normal, Uniform, Categorical

class CatalogPrior(object):
    def __init__(self,
                 max_objects: int,
                 img_width: int,
                 img_height: int,
                 pad: int,
                 min_flux: float):
        
        self.max_objects = max_objects
        self.D = self.max_objects + 1
        
        self.img_width = img_width
        self.img_height = img_height
        self.pad = pad
        
        self.min_flux = torch.tensor(min_flux)
        
        self.count_prior = Categorical((1/self.D) * torch.ones(self.D))
        self.flux_prior = Normal(10 * self.min_flux, 2 * self.min_flux)
        self.loc_prior = Uniform(torch.zeros(2) - self.pad*torch.ones(2), torch.tensor((self.img_width, self.img_height)) + self.pad*torch.ones(2))
    
    def sample(self,
               num_catalogs = 1,
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
        elif in_blocks is True:
            dim = num_blocks
            num_catalogs = num_blocks * catalogs_per_block
            counts = torch.ones(num_blocks * catalogs_per_block) * torch.arange(num_blocks).repeat_interleave(catalogs_per_block)
        
        count_indicator = torch.logical_and(torch.arange(dim).unsqueeze(0) <= counts.unsqueeze(1),
                                            torch.arange(dim).unsqueeze(0) > torch.zeros(num_catalogs).unsqueeze(1))
        
        fluxes = self.flux_prior.sample([num_catalogs, dim]) * count_indicator
        locs = self.loc_prior.sample([num_catalogs, dim]) * count_indicator.unsqueeze(2)
        
        return [counts, fluxes, locs]
    
    def log_prob(self,
                 counts, fluxes, locs):
        
        num_catalogs = fluxes.shape[0]
        dim = fluxes.shape[1]
        
        count_indicator = torch.logical_and(torch.arange(dim).unsqueeze(0) <= counts.unsqueeze(1),
                                            torch.arange(dim).unsqueeze(0) > torch.zeros(num_catalogs).unsqueeze(1))

        log_prior = self.count_prior.log_prob(counts)
        log_prior += (self.flux_prior.log_prob(fluxes) * count_indicator).sum(1)
        log_prior += (self.loc_prior.log_prob(locs) * count_indicator.unsqueeze(2)).sum(2).sum(1)
        
        return log_prior