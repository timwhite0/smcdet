import torch
from torch.distributions import Normal, Uniform, Categorical

class PointProcessPrior(object):
    def __init__(self,
                 max_objects: int,
                 img_dim: int,
                 pad = 0):
        
        self.max_objects = max_objects
        self.num_counts = self.max_objects + 1
        
        self.img_dim = img_dim
        self.pad = pad
        
        self.count_prior = Categorical((1 / self.num_counts) * torch.ones(self.num_counts))
        self.loc_prior = Uniform((0 - self.pad) * torch.ones(2), (self.img_dim + self.pad) * torch.ones(2))
    
    def sample(self,
               num_catalogs = 1,
               num_tiles_per_side = 1,
               stratify_by_count = False,
               num_catalogs_per_count = None):
        if stratify_by_count is True and num_catalogs_per_count is None:
            raise ValueError("If stratify_by_count is True, need to specify catalogs_per_count.")
        elif stratify_by_count is False and num_catalogs_per_count is not None:
            raise ValueError("If stratify_by_count is False, do not specify catalogs_per_count.")
        
        if stratify_by_count is False:
            self.num = num_catalogs
            counts = self.count_prior.sample([num_tiles_per_side, num_tiles_per_side, self.num])
        elif stratify_by_count is True:
            self.num = self.num_counts * num_catalogs_per_count
            strata = torch.arange(self.num_counts).repeat_interleave(num_catalogs_per_count)
            counts = strata * torch.ones(num_tiles_per_side, num_tiles_per_side, self.num)
        
        self.count_indicator = torch.arange(1, self.num_counts).unsqueeze(0) <= counts.unsqueeze(3)
        locs = self.loc_prior.sample([num_tiles_per_side, num_tiles_per_side, self.num, self.max_objects])
        locs *= self.count_indicator.unsqueeze(4)
        
        return [counts.squeeze([0,1]), locs.squeeze([0,1])]
    
    # we define log_prob for stratify_by_count = True, to be used within SMCsampler
    def log_prob(self, counts, locs):
        self.count_indicator = torch.arange(1, self.num_counts).unsqueeze(0) <= counts.unsqueeze(3)

        log_prior = self.count_prior.log_prob(counts)
        log_prior += (self.loc_prior.log_prob(locs) * self.count_indicator.unsqueeze(4)).sum([3,4])
        
        return log_prior


# TODO: Move all subclasses of PointProcessPrior to their own case_studies directory
class StarPrior(PointProcessPrior):
    def __init__(self,
                 *args,
                 min_flux: float,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.min_flux = min_flux
        self.feature_prior = Normal(10 * self.min_flux, 2 * self.min_flux)
    
    def sample(self,
               num_catalogs = 1,
               num_tiles_per_side = 1,
               stratify_by_count = False,
               num_catalogs_per_count = None):
        counts, locs = super().sample(num_catalogs, num_tiles_per_side,
                                      stratify_by_count, num_catalogs_per_count)
        
        features = self.feature_prior.sample([num_tiles_per_side, num_tiles_per_side,
                                              self.num, self.max_objects])
        features *= self.count_indicator
        
        return [counts, locs, features.squeeze([0,1])]
    
    # we define log_prob for stratify_by_count = True, to be used within SMCsampler
    def log_prob(self, counts, locs, features):
        log_prior = super().log_prob(counts, locs)
        
        return log_prior + (self.feature_prior.log_prob(features) * self.count_indicator).sum(3)
