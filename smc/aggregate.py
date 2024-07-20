import torch
from einops import rearrange, repeat
from copy import deepcopy

class Aggregate(object):
    def __init__(self,
                 Prior,
                 ImageModel,
                 data,
                 counts,
                 locs,
                 features,
                 weights):
        self.Prior = deepcopy(Prior)
        self.Prior.pad = 0
        self.Prior.update_attrs()
        self.ImageModel = deepcopy(ImageModel)
        
        self.data = data
        self.counts = counts
        self.locs = locs
        self.features = features
        self.weights = weights
        self.num_catalogs = self.weights.shape[-1]
        
        self.numH, self.numW, self.dimH, self.dimW = self.data.shape
        self.num_aggregation_levels = (2 * torch.tensor(self.numH).log2()).int().item()
        
        self.log_density_children = self.compute_log_density()
        self.log_density_parents = None
    
    
    def resample(self, method = "systematic", multiplier = 1):
        num = int(multiplier * self.num_catalogs)
        
        if method == "multinomial":
            weights_flat = self.weights.flatten(0,1)
            resampled_index_flat = weights_flat.multinomial(num, replacement = True)
            resampled_index = resampled_index_flat.unflatten(0, (self.numH, self.numW))
        elif method == "systematic":
            resampled_index = torch.zeros_like(self.weights)
            for h in range(self.numH):
                for w in range(self.numW):
                    u = (torch.arange(num) + torch.rand([1])) / num
                    bins = self.weights[h,w].cumsum(0)
                    resampled_index[h,w] = torch.bucketize(u, bins)
        
        resampled_index = resampled_index.int().clamp(min = 0, max = num - 1)
        
        if multiplier >= 1:
            counts = repeat(torch.zeros_like(self.counts), 'numH numW N -> numH numW (m N)', m = multiplier)
            locs = repeat(torch.zeros_like(self.locs), 'numH numW N ... -> numH numW (m N) ...', m = multiplier)
            features = repeat(torch.zeros_like(self.features), 'numH numW N ... -> numH numW (m N) ...', m = multiplier)
            weights = repeat(torch.zeros_like(self.weights), 'numH numW N -> numH numW (m N)', m = multiplier)
        if multiplier < 1:
            counts = torch.zeros_like(self.counts)[:,:,:num]
            locs = torch.zeros_like(self.locs)[:,:,:num,...]
            features = torch.zeros_like(self.features)[:,:,:num,...]
            weights = torch.zeros_like(self.weights)[:,:,:num]
        
        for h in range(self.numH):
            for w in range(self.numW):
                counts[h,w,:] = self.counts[h,w,resampled_index[h,w,:]]
                locs[h,w,:] = self.locs[h,w,resampled_index[h,w,:]]
                features[h,w,:] = self.features[h,w,resampled_index[h,w,:]]
                weights[h,w,:] = 1 / num
        
        self.counts = counts
        self.locs = locs
        self.features = features
        self.weights = weights
    
    
    def join(self, axis):
        if axis == 0:  # height axis
            self.numH = self.numH // 2
            self.dimH = self.dimH * 2
            self.ImageModel.image_height = self.ImageModel.image_height * 2
            self.Prior.image_height = self.Prior.image_height * 2
            self.data = rearrange(self.data.unfold(axis, 2, 2),
                                  'numH numW dimH dimW t -> numH numW (t dimH) dimW')
        elif axis == 1:  # width axis
            self.numW = self.numW // 2
            self.dimW = self.dimW * 2
            self.ImageModel.image_width = self.ImageModel.image_width * 2
            self.Prior.image_width = self.Prior.image_width * 2
            self.data = rearrange(self.data.unfold(axis, 2, 2),
                                  'numH numW dimH dimW t -> numH numW dimH (t dimW)')
        self.Prior.max_objects = self.Prior.max_objects * 2
        self.Prior.update_attrs()
        self.ImageModel.update_psf_grid()
        # counts
        self.counts = self.counts.unfold(axis, 2, 2).sum(3)
        # locs
        locs_unfolded = self.locs.unfold(axis, 2, 2)
        locs_unfolded_mask = (locs_unfolded != 0).int()
        locs_unfolded.select(-2, axis)[...,-1] += (self.dimH / 2) * (1 - axis) + (self.dimW / 2) * axis
        locs_adjusted = locs_unfolded * locs_unfolded_mask
        self.locs = rearrange(locs_adjusted,
                              'numH numW N M l t -> numH numW N (t M) l')
        locs_mask = (self.locs != 0).int()
        locs_index = torch.sort(locs_mask, dim = 3, descending = True)[1]
        self.locs = torch.gather(self.locs, dim = 3, index = locs_index)
        # features
        self.features = rearrange(self.features.unfold(axis, 2, 2),
                                  'numH numW N M t -> numH numW N (t M)')
        features_mask = (self.features != 0).int()
        features_index = torch.sort(features_mask, dim = 3, descending = True)[1]
        self.features = torch.gather(self.features, dim = 3, index = features_index)
        # log_density_children
        self.log_density_children = self.log_density_children.unfold(axis, 2, 2).sum(3)
    
    
    def compute_log_density(self):
        logprior = self.Prior.log_prob(self.counts, self.locs, self.features)
        loglik = self.ImageModel.loglikelihood(self.data, self.locs, self.features)
        return logprior + loglik
    
    
    def run(self):
        for level in range(self.num_aggregation_levels):
            self.resample()
            self.log_density_children = self.compute_log_density()
            if level % 2 == 0:
                self.join(axis = 0)
            elif level % 2 != 0:
                self.join(axis = 1)
            self.log_density_parents = self.compute_log_density()
            self.weights = (self.log_density_parents - self.log_density_children).softmax(-1)


    @property
    def ESS(self):
        return 1 / (self.weights**2).sum(-1)
    
    
    @property
    def posterior_mean_counts(self):
        return (self.weights * self.counts).sum(-1)


    @property
    def posterior_mean_total_flux(self):
        return (self.weights * self.features.sum(-1)).sum(-1)
