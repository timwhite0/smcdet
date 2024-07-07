import torch
from einops import rearrange
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
        self.ImageModel = deepcopy(ImageModel)
        
        self.data = data
        self.counts = counts
        self.locs = locs
        self.features = features
        self.weights = weights
        self.num_catalogs = self.weights.shape[-1]
        
        self.numH = self.data.shape[0]
        self.numW = self.data.shape[1]
        self.num_aggregation_levels = (2 * torch.tensor(self.numH).log2()).item()
        
        self.log_target_prev = self.compute_log_target()
        self.log_target_curr = None
    
    
    def resample(self):
        weights_flat = self.weights.flatten(0,1)
        resampled_index_flat = weights_flat.multinomial(self.num_catalogs,
                                                        replacement = True).clamp(min = 0,
                                                                                  max = self.num_catalogs - 1)
        resampled_index = resampled_index_flat.unflatten(0, (self.numH, self.numW))
        
        for h in range(self.numH):
            for w in range(self.numW):
                self.counts[h,w,:] = self.counts[h,w,resampled_index[h,w,:]]
                self.locs[h,w,:] = self.locs[h,w,resampled_index[h,w,:]]
                self.features[h,w,:] = self.features[h,w,resampled_index[h,w,:]]
                self.weights[h,w,:] = 1 / self.num_catalogs
    
    
    def join(self, axis):
        if axis == 0:  # height axis
            self.numH = self.numH // 2
            self.ImageModel.image_height = self.ImageModel.image_height * 2
            self.ImageModel.update_attrs()
            self.Prior.image_height = self.Prior.image_height * 2
            self.Prior.max_objects = self.Prior.max_objects * 2
            self.Prior.update_attrs()
            self.data = rearrange(self.data.unfold(axis, 2, 2),
                                  'numH numW dimH dimW t -> numH numW (t dimH) dimW')
        elif axis == 1:  # width axis
            self.numW = self.numW // 2
            self.ImageModel.image_width = self.ImageModel.image_width * 2
            self.ImageModel.update_attrs()
            self.Prior.image_width = self.Prior.image_width * 2
            self.Prior.max_objects = self.Prior.max_objects * 2
            self.Prior.update_attrs()
            self.data = rearrange(self.data.unfold(axis, 2, 2),
                                  'numH numW dimH dimW t -> numH numW dimH (t dimW)')
        # counts
        self.counts = self.counts.unfold(axis, 2, 2).sum(3)
        # locs
        self.locs = rearrange(self.locs.unfold(axis, 2, 2),
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
        # log_target_prev
        self.log_target_prev = self.log_target_prev.unfold(axis, 2, 2).sum(3)
    
    
    def compute_log_target(self):
        logprior = self.Prior.log_prob(self.counts, self.locs, self.features)
        loglik = self.ImageModel.loglikelihood(self.data, self.locs, self.features)
        return logprior + loglik
    
    
    def run(self):
        self.resample()
        self.log_target_prev = self.compute_log_target()
        self.join(axis = 0)
        self.log_target_curr = self.compute_log_target()
        self.weights = (self.log_target_curr - self.log_target_prev).softmax(-1)
        
        self.resample()
        self.log_target_prev = self.compute_log_target()
        self.join(axis = 1)
        self.log_target_curr = self.compute_log_target()
        self.weights = (self.log_target_curr - self.log_target_prev).softmax(-1)
