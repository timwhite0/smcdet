import torch

class Aggregate(object):
    def __init__(self,
                 tiled_image,
                 counts,
                 locs,
                 features,
                 weights):
        self.tiled_image = tiled_image
        self.counts = counts
        self.locs = locs
        self.features = features
        self.weights = weights
    
    def resample(self):
        ...
    
    def compute_weights(self):
        ...
    
    def merge(self):
        ...
    
    
        

# self.num_aggregation_levels = 2 * torch.tensor(self.tile_dim).log2()