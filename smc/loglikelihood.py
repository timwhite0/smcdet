# from torch.distributions import Poisson
# from smc.images import ImageAttributes

# def loglikelihood(locs, features):
#     psf = self.tile_attr.tilepsf(locs.shape[3], locs[:,:,:,:,0], locs[:,:,:,:,1])
    
#     rate = (psf * fluxes.unsqueeze(3).unsqueeze(3)).sum(5) + self.img_attr.background_intensity
#     rate = rate.permute((0, 1, 3, 4, 2))
    
#     loglik = Poisson(rate).log_prob(self.tiles.unsqueeze(4)).sum([2,3])

#     return loglik