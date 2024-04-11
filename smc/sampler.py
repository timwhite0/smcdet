import torch
from torch.distributions import Poisson, Normal, Uniform
from smc.distributions import TruncatedDiagonalMVN
from smc.images import ImageAttributes
import matplotlib.pyplot as plt
from scipy.optimize import brentq

class SMCsampler(object):
    def __init__(self,
                 img,
                 img_attr,
                 tile_side_length,
                 prior,
                 max_objects,
                 catalogs_per_block,
                 kernel_num_iters,
                 product_form_multiplier,
                 max_smc_iters):
        self.img = img
        self.img_attr = img_attr
        
        self.prior = prior
        
        self.num_blocks = max_objects + 1
        self.catalogs_per_block = catalogs_per_block
        self.num_catalogs = self.num_blocks * self.catalogs_per_block
        
        self.max_smc_iters = max_smc_iters
        
        self.kernel_num_iters = kernel_num_iters
        self.kernel_fluxes_stdev = 0.1*self.prior.flux_prior.stddev
        self.kernel_locs_stdev = 0.1*tile_side_length # could also be small multiple of self.prior.loc_prior.stddev
        
        self.product_form_multiplier = product_form_multiplier
        
        self.tile_side_length = tile_side_length
        self.num_tiles_h = self.img_attr.img_height//tile_side_length
        self.num_tiles_w = self.img_attr.img_width//tile_side_length
        self.tiles = img.unfold(0,
                                self.tile_side_length,
                                self.tile_side_length).unfold(1,
                                                              self.tile_side_length,
                                                              self.tile_side_length)
        self.tile_attr = ImageAttributes(self.tile_side_length,
                                         self.tile_side_length,
                                         self.prior.max_objects,
                                         self.img_attr.psf_stdev,
                                         self.img_attr.background_intensity)
        
        self.counts, self.fluxes, self.locs = self.prior.sample(num_tiles_h = self.num_tiles_h,
                                                                num_tiles_w = self.num_tiles_w,
                                                                in_blocks = True,
                                                                num_blocks = self.num_blocks,
                                                                catalogs_per_block = self.catalogs_per_block)
        
        self.temperature_prev = torch.zeros(1)
        self.temperature = torch.zeros(1)
        
        self.weights_log_unnorm = torch.zeros(self.num_tiles_h, self.num_tiles_w, self.num_catalogs)
        self.weights_intrablock = torch.stack(torch.split(self.weights_log_unnorm,
                                                          self.catalogs_per_block, dim=2), dim=2).softmax(3)
        self.weights_interblock = self.weights_log_unnorm.softmax(2)
        self.log_normalizing_constant = (self.weights_log_unnorm.exp().mean(2)).log()
        
        self.ESS_threshold_resampling = 0.5 * catalogs_per_block
        self.ESS_threshold_tempering = 0.5 * catalogs_per_block
        self.ESS = 1/(self.weights_intrablock**2).sum(3)
        
        self.has_run = False
        
    def tempered_log_likelihood(self, fluxes, locs, temperature):
        psf = self.tile_attr.tilePSF(locs.shape[3], locs[:,:,:,:,0], locs[:,:,:,:,1])
        
        rate = (psf * fluxes.unsqueeze(3).unsqueeze(3)).sum(5) + self.img_attr.background_intensity
        rate = rate.permute((0, 1, 3, 4, 2))
        
        loglik = Poisson(rate).log_prob(self.tiles.unsqueeze(4)).sum([2, 3])
        tempered_loglik = temperature * loglik

        return tempered_loglik

    def log_target(self, counts, fluxes, locs, temperature):
        return self.prior.log_prob(counts, fluxes, locs) + self.tempered_log_likelihood(fluxes, locs, temperature)

    def tempering_objective(self, log_likelihood, delta):
        log_numerator = 2 * ((delta * log_likelihood).logsumexp(0))
        log_denominator = (2 * delta * log_likelihood).logsumexp(0)

        return (log_numerator - log_denominator).exp() - self.ESS_threshold_tempering

    def temper(self):
        log_likelihood = self.tempered_log_likelihood(self.fluxes, self.locs, 1)
        
        solutions = torch.zeros(self.num_tiles_h, self.num_tiles_w)
        
        for h in range(self.num_tiles_h):
            for w in range(self.num_tiles_w):
                def func(delta):
                    return self.tempering_objective(log_likelihood[h,w], delta)
                
                if func(1 - self.temperature.item()) < 0:
                    solutions[h,w] = brentq(func, 0.0, 1 - self.temperature.item(),
                                            maxiter=500, xtol=1e-6, rtol=1e-6)
                else:
                    solutions[h,w] = 1 - self.temperature.item()
                
        delta = solutions.min()
        
        self.temperature_prev = self.temperature
        self.temperature = self.temperature + delta
    
    def resample(self):
        for block_num in range(self.num_blocks):
            resampled_index = self.weights_intrablock[:,:,block_num,:].flatten(0,1).multinomial(self.catalogs_per_block,
                                                                                                replacement = True).unflatten(0, (self.num_tiles_h, self.num_tiles_w))
            resampled_index = resampled_index.clamp(min = 0, max = self.catalogs_per_block - 1)
            
            lower = block_num*self.catalogs_per_block
            upper = (block_num+1)*self.catalogs_per_block
            
            for h in range(self.num_tiles_h):
                for w in range(self.num_tiles_w):
                    f = self.fluxes[h,w,lower:upper,:]
                    l = self.locs[h,w,lower:upper,:,:]
                    self.fluxes[h,w,lower:upper,:] = f[resampled_index[h,w,:],:]
                    self.locs[h,w,lower:upper,:,:] = l[resampled_index[h,w,:],:,:]
                    
            self.weights_intrablock[:,:,block_num,:] = (1/self.catalogs_per_block) * torch.ones(self.num_tiles_h, self.num_tiles_w, self.catalogs_per_block)
            self.weights_interblock[:,:,lower:upper] = (self.weights_interblock[:,:,lower:upper].sum(2)/self.catalogs_per_block).unsqueeze(2).repeat(1, 1, self.catalogs_per_block)
    
    def MH(self, num_iters, fluxes_stdev, locs_stdev):
        fluxes_proposal_stdev = fluxes_stdev * torch.ones(1)
        locs_proposal_stdev = locs_stdev * torch.ones(1)
        
        count_indicator = torch.arange(1, self.num_blocks).unsqueeze(0) <= self.counts.unsqueeze(3)
        
        fluxes_prev = self.fluxes
        locs_prev = self.locs
        
        for iter in range(num_iters):
            fluxes_proposed = Normal(fluxes_prev, fluxes_proposal_stdev).sample() * count_indicator
            locs_proposed = TruncatedDiagonalMVN(locs_prev, locs_proposal_stdev,
                                                 torch.tensor(0) - torch.tensor(self.prior.pad),
                                                 torch.tensor(self.img_attr.img_height) + torch.tensor(self.prior.pad)).sample() * count_indicator.unsqueeze(4)
            
            log_numerator = self.log_target(self.counts, fluxes_proposed, locs_proposed, self.temperature_prev)
            log_numerator += (TruncatedDiagonalMVN(locs_proposed, locs_proposal_stdev,
                                                   torch.tensor(0) - torch.tensor(self.prior.pad),
                                                   torch.tensor(self.img_attr.img_height) + torch.tensor(self.prior.pad)).log_prob(locs_prev) * count_indicator.unsqueeze(4)).sum([3,4])

            if iter == 0:
                log_denominator = self.log_target(self.counts, fluxes_prev, locs_prev, self.temperature_prev)
                log_denominator += (TruncatedDiagonalMVN(locs_prev, locs_proposal_stdev,
                                                         torch.tensor(0) - torch.tensor(self.prior.pad),
                                                         torch.tensor(self.img_attr.img_height) + torch.tensor(self.prior.pad)).log_prob(locs_proposed) * count_indicator.unsqueeze(4)).sum([3,4])
        
            alpha = (log_numerator - log_denominator).exp().clamp(max = 1)
            prob = Uniform(torch.zeros(self.num_tiles_h, self.num_tiles_w, self.num_catalogs),
                           torch.ones(self.num_tiles_h, self.num_tiles_w, self.num_catalogs)).sample()
            accept = prob <= alpha
            
            fluxes_new = fluxes_proposed * (accept).unsqueeze(3) + fluxes_prev * (~accept).unsqueeze(3)
            locs_new = locs_proposed * (accept).view(self.num_tiles_h, self.num_tiles_w, -1, 1, 1) + locs_prev * (~accept).view(self.num_tiles_h, self.num_tiles_w, -1, 1, 1)
        
            # Cache log_denominator for next iteration
            log_denominator = log_numerator * (accept) + log_denominator * (~accept)
            
            fluxes_prev = fluxes_new
            locs_prev = locs_new
        
        return [fluxes_new, locs_new]
    
    def propagate(self):
        self.fluxes, self.locs = self.MH(num_iters = self.kernel_num_iters,
                                         fluxes_stdev = self.kernel_fluxes_stdev,
                                         locs_stdev = self.kernel_locs_stdev)
        
    def update_weights(self):
        weights_log_incremental = self.tempered_log_likelihood(self.fluxes,
                                                               self.locs,
                                                               self.temperature - self.temperature_prev)
        
        self.weights_log_unnorm = self.weights_interblock.log() + weights_log_incremental
        self.weights_log_unnorm = torch.nan_to_num(self.weights_log_unnorm, -torch.inf)
        
        self.weights_intrablock = torch.stack(torch.split(self.weights_log_unnorm, self.catalogs_per_block, dim=2), dim=2).softmax(3)
        self.weights_interblock = self.weights_log_unnorm.softmax(2)
        
        m = self.weights_log_unnorm.max(2).values
        w = (self.weights_log_unnorm - m.unsqueeze(2)).exp()
        s = w.sum(2)
        self.log_normalizing_constant = self.log_normalizing_constant + m + (s/self.num_catalogs).log()
        
        self.ESS = 1/(self.weights_intrablock**2).sum(3)

    def run_tiles(self, print_progress = True):
        self.iter = 0
        
        print("Starting the tile samplers...")
        
        self.temper()
        self.update_weights()
        
        while 1 - self.temperature >= 1e-4 and self.iter <= self.max_smc_iters:
            self.iter += 1
            
            if print_progress == True and self.iter % 5 == 0:
                print(f"iteration {self.iter}, temperature = {self.temperature.item()}")
            
            self.resample()
            self.propagate()
            self.temper()
            self.update_weights()
        
        print("Done!\n")
    
    def resample_interblock(self, m):
        print("Combining the results...")
        
        resample_index = self.weights_interblock.flatten(0,1).multinomial(m * self.catalogs_per_block,
                                                                          replacement = True).unflatten(0, (self.num_tiles_h, self.num_tiles_w))
        
        c = torch.zeros(self.num_tiles_h, self.num_tiles_w, m * self.catalogs_per_block)
        f = torch.zeros(self.num_tiles_h, self.num_tiles_w, m * self.catalogs_per_block, self.prior.max_objects)
        l = torch.zeros(self.num_tiles_h, self.num_tiles_w, m * self.catalogs_per_block, self.prior.max_objects, 2)

        for h in range(self.num_tiles_h):
            for w in range(self.num_tiles_w):
                c[h,w] = self.counts[h,w,resample_index[h,w,:]]
                f[h,w] = self.fluxes[h,w,resample_index[h,w,:],:]
                l[h,w] = self.locs[h,w,resample_index[h,w,:],:,:]
        
        self.counts = c
        self.fluxes = f
        self.locs = l
        
        print("Done!\n")
    
    def prune(self):
        print("Pruning detections...")
        
        invalid_sources = torch.any(torch.logical_or(self.locs < 0,
                                                     self.locs > self.tile_side_length), dim = 4)
        invalid_catalogs = invalid_sources.sum(3)
        
        self.counts -= invalid_catalogs
        
        print("Done!\n")
    
    def run(self, print_progress = True):
        self.run_tiles(print_progress)
        
        self.resample_interblock(self.product_form_multiplier)
        
        self.prune()
        
        self.has_run = True
    
    @property
    def image_counts(self):
        if self.has_run == False:
            raise ValueError("Sampler hasn't been run yet.")
        return self.counts.sum([0,1])
    
    @property
    def image_total_flux(self):
        if self.has_run == False:
            raise ValueError("Sampler hasn't been run yet.")
        return self.fluxes.sum([0,1,3])
    
    @property
    def posterior_mean_count(self):
        if self.has_run == False:
            raise ValueError("Sampler hasn't been run yet.")
        return self.image_counts.mean()
    
    @property
    def posterior_mean_total_flux(self):
        if self.has_run == False:
            raise ValueError("Sampler hasn't been run yet.")
        return self.image_total_flux.mean()
    
    # @property
    # def reconstructed_image(self):
    #     if self.has_run == False:
    #         raise ValueError("Sampler hasn't been run yet.")
    #     argmax_index = self.weights_interblock.argmax()
    #     return ((self.img_attr.PSF(self.locs.shape[1],
    #                                self.locs[argmax_index,:,0],
    #                                self.locs[argmax_index,:,1]
    #             ) * self.fluxes[argmax_index,:].view(1, 1, -1)).sum(3) + self.img_attr.background_intensity).squeeze()
    
    def summarize(self, display_images = True):
        if self.has_run == False:
            raise ValueError("Sampler hasn't been run yet.")
        
        print(f"summary\nnumber of SMC iterations: {self.iter}")
        
        # print(f"log normalizing constant: {self.log_normalizing_constant}")
        
        print(f"posterior mean count: {self.posterior_mean_count}")
        print(f"posterior mean total flux: {self.posterior_mean_total_flux}\n\n\n")
        
        # if display_images == True:
        #     fig, (original, reconstruction) = plt.subplots(nrows = 1, ncols = 2)
        #     _ = original.imshow(self.img.cpu(), origin='lower')
        #     _ = original.set_title('original')
        #     _ = reconstruction.imshow(self.reconstructed_image.cpu(), origin='lower')
        #     _ = reconstruction.set_title('reconstruction')