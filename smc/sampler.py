import torch
from torch.distributions import Poisson, Normal, Uniform
from smc.distributions import TruncatedDiagonalMVN
from smc.images import ImageAttributes
import matplotlib.pyplot as plt

class SMCsampler(object):
    def __init__(self,
                 img,
                 img_attr,
                 tile_side_length,
                 prior,
                 num_blocks,
                 catalogs_per_block,
                 product_form_multiplier,
                 max_smc_iters):
        self.img = img
        self.img_attr = img_attr
        
        self.prior = prior
        
        self.num_blocks = num_blocks
        self.catalogs_per_block = catalogs_per_block
        self.num_catalogs = self.num_blocks * self.catalogs_per_block
        
        self.max_smc_iters = max_smc_iters
        
        self.tempering_tol = 1e-6
        self.tempering_max_iters = 100
        
        self.kernel_num_iters = 100
        self.kernel_fluxes_stdev = 1000
        self.kernel_locs_stdev = 0.25
        
        self.m = product_form_multiplier
        
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
        
        self.temperatures_prev = torch.zeros(self.num_tiles_h, self.num_tiles_w, self.num_blocks)
        self.temperatures = torch.zeros(self.num_tiles_h, self.num_tiles_w, self.num_blocks)
        
        self.weights_log_unnorm = torch.zeros(self.num_tiles_h, self.num_tiles_w, self.num_catalogs)
        self.weights_intrablock = torch.stack(torch.split(self.weights_log_unnorm,
                                                          self.catalogs_per_block, dim=2), dim=2).softmax(3)
        self.weights_interblock = self.weights_log_unnorm.softmax(2)
        self.log_normalizing_constant = (self.weights_log_unnorm.exp().mean(2)).log()
        
        self.ESS_threshold = 0.5 * catalogs_per_block
        self.ESS = 1/(self.weights_intrablock**2).sum(3)
        
        self.has_run = False
        
    def tempered_log_likelihood(self, fluxes, locs, temperatures):
        psf = self.tile_attr.tilePSF(locs.shape[3], locs[:,:,:,:,0], locs[:,:,:,:,1])
        
        rate = (psf * fluxes.unsqueeze(3).unsqueeze(3)).sum(5) + self.img_attr.background_intensity
        rate = rate.permute((0, 1, 3, 4, 2))
        
        loglik = Poisson(rate).log_prob(self.tiles.unsqueeze(4)).sum([2, 3])
        tempered_loglik = temperatures.unsqueeze(3) * torch.stack(torch.split(loglik,
                                                                              self.catalogs_per_block, dim=2), dim=2)

        return tempered_loglik

    def log_target(self, counts, fluxes, locs, temperature):
        return self.prior.log_prob(counts, fluxes, locs) + self.tempered_log_likelihood(fluxes, locs, temperature).flatten(2)

    def tempering_objective(self, delta):
        log_numerator = 2*self.tempered_log_likelihood(self.fluxes, self.locs, delta).logsumexp(dim=3)
        log_denominator = self.tempered_log_likelihood(self.fluxes, self.locs, 2*delta).logsumexp(dim=3)

        return (log_numerator - log_denominator).exp() - self.ESS_threshold

    def temper(self):
        a = torch.zeros(self.num_tiles_h, self.num_tiles_w, self.num_blocks)
        b = 1 - self.temperatures
        c = (a+b)/2
        
        f_a = torch.ones(self.num_tiles_h, self.num_tiles_w, self.num_blocks)
        f_b = torch.ones(self.num_tiles_h, self.num_tiles_w, self.num_blocks)
        f_c = torch.ones(self.num_tiles_h, self.num_tiles_w, self.num_blocks)
        
        # For every tile, compute increase in tau for every block using the bisection method
        for j in range(self.tempering_max_iters):
            if torch.all((b-a).abs() <= self.tempering_tol):
                break

            f_a = self.tempering_objective(a)
            f_b = self.tempering_objective(b)
            f_c = self.tempering_objective(c)
            
            a[f_a.sign() == f_c.sign()] = c[f_a.sign() == f_c.sign()]
            b[f_b.sign() == f_c.sign()] = c[f_b.sign() == f_c.sign()]

            c = (a+b)/2

        # Set the increase in tau to be the (small quantile)th increase across the tiles and blocks
        c = c.min().repeat(self.num_tiles_h, self.num_tiles_w, self.num_blocks)

        self.temperatures_prev = self.temperatures
        self.temperatures = self.temperatures + c
    
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
            
            log_numerator = self.log_target(self.counts, fluxes_proposed, locs_proposed, self.temperatures_prev)
            log_numerator += (TruncatedDiagonalMVN(locs_proposed, locs_proposal_stdev,
                                                   torch.tensor(0) - torch.tensor(self.prior.pad),
                                                   torch.tensor(self.img_attr.img_height) + torch.tensor(self.prior.pad)).log_prob(locs_prev) * count_indicator.unsqueeze(4)).sum([3,4])

            if iter == 0:
                log_denominator = self.log_target(self.counts, fluxes_prev, locs_prev, self.temperatures_prev)
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
                                                               self.temperatures - self.temperatures_prev).flatten(2)
        
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
        
        while 1 - self.temperatures.unique() >= 1e-4 and self.iter <= self.max_smc_iters:
            self.iter += 1
            
            if print_progress == True and self.iter % 5 == 0:
                print(f"iteration {self.iter}, temperature = {self.temperatures.unique().item()}")
            
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
        
        self.resample_interblock(self.m)
        
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