import torch
from torch.distributions import Poisson, Normal, Uniform
from smc.distributions import TruncatedDiagonalMVN
from smc.images import ImageAttributes
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from torchvision.transforms.v2.functional import gaussian_blur

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
                                         self.img_attr.psf_size,
                                         self.img_attr.psf_stdev,
                                         self.img_attr.background)
        
        if self.num_tiles_h == 1 and self.num_tiles_w == 1:
            self.product_form = False
        else:
            self.product_form = True
        
        self.product_form_multiplier = product_form_multiplier
        
        self.kernel_num_iters = kernel_num_iters
        
        self.kernel_fluors_stdev = 0.1*self.prior.fluor_prior.stddev * torch.ones(1)
        self.kernel_fluors_low = self.prior.fluor_prior.low * torch.ones(1)
        self.kernel_fluors_high = self.prior.fluor_prior.high * torch.ones(1)
        
        self.kernel_locs_stdev = 0.25 * torch.ones(1)
        self.kernel_locs_low = torch.zeros(1) - torch.tensor(self.prior.pad)
        self.kernel_locs_high = torch.tensor(self.tile_attr.img_height) + torch.tensor(self.prior.pad)
        
        self.kernel_axes_stdev = 0.2*self.prior.axis_prior.stddev.unique() * torch.ones(1)
        self.kernel_axes_low = self.prior.axis_prior.low.unique() * torch.ones(1)
        self.kernel_axes_high = self.prior.axis_prior.high.unique() * torch.ones(1)
        
        self.kernel_angles_stdev = 0.1*self.prior.angle_prior.stddev.unique() * torch.ones(1)
        self.kernel_angles_low = self.prior.angle_prior.low * torch.ones(1)
        self.kernel_angles_high = self.prior.angle_prior.high * torch.ones(1)
        
        
        catalogs = self.prior.sample(num_tiles_h = self.num_tiles_h,
                                     num_tiles_w = self.num_tiles_w,
                                     in_blocks = True,
                                     num_blocks = self.num_blocks,
                                     catalogs_per_block = self.catalogs_per_block)
        self.counts, self.fluors, self.locs, self.axes, self.angles = catalogs
        
        self.temperature_prev = torch.zeros(1)
        self.temperature = torch.zeros(1)
        
        self.loglik = self.tempered_log_likelihood(self.fluors, self.locs, self.axes, self.angles, 1) # for caching in tempering step before weight update
        
        self.weights_log_unnorm = torch.zeros(self.num_tiles_h, self.num_tiles_w, self.num_catalogs)
        self.weights_intrablock = torch.stack(torch.split(self.weights_log_unnorm,
                                                          self.catalogs_per_block, dim=2), dim=2).softmax(3)
        self.weights_interblock = self.weights_log_unnorm.softmax(2)
        self.log_normalizing_constant = (self.weights_log_unnorm.exp().mean(2)).log()
        
        self.ESS_threshold_resampling = 0.5 * catalogs_per_block
        self.ESS_threshold_tempering = 0.5 * catalogs_per_block
        self.ESS = 1/(self.weights_intrablock**2).sum(3)
        
        self.has_run = False
        
        
    def tempered_log_likelihood(self, fluors, locs, axes, angles, temperature):
        ellipse = self.tile_attr.tileEllipse(self.tile_attr.img_height, locs.shape[3], locs, axes, angles)
        
        cell_intensities = (fluors.unsqueeze(3).unsqueeze(3) * ellipse).sum(5)
        
        total_intensities = gaussian_blur(cell_intensities + self.tile_attr.background,
                                          kernel_size = self.tile_attr.psf_size, sigma = self.tile_attr.psf_stdev)
        total_intensities = total_intensities.permute((0, 1, 3, 4, 2))
        
        loglik = Poisson(total_intensities).log_prob(self.tiles.unsqueeze(4)).sum([2, 3])
        tempered_loglik = temperature * loglik

        return tempered_loglik


    def log_target(self, counts, fluors, locs, axes, angles, temperature):
        return self.prior.log_prob(counts, fluors, locs, axes, angles) + self.tempered_log_likelihood(fluors, locs, axes, angles, temperature)


    def tempering_objective(self, log_likelihood, delta):
        log_numerator = 2 * ((delta * log_likelihood).logsumexp(0))
        log_denominator = (2 * delta * log_likelihood).logsumexp(0)
        return (log_numerator - log_denominator).exp() - self.ESS_threshold_tempering


    def temper(self):
        self.loglik = self.tempered_log_likelihood(self.fluors, self.locs, self.axes, self.angles, 1)
        loglik = self.loglik.cpu()
        
        solutions = torch.zeros(self.num_tiles_h, self.num_tiles_w)
        
        for h in range(self.num_tiles_h):
            for w in range(self.num_tiles_w):
                def func(delta):
                    return self.tempering_objective(loglik[h,w], delta)
                
                if func(1 - self.temperature.item()) < 0:
                    solutions[h,w] = brentq(func, 0.0, 1 - self.temperature.item(),
                                            maxiter=500, xtol=1e-8, rtol=1e-8)
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
                    f = self.fluors[h,w,lower:upper,:]
                    l = self.locs[h,w,lower:upper,:,:]
                    ax = self.axes[h,w,lower:upper,:,:]
                    an = self.angles[h,w,lower:upper,:]
                    self.fluors[h,w,lower:upper,:] = f[resampled_index[h,w,:],:]
                    self.locs[h,w,lower:upper,:,:] = l[resampled_index[h,w,:],:,:]
                    self.axes[h,w,lower:upper,:,:] = ax[resampled_index[h,w,:],:,:]
                    self.angles[h,w,lower:upper,:] = an[resampled_index[h,w,:],:]
                    
            self.weights_intrablock[:,:,block_num,:] = (1/self.catalogs_per_block) * torch.ones(self.num_tiles_h, self.num_tiles_w, self.catalogs_per_block)
            self.weights_interblock[:,:,lower:upper] = (self.weights_interblock[:,:,lower:upper].sum(2)/self.catalogs_per_block).unsqueeze(2).repeat(1, 1, self.catalogs_per_block)
    
    
    def MH(self, num_iters):
        
        count_indicator = torch.arange(1, self.num_blocks).unsqueeze(0) <= self.counts.unsqueeze(3)
        
        fluors_prev = self.fluors
        locs_prev = self.locs
        axes_prev = self.axes
        angles_prev = self.angles
        
        for iter in range(num_iters):
            fluors_proposed = (TruncatedDiagonalMVN(fluors_prev, self.kernel_fluors_stdev,
                                                    self.kernel_fluors_low, self.kernel_fluors_high).sample().nan_to_num()) * count_indicator
            locs_proposed = TruncatedDiagonalMVN(locs_prev, self.kernel_locs_stdev,
                                                 self.kernel_locs_low, self.kernel_locs_high).sample() * count_indicator.unsqueeze(4)
            axes_proposed = (TruncatedDiagonalMVN(axes_prev, self.kernel_axes_stdev,
                                                  self.kernel_axes_low, self.kernel_axes_high).sample().nan_to_num()) * count_indicator.unsqueeze(4)
            angles_unconstrained = TruncatedDiagonalMVN(angles_prev, self.kernel_angles_stdev,
                                                        self.kernel_angles_low, self.kernel_angles_high).sample()
            angles_proposed = (angles_unconstrained + torch.pi * (angles_unconstrained <= 0.) - torch.pi * (angles_unconstrained >= torch.pi)) * count_indicator
            
            log_numerator = self.log_target(self.counts, fluors_proposed, locs_proposed, axes_proposed, angles_proposed, self.temperature_prev)
            log_numerator += (TruncatedDiagonalMVN(fluors_proposed, self.kernel_fluors_stdev,
                                                   self.kernel_fluors_low,
                                                   self.kernel_fluors_high).log_prob(fluors_prev +
                                                                                     self.prior.fluor_prior.mean * (fluors_prev==0.)) * count_indicator).nan_to_num().sum(3)
            log_numerator += (TruncatedDiagonalMVN(locs_proposed, self.kernel_locs_stdev,
                                                   self.kernel_locs_low, self.kernel_locs_high).log_prob(locs_prev) * count_indicator.unsqueeze(4)).sum([3,4])
            log_numerator += (TruncatedDiagonalMVN(axes_proposed, self.kernel_axes_stdev,
                                                   self.kernel_axes_low,
                                                   self.kernel_axes_high).log_prob(axes_prev +
                                                                                   self.prior.axis_prior.mean.unique() * (axes_prev==0.)) * count_indicator.unsqueeze(4)).nan_to_num().sum([3,4])
            log_numerator += TruncatedDiagonalMVN(angles_proposed, self.kernel_angles_stdev,
                                                  self.kernel_angles_low,
                                                  self.kernel_angles_high).log_prob(angles_prev).sum(3)

            if iter == 0:
                log_denominator = self.log_target(self.counts, fluors_prev, locs_prev, axes_prev, angles_prev, self.temperature_prev)
                log_denominator += (TruncatedDiagonalMVN(fluors_prev, self.kernel_fluors_stdev,
                                                         self.kernel_fluors_low,
                                                         self.kernel_fluors_high).log_prob(fluors_proposed +
                                                                                           self.prior.fluor_prior.mean * (fluors_proposed==0.)) * count_indicator).nan_to_num().sum(3)
                log_denominator += (TruncatedDiagonalMVN(locs_prev, self.kernel_locs_stdev,
                                                         self.kernel_locs_low, self.kernel_locs_high).log_prob(locs_proposed) * count_indicator.unsqueeze(4)).sum([3,4])
                log_denominator += (TruncatedDiagonalMVN(axes_prev, self.kernel_axes_stdev,
                                                         self.kernel_axes_low,
                                                         self.kernel_axes_high).log_prob(axes_proposed +
                                                                                         self.prior.axis_prior.mean.unique() * (axes_proposed==0.)) * count_indicator.unsqueeze(4)).nan_to_num().sum([3,4])
                log_denominator += TruncatedDiagonalMVN(angles_prev, self.kernel_angles_stdev,
                                                        self.kernel_angles_low,
                                                        self.kernel_angles_high).log_prob(angles_proposed).sum(3)
            
            log_alpha = (log_numerator - log_denominator).clamp(max = 0)
            log_p = Uniform(torch.zeros(self.num_tiles_h, self.num_tiles_w, self.num_catalogs),
                            torch.ones(self.num_tiles_h, self.num_tiles_w, self.num_catalogs)).sample().log()
            accept = log_p <= log_alpha
            
            fluors_new = fluors_proposed * (accept).unsqueeze(3) + fluors_prev * (~accept).unsqueeze(3)
            locs_new = locs_proposed * (accept).view(self.num_tiles_h, self.num_tiles_w, -1, 1, 1) + locs_prev * (~accept).view(self.num_tiles_h, self.num_tiles_w, -1, 1, 1)
            axes_new = axes_proposed * (accept).view(self.num_tiles_h, self.num_tiles_w, -1, 1, 1) + axes_prev * (~accept).view(self.num_tiles_h, self.num_tiles_w, -1, 1, 1)
            angles_new = angles_proposed * (accept).unsqueeze(3) + angles_prev * (~accept).unsqueeze(3)

            # Cache log_denominator for next iteration
            log_denominator = log_numerator * (accept) + log_denominator * (~accept)
            
            fluors_prev = fluors_new
            locs_prev = locs_new
            axes_prev = axes_new
            angles_prev = angles_new
        
        return [fluors_new, locs_new, axes_new, angles_new]
    
    
    def mutate(self):
        self.fluors, self.locs, self.axes, self.angles = self.MH(num_iters = self.kernel_num_iters)
        
        
    def update_weights(self):
        weights_log_incremental = (self.temperature - self.temperature_prev) * self.loglik
        
        self.weights_log_unnorm = self.weights_interblock.log() + weights_log_incremental
        self.weights_log_unnorm = torch.nan_to_num(self.weights_log_unnorm, -torch.inf)
        
        self.weights_intrablock = torch.stack(torch.split(self.weights_log_unnorm, self.catalogs_per_block, dim=2), dim=2).softmax(3)
        self.weights_interblock = self.weights_log_unnorm.softmax(2)
        
        self.ESS = 1/(self.weights_intrablock**2).sum(3)


    def run_tiles(self, print_progress = True):
        self.iter = 0
        
        print("Starting the tile samplers...")
        
        self.temper()
        self.update_weights()
        
        while self.temperature < 1 and self.iter <= self.max_smc_iters:
            self.iter += 1
            
            if print_progress == True and self.iter % 2 == 0:
                print(f"iteration {self.iter}, temperature = {self.temperature.item()}")
            
            self.resample()
            self.mutate()
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
        ax = torch.zeros(self.num_tiles_h, self.num_tiles_w, m * self.catalogs_per_block, self.prior.max_objects, 2)
        an = torch.zeros(self.num_tiles_h, self.num_tiles_w, m * self.catalogs_per_block, self.prior.max_objects)

        for h in range(self.num_tiles_h):
            for w in range(self.num_tiles_w):
                c[h,w] = self.counts[h,w,resample_index[h,w,:]]
                f[h,w] = self.fluors[h,w,resample_index[h,w,:],:]
                l[h,w] = self.locs[h,w,resample_index[h,w,:],:,:]
                ax[h,w] = self.axes[h,w,resample_index[h,w,:],:,:]
                an[h,w] = self.angles[h,w,resample_index[h,w,:],:]
        
        self.counts = c
        self.fluors = f
        self.locs = l
        self.axes = ax
        self.angles = an
        
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
        
        if self.product_form == True:
            self.resample_interblock(self.product_form_multiplier)
            self.prune()
        
        self.has_run = True
    
    
    @property
    def image_counts(self):
        if self.has_run == False:
            raise ValueError("Sampler hasn't been run yet.")
        
        if self.product_form == True:
            image_counts = self.counts.sum([0,1])
        elif self.product_form == False:
            image_counts = (self.counts.squeeze() * self.weights_interblock).sum()
        return image_counts
    
    
    @property
    def image_total_fluor(self):
        if self.has_run == False:
            raise ValueError("Sampler hasn't been run yet.")
        
        if self.product_form == True:
            image_total_fluor = self.fluors.sum([0,1,3])
        elif self.product_form == False:
            image_total_fluor = (self.fluors.squeeze().sum(1) * self.weights_interblock.squeeze()).sum()
        return image_total_fluor
    
    
    @property
    def posterior_mean_count(self):
        if self.has_run == False:
            raise ValueError("Sampler hasn't been run yet.")
        return self.image_counts.mean()
    
    
    @property
    def posterior_mean_total_fluor(self):
        if self.has_run == False:
            raise ValueError("Sampler hasn't been run yet.")
        return self.image_total_fluor.mean()
    
    
    def summarize(self, display_images = True):
        if self.has_run == False:
            raise ValueError("Sampler hasn't been run yet.")
        
        print(f"summary\nnumber of SMC iterations: {self.iter}")
                
        print(f"posterior mean count: {self.posterior_mean_count}")
        print(f"posterior mean total fluor: {self.posterior_mean_total_fluor}\n\n\n")