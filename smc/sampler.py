import torch
from torch.distributions import Poisson, Normal, Uniform
from smc.distributions import TruncatedDiagonalMVN
import matplotlib.pyplot as plt
from scipy.optimize import brentq

class SMCsampler(object):
    def __init__(self,
                 img,
                 img_attr,
                 prior,
                 num_blocks,
                 catalogs_per_block,
                 max_smc_iters,
                 wastefree = False,
                 wastefree_M = 1):
        self.img = img
        self.img_attr = img_attr
        
        self.prior = prior
        
        self.num_blocks = num_blocks
        self.catalogs_per_block = catalogs_per_block
        self.num_catalogs = self.num_blocks * self.catalogs_per_block
        
        self.max_smc_iters = max_smc_iters
        
        self.wastefree = wastefree
        self.wastefree_M = wastefree_M
        self.wastefree_P = self.catalogs_per_block // self.wastefree_M
        
        self.kernel_num_iters = 100
        self.kernel_fluxes_stdev = 0.1*self.prior.flux_prior.stddev
        self.kernel_locs_stdev = 0.1*self.prior.loc_prior.stddev
        
        self.counts, self.fluxes, self.locs = self.prior.sample(in_blocks = True,
                                                                num_blocks = self.num_blocks,
                                                                catalogs_per_block = self.catalogs_per_block)
        
        self.temperature_prev = torch.zeros(1)
        self.temperature = torch.zeros(1)
        
        self.weights_log_unnorm = torch.zeros(self.num_catalogs)
        self.weights_intrablock = torch.stack(torch.split(self.weights_log_unnorm,
                                                          self.catalogs_per_block, dim=0), dim=0).softmax(1)
        self.weights_interblock = self.weights_log_unnorm.softmax(0)
        self.log_normalizing_constant = 0 #(self.weights_log_unnorm.exp().mean()).log()
        
        self.ESS_threshold_resampling = 0.5 * self.catalogs_per_block
        self.ESS_threshold_tempering = 0.5 * self.num_catalogs
        self.ESS = 1/(self.weights_intrablock**2).sum(1)
        
        self.has_run = False
        
    def tempered_log_likelihood(self, fluxes, locs, temperature):
        psf = self.img_attr.PSF(locs.shape[1], locs[:,:,0], locs[:,:,1])
        
        rate = (psf * fluxes.view(-1, 1, 1, fluxes.shape[1])).sum(3) + self.img_attr.background_intensity
        rate = rate.permute((1,2,0))
        
        loglik = Poisson(rate).log_prob(self.img.view(self.img_attr.img_height,
                                                      self.img_attr.img_width, 1)).sum([0,1])
        tempered_loglik = temperature * loglik

        return tempered_loglik

    def log_target(self, counts, fluxes, locs, temperature):
        return self.prior.log_prob(counts, fluxes, locs) + self.tempered_log_likelihood(fluxes, locs, temperature)

    def tempering_objective(self, delta):
        log_numerator = 2*self.tempered_log_likelihood(self.fluxes, self.locs, delta).logsumexp(0)
        log_denominator = self.tempered_log_likelihood(self.fluxes, self.locs, 2*delta).logsumexp(0)

        return ((log_numerator - log_denominator).exp() - self.ESS_threshold_tempering)

    def temper(self):
        if self.tempering_objective(1 - self.temperature.item()) < 0:
            delta = brentq(self.tempering_objective, 0.0, 1 - self.temperature.item(), maxiter=500)
        else:
            delta = 1 - self.temperature.item()
        
        self.temperature_prev = self.temperature
        self.temperature = self.temperature + delta
    
    # ALTERNATIVE: Solving chi-sq dist separately for each block and taking the minimum of the solutions

    # def tempering_objective(self, fluxes, locs, delta):
    #     log_numerator = 2*self.tempered_log_likelihood(fluxes, locs, delta).logsumexp(0)
    #     log_denominator = self.tempered_log_likelihood(fluxes, locs, 2*delta).logsumexp(0)

    #     return ((log_numerator - log_denominator).exp() - self.ESS_threshold)

    # def temper(self):
    #     f = torch.stack(torch.split(self.fluxes, self.catalogs_per_block, dim=0), dim=0)
    #     l = torch.stack(torch.split(self.locs, self.catalogs_per_block, dim=0), dim=0)
        
    #     solutions = torch.zeros(self.num_blocks - 1)
        
    #     for block_num in range(1, self.num_blocks):
    #         def func(delta):
    #             return self.tempering_objective(f[block_num], l[block_num], delta)

    #         if func(1 - self.temperature.item()) < 0:
    #             solutions[block_num-1] = brentq(func, 0.0, 1 - self.temperature.item(), maxiter=500)
    #         else:
    #             solutions[block_num-1] = 1 - self.temperature.item()
        
    #     delta = solutions.min()
        
    #     self.temperature_prev = self.temperature
    #     self.temperature = self.temperature + delta

    def resample(self):
        if self.wastefree == False:
            num_resample_per_block = self.catalogs_per_block
            resample_threshold = self.ESS_threshold_resampling
        elif self.wastefree == True:
            num_resample_per_block = self.wastefree_M
            resample_threshold = self.catalogs_per_block + 1 # always resample
        
        for block_num in range(self.num_blocks):
            if self.ESS[block_num] < resample_threshold:
                u = (torch.arange(num_resample_per_block) + torch.rand(1))/num_resample_per_block
                bins = self.weights_intrablock[block_num,:].cumsum(0)
                resampled_index = torch.bucketize(u, bins).clamp(min = 0, max = self.catalogs_per_block - 1)
                
                lower = block_num*self.catalogs_per_block
                upper = lower + num_resample_per_block
                
                f = self.fluxes[lower:(lower + self.catalogs_per_block),:]
                l = self.locs[lower:(lower + self.catalogs_per_block),:,:]
                self.fluxes[lower:upper,:] = f[resampled_index,:]
                self.locs[lower:upper,:,:] = l[resampled_index,:,:]
                
                self.weights_intrablock[block_num,:] = (1/self.catalogs_per_block) * torch.ones(self.catalogs_per_block)
                self.weights_interblock[lower:(lower+self.catalogs_per_block)] = (self.weights_interblock[lower:(lower+self.catalogs_per_block)].sum(0)/self.catalogs_per_block).unsqueeze(0).expand(self.catalogs_per_block)
    
    def MH(self, num_iters, fluxes_stdev, locs_stdev):
        count_indicator = torch.arange(1, self.num_blocks).unsqueeze(0) <= self.counts.unsqueeze(1)
        
        fluxes_prev = self.fluxes
        locs_prev = self.locs
        
        for iter in range(num_iters):
            fluxes_proposed = Normal(fluxes_prev, fluxes_stdev).sample() * count_indicator
            locs_proposed = TruncatedDiagonalMVN(locs_prev, locs_stdev,
                                                 torch.tensor(0) - torch.tensor(self.prior.pad),
                                                 torch.tensor(self.img_attr.img_height) + torch.tensor(self.prior.pad)).sample() * count_indicator.unsqueeze(2)
            
            log_numerator = self.log_target(self.counts, fluxes_proposed, locs_proposed, self.temperature_prev)
            log_numerator += (TruncatedDiagonalMVN(locs_proposed, locs_stdev,
                                                   torch.tensor(0) - torch.tensor(self.prior.pad),
                                                   torch.tensor(self.img_attr.img_height) + torch.tensor(self.prior.pad)).log_prob(locs_prev) * count_indicator.unsqueeze(2)).sum([1,2])

            if iter == 0:
                log_denominator = self.log_target(self.counts, fluxes_prev, locs_prev, self.temperature_prev)
                log_denominator += (TruncatedDiagonalMVN(locs_prev, locs_stdev,
                                                         torch.tensor(0) - torch.tensor(self.prior.pad),
                                                         torch.tensor(self.img_attr.img_height) + torch.tensor(self.prior.pad)).log_prob(locs_proposed) * count_indicator.unsqueeze(2)).sum([1,2])
        
            alpha = (log_numerator - log_denominator).exp().clamp(max = 1)
            prob = Uniform(torch.zeros(self.num_catalogs), torch.ones(self.num_catalogs)).sample()
            accept = prob <= alpha
            
            fluxes_new = fluxes_proposed * (accept).unsqueeze(1) + fluxes_prev * (~accept).unsqueeze(1)
            locs_new = locs_proposed * (accept).view(-1, 1, 1) + locs_prev * (~accept).view(-1, 1, 1)
        
            # Cache log_denominator for next iteration
            log_denominator = log_numerator * (accept) + log_denominator * (~accept)
            
            fluxes_prev = fluxes_new
            locs_prev = locs_new
        
        return [fluxes_new, locs_new]
    
    def wastefreeMH(self, M, P, fluxes_stdev, locs_stdev):
        index_prev = (torch.arange(self.num_blocks).unsqueeze(1) * self.catalogs_per_block + torch.arange(M)).flatten().tolist()
        
        count_indicator = torch.arange(1, self.num_blocks).unsqueeze(0) <= self.counts[index_prev].unsqueeze(1)
        
        fluxes_prev = self.fluxes[index_prev]
        locs_prev = self.locs[index_prev]
        
        for p in range(1, P):
            fluxes_proposed = Normal(fluxes_prev, fluxes_stdev).sample() * count_indicator
            locs_proposed = TruncatedDiagonalMVN(locs_prev, locs_stdev,
                                                 torch.tensor(0) - torch.tensor(self.prior.pad),
                                                 torch.tensor(self.img_attr.img_height) + torch.tensor(self.prior.pad)).sample() * count_indicator.unsqueeze(2)
            
            log_numerator = self.log_target(self.counts[index_prev], fluxes_proposed, locs_proposed, self.temperature_prev)
            log_numerator += (TruncatedDiagonalMVN(locs_proposed, locs_stdev,
                                                   torch.tensor(0) - torch.tensor(self.prior.pad),
                                                   torch.tensor(self.img_attr.img_height) + torch.tensor(self.prior.pad)).log_prob(locs_prev) * count_indicator.unsqueeze(2)).sum([1,2])

            if p == 1:
                log_denominator = self.log_target(self.counts[index_prev], fluxes_prev, locs_prev, self.temperature_prev)
                log_denominator += (TruncatedDiagonalMVN(locs_prev, locs_stdev,
                                                         torch.tensor(0) - torch.tensor(self.prior.pad),
                                                         torch.tensor(self.img_attr.img_height) + torch.tensor(self.prior.pad)).log_prob(locs_proposed) * count_indicator.unsqueeze(2)).sum([1,2])
        
            alpha = (log_numerator - log_denominator).exp().clamp(max = 1)
            prob = Uniform(torch.zeros(alpha.shape[0]), torch.ones(alpha.shape[0])).sample()
            accept = prob <= alpha
            
            index_new = [M + i for i in index_prev]
            self.fluxes[index_new] = fluxes_proposed * (accept).unsqueeze(1) + fluxes_prev * (~accept).unsqueeze(1)
            self.locs[index_new] = locs_proposed * (accept).view(-1, 1, 1) + locs_prev * (~accept).view(-1, 1, 1)
            
            # Cache log_denominator for next iteration
            log_denominator = log_numerator * (accept) + log_denominator * (~accept)
            
            fluxes_prev = self.fluxes[index_new]
            locs_prev = self.locs[index_new]
            index_prev = index_new
    
    def propagate(self):
        if self.wastefree == False:
            self.fluxes, self.locs = self.MH(num_iters = self.kernel_num_iters,
                                             fluxes_stdev = self.kernel_fluxes_stdev,
                                             locs_stdev = self.kernel_locs_stdev)
        elif self.wastefree == True:
            self.wastefreeMH(M = self.wastefree_M, P = self.wastefree_P,
                             fluxes_stdev = self.kernel_fluxes_stdev,
                             locs_stdev = self.kernel_locs_stdev)
        
    def update_weights(self):
        weights_log_incremental = self.tempered_log_likelihood(self.fluxes,
                                                               self.locs,
                                                               self.temperature - self.temperature_prev)
        
        self.weights_log_unnorm = self.weights_interblock.log() + weights_log_incremental
        self.weights_log_unnorm = torch.nan_to_num(self.weights_log_unnorm, -torch.inf)
        
        self.weights_intrablock = torch.stack(torch.split(self.weights_log_unnorm, self.catalogs_per_block, dim=0), dim = 0).softmax(1)
        self.weights_interblock = self.weights_log_unnorm.softmax(0)
        
        m = self.weights_log_unnorm.max()
        w = (self.weights_log_unnorm - m).exp()
        s = w.sum()
        self.log_normalizing_constant = self.log_normalizing_constant + m + (s/self.num_catalogs).log()
        
        self.ESS = 1/(self.weights_intrablock**2).sum(1)

    def run(self, print_progress = True):
        self.iter = 0
        
        print("Starting the sampler...")
        
        self.temper()
        self.update_weights()
        
        while 1 - self.temperature >= 1e-4 and self.iter <= self.max_smc_iters:
            self.iter += 1
            
            if print_progress == True and self.iter % 5 == 0:
                print(f"iteration {self.iter}, temperature = {self.temperature.item()}, posterior mean count = {(self.weights_interblock * self.counts).sum()}")
            
            self.resample()
            self.propagate()
            self.temper()
            self.update_weights()
        
        print("Done!\n")
        
        self.has_run = True
    
    @property
    def posterior_mean_count(self):
        if self.has_run == False:
            raise ValueError("Sampler hasn't been run yet.")
        return (self.counts * self.weights_interblock).sum()
    
    @property
    def posterior_mean_total_flux(self):
        if self.has_run == False:
            raise ValueError("Sampler hasn't been run yet.")
        return (self.fluxes.sum(1) * self.weights_interblock).sum()
    
    @property
    def argmax_count(self):
        if self.has_run == False:
            raise ValueError("Sampler hasn't been run yet.")
        argmax_index = self.weights_interblock.argmax()
        return self.counts[argmax_index].item()
    
    @property
    def argmax_total_flux(self):
        if self.has_run == False:
            raise ValueError("Sampler hasn't been run yet.")
        argmax_index = self.weights_interblock.argmax()
        return self.fluxes[argmax_index].sum().item()
    
    @property
    def reconstructed_image(self):
        if self.has_run == False:
            raise ValueError("Sampler hasn't been run yet.")
        argmax_index = self.weights_interblock.argmax()
        return ((self.img_attr.PSF(self.locs.shape[1],
                                   self.locs[argmax_index,:,0],
                                   self.locs[argmax_index,:,1]
                ) * self.fluxes[argmax_index,:].view(1, 1, -1)).sum(3) + self.img_attr.background_intensity).squeeze()
    
    def summarize(self, display_images = True):
        if self.has_run == False:
            raise ValueError("Sampler hasn't been run yet.")
        
        print(f"summary\nnumber of SMC iterations: {self.iter}")
        
        print(f"log normalizing constant: {self.log_normalizing_constant}")
        
        print(f"posterior mean count: {self.posterior_mean_count}")
        print(f"posterior mean total flux: {self.posterior_mean_total_flux}")
        
        print(f"argmax count: {self.argmax_count}")
        print(f"argmax total flux: {self.argmax_total_flux}\n\n\n")
        
        if display_images == True:
            fig, (original, reconstruction) = plt.subplots(nrows = 1, ncols = 2)
            _ = original.imshow(self.img.cpu(), origin='lower')
            _ = original.set_title('original')
            _ = reconstruction.imshow(self.reconstructed_image.cpu(), origin='lower')
            _ = reconstruction.set_title('reconstruction')