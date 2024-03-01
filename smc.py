import torch
from torch.distributions import Poisson, Normal, Uniform
from distributions import TruncatedDiagonalMVN
from images import PSF
import matplotlib.pyplot as plt

class SMCsampler(object):
    def __init__(self,
                 img,
                 img_attr,
                 prior,
                 num_blocks,
                 catalogs_per_block,
                 max_smc_iters):
        self.img = img
        self.img_attr = img_attr
        
        self.prior = prior
        
        self.num_blocks = num_blocks
        self.catalogs_per_block = catalogs_per_block
        self.num_catalogs = self.num_blocks * self.catalogs_per_block
        
        self.max_smc_iters = max_smc_iters
        
        self.tempering_tol = 1e-6
        self.tempering_max_iters = 50
        
        self.kernel_num_iters = 80
        self.kernel_fluxes_stdev = 1000
        self.kernel_locs_stdev = 0.25
        
        self.counts, self.fluxes, self.locs = self.prior.sample(in_blocks = True,
                                                                num_blocks = self.num_blocks,
                                                                catalogs_per_block = self.catalogs_per_block)
        
        self.temperatures_prev = torch.zeros(self.num_blocks)
        self.temperatures = torch.zeros(self.num_blocks)
        
        self.weights_log_unnorm = torch.zeros(self.num_catalogs)
        self.weights_intrablock = torch.stack(torch.split(self.weights_log_unnorm, self.catalogs_per_block, dim=0), dim = 0).softmax(1)
        self.weights_interblock = self.weights_log_unnorm.softmax(0)
        
        self.ESS_threshold = 0.5 * catalogs_per_block
        self.ESS = 1/(self.weights_intrablock**2).sum(1)
        
        self.has_run = False
        
    def tempered_log_likelihood(self, fluxes, locs, temperatures):
        psf = PSF(self.img_attr.PSF_marginal_W, self.img_attr.PSF_marginal_H,
                  self.num_blocks, locs[:,:,0], locs[:,:,1], self.img_attr.psf_stdev)
        
        rate = (psf * fluxes.view(-1, 1, 1, self.num_blocks)).sum(3) + self.img_attr.background_intensity
        rate = rate.permute((1,2,0))
        
        loglik = Poisson(rate).log_prob(self.img.view(self.img_attr.img_width, self.img_attr.img_height, 1)).sum([0,1])
        tempered_loglik = temperatures.unsqueeze(1) * torch.stack(torch.split(loglik, self.catalogs_per_block, dim=0), dim=0)

        return tempered_loglik

    def log_target(self, counts, fluxes, locs, temperature):
        return self.prior.log_prob(counts, fluxes, locs) + self.tempered_log_likelihood(fluxes, locs, temperature).flatten(0)

    def tempering_objective(self, delta):
        log_numerator = 2*self.tempered_log_likelihood(self.fluxes, self.locs, delta).logsumexp(dim=1)
        log_denominator = self.tempered_log_likelihood(self.fluxes, self.locs, 2*delta).logsumexp(dim=1)

        return (log_numerator - log_denominator).exp() - self.ESS_threshold

    def temper(self):
        a = torch.zeros(self.num_blocks)
        b = 1 - self.temperatures
        c = (a+b)/2
        
        f_a = torch.ones(self.num_blocks)
        f_b = torch.ones(self.num_blocks)
        f_c = torch.ones(self.num_blocks)
        
        # Compute increase in tau for every block using the bisection method
        for j in range(self.tempering_max_iters):
            if torch.all((b-a).abs() <= self.tempering_tol):
                break

            f_a = self.tempering_objective(a)
            f_b = self.tempering_objective(b)
            f_c = self.tempering_objective(c)
            
            a[f_a.sign() == f_c.sign()] = c[f_a.sign() == f_c.sign()]
            b[f_b.sign() == f_c.sign()] = c[f_b.sign() == f_c.sign()]

            c = (a+b)/2

        # For all blocks, set the increase in tau to be the minimum increase across the blocks
        c = c.min(0).values.repeat(self.num_blocks)

        self.temperatures_prev = self.temperatures
        self.temperatures = self.temperatures + c
    
    def resample(self):
        for block_num in range(self.num_blocks):
            if self.ESS[block_num] < self.ESS_threshold:
                u = (torch.arange(self.catalogs_per_block) + torch.rand(1))/self.catalogs_per_block
                bins = self.weights_intrablock[block_num,:].cumsum(0)
                resampled_index = torch.bucketize(u, bins).clamp(min = 0, max = self.catalogs_per_block - 1)
                
                lower = block_num*self.catalogs_per_block
                upper = (block_num+1)*self.catalogs_per_block
                
                f = self.fluxes[lower:upper,:]
                l = self.locs[lower:upper,:,:]
                self.fluxes[lower:upper,:] = f[resampled_index,:]
                self.locs[lower:upper,:,:] = l[resampled_index,:,:]
                
                self.weights_intrablock[block_num,:] = (1/self.catalogs_per_block) * torch.ones(self.catalogs_per_block)
                self.weights_interblock[lower:upper] = (self.weights_interblock[lower:upper].sum(0)/self.catalogs_per_block).unsqueeze(0).expand(self.catalogs_per_block)
    
    def MH(self, num_iters, fluxes_stdev, locs_stdev):
        fluxes_proposal_stdev = fluxes_stdev * torch.ones(1)
        locs_proposal_stdev = locs_stdev * torch.ones(1)
        
        count_indicator = torch.logical_and(torch.arange(self.num_blocks).unsqueeze(0) <= self.counts.unsqueeze(1),
                                            torch.arange(self.num_blocks).unsqueeze(0) > torch.zeros(self.num_catalogs).unsqueeze(1))
        
        fluxes_prev = self.fluxes
        locs_prev = self.locs
        
        for iter in range(num_iters):
            fluxes_proposed = Normal(fluxes_prev, fluxes_proposal_stdev).sample() * count_indicator
            locs_proposed = TruncatedDiagonalMVN(locs_prev, locs_proposal_stdev, torch.tensor(0), torch.tensor(self.img_attr.img_height)).sample() * count_indicator.unsqueeze(2)
            
            log_numerator = self.log_target(self.counts, fluxes_proposed, locs_proposed, self.temperatures)
            log_numerator += (TruncatedDiagonalMVN(locs_proposed, locs_proposal_stdev, torch.tensor(0), torch.tensor(self.img_attr.img_height)).log_prob(locs_prev) * count_indicator.unsqueeze(2)).sum([1,2])

            if iter == 0:
                log_denominator = self.log_target(self.counts, fluxes_prev, locs_prev, self.temperatures)
                log_denominator += (TruncatedDiagonalMVN(locs_prev, locs_proposal_stdev, torch.tensor(0), torch.tensor(self.img_attr.img_height)).log_prob(locs_proposed) * count_indicator.unsqueeze(2)).sum([1,2])
        
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
    
    def propagate(self):
        self.fluxes, self.locs = self.MH(num_iters = self.kernel_num_iters,
                                         fluxes_stdev = self.kernel_fluxes_stdev,
                                         locs_stdev = self.kernel_locs_stdev)
        
    def update_weights(self):
        weights_log_incremental = self.tempered_log_likelihood(self.fluxes,
                                                               self.locs,
                                                               self.temperatures - self.temperatures_prev).flatten(0)
        
        self.weights_log_unnorm = self.weights_interblock.log() + weights_log_incremental
        self.weights_intrablock = torch.stack(torch.split(self.weights_log_unnorm, self.catalogs_per_block, dim=0), dim = 0).softmax(1)
        self.weights_interblock = self.weights_log_unnorm.softmax(0)
        self.ESS = 1/(self.weights_intrablock**2).sum(1)

    def run(self, print_progress = True):
        self.iter = 0
        
        print("Starting the sampler...")
        
        while 1 - self.temperatures.unique() >= 1e-4 and self.iter <= self.max_smc_iters:
            self.iter += 1
            
            if print_progress == True and self.iter % 5 == 0:
                print(f"iteration {self.iter}, temperature = {self.temperatures.unique().item()}, posterior mean count = {(self.weights_interblock * self.counts).sum()}")
            
            self.temper()
            self.resample()
            self.propagate()
            self.update_weights()
        
        print("Done!\n")
        
        self.has_run = True
    
    @property
    def posterior_mean_count(self):
        if self.has_run == False:
            raise ValueError("Sampler hasn't been run yet.")
        return (self.counts * self.weights_interblock).sum()
    
    def summarize(self, display_images = True):
        if self.has_run == False:
            raise ValueError("Sampler hasn't been run yet.")
        
        print(f"summary:\nnumber of SMC iterations: {self.iter}\n")
        
        print(f"posterior mean count: {self.posterior_mean_count}")
        
        argmax_index = self.weights_interblock.argmax()
        print(f"argmax count: {self.counts[argmax_index].item()}")
        print(f"argmax total flux: {self.fluxes[argmax_index].sum().item()}")
        
        if display_images == True:
            reconstructed_image = (PSF(self.img_attr.PSF_marginal_W, self.img_attr.PSF_marginal_H,
                                    self.num_blocks, self.locs[argmax_index,:,0],
                                    self.locs[argmax_index,:,1], self.img_attr.psf_stdev) * self.fluxes[argmax_index,:].view(1, 1, self.num_blocks)).sum(3) + self.img_attr.background_intensity
            fig, (original, reconstruction) = plt.subplots(nrows = 1, ncols = 2)
            _ = original.imshow(self.img.cpu(), origin='lower')
            _ = original.set_title('original')
            _ = reconstruction.imshow(reconstructed_image.squeeze().cpu(), origin='lower')
            _ = reconstruction.set_title('reconstruction')