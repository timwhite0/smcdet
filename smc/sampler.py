import torch
from scipy.optimize import brentq

class SMCsampler(object):
    def __init__(self,
                 image,
                 num_tiles_per_side,
                 Prior,
                 ImageModel,
                 MutationKernel,
                 max_objects,
                 num_catalogs,
                 max_smc_iters):
        self.image = image
        self.image_dim = image.shape[0]
        
        self.num_tiles_per_side = num_tiles_per_side
        self.tile_dim = self.image_dim // self.num_tiles_per_side
        self.tiled_image = image.unfold(0,
                                        self.tile_dim,
                                        self.tile_dim).unfold(1,
                                                            self.tile_dim,
                                                            self.tile_dim)
        
        self.Prior = Prior
        self.ImageModel = ImageModel
        self.MutationKernel = MutationKernel
        self.MutationKernel.locs_lower = 0 - self.Prior.pad
        self.MutationKernel.locs_upper = self.tile_dim + self.Prior.pad
        
        self.max_objects = max_objects
        self.num_counts = self.max_objects + 1  # num_counts = |{0,1,2,...,max_objects}|
        self.num_catalogs = num_catalogs
        self.num_catalogs_per_count = self.num_catalogs // self.num_counts
        
        self.max_smc_iters = max_smc_iters
        
        # initialize catalogs
        cats = self.Prior.sample(num_tiles_per_side = self.num_tiles_per_side,
                                 stratify_by_count = True,
                                 num_catalogs_per_count = self.num_catalogs_per_count)
        self.counts, self.locs, self.features = cats
        
        # initialize temperature
        self.temperature_prev = torch.zeros(1)
        self.temperature = torch.zeros(1)
        
        # cache loglikelihood for tempering step
        self.loglik = self.ImageModel.loglikelihood(self.tiled_image, self.locs, self.features)
        
        # initialize weights
        self.weights_log_unnorm = torch.zeros(self.num_tiles_per_side,
                                              self.num_tiles_per_side,
                                              self.num_catalogs)
        self.weights_intracount = torch.stack(torch.split(self.weights_log_unnorm,
                                                          self.num_catalogs_per_count,
                                                          dim = 2), dim = 2).softmax(3)
        self.weights_intercount = self.weights_log_unnorm.softmax(2)
        self.log_normalizing_constant = (self.weights_log_unnorm.exp().mean(2)).log()
        
        # set ESS thresholds
        self.ESS = 1 / (self.weights_intracount ** 2).sum(3)
        self.ESS_threshold_resampling = 0.5 * self.num_catalogs_per_count
        self.ESS_threshold_tempering = 0.5 * self.num_catalogs_per_count
        
        self.has_run = False


    def log_target(self, counts, locs, features, temperature):
        logprior = self.Prior.log_prob(counts, locs, features)
        loglik = self.ImageModel.loglikelihood(self.tiled_image, locs, features)
        
        return logprior + temperature * loglik


    def tempering_objective(self, loglikelihood, delta):
        log_numerator = 2 * ((delta * loglikelihood).logsumexp(0))
        log_denominator = (2 * delta * loglikelihood).logsumexp(0)

        return (log_numerator - log_denominator).exp() - self.ESS_threshold_tempering


    def temper(self):
        self.loglik = self.ImageModel.loglikelihood(self.tiled_image, self.locs, self.fluxes)
        
        solutions = torch.zeros(self.num_tiles_per_side, self.num_tiles_per_side)
        
        for h in range(self.num_tiles_per_side):
            for w in range(self.num_tiles_per_side):
                def func(delta):
                    return self.tempering_objective(self.loglik[h,w], delta)
                
                if func(1 - self.temperature.item()) < 0:
                    solutions[h,w] = brentq(func, 0.0, 1 - self.temperature.item(),
                                            maxiter = 500, xtol = 1e-8, rtol = 1e-8)
                else:
                    solutions[h,w] = 1 - self.temperature.item()
                
        delta = solutions.min()
        
        self.temperature_prev = self.temperature
        self.temperature = self.temperature + delta
    
    
    def resample(self):
        for count_num in range(self.num_counts):
            weights_intracount_flat = self.weights_intracount[:,:,count_num,:].flatten(0,1)
            resampled_index_flat = weights_intracount_flat.multinomial(self.num_catalogs_per_count, replacement = True)
            resampled_index = resampled_index_flat.unflatten(0, (self.num_tiles_per_side, self.num_tiles_per_side))
            resampled_index = resampled_index.clamp(min = 0, max = self.num_catalogs_per_count - 1)
            
            lower = count_num * self.num_catalogs_per_count
            upper = (count_num + 1) * self.num_catalogs_per_count
            
            for h in range(self.num_tiles_per_side):
                for w in range(self.num_tiles_per_side):
                    l = self.locs[h,w,lower:upper,:,:]
                    f = self.features[h,w,lower:upper,:]
                    self.locs[h,w,lower:upper,:,:] = l[resampled_index[h,w,:],:,:]
                    self.features[h,w,lower:upper,:] = f[resampled_index[h,w,:],:]
                    
            self.weights_intracount[:,:,count_num,:] = 1/self.num_catalogs_per_count
            tmp_weights_intercount = (self.weights_intercount[:,:,lower:upper].sum(2) / self.num_catalogs_per_count)
            self.weights_intercount[:,:,lower:upper] = tmp_weights_intercount.unsqueeze(2).repeat(1, 1, self.num_catalogs_per_count)
    
    
    def mutate(self):
        self.locs, self.features = self.MutationKernel.run(self.counts, self.locs, self.features,
                                                           self.temperature_prev, self.log_target)
    
    
    def update_weights(self):
        weights_log_incremental = (self.temperature - self.temperature_prev) * self.loglik
        
        self.weights_log_unnorm = self.weights_intercount.log() + weights_log_incremental
        self.weights_log_unnorm = torch.nan_to_num(self.weights_log_unnorm, -torch.inf)
        
        self.weights_intracount = torch.stack(torch.split(self.weights_log_unnorm,
                                                          self.num_catalogs_per_count,
                                                          dim = 2), dim = 2).softmax(3)
        self.weights_intercount = self.weights_log_unnorm.softmax(2)
        
        m = self.weights_log_unnorm.max(2).values
        w = (self.weights_log_unnorm - m.unsqueeze(2)).exp()
        s = w.sum(2)
        self.log_normalizing_constant = self.log_normalizing_constant + m + (s/self.num_catalogs).log()
        
        self.ESS = 1/(self.weights_intracount ** 2).sum(3)


    def run(self, print_progress = True):
        self.iter = 0
        
        print("Starting the tile samplers...")
        
        self.temper()
        self.update_weights()
        
        while self.temperature < 1 and self.iter <= self.max_smc_iters:
            self.iter += 1
            
            if print_progress == True and self.iter % 5 == 0:
                print(f"iteration {self.iter}, temperature = {self.temperature.item()}")
            
            self.resample()
            self.mutate()
            self.temper()
            self.update_weights()
        
        self.has_run = True
        
        print("Done!\n")
    
    
    @property
    def image_counts(self):
        if self.has_run == False:
            raise ValueError("Sampler hasn't been run yet.")
        
        if self.product_form == True:
            image_counts = self.counts.sum([0,1])
        elif self.product_form == False:
            image_counts = (self.counts.squeeze() * self.weights_intercount).sum()
        return image_counts
    
    
    @property
    def image_total_flux(self):
        if self.has_run == False:
            raise ValueError("Sampler hasn't been run yet.")
        
        if self.product_form == True:
            image_total_flux = self.fluxes.sum([0,1,3])
        elif self.product_form == False:
            image_total_flux = (self.fluxes.squeeze().sum(1) * self.weights_intercount).sum()
        return image_total_flux
    
    
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
    
    
    def summarize(self, display_images = True):
        if self.has_run == False:
            raise ValueError("Sampler hasn't been run yet.")
        
        print(f"summary\nnumber of SMC iterations: {self.iter}")
        
        print(f"posterior mean count: {self.posterior_mean_count}")
        print(f"posterior mean total flux: {self.posterior_mean_total_flux}\n\n\n")
        
        # if display_images == True:
        #     fig, (original, reconstruction) = plt.subplots(nrows = 1, ncols = 2)
        #     _ = original.imshow(self.image.cpu(), origin='lower')
        #     _ = original.set_title('original')
        #     _ = reconstruction.imshow(self.reconstructed_image.cpu(), origin='lower')
        #     _ = reconstruction.set_title('reconstruction')
