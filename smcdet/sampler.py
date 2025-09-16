import torch
from einops import repeat
from scipy.optimize import brentq


class SMCsampler(object):
    def __init__(
        self,
        image,
        tile_dim,
        Prior,
        ImageModel,
        MutationKernel,
        num_catalogs,
        ess_threshold_prop,
        resample_method,
        max_smc_iters,
        print_every=5,
    ):
        self.image = image
        self.image_dim = image.shape[0]

        self.tile_dim = tile_dim
        self.num_tiles_per_side = self.image_dim // self.tile_dim
        self.tiled_image = image.unfold(0, self.tile_dim, self.tile_dim).unfold(
            1, self.tile_dim, self.tile_dim
        )

        self.Prior = Prior
        self.ImageModel = ImageModel
        self.MutationKernel = MutationKernel
        self.MutationKernel.locs_min = self.Prior.loc_prior.low
        self.MutationKernel.locs_max = self.Prior.loc_prior.high
        self.mutation_acc_rates = None

        self.num_catalogs = num_catalogs

        if resample_method not in {"multinomial", "systematic"}:
            raise ValueError(
                "resample_method must be either multinomial or systematic."
            )
        self.resample_method = resample_method

        self.max_smc_iters = max_smc_iters

        self.print_every = print_every

        # initialize catalogs
        cats = self.Prior.sample(
            num_tiles_per_side=self.num_tiles_per_side,
            stratify_by_count=True,
            num_catalogs_per_count=self.num_catalogs,
        )
        self.counts, self.locs, self.fluxes = cats

        # initialize temperature
        self.temperature_prev = torch.zeros(
            self.num_tiles_per_side, self.num_tiles_per_side
        )
        self.temperature = torch.zeros(self.num_tiles_per_side, self.num_tiles_per_side)

        # cache loglikelihood for tempering step
        self.loglik = self.ImageModel.loglikelihood(
            self.tiled_image, self.locs, self.fluxes
        )

        # initialize weights
        self.weights_log_unnorm = torch.zeros(
            self.num_tiles_per_side, self.num_tiles_per_side, self.num_catalogs
        )
        self.weights = self.weights_log_unnorm.softmax(-1)
        self.log_normalizing_constant = self.weights_log_unnorm.exp().mean(-1).log()

        # set ESS thresholds
        self.ess = 1 / (self.weights**2).sum(-1)
        self.ess_threshold = ess_threshold_prop * num_catalogs

        self.has_run = False

    def log_target(self, data, counts, locs, fluxes, temperature):
        logprior = self.Prior.log_prob(counts, locs, fluxes)
        loglik = self.ImageModel.loglikelihood(data, locs, fluxes)

        return logprior + temperature.unsqueeze(-1) * loglik

    def tempering_objective(self, loglikelihood, delta):
        log_numerator = 2 * ((delta * loglikelihood).logsumexp(0))
        log_denominator = (2 * delta * loglikelihood).logsumexp(0)

        return (log_numerator - log_denominator).exp() - self.ess_threshold

    def temper(self):
        self.loglik = self.ImageModel.loglikelihood(
            self.tiled_image, self.locs, self.fluxes
        )
        loglik = self.loglik.cpu()

        delta = torch.zeros(self.num_tiles_per_side, self.num_tiles_per_side)

        for h in range(self.num_tiles_per_side):
            for w in range(self.num_tiles_per_side):

                def func(delta):
                    return self.tempering_objective(loglik[h, w], delta)

                if func(1 - self.temperature[h, w].item()) < 0:
                    delta[h, w] = brentq(
                        func,
                        0.0,
                        1 - self.temperature[h, w].item(),
                        xtol=1e-6,
                        rtol=1e-6,
                    )
                else:
                    delta[h, w] = 1 - self.temperature[h, w].item()

        self.temperature_prev = self.temperature
        self.temperature = self.temperature + delta

    def resample(self):
        if self.resample_method == "multinomial":
            resampled_index_flat = self.weights.flatten(0, 1).multinomial(
                self.num_catalogs, replacement=True
            )
            resampled_index = resampled_index_flat.unflatten(
                0, (self.num_tiles_per_side, self.num_tiles_per_side)
            )
        elif self.resample_method == "systematic":
            resampled_index = torch.zeros_like(self.weights, dtype=torch.int64)
            seq = repeat(
                torch.arange(self.num_catalogs),
                "n -> numH numW n",
                numH=self.num_tiles_per_side,
                numW=self.num_tiles_per_side,
            )
            rand = torch.rand([self.num_tiles_per_side, self.num_tiles_per_side])
            u = (seq + rand) / self.num_catalogs
            bins = self.weights.cumsum(-1)
            for h in range(self.num_tiles_per_side):
                for w in range(self.num_tiles_per_side):
                    resampled_index[h, w] = torch.bucketize(u[h, w], bins[h, w])

        resampled_index = resampled_index.clamp(min=0, max=self.num_catalogs - 1)
        self.counts = torch.gather(self.counts, -1, resampled_index)
        self.fluxes = torch.gather(
            self.fluxes,
            -2,
            repeat(
                resampled_index, "numH numW n -> numH numW n d", d=self.fluxes.shape[-1]
            ),
        )
        self.locs = torch.gather(
            self.locs,
            -3,
            repeat(
                resampled_index,
                "numH numW n -> numH numW n d t",
                d=self.locs.shape[-2],
                t=self.locs.shape[-1],
            ),
        )
        self.weights = 1 / self.num_catalogs

    def mutate(self):
        self.locs, self.fluxes, self.mutation_acc_rates = self.MutationKernel.run(
            self.tiled_image,
            self.counts,
            self.locs,
            self.fluxes,
            self.temperature,
            self.log_target,
        )

    def update_weights(self):
        self.weights_log_unnorm = torch.nan_to_num(
            (self.temperature - self.temperature_prev).unsqueeze(-1) * self.loglik,
            -torch.inf,
        )

        self.weights = self.weights_log_unnorm.softmax(-1)

        self.ess = 1 / (self.weights**2).sum(-1)

        m = self.weights_log_unnorm.max(-1).values
        w = (self.weights_log_unnorm - m.unsqueeze(-1)).exp()
        s = w.sum(-1)
        self.log_normalizing_constant = (
            self.log_normalizing_constant + m + (s / self.num_catalogs).log()
        )

    def run(self):
        self.iter = 0

        print("starting the tile samplers...")

        self.temper()
        self.update_weights()

        while torch.any(self.temperature < 1) and self.iter <= self.max_smc_iters:
            self.iter += 1

            if self.iter % self.print_every == 0:
                print(
                    (
                        f"iteration {self.iter}: "
                        f"temperature in [{round(self.temperature.min().item(), 2)}, "
                        f"{round(self.temperature.max().item(), 2)}], "
                        f"acceptance rate in [{round(self.mutation_acc_rates.min().item(), 2)}, "
                        f"{round(self.mutation_acc_rates.max().item(), 2)}]"
                    )
                )

            self.resample()
            self.mutate()
            self.temper()
            self.update_weights()

        self.has_run = True

        print("done!\n")

    @property
    def posterior_mean_counts(self):
        return (self.weights * self.counts).sum(-1)

    def summarize(self):
        if self.has_run is False:
            raise ValueError("Sampler hasn't been run yet.")

        print(f"summary:\nnumber of SMC iterations = {self.iter}\n")
        print(
            f"posterior mean count by tile (including padding):\n{self.posterior_mean_counts}"
        )
