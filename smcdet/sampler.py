import torch
from scipy.optimize import brentq


class SMCsampler(object):
    def __init__(
        self,
        image,
        tile_dim,
        Prior,
        ImageModel,
        MutationKernel,
        num_catalogs_per_count,
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

        self.min_objects = self.Prior.min_objects
        self.max_objects = self.Prior.max_objects
        self.num_counts = self.max_objects - self.min_objects + 1
        self.count_prior_log_probs = self.Prior.count_prior.log_prob(
            torch.arange(self.min_objects, self.max_objects + 1)
        )
        self.num_catalogs_per_count = num_catalogs_per_count
        self.num_catalogs = self.num_counts * self.num_catalogs_per_count

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
            num_catalogs_per_count=self.num_catalogs_per_count,
        )
        self.counts, self.locs, self.fluxes = cats
        self.fluxes = self.fluxes.clamp(
            min=self.MutationKernel.fluxes_min, max=self.MutationKernel.fluxes_max
        ) * (self.fluxes != 0)

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
        self.weights_intracount = torch.stack(
            torch.split(self.weights_log_unnorm, self.num_catalogs_per_count, dim=2),
            dim=2,
        ).softmax(3)
        self.weights_intercount = self.weights_log_unnorm.softmax(2)
        self.log_normalizing_constant = (
            torch.stack(
                torch.split(
                    self.weights_log_unnorm.exp(), self.num_catalogs_per_count, dim=-1
                ),
                dim=-1,
            )
            .mean(-2)
            .log()
        )

        # set ESS thresholds
        self.ess = 1 / (self.weights_intracount**2).sum(-1)
        self.ess_threshold = ess_threshold_prop * num_catalogs_per_count

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

        solutions = torch.zeros(
            self.num_tiles_per_side, self.num_tiles_per_side, self.num_counts
        )

        for c in range(self.num_counts):
            lower = c * self.num_catalogs_per_count
            upper = (c + 1) * self.num_catalogs_per_count

            for h in range(self.num_tiles_per_side):
                for w in range(self.num_tiles_per_side):

                    def func(delta):
                        return self.tempering_objective(
                            loglik[h, w, lower:upper], delta
                        )

                    if func(1 - self.temperature[h, w].item()) < 0:
                        solutions[h, w, c] = brentq(
                            func,
                            0.0,
                            1 - self.temperature[h, w].item(),
                            xtol=1e-6,
                            rtol=1e-6,
                        )
                    else:
                        solutions[h, w, c] = 1 - self.temperature[h, w].item()

        delta = solutions.min(-1).values

        self.temperature_prev = self.temperature
        self.temperature = self.temperature + delta

    def resample(self):
        for count_num in range(self.num_counts):
            weights = self.weights_intracount[:, :, count_num, :]

            if self.resample_method == "multinomial":
                weights_intracount_flat = weights.flatten(0, 1)
                resampled_index_flat = weights_intracount_flat.multinomial(
                    self.num_catalogs_per_count, replacement=True
                )
                resampled_index = resampled_index_flat.unflatten(
                    0, (self.num_tiles_per_side, self.num_tiles_per_side)
                )
            elif self.resample_method == "systematic":
                resampled_index = torch.zeros_like(weights)
                for h in range(self.num_tiles_per_side):
                    for w in range(self.num_tiles_per_side):
                        u = (
                            torch.arange(self.num_catalogs_per_count) + torch.rand([1])
                        ) / self.num_catalogs_per_count
                        bins = weights[h, w].cumsum(0)
                        resampled_index[h, w] = torch.bucketize(u, bins)

            resampled_index = resampled_index.int().clamp(
                min=0, max=self.num_catalogs_per_count - 1
            )

            lower = count_num * self.num_catalogs_per_count
            upper = (count_num + 1) * self.num_catalogs_per_count

            for h in range(self.num_tiles_per_side):
                for w in range(self.num_tiles_per_side):
                    l = self.locs[h, w, lower:upper, :, :]
                    f = self.fluxes[h, w, lower:upper, :]
                    self.locs[h, w, lower:upper, :, :] = l[
                        resampled_index[h, w, :], :, :
                    ]
                    self.fluxes[h, w, lower:upper, :] = f[resampled_index[h, w, :], :]

            self.weights_intracount[:, :, count_num, :] = (
                1 / self.num_catalogs_per_count
            )

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

        self.weights_intracount = torch.stack(
            torch.split(self.weights_log_unnorm, self.num_catalogs_per_count, dim=2),
            dim=2,
        ).softmax(3)

        self.ess = 1 / (self.weights_intracount**2).sum(3)

        m = (
            torch.stack(
                torch.split(
                    self.weights_log_unnorm, self.num_catalogs_per_count, dim=-1
                ),
                dim=-1,
            )
            .max(-2)
            .values
        )
        w = (
            torch.stack(
                torch.split(
                    self.weights_log_unnorm, self.num_catalogs_per_count, dim=-1
                ),
                dim=-1,
            )
            - m.unsqueeze(-2)
        ).exp()
        s = w.sum(-2)
        self.log_normalizing_constant = (
            self.log_normalizing_constant + m + (s / self.num_catalogs_per_count).log()
        )

        self.weights_intercount = (
            self.weights_intracount
            * (self.count_prior_log_probs + self.log_normalizing_constant)
            .softmax(-1)
            .unsqueeze(-1)
        ).flatten(-2, -1)

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
        return (self.weights_intercount * self.counts).sum(-1)

    def summarize(self):
        if self.has_run is False:
            raise ValueError("Sampler hasn't been run yet.")

        print(f"summary:\nnumber of SMC iterations = {self.iter}\n")
        print(
            f"posterior mean count by tile (including padding):\n{self.posterior_mean_counts}"
        )
