import torch
from einops import repeat
from scipy.optimize import brentq
from torch.distributions import Multinomial, Uniform

from smcdet.distributions import TruncatedDiagonalMVN


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
        flux_detection_threshold,
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

        self.num_catalogs = num_catalogs

        self.ess_threshold = ess_threshold_prop * num_catalogs

        if resample_method not in {"multinomial", "systematic"}:
            raise ValueError(
                "resample_method must be either multinomial or systematic."
            )
        self.resample_method = resample_method

        self.flux_detection_threshold = flux_detection_threshold

        self.max_smc_iters = max_smc_iters

        self.print_every = print_every

        self.has_run = False

    def initialize(self):
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

        # initialize weights and normalizing constant
        self.weights_log_unnorm = torch.zeros(
            self.num_tiles_per_side, self.num_tiles_per_side, self.num_catalogs
        )
        self.weights = self.weights_log_unnorm.softmax(-1)
        self.log_normalizing_constant = self.weights_log_unnorm.exp().mean(-1).log()

        # compute initial ess
        self.ess = 1 / (self.weights**2).sum(-1)

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
            u = (seq + rand.unsqueeze(-1)) / self.num_catalogs
            bins = self.weights.cumsum(-1)
            for h in range(self.num_tiles_per_side):
                for w in range(self.num_tiles_per_side):
                    resampled_index[h, w] = torch.bucketize(u[h, w], bins[h, w])

        resampled_index = resampled_index.clamp(min=0, max=self.num_catalogs - 1)
        self.counts = torch.gather(self.counts, -1, resampled_index)
        self.fluxes = torch.gather(
            self.fluxes,
            2,
            repeat(
                resampled_index, "numH numW n -> numH numW n d", d=self.fluxes.shape[-1]
            ),
        )
        self.locs = torch.gather(
            self.locs,
            2,
            repeat(
                resampled_index,
                "numH numW n -> numH numW n d t",
                d=self.locs.shape[-2],
                t=self.locs.shape[-1],
            ),
        )
        self.weights = (1 / self.num_catalogs) * torch.ones_like(self.weights)

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

    def prune(self, locs, fluxes):
        mask = torch.all(
            torch.logical_and(
                locs > 0, locs < torch.tensor((self.tile_dim, self.tile_dim))
            ),
            dim=-1,
        )
        mask *= fluxes > self.flux_detection_threshold

        counts = mask.sum(-1)

        locs = mask.unsqueeze(-1) * locs
        locs_mask = (locs != 0).int()
        locs_index = torch.sort(locs_mask, dim=3, descending=True)[1]
        locs = torch.gather(locs, dim=3, index=locs_index)

        fluxes = mask * fluxes
        fluxes_mask = (fluxes != 0).int()
        fluxes_index = torch.sort(fluxes_mask, dim=3, descending=True)[1]
        fluxes = torch.gather(fluxes, dim=3, index=fluxes_index)

        return counts, locs, fluxes

    def run(self):
        self.iter = 0

        print("starting...")

        self.initialize()
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

        self.resample()
        self.pruned_counts, self.pruned_locs, self.pruned_fluxes = self.prune(
            self.locs, self.fluxes
        )

        self.has_run = True

        print("done!\n")

    def posterior_mean_count(self, counts):
        return (self.weights * counts).sum(-1)

    def posterior_mean_total_flux(self, fluxes):
        return (self.weights * fluxes.sum(-1)).sum(-1)

    @property
    def posterior_predictive_total_observed_flux(self):
        return self.ImageModel.sample(self.locs, self.fluxes).sum([-2, -3]).squeeze()

    def summarize(self):
        if self.has_run is False:
            raise ValueError("Sampler hasn't been run yet.")

        print(
            "posterior distribution of number of detectable stars within image boundary:"
        )
        print(self.pruned_counts.unique(return_counts=True)[0].cpu())
        print(
            (
                self.pruned_counts.unique(return_counts=True)[1]
                / self.pruned_counts.shape[-1]
            )
            .round(decimals=3)
            .cpu(),
            "\n",
        )

        print(
            "posterior mean total intrinsic flux (including undetectable and/or in padding) =",
            f"{self.posterior_mean_total_flux(self.fluxes).item()}\n",
        )

        print(
            "posterior mean total intrinsic flux of detectable stars within image boundary =",
            f"{self.posterior_mean_total_flux(self.pruned_fluxes).item()}\n",
        )

        print(
            f"number of unique catalogs = {self.fluxes[0, 0].sum(-1).unique(dim=0).shape[0]}"
        )


class MHsampler(object):
    def __init__(
        self,
        image,
        tile_dim,
        Prior,
        ImageModel,
        locs_stdev,
        fluxes_stdev,
        flux_detection_threshold,
        num_iters,
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

        self.locs_stdev = torch.tensor(locs_stdev)
        self.locs_min = Prior.loc_prior.low
        self.locs_max = Prior.loc_prior.high
        self.fluxes_stdev = torch.tensor(fluxes_stdev)
        self.fluxes_min = torch.tensor(Prior.flux_lower)
        self.fluxes_max = torch.tensor(Prior.flux_upper)
        self.flux_detection_threshold = flux_detection_threshold

        self.num_iters = num_iters

        self.counts = (
            torch.ones(self.num_tiles_per_side, self.num_tiles_per_side, num_iters)
            * Prior.max_objects
        )
        self.locs = torch.zeros(
            self.num_tiles_per_side,
            self.num_tiles_per_side,
            num_iters,
            Prior.max_objects,
            2,
        )
        self.fluxes = torch.zeros(
            self.num_tiles_per_side,
            self.num_tiles_per_side,
            num_iters,
            Prior.max_objects,
        )

        _, l, f = self.Prior.sample(
            num_tiles_per_side=self.num_tiles_per_side,
            stratify_by_count=True,
            num_catalogs_per_count=1,
        )
        self.locs[..., 0, :, :] = l[..., 0, :, :]
        self.fluxes[..., 0, :] = f[..., 0, :]

        self.component_multinom = Multinomial(
            total_count=1,
            probs=1 / self.fluxes.shape[-1] * torch.ones_like(self.fluxes[..., 0, :]),
        )
        self.AcceptRejectDist = Uniform(
            torch.zeros(self.num_tiles_per_side, self.num_tiles_per_side, 1),
            torch.ones(self.num_tiles_per_side, self.num_tiles_per_side, 1),
        )

        self.has_run = False

    def log_target(self, data, counts, locs, fluxes):
        logprior = self.Prior.log_prob(counts, locs, fluxes)
        loglik = self.ImageModel.loglikelihood(data, locs, fluxes)

        return logprior + loglik

    def prune(self, locs, fluxes):
        mask = torch.all(
            torch.logical_and(
                locs > 0, locs < torch.tensor((self.tile_dim, self.tile_dim))
            ),
            dim=-1,
        )
        mask *= fluxes > self.flux_detection_threshold

        counts = mask.sum(-1)

        locs = mask.unsqueeze(-1) * locs
        locs_mask = (locs != 0).int()
        locs_index = torch.sort(locs_mask, dim=3, descending=True)[1]
        locs = torch.gather(locs, dim=3, index=locs_index)

        fluxes = mask * fluxes
        fluxes_mask = (fluxes != 0).int()
        fluxes_index = torch.sort(fluxes_mask, dim=3, descending=True)[1]
        fluxes = torch.gather(fluxes, dim=3, index=fluxes_index)

        return counts, locs, fluxes

    def run(self):
        for n in range(self.num_iters - 1):
            component_mask = self.component_multinom.sample()

            # propose locs and fluxes
            locs_proposed = self.locs[..., n, :, :].unsqueeze(2) * (
                1 - component_mask.unsqueeze(-1)
            ) + (
                TruncatedDiagonalMVN(
                    self.locs[..., n, :, :].unsqueeze(2),
                    self.locs_stdev,
                    self.locs_min,
                    self.locs_max,
                ).sample()
                * component_mask.unsqueeze(-1)
            )
            fluxes_proposed = self.fluxes[..., n, :].unsqueeze(2) * (
                1 - component_mask
            ) + (
                TruncatedDiagonalMVN(
                    self.fluxes[..., n, :].unsqueeze(2),
                    self.fluxes_stdev,
                    self.fluxes_min,
                    self.fluxes_max,
                ).sample()
                * component_mask
            )

            # compute log numerator
            log_num_target = self.log_target(
                self.tiled_image,
                self.counts[..., n].unsqueeze(2),
                locs_proposed,
                fluxes_proposed,
            )
            log_num_qlocs = (
                TruncatedDiagonalMVN(
                    locs_proposed, self.locs_stdev, self.locs_min, self.locs_max
                ).log_prob(self.locs[..., n, :, :].unsqueeze(2))
                * component_mask.unsqueeze(-1)
            ).sum([-2, -1])
            log_num_qfluxes = (
                TruncatedDiagonalMVN(
                    fluxes_proposed,
                    self.fluxes_stdev,
                    self.fluxes_min,
                    self.fluxes_max,
                ).log_prob(self.fluxes[..., n, :].unsqueeze(2))
                * component_mask
            ).sum(-1)
            log_numerator = log_num_target + log_num_qlocs + log_num_qfluxes

            # compute log denominator
            if n == 0:
                log_denom_target = self.log_target(
                    self.tiled_image,
                    self.counts[..., n].unsqueeze(2),
                    self.locs[..., n, :, :].unsqueeze(2),
                    self.fluxes[..., n, :].unsqueeze(2),
                )
            log_denom_qlocs = (
                TruncatedDiagonalMVN(
                    self.locs[..., n, :, :].unsqueeze(2),
                    self.locs_stdev,
                    self.locs_min,
                    self.locs_max,
                ).log_prob(locs_proposed)
                * component_mask.unsqueeze(-1)
            ).sum([-2, -1])
            log_denom_qfluxes = (
                TruncatedDiagonalMVN(
                    self.fluxes[..., n, :].unsqueeze(2),
                    self.fluxes_stdev,
                    self.fluxes_min,
                    self.fluxes_max,
                ).log_prob(fluxes_proposed)
                * component_mask
            ).sum(-1)
            log_denominator = log_denom_target + log_denom_qlocs + log_denom_qfluxes

            alpha = (log_numerator - log_denominator).exp().clamp(max=1)
            prob = self.AcceptRejectDist.sample()
            accept = prob <= alpha

            accept_l = (accept).unsqueeze(-1).unsqueeze(-1)
            self.locs[..., n + 1, :, :] = locs_proposed * (accept_l) + self.locs[
                ..., n, :, :
            ].unsqueeze(2) * (~accept_l)

            accept_f = (accept).unsqueeze(-1)
            self.fluxes[..., n + 1, :] = fluxes_proposed * (accept_f) + self.fluxes[
                ..., n, :
            ].unsqueeze(2) * (~accept_f)

            # cache denominator loglik for next iteration
            log_denom_target = log_num_target * (accept) + log_denom_target * (~accept)

        self.pruned_counts, self.pruned_locs, self.pruned_fluxes = self.prune(
            self.locs, self.fluxes
        )

        self.has_run = True
