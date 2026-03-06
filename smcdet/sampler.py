import torch
from einops import rearrange, repeat
from torch.distributions import Multinomial, Uniform

from smcdet.distributions import TruncatedDiagonalMVN


class SMCsampler(object):
    def __init__(
        self,
        image,
        Prior,
        ImageModel,
        MutationKernel,
        num_catalogs,
        ess_threshold_prop,
        resample_method,
        max_smc_iters,
        prune_flux_lower,
        prune_h_lower=-torch.inf,
        prune_h_upper=torch.inf,
        prune_w_lower=-torch.inf,
        prune_w_upper=torch.inf,
        print_every=5,
        locs_cond=None,
        fluxes_cond=None,
    ):
        self.image = image
        self.image_height = image.shape[-2]
        self.image_width = image.shape[-1]

        self.num_tiles_h = image.shape[0]
        self.num_tiles_w = image.shape[1]

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

        self.pruned_flux_lower = prune_flux_lower

        self.prune_h_lower = torch.as_tensor(prune_h_lower).float()
        self.prune_h_upper = torch.as_tensor(prune_h_upper).float()
        self.prune_w_lower = torch.as_tensor(prune_w_lower).float()
        self.prune_w_upper = torch.as_tensor(prune_w_upper).float()

        self.max_smc_iters = max_smc_iters

        self.print_every = print_every

        self.locs_cond = locs_cond
        self.fluxes_cond = fluxes_cond

        self.has_run = False

    def initialize(self):
        # initialize catalogs
        cats = self.Prior.sample(
            num_tiles_h=self.num_tiles_h,
            num_tiles_w=self.num_tiles_w,
            stratify_by_count=True,
            num_catalogs_per_count=self.num_catalogs,
        )
        self.counts, self.locs, self.fluxes = cats

        # initialize temperature
        self.temperature_prev = torch.zeros(self.num_tiles_h, self.num_tiles_w)
        self.temperature = torch.zeros(self.num_tiles_h, self.num_tiles_w)

        # cache loglikelihood for tempering step
        self.loglik = self.ImageModel.loglikelihood(
            self.image,
            self.locs,
            self.fluxes,
            self.locs_cond,
            self.fluxes_cond,
        )

        # initialize weights and normalizing constant
        self.weights_log_unnorm = torch.zeros(
            self.num_tiles_h, self.num_tiles_w, self.num_catalogs
        )
        self.weights = self.weights_log_unnorm.softmax(-1)
        self.log_normalizing_constant = self.weights_log_unnorm.exp().mean(-1).log()

        # compute initial ess
        self.ess = 1 / (self.weights**2).sum(-1)

    def log_target(self, data, counts, locs, fluxes, temperature):
        logprior = self.Prior.log_prob(counts, locs, fluxes)
        loglik = self.ImageModel.loglikelihood(
            data, locs, fluxes, self.locs_cond, self.fluxes_cond
        )

        return logprior + temperature.unsqueeze(-1) * loglik

    def tempering_objective(self, loglikelihood, delta):
        log_numerator = 2 * ((delta * loglikelihood).logsumexp(0))
        log_denominator = (2 * delta * loglikelihood).logsumexp(0)

        return (log_numerator - log_denominator).exp() - self.ess_threshold

    def _batched_tempering_objective(self, loglik, delta):
        """Evaluate ESS objective for all tiles at once.

        Args:
            loglik: [H, W, N] log-likelihood values
            delta: [H, W] temperature increments

        Returns:
            [H, W] objective values
        """
        delta_expanded = delta.unsqueeze(-1)
        log_numerator = 2 * (delta_expanded * loglik).logsumexp(-1)
        log_denominator = (2 * delta_expanded * loglik).logsumexp(-1)
        return (log_numerator - log_denominator).exp() - self.ess_threshold

    def _vectorized_bisect(self, loglik, upper, mask, num_iters=50, tol=1e-6):
        """Batched bisection over all tiles that need search.

        Args:
            loglik: [H, W, N] log-likelihood values
            upper: [H, W] upper bounds for bisection
            mask: [H, W] bool, which tiles need bisection
            num_iters: maximum bisection iterations
            tol: convergence tolerance

        Returns:
            [H, W] found delta values (only meaningful where mask is True)
        """
        lo = torch.zeros_like(upper)
        hi = upper.clone()

        for _ in range(num_iters):
            mid = (lo + hi) / 2
            obj = self._batched_tempering_objective(loglik, mid)
            # Where objective < 0, root is in [lo, mid]; else in [mid, hi]
            hi = torch.where(mask & (obj < 0), mid, hi)
            lo = torch.where(mask & (obj >= 0), mid, lo)
            if (hi - lo)[mask].max() < tol:
                break

        return (lo + hi) / 2

    def temper(self):
        self.loglik = self.ImageModel.loglikelihood(
            self.image,
            self.locs,
            self.fluxes,
            self.locs_cond,
            self.fluxes_cond,
        )
        loglik = self.loglik

        upper = 1 - self.temperature
        obj_at_upper = self._batched_tempering_objective(loglik, upper)
        needs_search = obj_at_upper < 0

        delta = upper.clone()
        if needs_search.any():
            searched = self._vectorized_bisect(loglik, upper, needs_search)
            delta = torch.where(needs_search, searched, delta)

        self.temperature_prev = self.temperature
        self.temperature = self.temperature + delta

    def resample(self):
        if self.resample_method == "multinomial":
            resampled_index_flat = self.weights.flatten(0, 1).multinomial(
                self.num_catalogs, replacement=True
            )
            resampled_index = resampled_index_flat.unflatten(
                0, (self.num_tiles_h, self.num_tiles_w)
            )
        elif self.resample_method == "systematic":
            seq = repeat(
                torch.arange(self.num_catalogs),
                "n -> numH numW n",
                numH=self.num_tiles_h,
                numW=self.num_tiles_w,
            )
            rand = torch.rand([self.num_tiles_h, self.num_tiles_w])
            u = (seq + rand.unsqueeze(-1)) / self.num_catalogs
            bins = self.weights.cumsum(-1)
            resampled_index = torch.searchsorted(bins, u)

        resampled_index = resampled_index.clamp(min=0, max=self.num_catalogs - 1)
        self.counts = torch.gather(self.counts, -1, resampled_index)
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
        if self.locs_cond is not None:
            self.locs_cond = torch.gather(
                self.locs_cond,
                2,
                repeat(
                    resampled_index,
                    "numH numW n -> numH numW n d t",
                    d=self.locs_cond.shape[-2],
                    t=self.locs_cond.shape[-1],
                ),
            )
        self.fluxes = torch.gather(
            self.fluxes,
            2,
            repeat(
                resampled_index, "numH numW n -> numH numW n d", d=self.fluxes.shape[-1]
            ),
        )
        if self.fluxes_cond is not None:
            self.fluxes_cond = torch.gather(
                self.fluxes_cond,
                2,
                repeat(
                    resampled_index,
                    "numH numW n -> numH numW n d",
                    d=self.fluxes_cond.shape[-1],
                ),
            )
        self.weights = (1 / self.num_catalogs) * torch.ones_like(self.weights)

    def mutate(self):
        if hasattr(self.MutationKernel, "run_incremental"):
            self.locs, self.fluxes, self.mutation_acc_rates = (
                self.MutationKernel.run_incremental(
                    self.image,
                    self.counts,
                    self.locs,
                    self.fluxes,
                    self.temperature,
                    self.ImageModel,
                    self.Prior,
                    self.locs_cond,
                    self.fluxes_cond,
                )
            )
        else:
            self.locs, self.fluxes, self.mutation_acc_rates = self.MutationKernel.run(
                self.image,
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
        lower = torch.stack([self.prune_h_lower, self.prune_w_lower], dim=-1)
        upper = torch.stack([self.prune_h_upper, self.prune_w_upper], dim=-1)
        # Expand bounds to broadcast against locs [numH, numW, n, d, 2]
        # [numH, numW, 2] -> [numH, numW, 1, 1, 2]; [2] -> [1, 1, 1, 1, 2]
        while lower.dim() < locs.dim():
            lower = lower.unsqueeze(-2)
        while upper.dim() < locs.dim():
            upper = upper.unsqueeze(-2)
        mask = torch.all(
            torch.logical_and(locs > lower, locs < upper),
            dim=-1,
        )
        mask *= fluxes > self.pruned_flux_lower

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
        num_samples_total,
        num_samples_burnin,
        keep_every_k: int = 1,
        print_every: int = 1000,
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

        self.num_samples_total = num_samples_total
        self.burn_thin_idx = torch.arange(
            num_samples_burnin, num_samples_total, step=keep_every_k
        )

        self.counts = (
            torch.ones(
                self.num_tiles_per_side, self.num_tiles_per_side, num_samples_total
            )
            * Prior.max_objects
        )
        self.locs = torch.zeros(
            self.num_tiles_per_side,
            self.num_tiles_per_side,
            num_samples_total,
            Prior.max_objects,
            2,
        )
        self.fluxes = torch.zeros(
            self.num_tiles_per_side,
            self.num_tiles_per_side,
            num_samples_total,
            Prior.max_objects,
        )

        _, locs_new, fluxes_new = self.Prior.sample(
            num_tiles_per_side=self.num_tiles_per_side,
            stratify_by_count=True,
            num_catalogs_per_count=1,
        )
        self.locs[..., 0, :, :] = locs_new[..., 0, :, :]
        self.fluxes[..., 0, :] = fluxes_new[..., 0, :]

        self.component_multinom = Multinomial(
            total_count=1,
            probs=1
            / self.fluxes.shape[-1]
            * torch.ones_like(self.fluxes[..., 0, :].unsqueeze(2)),
        )
        self.AcceptRejectDist = Uniform(
            torch.zeros(self.num_tiles_per_side, self.num_tiles_per_side),
            torch.ones(self.num_tiles_per_side, self.num_tiles_per_side),
        )

        self.accept = torch.zeros(
            self.num_tiles_per_side,
            self.num_tiles_per_side,
            num_samples_total - 1,
            dtype=torch.int,
        )

        self.print_every = print_every

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
        locs_prev = self.locs[..., 0, :, :].unsqueeze(2)
        fluxes_prev = self.fluxes[..., 0, :].unsqueeze(2)

        for n in range(self.num_samples_total - 1):
            if (n > 0) and (n % self.print_every == 0):
                mean_acc = self.accept[..., (n - self.print_every) : n].float().mean()
                print(
                    (
                        f"iteration {n}, "
                        f"acceptance rate in past {self.print_every} iters = {mean_acc:.2f}\n"
                    )
                )

            component_mask = self.component_multinom.sample()

            # propose locs and fluxes
            locs_proposed = locs_prev * (1 - component_mask.unsqueeze(-1)) + (
                TruncatedDiagonalMVN(
                    locs_prev,
                    self.locs_stdev,
                    self.locs_min,
                    self.locs_max,
                ).sample()
                * component_mask.unsqueeze(-1)
            )
            fluxes_proposed = fluxes_prev * (1 - component_mask) + (
                TruncatedDiagonalMVN(
                    fluxes_prev,
                    self.fluxes_stdev,
                    self.fluxes_min,
                    self.fluxes_max,
                ).sample()
                * component_mask
            )

            # compute log numerator
            log_num_target = self.log_target(
                self.tiled_image,
                self.counts[..., n + 1].unsqueeze(2),
                locs_proposed,
                fluxes_proposed,
            )
            log_num_qlocs = (
                TruncatedDiagonalMVN(
                    locs_proposed, self.locs_stdev, self.locs_min, self.locs_max
                ).log_prob(locs_prev)
                * component_mask.unsqueeze(-1)
            ).sum([-2, -1])
            log_num_qfluxes = (
                TruncatedDiagonalMVN(
                    fluxes_proposed,
                    self.fluxes_stdev,
                    self.fluxes_min,
                    self.fluxes_max,
                ).log_prob(fluxes_prev)
                * component_mask
            ).sum(-1)
            log_numerator = log_num_target + log_num_qlocs + log_num_qfluxes

            # compute log denominator
            if n == 0:
                log_denom_target = self.log_target(
                    self.tiled_image,
                    self.counts[..., n].unsqueeze(2),
                    locs_prev,
                    fluxes_prev,
                )
            log_denom_qlocs = (
                TruncatedDiagonalMVN(
                    locs_prev,
                    self.locs_stdev,
                    self.locs_min,
                    self.locs_max,
                ).log_prob(locs_proposed)
                * component_mask.unsqueeze(-1)
            ).sum([-2, -1])
            log_denom_qfluxes = (
                TruncatedDiagonalMVN(
                    fluxes_prev,
                    self.fluxes_stdev,
                    self.fluxes_min,
                    self.fluxes_max,
                ).log_prob(fluxes_proposed)
                * component_mask
            ).sum(-1)
            log_denominator = log_denom_target + log_denom_qlocs + log_denom_qfluxes

            alpha = (log_numerator - log_denominator).exp().clamp(max=1).squeeze(-1)
            prob = self.AcceptRejectDist.sample()
            self.accept[..., n] = prob <= alpha

            accept_l = rearrange(self.accept[..., n], "numH numW -> numH numW 1 1 1")
            locs_new = locs_proposed * (accept_l) + locs_prev * (1 - accept_l)
            self.locs[..., n + 1, :, :] = locs_new.squeeze(2)
            locs_prev = locs_new

            accept_f = rearrange(self.accept[..., n], "numH numW -> numH numW 1 1")
            fluxes_new = fluxes_proposed * (accept_f) + fluxes_prev * (1 - accept_f)
            self.fluxes[..., n + 1, :] = fluxes_new.squeeze(2)
            fluxes_prev = fluxes_new

            # cache denominator loglik for next iteration
            accept_ldt = rearrange(self.accept[..., n], "numH numW -> numH numW 1")
            log_denom_target = log_num_target * accept_ldt + log_denom_target * (
                1 - accept_ldt
            )

        # discard burn-in samples and thin the chain
        self.counts = self.counts[..., self.burn_thin_idx]
        self.locs = self.locs[..., self.burn_thin_idx, :, :]
        self.fluxes = self.fluxes[..., self.burn_thin_idx, :]

        # apply location and flux thresholds
        self.pruned_counts, self.pruned_locs, self.pruned_fluxes = self.prune(
            self.locs, self.fluxes
        )

        self.has_run = True

    def posterior_mean_count(self, counts):
        return counts.float().mean(-1)

    def posterior_mean_total_flux(self, fluxes):
        return fluxes.sum(-1).mean()

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
