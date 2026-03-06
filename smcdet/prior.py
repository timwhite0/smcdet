# isort: skip_file
import torch
from torch.distributions import Geometric, Normal, Pareto, Poisson, Uniform

from smcdet.distributions import DiscreteUniform, TruncatedPareto


class PointProcessPrior(object):
    def __init__(self, min_objects, max_objects, h_lower, h_upper, w_lower, w_upper):
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.h_lower = torch.tensor(h_lower).float()
        self.h_upper = torch.tensor(h_upper).float()
        self.w_lower = torch.tensor(w_lower).float()
        self.w_upper = torch.tensor(w_upper).float()
        self.update_attrs()

    def update_attrs(self):
        self.num_counts = self.max_objects - self.min_objects + 1
        self.count_prior = DiscreteUniform(self.min_objects, self.max_objects)
        self.loc_prior = Uniform(
            torch.tensor((self.h_lower, self.w_lower)),
            torch.tensor((self.h_upper, self.w_upper)),
        )

    def sample(
        self,
        num_catalogs=1,
        num_tiles_per_side=None,
        num_tiles_h=1,
        num_tiles_w=1,
        stratify_by_count=False,
        num_catalogs_per_count=None,
    ):
        if num_tiles_per_side is not None:
            num_tiles_h = num_tiles_per_side
            num_tiles_w = num_tiles_per_side

        if stratify_by_count is True and num_catalogs_per_count is None:
            raise ValueError(
                "If stratify_by_count is True, need to specify catalogs_per_count."
            )
        elif stratify_by_count is False and num_catalogs_per_count is not None:
            raise ValueError(
                "If stratify_by_count is False, do not specify catalogs_per_count."
            )

        if stratify_by_count is False:
            self.num = num_catalogs
            count_indices = self.count_prior.sample(
                [num_tiles_h, num_tiles_w, self.num]
            ).int()
            counts = torch.arange(self.min_objects, self.max_objects + 1)[count_indices]
        elif stratify_by_count is True:
            self.num = self.num_counts * num_catalogs_per_count
            strata = torch.arange(
                self.min_objects, self.max_objects + 1
            ).repeat_interleave(num_catalogs_per_count)
            counts = strata * torch.ones(num_tiles_h, num_tiles_w, self.num)

        self.counts_mask = torch.arange(0, self.max_objects).unsqueeze(
            0
        ) < counts.unsqueeze(3)
        locs = self.loc_prior.sample(
            [num_tiles_h, num_tiles_w, self.num, self.max_objects]
        )
        locs *= self.counts_mask.unsqueeze(4)

        return [counts, locs]

    # we define log_prob for stratify_by_count = True, to be used within SMCsampler
    def log_prob(self, counts, locs):
        self.counts_mask = torch.arange(0, self.max_objects).unsqueeze(
            0
        ) < counts.unsqueeze(-1)

        log_prior = self.count_prior.log_prob(counts)
        log_prior += (self.loc_prior.log_prob(locs).sum(-1) * self.counts_mask).sum(-1)

        return log_prior


class PoissonProcessPrior(PointProcessPrior):
    def __init__(
        self,
        min_objects,
        max_objects,
        counts_rate,
        h_lower,
        h_upper,
        w_lower,
        w_upper,
    ):
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.counts_rate = counts_rate
        self.h_lower = torch.tensor(h_lower).float()
        self.h_upper = torch.tensor(h_upper).float()
        self.w_lower = torch.tensor(w_lower).float()
        self.w_upper = torch.tensor(w_upper).float()
        self.update_attrs()

    # Override to change count_prior to Poisson (but keep loc_prior the same)
    def update_attrs(self):
        self.num_counts = self.max_objects - self.min_objects + 1
        self.count_prior = Poisson(
            self.counts_rate
            * (self.h_upper - self.h_lower)
            * (self.w_upper - self.w_lower)
        )
        self.loc_prior = Uniform(
            torch.tensor((self.h_lower, self.w_lower)),
            torch.tensor((self.h_upper, self.w_upper)),
        )


class GeometricProcessPrior(PointProcessPrior):
    def __init__(self, min_objects, max_objects, h_lower, h_upper, w_lower, w_upper):
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.h_lower = torch.tensor(h_lower).float()
        self.h_upper = torch.tensor(h_upper).float()
        self.w_lower = torch.tensor(w_lower).float()
        self.w_upper = torch.tensor(w_upper).float()
        self.update_attrs()

    # Override to change count_prior to Geometric (but keep loc_prior the same)
    def update_attrs(self):
        self.num_counts = self.max_objects - self.min_objects + 1
        self.count_prior = Geometric(
            1 - torch.exp(torch.tensor(-1.5))
        )  # see Feder et al 2020
        self.loc_prior = Uniform(
            torch.tensor((self.h_lower, self.w_lower)),
            torch.tensor((self.h_upper, self.w_upper)),
        )


class StarPrior(PointProcessPrior):
    def __init__(self, *args, flux_mean, flux_stdev, **kwargs):
        super().__init__(*args, **kwargs)
        self.flux_mean = flux_mean
        self.flux_stdev = flux_stdev
        self.flux_prior = Normal(self.flux_mean, self.flux_stdev)

    def sample(
        self,
        num_catalogs=1,
        num_tiles_per_side=None,
        num_tiles_h=1,
        num_tiles_w=1,
        stratify_by_count=False,
        num_catalogs_per_count=None,
    ):
        if num_tiles_per_side is not None:
            num_tiles_h = num_tiles_per_side
            num_tiles_w = num_tiles_per_side

        counts, locs = super().sample(
            num_catalogs,
            num_tiles_h=num_tiles_h,
            num_tiles_w=num_tiles_w,
            stratify_by_count=stratify_by_count,
            num_catalogs_per_count=num_catalogs_per_count,
        )

        fluxes = self.flux_prior.sample(
            [num_tiles_h, num_tiles_w, self.num, self.max_objects]
        )
        fluxes *= self.counts_mask

        return [counts, locs, fluxes]

    # we define log_prob for stratify_by_count = True, to be used within SMCsampler
    def log_prob(self, counts, locs, fluxes):
        log_prior = super().log_prob(counts, locs)

        return log_prior + (self.flux_prior.log_prob(fluxes) * self.counts_mask).sum(-1)


class ParetoStarPrior(PointProcessPrior):
    def __init__(self, *args, flux_scale, flux_alpha, **kwargs):
        super().__init__(*args, **kwargs)
        self.flux_scale = flux_scale
        self.flux_alpha = flux_alpha
        self.flux_prior = Pareto(self.flux_scale, self.flux_alpha)

    def sample(
        self,
        num_catalogs=1,
        num_tiles_per_side=None,
        num_tiles_h=1,
        num_tiles_w=1,
        stratify_by_count=False,
        num_catalogs_per_count=None,
    ):
        if num_tiles_per_side is not None:
            num_tiles_h = num_tiles_per_side
            num_tiles_w = num_tiles_per_side

        counts, locs = super().sample(
            num_catalogs,
            num_tiles_h=num_tiles_h,
            num_tiles_w=num_tiles_w,
            stratify_by_count=stratify_by_count,
            num_catalogs_per_count=num_catalogs_per_count,
        )

        fluxes = self.flux_prior.sample(
            [num_tiles_h, num_tiles_w, self.num, self.max_objects]
        )
        fluxes *= self.counts_mask

        return [counts, locs, fluxes]

    # we define log_prob for stratify_by_count = True, to be used within SMCsampler
    def log_prob(self, counts, locs, fluxes):
        log_prior = super().log_prob(counts, locs)

        return log_prior + (
            self.flux_prior.log_prob(fluxes + self.flux_scale * (fluxes == 0))
            * self.counts_mask
        ).sum(-1)


class M71Prior(PoissonProcessPrior):
    def __init__(self, *args, flux_alpha, flux_lower, flux_upper, **kwargs):
        super().__init__(*args, **kwargs)
        self.flux_alpha = flux_alpha
        self.flux_lower = flux_lower
        self.flux_upper = flux_upper

        self.flux_prior = TruncatedPareto(flux_alpha, flux_lower, flux_upper)

    def sample(
        self,
        num_catalogs=1,
        num_tiles_per_side=None,
        num_tiles_h=1,
        num_tiles_w=1,
        stratify_by_count=False,
        num_catalogs_per_count=None,
    ):
        if num_tiles_per_side is not None:
            num_tiles_h = num_tiles_per_side
            num_tiles_w = num_tiles_per_side

        counts, locs = super().sample(
            num_catalogs,
            num_tiles_h=num_tiles_h,
            num_tiles_w=num_tiles_w,
            stratify_by_count=stratify_by_count,
            num_catalogs_per_count=num_catalogs_per_count,
        )

        fluxes = self.flux_prior.sample(
            [num_tiles_h, num_tiles_w, self.num, self.max_objects]
        )
        fluxes *= self.counts_mask

        return [counts, locs, fluxes]

    # we define log_prob for stratify_by_count = True, to be used within SMCsampler
    def log_prob(self, counts, locs, fluxes):
        log_prior = super().log_prob(counts, locs)

        return log_prior + (
            self.flux_prior.log_prob(fluxes + self.flux_prior.lower * (fluxes == 0))
            * self.counts_mask
        ).sum(-1)
