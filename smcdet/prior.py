# isort: skip_file
import torch
from torch.distributions import Normal, Pareto, Uniform

from smcdet.distributions import TruncatedPareto


class PointProcessPrior(object):
    def __init__(self, num_objects, h_lower, h_upper, w_lower, w_upper):
        self.num_objects = num_objects
        self.max_objects = num_objects
        self.h_lower = torch.tensor(h_lower).float()
        self.h_upper = torch.tensor(h_upper).float()
        self.w_lower = torch.tensor(w_lower).float()
        self.w_upper = torch.tensor(w_upper).float()
        self.update_attrs()

    def update_attrs(self):
        self.loc_prior = Uniform(
            torch.tensor((self.h_lower, self.w_lower)),
            torch.tensor((self.h_upper, self.w_upper)),
        )

    def sample(self, num_catalogs=1, num_tiles_h=1, num_tiles_w=1):
        self.num = num_catalogs
        counts = self.num_objects * torch.ones(num_tiles_h, num_tiles_w, num_catalogs)
        self.counts_mask = torch.arange(self.num_objects).unsqueeze(
            0
        ) < counts.unsqueeze(-1)
        locs = self.loc_prior.sample(
            [num_tiles_h, num_tiles_w, self.num, self.num_objects]
        )
        locs *= self.counts_mask.unsqueeze(-1)

        return [counts, locs]

    def log_prob(self, counts, locs):
        self.counts_mask = torch.arange(0, self.max_objects).unsqueeze(
            0
        ) < counts.unsqueeze(-1)

        log_prior = (self.loc_prior.log_prob(locs).sum(-1) * self.counts_mask).sum(-1)

        return log_prior


class PoissonProcessPrior(PointProcessPrior):
    def __init__(self, num_objects, counts_rate, h_lower, h_upper, w_lower, w_upper):
        self.counts_rate = counts_rate
        super().__init__(num_objects, h_lower, h_upper, w_lower, w_upper)


class GeometricProcessPrior(PointProcessPrior):
    pass


class StarPrior(PointProcessPrior):
    def __init__(self, *args, flux_mean, flux_stdev, **kwargs):
        super().__init__(*args, **kwargs)
        self.flux_mean = flux_mean
        self.flux_stdev = flux_stdev
        self.flux_prior = Normal(self.flux_mean, self.flux_stdev)

    def sample(self, num_catalogs=1, num_tiles_h=1, num_tiles_w=1):
        counts, locs = super().sample(
            num_catalogs,
            num_tiles_h=num_tiles_h,
            num_tiles_w=num_tiles_w,
        )

        fluxes = self.flux_prior.sample(
            [num_tiles_h, num_tiles_w, self.num, self.num_objects]
        )
        fluxes *= self.counts_mask

        return [counts, locs, fluxes]

    def log_prob(self, counts, locs, fluxes):
        log_prior = super().log_prob(counts, locs)

        return log_prior + (self.flux_prior.log_prob(fluxes) * self.counts_mask).sum(-1)


class ParetoStarPrior(PointProcessPrior):
    def __init__(self, *args, flux_scale, flux_alpha, **kwargs):
        super().__init__(*args, **kwargs)
        self.flux_scale = flux_scale
        self.flux_alpha = flux_alpha
        self.flux_prior = Pareto(self.flux_scale, self.flux_alpha)

    def sample(self, num_catalogs=1, num_tiles_h=1, num_tiles_w=1):
        counts, locs = super().sample(
            num_catalogs,
            num_tiles_h=num_tiles_h,
            num_tiles_w=num_tiles_w,
        )

        fluxes = self.flux_prior.sample(
            [num_tiles_h, num_tiles_w, self.num, self.num_objects]
        )
        fluxes *= self.counts_mask

        return [counts, locs, fluxes]

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

    def sample(self, num_catalogs=1, num_tiles_h=1, num_tiles_w=1):
        counts, locs = super().sample(
            num_catalogs,
            num_tiles_h=num_tiles_h,
            num_tiles_w=num_tiles_w,
        )

        fluxes = self.flux_prior.sample(
            [num_tiles_h, num_tiles_w, self.num, self.num_objects]
        )
        fluxes *= self.counts_mask

        return [counts, locs, fluxes]

    def log_prob(self, counts, locs, fluxes):
        log_prior = super().log_prob(counts, locs)

        return log_prior + (
            self.flux_prior.log_prob(fluxes + self.flux_prior.lower * (fluxes == 0))
            * self.counts_mask
        ).sum(-1)
