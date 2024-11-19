import torch
from torch.distributions import Categorical, Normal, Pareto, Uniform


class PointProcessPrior(object):
    def __init__(self, max_objects, image_height, image_width, pad=0):
        self.max_objects = max_objects
        self.image_height = image_height
        self.image_width = image_width
        self.pad = pad
        self.update_attrs()

    def update_attrs(self):
        self.num_counts = self.max_objects + 1
        self.count_prior = Categorical(
            (1 / self.num_counts) * torch.ones(self.num_counts)
        )
        self.loc_prior = Uniform(
            (0 - self.pad) * torch.ones(2),
            torch.tensor((self.image_height + self.pad, self.image_width + self.pad)),
        )

    def sample(
        self,
        num_catalogs=1,
        num_tiles_per_side=1,
        stratify_by_count=False,
        num_catalogs_per_count=None,
    ):
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
            counts = self.count_prior.sample(
                [num_tiles_per_side, num_tiles_per_side, self.num]
            )
        elif stratify_by_count is True:
            self.num = self.num_counts * num_catalogs_per_count
            strata = torch.arange(self.num_counts).repeat_interleave(
                num_catalogs_per_count
            )
            counts = strata * torch.ones(
                num_tiles_per_side, num_tiles_per_side, self.num
            )

        self.counts_mask = torch.arange(1, self.num_counts).unsqueeze(
            0
        ) <= counts.unsqueeze(3)
        locs = self.loc_prior.sample(
            [num_tiles_per_side, num_tiles_per_side, self.num, self.max_objects]
        )
        locs *= self.counts_mask.unsqueeze(4)

        return [counts, locs]

    # we define log_prob for stratify_by_count = True, to be used within SMCsampler
    def log_prob(self, counts, locs):
        self.counts_mask = torch.arange(1, self.num_counts).unsqueeze(
            0
        ) <= counts.unsqueeze(-1)

        log_prior = self.count_prior.log_prob(counts)
        log_prior += (
            self.loc_prior.log_prob(locs) * self.counts_mask.unsqueeze(-1)
        ).sum([-1, -2])

        return log_prior


# TODO: Move each subclass of PointProcessPrior to its own case_studies directory
class StarPrior(PointProcessPrior):
    def __init__(self, *args, flux_mean, flux_stdev, **kwargs):
        super().__init__(*args, **kwargs)
        self.flux_mean = flux_mean
        self.flux_stdev = flux_stdev
        self.flux_prior = Normal(self.flux_mean, self.flux_stdev)

    def sample(
        self,
        num_catalogs=1,
        num_tiles_per_side=1,
        stratify_by_count=False,
        num_catalogs_per_count=None,
    ):
        counts, locs = super().sample(
            num_catalogs, num_tiles_per_side, stratify_by_count, num_catalogs_per_count
        )

        fluxes = self.flux_prior.sample(
            [num_tiles_per_side, num_tiles_per_side, self.num, self.max_objects]
        )
        fluxes *= self.counts_mask

        return [counts, locs, fluxes]

    # we define log_prob for stratify_by_count = True, to be used within SMCsampler
    def log_prob(self, counts, locs, fluxes):
        log_prior = super().log_prob(counts, locs)

        return log_prior + (self.flux_prior.log_prob(fluxes) * self.counts_mask).sum(-1)


class M2StarPrior(PointProcessPrior):
    def __init__(self, *args, flux_scale, flux_alpha, **kwargs):
        super().__init__(*args, **kwargs)
        self.flux_scale = flux_scale
        self.flux_alpha = flux_alpha
        self.flux_prior = Pareto(self.flux_scale, self.flux_alpha)

    def sample(
        self,
        num_catalogs=1,
        num_tiles_per_side=1,
        stratify_by_count=False,
        num_catalogs_per_count=None,
    ):
        counts, locs = super().sample(
            num_catalogs, num_tiles_per_side, stratify_by_count, num_catalogs_per_count
        )

        fluxes = self.flux_prior.sample(
            [num_tiles_per_side, num_tiles_per_side, self.num, self.max_objects]
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
