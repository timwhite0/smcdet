import torch
from torch.distributions import Uniform

from smc.distributions import TruncatedDiagonalMVN


class MetropolisHastings(object):
    def __init__(
        self, num_iters, locs_stdev, features_stdev, features_min, features_max
    ):
        self.num_iters = num_iters

        self.locs_stdev = torch.tensor(locs_stdev)
        self.locs_min = None  # defined automatically within SMCsampler
        self.locs_max = None  # defined automatically within SMCsampler

        self.features_stdev = features_stdev * torch.ones(1)
        self.features_min = features_min * torch.ones(1)
        self.features_max = features_max * torch.ones(1)

    def run(self, data, counts, locs, features, temperature, log_target):
        count_indicator = torch.arange(1, locs.shape[-2] + 1).unsqueeze(
            0
        ) <= counts.unsqueeze(3)

        locs_prev = locs
        features_prev = features

        for _ in range(self.num_iters):
            locs_proposed = TruncatedDiagonalMVN(
                locs_prev, self.locs_stdev, self.locs_min, self.locs_max
            ).sample() * count_indicator.unsqueeze(4)
            features_proposed = (
                TruncatedDiagonalMVN(
                    features_prev,
                    self.features_stdev,
                    self.features_min,
                    self.features_max,
                ).sample()
                * count_indicator
            )

            log_numerator = log_target(
                data, counts, locs_proposed, features_proposed, temperature
            )
            log_numerator += (
                TruncatedDiagonalMVN(
                    locs_proposed, self.locs_stdev, self.locs_min, self.locs_max
                ).log_prob(locs_prev)
                * count_indicator.unsqueeze(4)
            ).sum([3, 4])
            log_numerator += (
                TruncatedDiagonalMVN(
                    features_proposed + self.features_min * (features_proposed == 0),
                    self.features_stdev,
                    self.features_min,
                    self.features_max,
                ).log_prob(features_prev + self.features_min * (features_prev == 0))
                * count_indicator
            ).sum(3)

            log_denominator = log_target(
                data, counts, locs_prev, features_prev, temperature
            )
            log_denominator += (
                TruncatedDiagonalMVN(
                    locs_prev, self.locs_stdev, self.locs_min, self.locs_max
                ).log_prob(locs_proposed)
                * count_indicator.unsqueeze(4)
            ).sum([3, 4])
            log_denominator += (
                TruncatedDiagonalMVN(
                    features_prev + self.features_min * (features_prev == 0),
                    self.features_stdev,
                    self.features_min,
                    self.features_max,
                ).log_prob(
                    features_proposed + self.features_min * (features_proposed == 0)
                )
                * count_indicator
            ).sum(3)

            alpha = (log_numerator - log_denominator).exp().clamp(max=1)
            prob = Uniform(torch.zeros_like(counts), torch.ones_like(counts)).sample()
            accept = prob <= alpha

            locs_new = locs_proposed * (accept).unsqueeze(3).unsqueeze(
                4
            ) + locs_prev * (~accept).unsqueeze(3).unsqueeze(4)
            features_new = features_proposed * (accept).unsqueeze(3) + features_prev * (
                ~accept
            ).unsqueeze(3)

            locs_prev = locs_new
            features_prev = features_new

        return [locs_new, features_new]
