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
        counts_mask = torch.arange(1, locs.shape[-2] + 1).unsqueeze(
            0
        ) <= counts.unsqueeze(3)

        locs_prev = locs
        features_prev = features

        for iter in range(self.num_iters):
            locs_proposed = TruncatedDiagonalMVN(
                locs_prev, self.locs_stdev, self.locs_min, self.locs_max
            ).sample() * counts_mask.unsqueeze(4)
            features_proposed = (
                TruncatedDiagonalMVN(
                    features_prev,
                    self.features_stdev,
                    self.features_min,
                    self.features_max,
                ).sample()
                * counts_mask
            )

            log_num_target = log_target(
                data, counts, locs_proposed, features_proposed, temperature
            )
            log_num_qlocs = (
                TruncatedDiagonalMVN(
                    locs_proposed, self.locs_stdev, self.locs_min, self.locs_max
                ).log_prob(locs_prev)
                * counts_mask.unsqueeze(4)
            ).sum([3, 4])
            log_num_qfeatures = (
                TruncatedDiagonalMVN(
                    features_proposed + self.features_min * (features_proposed == 0),
                    self.features_stdev,
                    self.features_min,
                    self.features_max,
                ).log_prob(features_prev + self.features_min * (features_prev == 0))
                * counts_mask
            ).sum(3)
            log_numerator = log_num_target + log_num_qlocs + log_num_qfeatures

            if iter == 0:
                log_denom_target = log_target(
                    data, counts, locs_prev, features_prev, temperature
                )
            log_denom_qlocs = (
                TruncatedDiagonalMVN(
                    locs_prev, self.locs_stdev, self.locs_min, self.locs_max
                ).log_prob(locs_proposed)
                * counts_mask.unsqueeze(4)
            ).sum([3, 4])
            log_denom_qfeatures = (
                TruncatedDiagonalMVN(
                    features_prev + self.features_min * (features_prev == 0),
                    self.features_stdev,
                    self.features_min,
                    self.features_max,
                ).log_prob(
                    features_proposed + self.features_min * (features_proposed == 0)
                )
                * counts_mask
            ).sum(3)
            log_denominator = log_denom_target + log_denom_qlocs + log_denom_qfeatures

            alpha = (log_numerator - log_denominator).exp().clamp(max=1)
            prob = Uniform(torch.zeros_like(counts), torch.ones_like(counts)).sample()
            accept = prob <= alpha

            accept_l = (accept).unsqueeze(3).unsqueeze(4)
            locs_new = locs_proposed * (accept_l) + locs_prev * (~accept_l)

            accept_f = (accept).unsqueeze(3)
            features_new = features_proposed * (accept_f) + features_prev * (~accept_f)

            # cache denominator loglik for next iteration
            log_denom_target = log_num_target * (accept) + log_denom_target * (~accept)

            locs_prev = locs_new
            features_prev = features_new

        return [locs_new, features_new]
