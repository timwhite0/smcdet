import torch
from torch.distributions import Uniform

from smc.distributions import TruncatedDiagonalMVN


class MetropolisHastings(object):
    def __init__(self, num_iters, locs_stdev, fluxes_stdev, fluxes_min, fluxes_max):
        self.num_iters = num_iters

        self.locs_stdev = torch.tensor(locs_stdev)
        self.locs_min = None  # defined automatically within SMCsampler
        self.locs_max = None  # defined automatically within SMCsampler

        self.fluxes_stdev = fluxes_stdev * torch.ones(1)
        self.fluxes_min = fluxes_min * torch.ones(1)
        self.fluxes_max = fluxes_max * torch.ones(1)

    def run(self, data, counts, locs, fluxes, temperature, log_target):
        counts_mask = torch.arange(1, locs.shape[-2] + 1).unsqueeze(
            0
        ) <= counts.unsqueeze(3)

        locs_prev = locs
        fluxes_prev = fluxes

        for iter in range(self.num_iters):
            locs_proposed = TruncatedDiagonalMVN(
                locs_prev, self.locs_stdev, self.locs_min, self.locs_max
            ).sample() * counts_mask.unsqueeze(4)
            fluxes_proposed = (
                TruncatedDiagonalMVN(
                    fluxes_prev,
                    self.fluxes_stdev,
                    self.fluxes_min,
                    self.fluxes_max,
                ).sample()
                * counts_mask
            )

            log_num_target = log_target(
                data, counts, locs_proposed, fluxes_proposed, temperature
            )
            log_num_qlocs = (
                TruncatedDiagonalMVN(
                    locs_proposed, self.locs_stdev, self.locs_min, self.locs_max
                ).log_prob(locs_prev)
                * counts_mask.unsqueeze(4)
            ).sum([3, 4])
            log_num_qfluxes = (
                TruncatedDiagonalMVN(
                    fluxes_proposed + self.fluxes_min * (fluxes_proposed == 0),
                    self.fluxes_stdev,
                    self.fluxes_min,
                    self.fluxes_max,
                ).log_prob(fluxes_prev + self.fluxes_min * (fluxes_prev == 0))
                * counts_mask
            ).sum(3)
            log_numerator = log_num_target + log_num_qlocs + log_num_qfluxes

            if iter == 0:
                log_denom_target = log_target(
                    data, counts, locs_prev, fluxes_prev, temperature
                )
            log_denom_qlocs = (
                TruncatedDiagonalMVN(
                    locs_prev, self.locs_stdev, self.locs_min, self.locs_max
                ).log_prob(locs_proposed)
                * counts_mask.unsqueeze(4)
            ).sum([3, 4])
            log_denom_qfluxes = (
                TruncatedDiagonalMVN(
                    fluxes_prev + self.fluxes_min * (fluxes_prev == 0),
                    self.fluxes_stdev,
                    self.fluxes_min,
                    self.fluxes_max,
                ).log_prob(fluxes_proposed + self.fluxes_min * (fluxes_proposed == 0))
                * counts_mask
            ).sum(3)
            log_denominator = log_denom_target + log_denom_qlocs + log_denom_qfluxes

            alpha = (log_numerator - log_denominator).exp().clamp(max=1)
            prob = Uniform(torch.zeros_like(counts), torch.ones_like(counts)).sample()
            accept = prob <= alpha

            accept_l = (accept).unsqueeze(3).unsqueeze(4)
            locs_new = locs_proposed * (accept_l) + locs_prev * (~accept_l)

            accept_f = (accept).unsqueeze(3)
            fluxes_new = fluxes_proposed * (accept_f) + fluxes_prev * (~accept_f)

            # cache denominator loglik for next iteration
            log_denom_target = log_num_target * (accept) + log_denom_target * (~accept)

            locs_prev = locs_new
            fluxes_prev = fluxes_new

        return [locs_new, fluxes_new]
