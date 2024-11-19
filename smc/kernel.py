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
            ).sample() * counts_mask.unsqueeze(-1)
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
                * counts_mask.unsqueeze(-1)
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
                * counts_mask.unsqueeze(-1)
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


class MetropolisAdjustedLangevin(object):
    def __init__(self, num_iters, locs_step, fluxes_step, fluxes_min, fluxes_max):
        self.num_iters = num_iters

        self.locs_step = torch.tensor(locs_step)
        self.locs_min = None  # defined automatically within SMCsampler
        self.locs_max = None  # defined automatically within SMCsampler

        self.fluxes_step = torch.tensor(fluxes_step)
        self.fluxes_min = fluxes_min * torch.ones(1)
        self.fluxes_max = fluxes_max * torch.ones(1)

    def run(self, data, counts, locs, fluxes, temperature, log_target):
        counts_mask = torch.arange(1, locs.shape[-2] + 1).unsqueeze(
            0
        ) <= counts.unsqueeze(3)

        for iter in range(self.num_iters):
            # compute gradients
            locs.requires_grad_(True)
            fluxes.requires_grad_(True)
            logtarg = log_target(data, counts, locs, fluxes, temperature)
            locs_grad, fluxes_grad = torch.autograd.grad(
                logtarg,
                [locs, fluxes],
                grad_outputs=torch.ones_like(logtarg),
            )
            locs_grad = locs_grad * counts_mask.unsqueeze(-1)
            fluxes_grad = fluxes_grad * counts_mask

            # propose locs
            with torch.no_grad():
                locs_proposed_qmean = locs + 0.5 * (self.locs_step**2) * locs_grad
                locs_proposed = TruncatedDiagonalMVN(
                    locs_proposed_qmean, self.locs_step, self.locs_min, self.locs_max
                ).sample() * counts_mask.unsqueeze(-1)

            # propose fluxes
            with torch.no_grad():
                fluxes_proposed_qmean = (
                    fluxes + 0.5 * (self.fluxes_step**2) * fluxes_grad
                )
                fluxes_proposed = (
                    TruncatedDiagonalMVN(
                        fluxes_proposed_qmean,
                        self.fluxes_step,
                        self.fluxes_min,
                        self.fluxes_max,
                    ).sample()
                    * counts_mask
                )

            # compute log numerator
            locs_proposed.requires_grad_(True)
            fluxes_proposed.requires_grad_(True)

            log_num_target = log_target(
                data, counts, locs_proposed, fluxes_proposed, temperature
            )
            locs_proposed_grad, fluxes_proposed_grad = torch.autograd.grad(
                log_num_target,
                [locs_proposed, fluxes_proposed],
                grad_outputs=torch.ones_like(log_num_target),
            )
            locs_proposed_grad = locs_proposed_grad * counts_mask.unsqueeze(-1)
            fluxes_proposed_grad = fluxes_proposed_grad * counts_mask

            with torch.no_grad():
                locs_qmean = (
                    locs_proposed + 0.5 * (self.locs_step**2) * locs_proposed_grad
                )
                fluxes_qmean = (
                    fluxes_proposed + 0.5 * (self.fluxes_step**2) * fluxes_proposed_grad
                )

            log_num_qlocs = (
                TruncatedDiagonalMVN(
                    locs_qmean, self.locs_step, self.locs_min, self.locs_max
                ).log_prob(locs)
                * counts_mask.unsqueeze(-1)
            ).sum([3, 4])
            log_num_qfluxes = (
                TruncatedDiagonalMVN(
                    fluxes_qmean + self.fluxes_min * (fluxes_proposed == 0),
                    self.fluxes_step,
                    self.fluxes_min,
                    self.fluxes_max,
                ).log_prob(fluxes + self.fluxes_min * (fluxes == 0))
                * counts_mask
            ).sum(3)
            log_numerator = log_num_target + log_num_qlocs + log_num_qfluxes

            # compute log denominator
            if iter == 0:
                log_denom_target = logtarg

            log_denom_qlocs = (
                TruncatedDiagonalMVN(
                    locs_proposed_qmean, self.locs_step, self.locs_min, self.locs_max
                ).log_prob(locs_proposed)
                * counts_mask.unsqueeze(-1)
            ).sum([3, 4])
            log_denom_qfluxes = (
                TruncatedDiagonalMVN(
                    fluxes_proposed_qmean + self.fluxes_min * (fluxes == 0),
                    self.fluxes_step,
                    self.fluxes_min,
                    self.fluxes_max,
                ).log_prob(fluxes_proposed + self.fluxes_min * (fluxes_proposed == 0))
                * counts_mask
            ).sum(3)
            log_denominator = log_denom_target + log_denom_qlocs + log_denom_qfluxes

            # accept or reject
            alpha = (log_numerator - log_denominator).exp().clamp(max=1)
            prob = torch.rand_like(alpha)
            accept = prob <= alpha

            accept_l = accept.unsqueeze(3).unsqueeze(4)
            locs = torch.where(accept_l, locs_proposed, locs).detach()
            fluxes = torch.where(accept.unsqueeze(3), fluxes_proposed, fluxes).detach()

            # cache log denom target for next iteration
            log_denom_target = torch.where(accept, log_num_target, log_denom_target)

        return [locs, fluxes]
