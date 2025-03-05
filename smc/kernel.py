import torch
from torch.distributions import Multinomial, Uniform

from smc.distributions import TruncatedDiagonalMVN


class SingleComponentMH(object):
    def __init__(
        self,
        max_iters,
        sqjumpdist_tol,
        locs_stdev,
        fluxes_stdev,
        fluxes_min,
        fluxes_max,
    ):
        self.max_iters = max_iters
        self.sqjumpdist_tol = sqjumpdist_tol

        self.locs_stdev = torch.tensor(locs_stdev)
        self.locs_min = None  # defined automatically within SMCsampler
        self.locs_max = None  # defined automatically within SMCsampler

        self.fluxes_stdev = fluxes_stdev * torch.ones(1)
        self.fluxes_min = fluxes_min * torch.ones(1)
        self.fluxes_max = fluxes_max * torch.ones(1)

    def run(
        self,
        data,
        counts,
        locs,
        fluxes,
        temperature,
        log_target,
        unjoin=None,
        axis=None,
        ChildImageModel=None,
    ):
        counts_mask = torch.arange(1, locs.shape[-2] + 1).unsqueeze(
            0
        ) <= counts.unsqueeze(3)
        component_multinom = Multinomial(
            total_count=1, probs=1 / locs.shape[-2] * counts_mask + 1e-8
        )

        locs_prev = locs
        fluxes_prev = fluxes

        locs_sqjumpdist_prev = self.sqjumpdist_tol
        fluxes_sqjumpdist_prev = self.sqjumpdist_tol

        for iter in range(self.max_iters):
            # choose component to update for each catalog
            component_mask = torch.where(fluxes == 0, 0, component_multinom.sample())

            # propose locs and fluxes
            locs_proposed = locs_prev * (1 - component_mask.unsqueeze(-1)) + (
                TruncatedDiagonalMVN(
                    locs_prev, self.locs_stdev, self.locs_min, self.locs_max
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
            if unjoin is not None and axis is not None and ChildImageModel is not None:
                child_data, _, child_locs_proposed, child_fluxes_proposed = unjoin(
                    axis,
                    data,
                    locs_proposed,
                    fluxes_proposed,
                )
                log_num_target = log_target(
                    axis,
                    ChildImageModel,
                    child_data,
                    child_locs_proposed,
                    child_fluxes_proposed,
                    data,
                    counts,
                    locs_proposed,
                    fluxes_proposed,
                    temperature,
                )
            elif unjoin is None and axis is None and ChildImageModel is None:
                log_num_target = log_target(
                    data,
                    counts,
                    locs_proposed,
                    fluxes_proposed,
                    temperature,
                )
            log_num_qlocs = (
                TruncatedDiagonalMVN(
                    locs_proposed, self.locs_stdev, self.locs_min, self.locs_max
                ).log_prob(locs_prev)
                * component_mask.unsqueeze(-1)
            ).sum([3, 4])
            log_num_qfluxes = (
                TruncatedDiagonalMVN(
                    fluxes_proposed + self.fluxes_min * (fluxes_proposed == 0),
                    self.fluxes_stdev,
                    self.fluxes_min,
                    self.fluxes_max,
                ).log_prob(fluxes_prev + self.fluxes_min * (fluxes_prev == 0))
                * component_mask
            ).sum(3)
            log_numerator = log_num_target + log_num_qlocs + log_num_qfluxes

            # compute log denominator
            if iter == 0:
                if unjoin is not None:
                    child_data, _, child_locs_prev, child_fluxes_prev = unjoin(
                        axis,
                        data,
                        locs_prev,
                        fluxes_prev,
                    )
                    log_denom_target = log_target(
                        axis,
                        ChildImageModel,
                        child_data,
                        child_locs_prev,
                        child_fluxes_prev,
                        data,
                        counts,
                        locs_prev,
                        fluxes_prev,
                        temperature,
                    )
                elif unjoin is None:
                    log_denom_target = log_target(
                        data,
                        counts,
                        locs_prev,
                        fluxes_prev,
                        temperature,
                    )
            log_denom_qlocs = (
                TruncatedDiagonalMVN(
                    locs_prev, self.locs_stdev, self.locs_min, self.locs_max
                ).log_prob(locs_proposed)
                * component_mask.unsqueeze(-1)
            ).sum([3, 4])
            log_denom_qfluxes = (
                TruncatedDiagonalMVN(
                    fluxes_prev + self.fluxes_min * (fluxes_prev == 0),
                    self.fluxes_stdev,
                    self.fluxes_min,
                    self.fluxes_max,
                ).log_prob(fluxes_proposed + self.fluxes_min * (fluxes_proposed == 0))
                * component_mask
            ).sum(3)
            log_denominator = log_denom_target + log_denom_qlocs + log_denom_qfluxes

            alpha = (log_numerator - log_denominator).exp().clamp(max=1)
            prob = Uniform(torch.zeros_like(counts), torch.ones_like(counts)).sample()
            accept = prob <= alpha

            accept_l = (accept).unsqueeze(-1).unsqueeze(-1)
            locs_new = locs_proposed * (accept_l) + locs_prev * (~accept_l)

            accept_f = (accept).unsqueeze(-1)
            fluxes_new = fluxes_proposed * (accept_f) + fluxes_prev * (~accept_f)

            # cache denominator loglik for next iteration
            log_denom_target = log_num_target * (accept) + log_denom_target * (~accept)

            # check relative increase in squared jumping distance
            locs_sqjumpdist = ((locs_new - locs) ** 2).sum()
            locs_stop = (
                locs_sqjumpdist - locs_sqjumpdist_prev
            ) / locs_sqjumpdist_prev < self.sqjumpdist_tol
            fluxes_sqjumpdist = ((fluxes_new - fluxes) ** 2).sum()
            fluxes_stop = (
                fluxes_sqjumpdist - fluxes_sqjumpdist_prev
            ) / fluxes_sqjumpdist_prev < self.sqjumpdist_tol
            if locs_stop and fluxes_stop and iter > 0.1 * self.max_iters:
                break

            locs_prev = locs_new
            locs_sqjumpdist_prev = locs_sqjumpdist
            fluxes_prev = fluxes_new
            fluxes_sqjumpdist_prev = fluxes_sqjumpdist

        return [locs_new, fluxes_new, accept.float().mean(-1)]


# class SingleComponentMALA(object):
#     def __init__(
#         self, max_iters, sqjumpdist_tol, locs_step, fluxes_step, fluxes_min, fluxes_max
#     ):
#         self.max_iters = max_iters
#         self.sqjumpdist_tol = sqjumpdist_tol

#         self.locs_step = torch.tensor(locs_step)
#         self.locs_min = None  # defined automatically within SMCsampler
#         self.locs_max = None  # defined automatically within SMCsampler

#         self.fluxes_step = torch.tensor(fluxes_step)
#         self.fluxes_min = fluxes_min * torch.ones(1)
#         self.fluxes_max = fluxes_max * torch.ones(1)

#     def run(self, data, counts, locs, fluxes, temperature, log_target):
#         counts_mask = torch.arange(1, locs.shape[-2] + 1).unsqueeze(
#             0
#         ) <= counts.unsqueeze(3)
#         component_multinom = Multinomial(
#             total_count=1, probs=1 / locs.shape[-2] * counts_mask + 1e-8
#         )

#         locs_orig = locs
#         fluxes_orig = fluxes

#         locs_sqjumpdist_prev = self.sqjumpdist_tol
#         fluxes_sqjumpdist_prev = self.sqjumpdist_tol

#         for iter in range(self.max_iters):
#             # choose component to update for each catalog
#             component_mask = torch.where(fluxes == 0, 0, component_multinom.sample())

#             # compute gradients
#             locs.requires_grad_(True)
#             fluxes.requires_grad_(True)
#             logtarg = log_target(data, counts, locs, fluxes, temperature)
#             locs_grad, fluxes_grad = torch.autograd.grad(
#                 logtarg,
#                 [locs, fluxes],
#                 grad_outputs=torch.ones_like(logtarg),
#             )
#             locs_grad = locs_grad * component_mask.unsqueeze(-1)
#             fluxes_grad = fluxes_grad * component_mask

#             with torch.no_grad():
#                 # propose locs
#                 locs_proposed_qmean = (
#                     locs + 0.5 * (self.locs_step**2) * locs_grad
#                 ) * component_mask.unsqueeze(-1)
#                 locs_proposed = locs * (
#                     1 - component_mask.unsqueeze(-1)
#                 ) + TruncatedDiagonalMVN(
#                     locs_proposed_qmean, self.locs_step, self.locs_min, self.locs_max
#                 ).sample() * component_mask.unsqueeze(
#                     -1
#                 )

#                 # propose fluxes
#                 fluxes_proposed_qmean = (
#                     fluxes + 0.5 * (self.fluxes_step**2) * fluxes_grad
#                 ) * component_mask
#                 fluxes_proposed = fluxes * (1 - component_mask) + (
#                     TruncatedDiagonalMVN(
#                         fluxes_proposed_qmean,
#                         self.fluxes_step,
#                         self.fluxes_min,
#                         self.fluxes_max,
#                     ).sample()
#                     * component_mask
#                 )

#             # compute log numerator
#             locs_proposed.requires_grad_(True)
#             fluxes_proposed.requires_grad_(True)

#             log_num_target = log_target(
#                 data, counts, locs_proposed, fluxes_proposed, temperature
#             )
#             locs_proposed_grad, fluxes_proposed_grad = torch.autograd.grad(
#                 log_num_target,
#                 [locs_proposed, fluxes_proposed],
#                 grad_outputs=torch.ones_like(log_num_target),
#             )
#             locs_proposed_grad = locs_proposed_grad * component_mask.unsqueeze(-1)
#             fluxes_proposed_grad = fluxes_proposed_grad * component_mask

#             with torch.no_grad():
#                 locs_qmean = (
#                     locs_proposed + 0.5 * (self.locs_step**2) * locs_proposed_grad
#                 ) * component_mask.unsqueeze(-1)
#                 fluxes_qmean = (
#                     fluxes_proposed + 0.5 * (self.fluxes_step**2) * fluxes_proposed_grad
#                 ) * component_mask

#                 log_num_qlocs = (
#                     TruncatedDiagonalMVN(
#                         locs_qmean, self.locs_step, self.locs_min, self.locs_max
#                     ).log_prob(locs)
#                     * component_mask.unsqueeze(-1)
#                 ).sum([3, 4])
#                 log_num_qfluxes = (
#                     TruncatedDiagonalMVN(
#                         fluxes_qmean + self.fluxes_min * (fluxes_proposed == 0),
#                         self.fluxes_step,
#                         self.fluxes_min,
#                         self.fluxes_max,
#                     ).log_prob(fluxes + self.fluxes_min * (fluxes == 0))
#                     * component_mask
#                 ).sum(3)

#                 log_numerator = log_num_target + log_num_qlocs + log_num_qfluxes

#                 # compute log denominator
#                 if iter == 0:
#                     log_denom_target = logtarg

#                 log_denom_qlocs = (
#                     TruncatedDiagonalMVN(
#                         locs_proposed_qmean,
#                         self.locs_step,
#                         self.locs_min,
#                         self.locs_max,
#                     ).log_prob(locs_proposed)
#                     * component_mask.unsqueeze(-1)
#                 ).sum([3, 4])
#                 log_denom_qfluxes = (
#                     TruncatedDiagonalMVN(
#                         fluxes_proposed_qmean + self.fluxes_min * (fluxes == 0),
#                         self.fluxes_step,
#                         self.fluxes_min,
#                         self.fluxes_max,
#                     ).log_prob(
#                         fluxes_proposed + self.fluxes_min * (fluxes_proposed == 0)
#                     )
#                     * component_mask
#                 ).sum(3)
#                 log_denominator = log_denom_target + log_denom_qlocs + log_denom_qfluxes

#                 # accept or reject
#                 alpha = (log_numerator - log_denominator).exp().clamp(max=1)
#                 prob = torch.rand_like(alpha)
#                 accept = prob <= alpha

#                 accept_l = accept.unsqueeze(-1).unsqueeze(-1)
#                 locs = torch.where(accept_l, locs_proposed, locs).detach()
#                 fluxes = torch.where(
#                     accept.unsqueeze(-1), fluxes_proposed, fluxes
#                 ).detach()

#                 # cache log denom target for next iteration
#                 log_denom_target = torch.where(accept, log_num_target, log_denom_target)

#                 # check relative increase in squared jumping distance
#                 locs_sqjumpdist = ((locs - locs_orig) ** 2).sum()
#                 locs_stop = (
#                     locs_sqjumpdist - locs_sqjumpdist_prev
#                 ) / locs_sqjumpdist_prev < self.sqjumpdist_tol
#                 fluxes_sqjumpdist = ((fluxes - fluxes_orig) ** 2).sum()
#                 fluxes_stop = (
#                     fluxes_sqjumpdist - fluxes_sqjumpdist_prev
#                 ) / fluxes_sqjumpdist_prev < self.sqjumpdist_tol
#                 if locs_stop and fluxes_stop and iter > 0.1 * self.max_iters:
#                     break

#                 locs_sqjumpdist_prev = locs_sqjumpdist
#                 fluxes_sqjumpdist_prev = fluxes_sqjumpdist

#         return [locs, fluxes, accept.float().mean(-1)]
