import torch
from einops import rearrange
from torch.distributions import Multinomial

from smcdet.distributions import TruncatedDiagonalMVN


class SingleComponentMH(object):
    def __init__(
        self,
        num_iters,
        locs_stdev,
        fluxes_stdev,
        fluxes_min,
        fluxes_max,
    ):
        self.num_iters = num_iters

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
    ):
        component_multinom = Multinomial(
            total_count=1, probs=1 / fluxes.shape[-1] * torch.ones_like(fluxes)
        )

        locs_prev = locs
        fluxes_prev = fluxes

        for iter in range(self.num_iters):
            # choose component to update for each catalog
            component_mask = component_multinom.sample()

            # propose locs and fluxes
            q_locs_fwd = TruncatedDiagonalMVN(
                locs_prev, self.locs_stdev, self.locs_min, self.locs_max
            )
            q_fluxes_fwd = TruncatedDiagonalMVN(
                fluxes_prev,
                self.fluxes_stdev,
                self.fluxes_min,
                self.fluxes_max,
            )
            locs_proposed = locs_prev * (1 - component_mask.unsqueeze(-1)) + (
                q_locs_fwd.sample() * component_mask.unsqueeze(-1)
            )
            fluxes_proposed = fluxes_prev * (1 - component_mask) + (
                q_fluxes_fwd.sample() * component_mask
            )

            # compute reverse proposal distributions
            q_locs_rev = TruncatedDiagonalMVN(
                locs_proposed, self.locs_stdev, self.locs_min, self.locs_max
            )
            q_fluxes_rev = TruncatedDiagonalMVN(
                fluxes_proposed,
                self.fluxes_stdev,
                self.fluxes_min,
                self.fluxes_max,
            )

            # compute log numerator
            log_num_target = log_target(
                data,
                counts,
                locs_proposed,
                fluxes_proposed,
                temperature,
            )
            log_num_qlocs = (
                q_locs_rev.log_prob(locs_prev) * component_mask.unsqueeze(-1)
            ).sum([-2, -1])
            log_num_qfluxes = (q_fluxes_rev.log_prob(fluxes_prev) * component_mask).sum(
                -1
            )
            log_numerator = log_num_target + log_num_qlocs + log_num_qfluxes

            # compute log denominator
            if iter == 0:
                log_denom_target = log_target(
                    data,
                    counts,
                    locs_prev,
                    fluxes_prev,
                    temperature,
                )
            log_denom_qlocs = (
                q_locs_fwd.log_prob(locs_proposed) * component_mask.unsqueeze(-1)
            ).sum([-2, -1])
            log_denom_qfluxes = (
                q_fluxes_fwd.log_prob(fluxes_proposed) * component_mask
            ).sum(-1)
            log_denominator = log_denom_target + log_denom_qlocs + log_denom_qfluxes

            alpha = (log_numerator - log_denominator).exp().clamp(max=1)
            prob = torch.rand_like(alpha)
            accept = prob <= alpha

            accept_l = (accept).unsqueeze(-1).unsqueeze(-1)
            locs_new = locs_proposed * (accept_l) + locs_prev * (~accept_l)

            accept_f = (accept).unsqueeze(-1)
            fluxes_new = fluxes_proposed * (accept_f) + fluxes_prev * (~accept_f)

            # cache denominator loglik for next iteration
            log_denom_target = log_num_target * (accept) + log_denom_target * (~accept)

            locs_prev = locs_new
            fluxes_prev = fluxes_new

        return [locs_new, fluxes_new, accept.float().mean(-1)]

    def run_incremental(
        self,
        data,
        counts,
        locs,
        fluxes,
        temperature,
        image_model,
        prior,
        locs_cond=None,
        fluxes_cond=None,
    ):
        numH, numW, n, d, _ = locs.shape
        fs = image_model.flux_scale

        # Combine with conditioning stars for initial rate
        if locs_cond is not None:
            all_locs = torch.cat((locs, locs_cond), dim=-2)
            all_fluxes = torch.cat((fluxes, fluxes_cond), dim=-1)
        else:
            all_locs = locs
            all_fluxes = fluxes

        # Full PSF and rate (one-time cost)
        psf_all = image_model.psf(all_locs)
        rate = (
            psf_all * rearrange(fs * all_fluxes, "numH numW n d -> numH numW 1 1 n d")
        ).sum(-1) + image_model.background

        # Cache per-star rate contributions (mutable stars only)
        star_contribs = psf_all[..., :d] * rearrange(
            fs * fluxes, "numH numW n d -> numH numW 1 1 n d"
        )  # [numH, numW, H, W, n, d]

        # Initial log target
        logprior = prior.log_prob(counts, locs, fluxes)
        loglik = image_model.loglikelihood_from_rate(data, rate)
        log_denom_target = logprior + temperature.unsqueeze(-1) * loglik

        for it in range(self.num_iters):
            j = it % d

            # Save current star j
            locs_j = locs[:, :, :, j : j + 1, :].clone()
            fluxes_j = fluxes[:, :, :, j : j + 1].clone()

            # Propose new star j
            q_locs_fwd = TruncatedDiagonalMVN(
                locs_j, self.locs_stdev, self.locs_min, self.locs_max
            )
            q_fluxes_fwd = TruncatedDiagonalMVN(
                fluxes_j, self.fluxes_stdev, self.fluxes_min, self.fluxes_max
            )
            locs_j_new = q_locs_fwd.sample()
            fluxes_j_new = q_fluxes_fwd.sample()

            # PSF for proposed star j only (d=1)
            psf_j_new = image_model.psf(locs_j_new)
            new_contrib = (
                psf_j_new
                * rearrange(fs * fluxes_j_new, "numH numW n d -> numH numW 1 1 n d")
            ).squeeze(-1)
            old_contrib = star_contribs[..., j]

            # Incremental rate update
            rate_proposed = rate - old_contrib + new_contrib

            # Temporarily set star j to proposed for prior computation
            locs[:, :, :, j, :] = locs_j_new.squeeze(-2)
            fluxes[:, :, :, j] = fluxes_j_new.squeeze(-1)

            # Log numerator
            logprior_proposed = prior.log_prob(counts, locs, fluxes)
            loglik_proposed = image_model.loglikelihood_from_rate(data, rate_proposed)
            log_num_target = (
                logprior_proposed + temperature.unsqueeze(-1) * loglik_proposed
            )

            # Proposal log probs
            q_locs_rev = TruncatedDiagonalMVN(
                locs_j_new, self.locs_stdev, self.locs_min, self.locs_max
            )
            q_fluxes_rev = TruncatedDiagonalMVN(
                fluxes_j_new, self.fluxes_stdev, self.fluxes_min, self.fluxes_max
            )

            log_num_qlocs = q_locs_rev.log_prob(locs_j).sum([-2, -1])
            log_num_qfluxes = q_fluxes_rev.log_prob(fluxes_j).sum(-1)
            log_numerator = log_num_target + log_num_qlocs + log_num_qfluxes

            log_denom_qlocs = q_locs_fwd.log_prob(locs_j_new).sum([-2, -1])
            log_denom_qfluxes = q_fluxes_fwd.log_prob(fluxes_j_new).sum(-1)
            log_denominator = log_denom_target + log_denom_qlocs + log_denom_qfluxes

            # Accept/reject
            alpha = (log_numerator - log_denominator).exp().clamp(max=1)
            accept = torch.rand_like(alpha) <= alpha

            # Update star j: keep proposed if accepted, restore if rejected
            locs[:, :, :, j, :] = torch.where(
                accept.unsqueeze(-1),
                locs_j_new.squeeze(-2),
                locs_j.squeeze(-2),
            )
            fluxes[:, :, :, j] = torch.where(
                accept, fluxes_j_new.squeeze(-1), fluxes_j.squeeze(-1)
            )

            # Update rate and contribution cache
            accept_r = accept[:, :, None, None, :]
            rate = torch.where(accept_r, rate_proposed, rate)
            star_contribs[..., j] = torch.where(accept_r, new_contrib, old_contrib)

            # Cache log_denom for next iteration
            log_denom_target = torch.where(accept, log_num_target, log_denom_target)

        return [locs, fluxes, accept.float().mean(-1)]


class SingleComponentMALA(object):
    def __init__(self, num_iters, locs_step, fluxes_step, fluxes_min, fluxes_max):
        self.num_iters = num_iters

        self.locs_step = torch.tensor(locs_step)
        self.locs_min = None  # defined automatically within SMCsampler
        self.locs_max = None  # defined automatically within SMCsampler

        self.fluxes_step = torch.tensor(fluxes_step)
        self.fluxes_min = fluxes_min * torch.ones(1)
        self.fluxes_max = fluxes_max * torch.ones(1)

    def run(self, data, counts, locs, fluxes, temperature, log_target):
        component_multinom = Multinomial(
            total_count=1, probs=1 / fluxes.shape[-1] * torch.ones_like(fluxes)
        )

        locs.requires_grad_(True)
        fluxes.requires_grad_(True)

        for iter in range(self.num_iters):
            # choose component to update for each catalog
            component_mask = component_multinom.sample()

            # compute gradients
            locs.requires_grad_(True)
            fluxes.requires_grad_(True)
            logtarg = log_target(data, counts, locs, fluxes, temperature)
            locs_grad, fluxes_grad = torch.autograd.grad(
                logtarg,
                [locs, fluxes],
                grad_outputs=torch.ones_like(logtarg),
            )
            locs_grad = locs_grad * component_mask.unsqueeze(-1)
            fluxes_grad = fluxes_grad * component_mask

            with torch.no_grad():
                # propose locs
                locs_proposed_qmean = (
                    locs + 0.5 * (self.locs_step**2) * locs_grad
                ) * component_mask.unsqueeze(-1)
                locs_proposed = locs * (
                    1 - component_mask.unsqueeze(-1)
                ) + TruncatedDiagonalMVN(
                    locs_proposed_qmean, self.locs_step, self.locs_min, self.locs_max
                ).sample() * component_mask.unsqueeze(-1)

                # propose fluxes
                fluxes_proposed_qmean = (
                    fluxes + 0.5 * (self.fluxes_step**2) * fluxes_grad
                ) * component_mask
                fluxes_proposed = fluxes * (1 - component_mask) + (
                    TruncatedDiagonalMVN(
                        fluxes_proposed_qmean,
                        self.fluxes_step,
                        self.fluxes_min,
                        self.fluxes_max,
                    ).sample()
                    * component_mask
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
            locs_proposed_grad = locs_proposed_grad * component_mask.unsqueeze(-1)
            fluxes_proposed_grad = fluxes_proposed_grad * component_mask

            with torch.no_grad():
                locs_qmean = (
                    locs_proposed + 0.5 * (self.locs_step**2) * locs_proposed_grad
                ) * component_mask.unsqueeze(-1)
                fluxes_qmean = (
                    fluxes_proposed + 0.5 * (self.fluxes_step**2) * fluxes_proposed_grad
                ) * component_mask

                log_num_qlocs = (
                    TruncatedDiagonalMVN(
                        locs_qmean, self.locs_step, self.locs_min, self.locs_max
                    ).log_prob(locs)
                    * component_mask.unsqueeze(-1)
                ).sum([-2, -1])
                log_num_qfluxes = (
                    TruncatedDiagonalMVN(
                        fluxes_qmean,
                        self.fluxes_step,
                        self.fluxes_min,
                        self.fluxes_max,
                    ).log_prob(fluxes)
                    * component_mask
                ).sum(-1)

                log_numerator = log_num_target + log_num_qlocs + log_num_qfluxes

                # compute log denominator
                if iter == 0:
                    log_denom_target = logtarg

                log_denom_qlocs = (
                    TruncatedDiagonalMVN(
                        locs_proposed_qmean,
                        self.locs_step,
                        self.locs_min,
                        self.locs_max,
                    ).log_prob(locs_proposed)
                    * component_mask.unsqueeze(-1)
                ).sum([-2, -1])
                log_denom_qfluxes = (
                    TruncatedDiagonalMVN(
                        fluxes_proposed_qmean,
                        self.fluxes_step,
                        self.fluxes_min,
                        self.fluxes_max,
                    ).log_prob(fluxes_proposed)
                    * component_mask
                ).sum(-1)
                log_denominator = log_denom_target + log_denom_qlocs + log_denom_qfluxes

                # accept or reject
                alpha = (log_numerator - log_denominator).exp().clamp(max=1)
                prob = torch.rand_like(alpha)
                accept = prob <= alpha

                accept_l = accept.unsqueeze(-1).unsqueeze(-1)
                locs = torch.where(accept_l, locs_proposed, locs).detach()
                fluxes = torch.where(
                    accept.unsqueeze(-1), fluxes_proposed, fluxes
                ).detach()

                # cache log denom target for next iteration
                log_denom_target = torch.where(accept, log_num_target, log_denom_target)

        return [locs, fluxes, accept.float().mean(-1)]
