from copy import deepcopy

import torch
from einops import rearrange
from scipy.optimize import brentq


class Aggregate(object):
    def __init__(
        self,
        Prior,
        ImageModel,
        MutationKernel,
        data,
        counts,
        locs,
        fluxes,
        weights,
        log_normalizing_constant,
        resample_method,
        ess_threshold_prop,
        print_every=5,
    ):
        self.Prior = deepcopy(Prior)
        self.ImageModel = deepcopy(ImageModel)
        self.MutationKernel = deepcopy(MutationKernel)
        self.MutationKernel.locs_min = self.Prior.loc_prior.low
        self.MutationKernel.locs_max = self.Prior.loc_prior.high
        self.mutation_acc_rates = None

        self.data = data
        self.counts = counts
        self.locs = locs
        self.fluxes = fluxes
        self.weights = weights
        self.weights_intracount = None

        self.numH, self.numW, self.dimH, self.dimW = self.data.shape
        self.num_aggregation_levels = (2 * torch.tensor(self.numH).log2()).int().item()

        self.log_normalizing_constant = [
            [log_normalizing_constant[h, w].tolist() for w in range(self.numW)]
            for h in range(self.numH)
        ]

        self.num_catalogs = self.weights.shape[-1]
        self.num_catalogs_per_count = [
            [None for _ in range(self.numW)] for _ in range(self.numH)
        ]

        self.temperature_prev = torch.zeros(self.numH, self.numW)
        self.temperature = torch.zeros(self.numH, self.numW)

        if resample_method not in {"multinomial", "systematic"}:
            raise ValueError(
                "resample_method must be either multinomial or systematic."
            )
        self.resample_method = resample_method

        self.ess_threshold_prop = ess_threshold_prop

        self.print_every = print_every

        self.has_run = False

    def get_resampled_index(self, weights, multiplier):
        num = int(multiplier * weights.shape[-1])

        if self.resample_method == "multinomial":
            weights_flat = weights.flatten(0, 1)
            resampled_index_flat = weights_flat.multinomial(num, replacement=True)
            resampled_index = resampled_index_flat.unflatten(0, (self.numH, self.numW))
        elif self.resample_method == "systematic":
            resampled_index = torch.zeros([self.numH, self.numW, num])
            for h in range(self.numH):
                for w in range(self.numW):
                    u = (torch.arange(num) + torch.rand([1])) / num
                    bins = weights[h, w].cumsum(0)
                    resampled_index[h, w] = torch.bucketize(u, bins)

        return resampled_index.int().clamp(min=0, max=num - 1)

    def apply_resampled_index(self, resampled_index, counts, locs, fluxes):
        num = resampled_index.shape[-1]
        numH = fluxes.shape[0]
        numW = fluxes.shape[1]
        max_objects = fluxes.shape[-1]
        cs = torch.zeros(numH, numW, num)
        ls = torch.zeros(numH, numW, num, max_objects, 2)
        fs = torch.zeros(numH, numW, num, max_objects)
        ws = torch.zeros(numH, numW, num)

        for h in range(self.numH):
            for w in range(self.numW):
                cs[h, w, :] = counts[h, w, resampled_index[h, w, :]]
                ls[h, w, :] = locs[h, w, resampled_index[h, w, :]]
                fs[h, w, :] = fluxes[h, w, resampled_index[h, w, :]]
                ws[h, w, :] = 1 / num

        return cs, ls, fs, ws

    def log_target(
        self,
        axis,
        ChildImageModel,
        child_data,
        child_locs,
        child_fluxes,
        parent_data,
        parent_counts,
        parent_locs,
        parent_fluxes,
        temperature,
    ):
        logprior = self.Prior.log_prob(parent_counts, parent_locs, parent_fluxes)
        child_loglik = ChildImageModel.loglikelihood(
            child_data, child_locs, child_fluxes
        )
        parent_loglik = self.ImageModel.loglikelihood(
            parent_data, parent_locs, parent_fluxes
        )

        return (
            logprior
            + (1 - temperature).unsqueeze(-1) * child_loglik.unfold(axis, 2, 2).sum(-1)
            + temperature.unsqueeze(-1) * parent_loglik
        )

    def tempering_objective(self, loglikelihood, delta):
        log_numerator = 2 * ((delta * loglikelihood).logsumexp(0))
        log_denominator = (2 * delta * loglikelihood).logsumexp(0)

        return (
            log_numerator - log_denominator
        ).exp() - self.ess_threshold_prop * loglikelihood.shape[0]

    def temper(self):
        loglik_diff = self.loglik_diff.cpu()

        solutions = torch.zeros(self.numH, self.numW)

        for h in range(self.numH):
            for w in range(self.numW):
                loglik_diff_split = torch.split(
                    loglik_diff[h, w], self.num_catalogs_per_count[h][w], dim=0
                )

                solutions_within_tile = torch.zeros(
                    len(self.num_catalogs_per_count[h][w])
                )

                for c in range(len(self.num_catalogs_per_count[h][w])):

                    def func(delta):
                        return self.tempering_objective(loglik_diff_split[c], delta)

                    if func(1 - self.temperature[h, w].item()) < 0:
                        solutions_within_tile[c] = brentq(
                            func,
                            0.0,
                            1 - self.temperature[h, w].item(),
                            xtol=1e-6,
                            rtol=1e-6,
                        )
                    else:
                        solutions_within_tile[c] = 1 - self.temperature[h, w].item()

                solutions[h, w] = solutions_within_tile.min()

        self.temperature_prev = self.temperature
        self.temperature = self.temperature + solutions

    def mutate(self, axis, ChildImageModel):
        self.locs, self.fluxes, self.mutation_acc_rates = self.MutationKernel.run(
            self.data,
            self.counts,
            self.locs,
            self.fluxes,
            self.temperature,
            self.log_target,
            self.unjoin,
            axis,
            ChildImageModel,
        )

    def drop_sources_from_overlap(self, axis, counts, locs, fluxes):
        if axis == 0:  # height axis
            sources_to_keep_even = torch.logical_and(
                locs[0::2, :, ..., 0] < self.dimH, locs[0::2, :, ..., 0] != 0
            )
            counts[0::2, ...] = sources_to_keep_even.sum(-1)
            locs[0::2, ...] *= sources_to_keep_even.unsqueeze(-1)
            fluxes[0::2, ...] *= sources_to_keep_even

            sources_to_keep_odd = locs[1::2, :, ..., 0] > 0
            counts[1::2, ...] = sources_to_keep_odd.sum(-1)
            locs[1::2, ...] *= sources_to_keep_odd.unsqueeze(-1)
            fluxes[1::2, ...] *= sources_to_keep_odd
        elif axis == 1:  # width axis
            sources_to_keep_even = torch.logical_and(
                locs[:, 0::2, ..., 1] < self.dimW, locs[:, 0::2, ..., 1] != 0
            )
            counts[:, 0::2, ...] = sources_to_keep_even.sum(-1)
            locs[:, 0::2, ...] *= sources_to_keep_even.unsqueeze(-1)
            fluxes[:, 0::2, ...] *= sources_to_keep_even

            sources_to_keep_odd = locs[:, 1::2, ..., 1] > 0
            counts[:, 1::2, ...] = sources_to_keep_odd.sum(-1)
            locs[:, 1::2, ...] *= sources_to_keep_odd.unsqueeze(-1)
            fluxes[:, 1::2, ...] *= sources_to_keep_odd

        return counts, locs, fluxes

    def join(self, axis, data, counts, locs, fluxes):
        if axis == 0:  # height axis
            self.numH = self.numH // 2
            self.dimH = self.dimH * 2
            self.ImageModel.image_height = self.ImageModel.image_height * 2
            self.Prior.image_height = self.Prior.image_height * 2
            dat = rearrange(
                data.unfold(axis, 2, 2),
                "numH numW dimH dimW t -> numH numW (t dimH) dimW",
            )
        elif axis == 1:  # width axis
            self.numW = self.numW // 2
            self.dimW = self.dimW * 2
            self.ImageModel.image_width = self.ImageModel.image_width * 2
            self.Prior.image_width = self.Prior.image_width * 2
            dat = rearrange(
                data.unfold(axis, 2, 2),
                "numH numW dimH dimW t -> numH numW dimH (t dimW)",
            )

        cs = counts.unfold(axis, 2, 2).sum(3)

        self.Prior.max_objects = max(1, cs.max().int().item())  # max objects detected
        self.Prior.update_attrs()
        self.ImageModel.update_psf_grid()
        self.MutationKernel.locs_min = self.Prior.loc_prior.low
        self.MutationKernel.locs_max = self.Prior.loc_prior.high

        locs_unfolded = locs.unfold(axis, 2, 2)
        locs_unfolded_mask = (locs_unfolded != 0).int()
        locs_unfolded.select(-2, axis)[..., -1] += (self.dimH / 2) * (1 - axis) + (
            self.dimW / 2
        ) * axis
        locs_adjusted = locs_unfolded * locs_unfolded_mask
        ls = rearrange(locs_adjusted, "numH numW N M l t -> numH numW N (t M) l")
        locs_mask = (ls != 0).int()
        locs_index = torch.sort(locs_mask, dim=3, descending=True)[1]
        ls = torch.gather(ls, dim=3, index=locs_index)[..., : self.Prior.max_objects, :]

        fs = rearrange(
            fluxes.unfold(axis, 2, 2), "numH numW N M t -> numH numW N (t M)"
        )
        fluxes_mask = (fs != 0).int()
        fluxes_index = torch.sort(fluxes_mask, dim=3, descending=True)[1]
        fs = torch.gather(fs, dim=3, index=fluxes_index)[..., : self.Prior.max_objects]

        return dat, cs, ls, fs

    def unjoin(self, axis, data, locs, fluxes):
        if axis == 0:
            dat = rearrange(
                data, "numH numW (t dimH) dimW -> (t numH) numW dimH dimW", t=2
            )
        elif axis == 1:
            dat = rearrange(
                data, "numH numW dimH (t dimW) -> numH (t numW) dimH dimW", t=2
            )

        mask = (
            locs.select(-1, axis) <= (self.dimH * (1 - axis) + self.dimW * axis) / 2
        ).int()

        # split up locs
        region1_locs = mask.unsqueeze(-1) * locs
        region1_locs_mask = (region1_locs != 0).int()
        region1_locs_index = torch.sort(region1_locs_mask, dim=-2, descending=True)[1]
        region1_locs = torch.gather(region1_locs, dim=-2, index=region1_locs_index)

        region2_locs = (1 - mask).unsqueeze(-1) * locs
        region2_locs_mask = (region2_locs != 0).int()
        region2_locs_index = torch.sort(region2_locs_mask, dim=-2, descending=True)[1]
        region2_locs = torch.gather(region2_locs, dim=-2, index=region2_locs_index)

        region2_locs_sorted_mask = (region2_locs != 0).int()
        region2_locs.select(-1, axis)[...] -= (self.dimH / 2) * (1 - axis) + (
            self.dimW / 2
        ) * axis
        region2_locs *= region2_locs_sorted_mask

        new_locs = torch.cat((region1_locs, region2_locs), dim=axis)

        # split up fluxes
        region1_fluxes = mask * fluxes
        region1_fluxes_mask = (region1_fluxes != 0).int()
        region1_fluxes_index = torch.sort(region1_fluxes_mask, dim=-1, descending=True)[
            1
        ]
        region1_fluxes = torch.gather(
            region1_fluxes, dim=-1, index=region1_fluxes_index
        )

        region2_fluxes = (1 - mask) * fluxes
        region2_fluxes_mask = (region2_fluxes != 0).int()
        region2_fluxes_index = torch.sort(region2_fluxes_mask, dim=-1, descending=True)[
            1
        ]
        region2_fluxes = torch.gather(
            region2_fluxes, dim=-1, index=region2_fluxes_index
        )

        new_fluxes = torch.cat((region1_fluxes, region2_fluxes), dim=axis)

        # split up counts
        tile1_counts = (mask * (locs != 0).all(-1)).sum(-1)
        tile2_counts = ((1 - mask) * (locs != 0).all(-1)).sum(-1)
        new_counts = torch.cat((tile1_counts, tile2_counts), dim=axis)

        return dat, new_counts, new_locs, new_fluxes

    def prune(self, locs, fluxes):
        in_bounds = torch.all(
            torch.logical_and(locs > 0, locs < torch.tensor((self.dimH, self.dimW))),
            dim=-1,
        )

        counts = in_bounds.sum(-1)

        locs = in_bounds.unsqueeze(-1) * locs
        locs_mask = (locs != 0).int()
        locs_index = torch.sort(locs_mask, dim=3, descending=True)[1]
        locs = torch.gather(locs, dim=3, index=locs_index)

        fluxes = in_bounds * fluxes
        fluxes_mask = (fluxes != 0).int()
        fluxes_index = torch.sort(fluxes_mask, dim=3, descending=True)[1]
        fluxes = torch.gather(fluxes, dim=3, index=fluxes_index)

        return counts, locs, fluxes

    def merge(self, level):
        marginal_counts = self.counts

        index = self.get_resampled_index(self.weights, 1)
        cs_resampled, ls_resampled, fs_resampled, ws_resampled = (
            self.apply_resampled_index(index, self.counts, self.locs, self.fluxes)
        )

        cs_pruned, ls_pruned, fs_pruned = self.drop_sources_from_overlap(
            level % 2, cs_resampled, ls_resampled, fs_resampled
        )
        self.data, self.counts, self.locs, self.fluxes = self.join(
            level % 2, self.data, cs_pruned, ls_pruned, fs_pruned
        )

        # compute initial log normalizing constant
        marginal_numH = self.numH * (1 + (1 - level % 2))
        marginal_numW = self.numW * (1 + (level % 2))

        marginal_log_normalizing_constant = [
            [None for _ in range(marginal_numW)] for _ in range(marginal_numH)
        ]

        for marginal_h in range(marginal_numH):
            for marginal_w in range(marginal_numW):
                joint_h = marginal_h // (1 + (1 - level % 2))
                joint_w = marginal_w // (1 + (level % 2))

                unique_joint_counts = self.counts[joint_h, joint_w].unique()
                unique_marginal_counts = marginal_counts[
                    marginal_h, marginal_w
                ].unique()

                marginal_log_normalizing_constant[marginal_h][marginal_w] = torch.zeros(
                    len(unique_joint_counts)
                )

                for j in range(len(unique_joint_counts)):
                    merge_pmf = torch.zeros(len(unique_marginal_counts))

                    for k in range(len(unique_marginal_counts)):
                        merge_pmf[k] = (
                            (
                                cs_resampled[marginal_h, marginal_w][
                                    self.counts[joint_h, joint_w]
                                    == unique_joint_counts[j]
                                ]
                                == unique_marginal_counts[k]
                            )
                            .float()
                            .mean()
                        )

                    marginal_log_normalizing_constant[marginal_h][marginal_w][j] = (
                        (
                            torch.tensor(
                                self.log_normalizing_constant[marginal_h][marginal_w]
                            )
                            + merge_pmf.log().nan_to_num()
                        ).logsumexp(-1)
                    ).tolist()

        self.log_normalizing_constant = [
            [None for _ in range(self.numW)] for _ in range(self.numH)
        ]

        for joint_h in range(self.numH):
            for joint_w in range(self.numW):
                self.log_normalizing_constant[joint_h][joint_w] = (
                    marginal_log_normalizing_constant[joint_h * (1 + (1 - level % 2))][
                        joint_w * (1 + level % 2)
                    ]
                    + marginal_log_normalizing_constant[
                        joint_h * (1 + (1 - level % 2)) + (1 - level % 2)
                    ][joint_w * (1 + level % 2) + (level % 2)]
                )

    def sort_by_count(self):
        self.counts, indices = self.counts.sort()

        self.num_catalogs_per_count = [
            [None for _ in range(self.numW)] for _ in range(self.numH)
        ]

        for h in range(self.numH):
            for w in range(self.numW):
                self.num_catalogs_per_count[h][w] = (
                    self.counts[h, w].unique(return_counts=True)[-1].tolist()
                )
                self.fluxes[h, w] = self.fluxes[h, w].index_select(0, indices[h, w])
                self.locs[h, w] = self.locs[h, w].index_select(0, indices[h, w])

    def update_weights(self):
        weights_log_unnorm = (self.temperature - self.temperature_prev).unsqueeze(
            -1
        ) * self.loglik_diff

        self.weights_intracount = torch.zeros_like(weights_log_unnorm)
        self.weights = torch.zeros_like(weights_log_unnorm)

        for h in range(self.numH):
            for w in range(self.numW):
                weights_log_unnorm_split = torch.split(
                    weights_log_unnorm[h, w], self.num_catalogs_per_count[h][w], dim=0
                )
                self.weights_intracount[h, w] = torch.cat(
                    [wt.softmax(-1) for wt in weights_log_unnorm_split], dim=0
                )

                lnc_update = [
                    (wt - wt.max()).exp().mean().log() + wt.max()
                    for wt in weights_log_unnorm_split
                ]
                self.log_normalizing_constant[h][w] = [
                    prev + new
                    for prev, new in zip(
                        self.log_normalizing_constant[h][w], lnc_update
                    )
                ]

                weights_intracount_split = torch.split(
                    self.weights_intracount[h, w],
                    self.num_catalogs_per_count[h][w],
                    dim=0,
                )
                lnc_softmax = (
                    torch.tensor(self.log_normalizing_constant[h][w])
                    .softmax(0)
                    .tolist()
                )
                self.weights[h, w] = torch.cat(
                    [
                        wt * lnc
                        for wt, lnc in zip(weights_intracount_split, lnc_softmax)
                    ],
                    dim=0,
                )

    def resample_intracount(self):
        for h in range(self.numH):
            for w in range(self.numW):
                weights_intracount_split = list(
                    torch.split(
                        self.weights_intracount[h, w],
                        self.num_catalogs_per_count[h][w],
                        dim=0,
                    )
                )
                locs = list(
                    torch.split(
                        self.locs[h, w], self.num_catalogs_per_count[h][w], dim=0
                    )
                )
                fluxes = list(
                    torch.split(
                        self.fluxes[h, w], self.num_catalogs_per_count[h][w], dim=0
                    )
                )

                for c in range(len(self.num_catalogs_per_count[h][w])):
                    idx = weights_intracount_split[c].multinomial(
                        weights_intracount_split[c].shape[0], replacement=True
                    )
                    locs[c] = locs[c][idx]
                    fluxes[c] = fluxes[c][idx]
                    weights_intracount_split[c] = (
                        torch.ones(self.num_catalogs_per_count[h][w][c])
                        / self.num_catalogs_per_count[h][w][c]
                    )

                self.locs[h, w] = torch.cat(locs, dim=0)
                self.fluxes[h, w] = torch.cat(fluxes, dim=0)
                self.weights_intracount[h, w] = torch.cat(
                    tuple(weights_intracount_split), dim=0
                )

    def run(self):
        print("aggregating tile catalogs...")

        for level in range(self.num_aggregation_levels):
            print(f"level {level}")

            ChildImageModel = deepcopy(self.ImageModel)
            self.merge(level)
            self.sort_by_count()

            child_data, child_counts, child_locs, child_fluxes = self.unjoin(
                level % 2, self.data, self.locs, self.fluxes
            )
            child_loglik = ChildImageModel.loglikelihood(
                child_data, child_locs, child_fluxes
            )
            self.loglik_diff = self.ImageModel.loglikelihood(
                self.data, self.locs, self.fluxes
            ) - child_loglik.unfold(level % 2, 2, 2).sum(-1)

            self.temperature_prev = torch.zeros(self.numH, self.numW)
            self.temperature = torch.zeros(self.numH, self.numW)
            self.temper()

            self.update_weights()

            self.iter = 0

            while torch.any(self.temperature < 1):
                self.iter += 1

                if self.iter % self.print_every == 0:
                    print(
                        (
                            f"iteration {self.iter}: "
                            f"temperature in [{round(self.temperature.min().item(), 2)}, "
                            f"{round(self.temperature.max().item(), 2)}], "
                            f"accept rate in [{round(self.mutation_acc_rates.min().item(), 2)}, "
                            f"{round(self.mutation_acc_rates.max().item(), 2)}]"
                        )
                    )

                self.resample_intracount()

                self.mutate(axis=level % 2, ChildImageModel=ChildImageModel)

                child_data, child_counts, child_locs, child_fluxes = self.unjoin(
                    level % 2, self.data, self.locs, self.fluxes
                )
                child_loglik = ChildImageModel.loglikelihood(
                    child_data, child_locs, child_fluxes
                )
                self.loglik_diff = self.ImageModel.loglikelihood(
                    self.data, self.locs, self.fluxes
                ) - child_loglik.unfold(level % 2, 2, 2).sum(-1)

                self.temper()

                self.update_weights()

        index = self.get_resampled_index(self.weights, 1)
        res = self.apply_resampled_index(index, self.counts, self.locs, self.fluxes)
        self.counts, self.locs, self.fluxes, self.weights = res

        self.pruned_counts, self.pruned_locs, self.pruned_fluxes = self.prune(
            self.locs, self.fluxes
        )

        self.has_run = True

        print("done!\n")

    @property
    def ess(self):
        return 1 / (self.weights**2).sum(-1)

    def posterior_mean_count(self, counts):
        return (self.weights * counts).sum(-1)

    def posterior_mean_total_flux(self, fluxes):
        return (self.weights * fluxes.sum(-1)).sum(-1)

    @property
    def posterior_predictive_total_observed_flux(self):
        return self.ImageModel.sample(self.locs, self.fluxes).sum([-2, -3]).squeeze()

    def summarize(self):
        if self.has_run is False:
            raise ValueError("aggregation procedure hasn't been run yet.")

        print(
            "summary:\n\nposterior distribution of number of stars including padding:"
        )
        print(self.counts.unique(return_counts=True)[0].cpu())
        print(
            (self.counts.unique(return_counts=True)[1] / self.counts.shape[-1])
            .round(decimals=3)
            .cpu(),
            "\n",
        )

        print("posterior distribution of number of stars within image boundary:")
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
            "posterior mean total intrinsic flux including padding =",
            f"{self.posterior_mean_total_flux(self.fluxes).round().item()}\n",
        )

        print(
            "posterior mean total intrinsic flux within boundary =",
            f"{self.posterior_mean_total_flux(self.pruned_fluxes).round().item()}\n",
        )

        print(
            f"number of unique catalogs = {self.fluxes[0, 0].sum(-1).unique(dim=0).shape[0]}"
        )
