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
        resample_method,
        merge_method,
        merge_multiplier,
        ess_threshold,
    ):
        self.Prior = deepcopy(Prior)
        self.ImageModel = deepcopy(ImageModel)
        self.MutationKernel = deepcopy(MutationKernel)
        self.MutationKernel.locs_min = self.Prior.loc_prior.low
        self.MutationKernel.locs_max = self.Prior.loc_prior.high

        self.temperature_prev = torch.zeros(1)
        self.temperature = torch.zeros(1)

        self.data = data
        self.counts = counts
        self.locs = locs
        self.fluxes = fluxes
        self.weights = weights
        self.num_catalogs = self.weights.shape[-1]

        self.numH, self.numW, self.dimH, self.dimW = self.data.shape
        self.num_aggregation_levels = (2 * torch.tensor(self.numH).log2()).int().item()

        if resample_method not in {"multinomial", "systematic"}:
            raise ValueError(
                "resample_method must be either multinomial or systematic."
            )
        self.resample_method = resample_method

        if merge_method not in {"naive", "lw_mixture"}:
            raise ValueError("merge_method must be either naive or lw_mixture.")
        self.merge_method = merge_method

        if merge_method == "naive":
            self.merge_multiplier = 1.0
        elif merge_method == "lw_mixture":
            self.merge_multiplier = float(merge_multiplier)

        self.ess_threshold = ess_threshold

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

    def log_density(self, data, counts, locs, fluxes, temperature=1):
        logprior = self.Prior.log_prob(counts, locs, fluxes)
        loglik = self.ImageModel.loglikelihood(data, locs, fluxes)
        return logprior + temperature * loglik

    def tempering_objective(self, loglikelihood, delta):
        log_numerator = 2 * ((delta * loglikelihood).logsumexp(0))
        log_denominator = (2 * delta * loglikelihood).logsumexp(0)

        return (log_numerator - log_denominator).exp() - self.ess_threshold

    def temper(self):
        self.loglik = self.ImageModel.loglikelihood(self.data, self.locs, self.fluxes)

        solutions = torch.zeros(self.numH, self.numW)

        for h in range(self.numH):
            for w in range(self.numW):

                def func(delta):
                    return self.tempering_objective(self.loglik[h, w], delta)

                if func(1 - self.temperature.item()) < 0:
                    solutions[h, w] = brentq(
                        func, 0.0, 1 - self.temperature.item(), xtol=1e-6, rtol=1e-6
                    )
                else:
                    solutions[h, w] = 1 - self.temperature.item()

        delta = solutions.min()

        self.temperature_prev = self.temperature
        self.temperature = self.temperature + delta

    def mutate(self):
        self.locs, self.fluxes = self.MutationKernel.run(
            self.data,
            self.counts,
            self.locs,
            self.fluxes,
            self.temperature_prev,
            self.log_density,
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

    def prune(self):
        in_bounds = torch.all(
            torch.logical_and(
                self.locs > 0, self.locs < torch.tensor((self.dimH, self.dimW))
            ),
            dim=-1,
        )

        self.counts = in_bounds.sum(-1)

        self.locs = in_bounds.unsqueeze(-1) * self.locs
        locs_mask = (self.locs != 0).int()
        locs_index = torch.sort(locs_mask, dim=3, descending=True)[1]
        self.locs = torch.gather(self.locs, dim=3, index=locs_index)

        self.fluxes = in_bounds * self.fluxes
        fluxes_mask = (self.fluxes != 0).int()
        fluxes_index = torch.sort(fluxes_mask, dim=3, descending=True)[1]
        self.fluxes = torch.gather(self.fluxes, dim=3, index=fluxes_index)

    def merge(self, level):
        if self.merge_method == "naive":
            index = self.get_resampled_index(self.weights, 1)
            cs, ls, fs, ws = self.apply_resampled_index(
                index, self.counts, self.locs, self.fluxes
            )

            if level % 2 == 0:
                cs, ls, fs = self.drop_sources_from_overlap(0, cs, ls, fs)
                self.data, self.counts, self.locs, self.fluxes = self.join(
                    0, self.data, cs, ls, fs
                )
            elif level % 2 != 0:
                cs, ls, fs = self.drop_sources_from_overlap(1, cs, ls, fs)
                self.data, self.counts, self.locs, self.fluxes = self.join(
                    1, self.data, cs, ls, fs
                )
        elif self.merge_method == "lw_mixture":
            index = self.get_resampled_index(self.weights, self.merge_multiplier)
            cs, ls, fs, ws = self.apply_resampled_index(
                index, self.counts, self.locs, self.fluxes
            )

            if level % 2 == 0:
                cs, ls, fs = self.drop_sources_from_overlap(0, cs, ls, fs)
                self.data, cs, ls, fs = self.join(0, self.data, cs, ls, fs)
            elif level % 2 != 0:
                cs, ls, fs = self.drop_sources_from_overlap(1, cs, ls, fs)
                self.data, cs, ls, fs = self.join(1, self.data, cs, ls, fs)

            ld = self.log_density(self.data, cs, ls, fs)
            ws = ld.softmax(-1)

            index = self.get_resampled_index(ws, 1 / self.merge_multiplier)
            res = self.apply_resampled_index(index, cs, ls, fs)
            self.counts, self.locs, self.fluxes, self.weights = res

    def run(self):
        for level in range(self.num_aggregation_levels):
            print(f"level {level}")

            self.iter = 0

            self.merge(level)

            self.temper()

            self.weights = (
                (self.temperature - self.temperature_prev)
                * self.ImageModel.loglikelihood(self.data, self.locs, self.fluxes)
            ).softmax(-1)

            while self.temperature < 1:
                self.iter += 1

                if self.iter % 5 == 0:
                    print(
                        f"iteration {self.iter}, temperature = {self.temperature.item()}"
                    )

                index = self.get_resampled_index(self.weights, 1)
                res = self.apply_resampled_index(
                    index, self.counts, self.locs, self.fluxes
                )
                self.counts, self.locs, self.fluxes, self.weights = res

                self.mutate()

                self.temper()

                self.weights = (
                    (self.temperature - self.temperature_prev)
                    * self.ImageModel.loglikelihood(self.data, self.locs, self.fluxes)
                ).softmax(-1)

            # reset for next merge
            self.temperature_prev = torch.zeros(1)
            self.temperature = torch.zeros(1)

        index = self.get_resampled_index(self.weights, 1)
        res = self.apply_resampled_index(index, self.counts, self.locs, self.fluxes)
        self.counts, self.locs, self.fluxes, self.weights = res

        self.prune()

    @property
    def ESS(self):
        return 1 / (self.weights**2).sum(-1)

    @property
    def posterior_mean_counts(self):
        return (self.weights * self.counts).sum(-1)

    @property
    def posterior_mean_total_flux(self):
        return (self.weights * self.fluxes.sum(-1)).sum(-1)
