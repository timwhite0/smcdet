from copy import deepcopy

import torch
from einops import rearrange


class Aggregate(object):
    def __init__(self, Prior, ImageModel, data, counts, locs, features, weights):
        self.Prior = deepcopy(Prior)
        self.ImageModel = deepcopy(ImageModel)

        self.data = data
        self.counts = counts
        self.locs = locs
        self.features = features
        self.weights = weights
        self.num_catalogs = self.weights.shape[-1]

        self.numH, self.numW, self.dimH, self.dimW = self.data.shape
        self.num_aggregation_levels = (2 * torch.tensor(self.numH).log2()).int().item()

    def get_resampled_index(self, weights, method="systematic", multiplier=1):
        num = int(multiplier * weights.shape[-1])

        if method == "multinomial":
            weights_flat = weights.flatten(0, 1)
            resampled_index_flat = weights_flat.multinomial(num, replacement=True)
            resampled_index = resampled_index_flat.unflatten(0, (self.numH, self.numW))
        elif method == "systematic":
            resampled_index = torch.zeros([self.numH, self.numW, num])
            for h in range(self.numH):
                for w in range(self.numW):
                    u = (torch.arange(num) + torch.rand([1])) / num
                    bins = weights[h, w].cumsum(0)
                    resampled_index[h, w] = torch.bucketize(u, bins)

        return resampled_index.int().clamp(min=0, max=num - 1)

    def apply_resampled_index(self, resampled_index, counts, locs, features):
        num = resampled_index.shape[-1]
        numH = features.shape[0]
        numW = features.shape[1]
        max_objects = features.shape[-1]
        cs = torch.zeros(numH, numW, num)
        ls = torch.zeros(numH, numW, num, max_objects, 2)
        fs = torch.zeros(numH, numW, num, max_objects)
        ws = torch.zeros(numH, numW, num)

        for h in range(self.numH):
            for w in range(self.numW):
                cs[h, w, :] = counts[h, w, resampled_index[h, w, :]]
                ls[h, w, :] = locs[h, w, resampled_index[h, w, :]]
                fs[h, w, :] = features[h, w, resampled_index[h, w, :]]
                ws[h, w, :] = 1 / num

        return cs, ls, fs, ws

    def log_density(self, data, counts, locs, features, temperature=1):
        logprior = self.Prior.log_prob(counts, locs, features)
        loglik = self.ImageModel.loglikelihood(data, locs, features)
        return logprior + temperature * loglik

    def drop_sources_from_overlap(self, axis, counts, locs, features):
        if axis == 0:  # height axis
            sources_to_keep_even = torch.logical_and(
                locs[0::2, :, ..., 0] < self.dimH, locs[0::2, :, ..., 0] != 0
            )
            counts[0::2, ...] = sources_to_keep_even.sum(-1)
            locs[0::2, ...] *= sources_to_keep_even.unsqueeze(-1)
            features[0::2, ...] *= sources_to_keep_even

            sources_to_keep_odd = locs[1::2, :, ..., 0] > 0
            counts[1::2, ...] = sources_to_keep_odd.sum(-1)
            locs[1::2, ...] *= sources_to_keep_odd.unsqueeze(-1)
            features[1::2, ...] *= sources_to_keep_odd
        elif axis == 1:  # width axis
            sources_to_keep_even = torch.logical_and(
                locs[:, 0::2, ..., 1] < self.dimW, locs[:, 0::2, ..., 1] != 0
            )
            counts[:, 0::2, ...] = sources_to_keep_even.sum(-1)
            locs[:, 0::2, ...] *= sources_to_keep_even.unsqueeze(-1)
            features[:, 0::2, ...] *= sources_to_keep_even

            sources_to_keep_odd = locs[:, 1::2, ..., 1] > 0
            counts[:, 1::2, ...] = sources_to_keep_odd.sum(-1)
            locs[:, 1::2, ...] *= sources_to_keep_odd.unsqueeze(-1)
            features[:, 1::2, ...] *= sources_to_keep_odd

        return counts, locs, features

    def join(self, axis, data, counts, locs, features):
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
            features.unfold(axis, 2, 2), "numH numW N M t -> numH numW N (t M)"
        )
        features_mask = (fs != 0).int()
        features_index = torch.sort(features_mask, dim=3, descending=True)[1]
        fs = torch.gather(fs, dim=3, index=features_index)[
            ..., : self.Prior.max_objects
        ]

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

        self.features = in_bounds * self.features
        features_mask = (self.features != 0).int()
        features_index = torch.sort(features_mask, dim=3, descending=True)[1]
        self.features = torch.gather(self.features, dim=3, index=features_index)

    def merge(self, level, method="naive"):
        if method == "naive":
            index = self.get_resampled_index(self.weights)
            res = self.apply_resampled_index(
                index, self.counts, self.locs, self.features
            )
            self.counts, self.locs, self.features, self.weights = res

            if level % 2 == 0:
                self.counts, self.locs, self.features = self.drop_sources_from_overlap(
                    0, self.counts, self.locs, self.features
                )
                self.data, self.counts, self.locs, self.features = self.join(
                    0, self.data, self.counts, self.locs, self.features
                )
            elif level % 2 != 0:
                self.counts, self.locs, self.features = self.drop_sources_from_overlap(
                    1, self.counts, self.locs, self.features
                )
                self.data, self.counts, self.locs, self.features = self.join(
                    1, self.data, self.counts, self.locs, self.features
                )
        elif method == "lw_mixture":
            index = self.get_resampled_index(self.weights, multiplier=5)
            cs, ls, fs, ws = self.apply_resampled_index(
                index, self.counts, self.locs, self.features
            )

            if level % 2 == 0:
                cs, ls, fs = self.drop_sources_from_overlap(0, cs, ls, fs)
                self.data, cs, ls, fs = self.join(0, self.data, cs, ls, fs)
            elif level % 2 != 0:
                cs, ls, fs = self.drop_sources_from_overlap(1, cs, ls, fs)
                self.data, cs, ls, fs = self.join(1, self.data, cs, ls, fs)

            ld = self.log_density(self.data, cs, ls, fs)
            ws = ld.softmax(-1)

            index = self.get_resampled_index(ws, multiplier=0.2)

            res = self.apply_resampled_index(index, cs, ls, fs)
            self.counts, self.locs, self.features, self.weights = res

    def run(self):
        for level in range(self.num_aggregation_levels):
            self.merge(level, method="lw_mixture")

            ld = self.log_density(self.data, self.counts, self.locs, self.features)
            self.weights = ld.softmax(-1)

        index = self.get_resampled_index(self.weights)
        res = self.apply_resampled_index(index, self.counts, self.locs, self.features)
        self.counts, self.locs, self.features, self.weights = res

        self.prune()

    @property
    def ESS(self):
        return 1 / (self.weights**2).sum(-1)

    @property
    def posterior_mean_counts(self):
        return (self.weights * self.counts).sum(-1)

    @property
    def posterior_mean_total_flux(self):
        return (self.weights * self.features.sum(-1)).sum(-1)
