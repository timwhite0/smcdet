import math
from copy import deepcopy
from types import SimpleNamespace

import torch

from smcdet.sampler import SMCsampler


def checkerboard_color(i, j):
    """Return the color (0-3) of tile at row i, col j.

    Pattern (counter-clockwise within each 2x2 block, color 1 south of color 0):
        0 3 0 3 ...
        1 2 1 2 ...
        0 3 0 3 ...
        1 2 1 2 ...
    """
    return (i % 2) + (j % 2) * (3 - 2 * (i % 2))


def _color_subgrid(color, R, C):
    """Return (numH, numW, mapping) for a given color.

    mapping: list of (si, sj, gi, gj) tuples mapping sub-grid to global indices.
    """
    tiles = []
    for i in range(R):
        for j in range(C):
            if checkerboard_color(i, j) == color:
                tiles.append((i, j))

    if not tiles:
        return 0, 0, []

    # Determine sub-grid dimensions from the color pattern
    if color == 0:
        numH, numW = math.ceil(R / 2), math.ceil(C / 2)
    elif color == 1:
        numH, numW = R // 2, math.ceil(C / 2)
    elif color == 2:
        numH, numW = R // 2, C // 2
    elif color == 3:
        numH, numW = math.ceil(R / 2), C // 2

    # Build mapping from sub-grid to global
    mapping = []
    for si in range(numH):
        for sj in range(numW):
            if color == 0:
                gi, gj = 2 * si, 2 * sj
            elif color == 1:
                gi, gj = 2 * si + 1, 2 * sj
            elif color == 2:
                gi, gj = 2 * si + 1, 2 * sj + 1
            elif color == 3:
                gi, gj = 2 * si, 2 * sj + 1
            mapping.append((si, sj, gi, gj))

    return numH, numW, mapping


class BatchPrior:
    """Wraps a 2D grid of individual Prior instances for batched SMCsampler."""

    def __init__(self, priors_grid):
        """priors_grid: list of lists of Prior instances, shape [numH][numW]."""
        self.priors_grid = priors_grid
        self.numH = len(priors_grid)
        self.numW = len(priors_grid[0])
        self._ref = priors_grid[0][0]

        # Forward attributes from the reference prior
        self.max_objects = self._ref.max_objects

    @property
    def loc_prior(self):
        """Returns a namespace with .low/.high as [numH, numW, 2] tensors."""
        lows = torch.stack(
            [
                torch.stack(
                    [self.priors_grid[i][j].loc_prior.low for j in range(self.numW)]
                )
                for i in range(self.numH)
            ]
        )
        highs = torch.stack(
            [
                torch.stack(
                    [self.priors_grid[i][j].loc_prior.high for j in range(self.numW)]
                )
                for i in range(self.numH)
            ]
        )
        return SimpleNamespace(low=lows, high=highs)

    def sample(self, num_tiles_h=None, num_tiles_w=None, **kwargs):
        """Sample from each prior, stack into [numH, numW, n, d, ...]."""
        results = []
        for i in range(self.numH):
            row = []
            for j in range(self.numW):
                row.append(
                    self.priors_grid[i][j].sample(
                        num_tiles_h=1, num_tiles_w=1, **kwargs
                    )
                )
            results.append(row)

        # Each result is a list [counts, locs, fluxes] with shapes [1, 1, n, ...]
        num_fields = len(results[0][0])
        stacked = []
        for f in range(num_fields):
            # Stack [1,1,...] tiles into [numH, numW, ...]
            rows = []
            for i in range(self.numH):
                cols = torch.cat([results[i][j][f] for j in range(self.numW)], dim=1)
                rows.append(cols)
            stacked.append(torch.cat(rows, dim=0))

        # Aggregate counts_mask from individual priors
        cm_rows = []
        for i in range(self.numH):
            cm_cols = torch.cat(
                [self.priors_grid[i][j].counts_mask for j in range(self.numW)], dim=1
            )
            cm_rows.append(cm_cols)
        self.counts_mask = torch.cat(cm_rows, dim=0)

        # Store self.num from the reference prior (all should be the same)
        self.num = self._ref.num

        return stacked

    def log_prob(self, counts, locs, fluxes):
        """Compute log_prob per tile, return [numH, numW, n]."""
        result = torch.zeros(self.numH, self.numW, counts.shape[-1])
        for i in range(self.numH):
            for j in range(self.numW):
                result[i, j] = (
                    self.priors_grid[i][j]
                    .log_prob(
                        counts[i : i + 1, j : j + 1],
                        locs[i : i + 1, j : j + 1],
                        fluxes[i : i + 1, j : j + 1],
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
        return result


class CheckerboardSMC:
    def __init__(
        self,
        image,
        tile_dim,
        pad,
        image_pad,
        PriorClass,
        prior_kwargs,
        ImageModelClass,
        image_model_kwargs,
        MutationKernelClass,
        kernel_kwargs,
        num_catalogs,
        ess_threshold_prop,
        resample_method,
        max_smc_iters,
        prune_flux_lower,
        print_every=5,
    ):
        self.image = image
        self.H, self.W = image.shape[-2], image.shape[-1]
        self.tile_dim = tile_dim
        self.pad = pad
        self.image_pad = image_pad

        self.num_rows = self.H // tile_dim
        self.num_cols = self.W // tile_dim

        self.PriorClass = PriorClass
        self.prior_kwargs = prior_kwargs
        self.ImageModelClass = ImageModelClass
        self.image_model_kwargs = image_model_kwargs
        self.MutationKernelClass = MutationKernelClass
        self.kernel_kwargs = kernel_kwargs

        self.num_catalogs = num_catalogs
        self.ess_threshold_prop = ess_threshold_prop
        self.resample_method = resample_method
        self.max_smc_iters = max_smc_iters
        self.prune_flux_lower = prune_flux_lower
        self.print_every = print_every

        # How far away (in tiles) a neighbor's sources can still contribute
        self.neighbor_radius = math.ceil(self.pad / self.tile_dim)

        # Pre-pad the full image with background
        bg = image_model_kwargs["background"]
        self.padded_image = torch.full(
            (self.H + 2 * image_pad, self.W + 2 * image_pad), bg
        )
        self.padded_image[
            image_pad : image_pad + self.H, image_pad : image_pad + self.W
        ] = image

        # Storage for pruned results per tile: (counts, locs, fluxes)
        # Used during run() for conditioning between colors
        self.tile_results = {}

    def _get_tile_bounds(self, i, j):
        """Compute pixel, prior, and image model bounds for tile (i, j)."""
        td = self.tile_dim

        # Tile pixel region (strict)
        tile_h0 = i * td
        tile_h1 = (i + 1) * td
        tile_w0 = j * td
        tile_w1 = (j + 1) * td

        # Prior bounds (padded latent space)
        prior_h_lower = tile_h0 - self.pad
        prior_h_upper = tile_h1 + self.pad
        prior_w_lower = tile_w0 - self.pad
        prior_w_upper = tile_w1 + self.pad

        # ImageModel bounds (using image_pad, no clipping needed — image is pre-padded)
        img_h_lower = tile_h0 - self.image_pad
        img_h_upper = tile_h1 + self.image_pad
        img_w_lower = tile_w0 - self.image_pad
        img_w_upper = tile_w1 + self.image_pad

        return {
            "tile_h0": tile_h0,
            "tile_h1": tile_h1,
            "tile_w0": tile_w0,
            "tile_w1": tile_w1,
            "prior_h_lower": prior_h_lower,
            "prior_h_upper": prior_h_upper,
            "prior_w_lower": prior_w_lower,
            "prior_w_upper": prior_w_upper,
            "img_h_lower": img_h_lower,
            "img_h_upper": img_h_upper,
            "img_w_lower": img_w_lower,
            "img_w_upper": img_w_upper,
        }

    def _gather_conditioning(self, i, j):
        """Gather locs_cond/fluxes_cond from previously-sampled nearby tiles.

        Returns (locs_cond, fluxes_cond, provenance) where provenance is a list
        of (ni, nj, num_sources) tuples tracking which global tiles contributed.
        """
        all_locs, all_fluxes, provenance = [], [], []

        nr = self.neighbor_radius
        for di in range(-nr, nr + 1):
            for dj in range(-nr, nr + 1):
                ni, nj = i + di, j + dj
                if (ni, nj) == (i, j) or (ni, nj) not in self.tile_results:
                    continue

                _, locs, fluxes = self.tile_results[(ni, nj)]
                provenance.append((ni, nj, locs.shape[-2]))
                all_locs.append(locs)
                all_fluxes.append(fluxes)

        if not all_locs:
            return None, None, []

        return torch.cat(all_locs, dim=-2), torch.cat(all_fluxes, dim=-1), provenance

    def _extract_tile_image(self, i, j):
        """Extract tile image from padded image. Returns [tile_dim+2*image_pad, ...]."""
        td = self.tile_dim
        ip = self.image_pad
        # In padded image, tile (i,j) starts at (i*td, j*td) (the padding offset is built in)
        h0 = i * td
        h1 = i * td + td + 2 * ip
        w0 = j * td
        w1 = j * td + td + 2 * ip
        return self.padded_image[h0:h1, w0:w1]

    def run(self):
        for color in range(4):
            numH, numW, mapping = _color_subgrid(color, self.num_rows, self.num_cols)

            if numH == 0 or numW == 0:
                print(f"Color {color}: 0 tile(s)")
                continue

            print(f"Color {color}: {numH * numW} tile(s) ({numH}x{numW} sub-grid)")

            # Build tile images tensor [numH, numW, imgH, imgW]
            tile_images = torch.zeros(
                numH,
                numW,
                self.tile_dim + 2 * self.image_pad,
                self.tile_dim + 2 * self.image_pad,
            )

            # Build per-tile priors, bounds, and conditioning
            priors_grid = [[None] * numW for _ in range(numH)]
            prune_h_lower = torch.zeros(numH, numW)
            prune_h_upper = torch.zeros(numH, numW)
            prune_w_lower = torch.zeros(numH, numW)
            prune_w_upper = torch.zeros(numH, numW)
            img_h_lower = torch.zeros(numH, numW)
            img_h_upper = torch.zeros(numH, numW)
            img_w_lower = torch.zeros(numH, numW)
            img_w_upper = torch.zeros(numH, numW)

            # Gather per-tile conditioning
            per_tile_locs_cond = {}
            per_tile_fluxes_cond = {}
            per_tile_provenance = {}
            max_cond = 0

            for si, sj, gi, gj in mapping:
                bounds = self._get_tile_bounds(gi, gj)

                # Extract tile image
                tile_images[si, sj] = self._extract_tile_image(gi, gj)

                # Prior
                pkw = dict(self.prior_kwargs)
                pkw["h_lower"] = bounds["prior_h_lower"]
                pkw["h_upper"] = bounds["prior_h_upper"]
                pkw["w_lower"] = bounds["prior_w_lower"]
                pkw["w_upper"] = bounds["prior_w_upper"]
                priors_grid[si][sj] = self.PriorClass(**pkw)

                # Prune bounds
                prune_h_lower[si, sj] = bounds["tile_h0"]
                prune_h_upper[si, sj] = bounds["tile_h1"]
                prune_w_lower[si, sj] = bounds["tile_w0"]
                prune_w_upper[si, sj] = bounds["tile_w1"]

                # Image model bounds
                img_h_lower[si, sj] = bounds["img_h_lower"]
                img_h_upper[si, sj] = bounds["img_h_upper"]
                img_w_lower[si, sj] = bounds["img_w_lower"]
                img_w_upper[si, sj] = bounds["img_w_upper"]

                # Conditioning
                locs_c, fluxes_c, prov = self._gather_conditioning(gi, gj)
                per_tile_locs_cond[(si, sj)] = locs_c
                per_tile_fluxes_cond[(si, sj)] = fluxes_c
                per_tile_provenance[(si, sj)] = prov
                if locs_c is not None:
                    max_cond = max(max_cond, locs_c.shape[-2])

                print(
                    f"  Tile ({gi}, {gj}) -> sub-grid ({si}, {sj}): "
                    f"prior [{bounds['prior_h_lower']}, {bounds['prior_h_upper']}] x "
                    f"[{bounds['prior_w_lower']}, {bounds['prior_w_upper']}], "
                    f"image [{bounds['img_h_lower']}, {bounds['img_h_upper']}] x "
                    f"[{bounds['img_w_lower']}, {bounds['img_w_upper']}]"
                    + (
                        f", cond sources: {locs_c.shape[-2]}"
                        if locs_c is not None
                        else ""
                    )
                )

            # Build BatchPrior
            batch_prior = BatchPrior(priors_grid)

            # Build ImageModel with per-tile tensor bounds
            ikw = dict(self.image_model_kwargs)
            ikw["h_lower"] = img_h_lower
            ikw["h_upper"] = img_h_upper
            ikw["w_lower"] = img_w_lower
            ikw["w_upper"] = img_w_upper
            tile_image_model = self.ImageModelClass(**ikw)

            # Build MutationKernel
            tile_kernel = deepcopy(self.MutationKernelClass(**self.kernel_kwargs))

            # Build batched conditioning tensors
            locs_cond = None
            fluxes_cond = None
            if max_cond > 0:
                # Pad each tile's conditioning to max_cond sources with zeros
                # (zero-flux sources are inert in PSF)
                locs_cond = torch.zeros(numH, numW, self.num_catalogs, max_cond, 2)
                fluxes_cond = torch.zeros(numH, numW, self.num_catalogs, max_cond)
                for si, sj, gi, gj in mapping:
                    lc = per_tile_locs_cond[(si, sj)]
                    fc = per_tile_fluxes_cond[(si, sj)]
                    if lc is not None:
                        nc = lc.shape[-2]
                        # lc shape: [1, 1, num_catalogs, nc, 2]
                        locs_cond[si, sj, :, :nc, :] = lc.squeeze(0).squeeze(0)
                        fluxes_cond[si, sj, :, :nc] = fc.squeeze(0).squeeze(0)

            # Run one SMCsampler for all tiles of this color
            sampler = SMCsampler(
                image=tile_images,
                Prior=batch_prior,
                ImageModel=tile_image_model,
                MutationKernel=tile_kernel,
                num_catalogs=self.num_catalogs,
                ess_threshold_prop=self.ess_threshold_prop,
                resample_method=self.resample_method,
                max_smc_iters=self.max_smc_iters,
                prune_flux_lower=self.prune_flux_lower,
                prune_h_lower=prune_h_lower,
                prune_h_upper=prune_h_upper,
                prune_w_lower=prune_w_lower,
                prune_w_upper=prune_w_upper,
                print_every=self.print_every,
                locs_cond=locs_cond,
                fluxes_cond=fluxes_cond,
            )
            sampler.run()

            # Store pruned results per tile for conditioning by subsequent colors
            for si, sj, gi, gj in mapping:
                self.tile_results[(gi, gj)] = (
                    sampler.pruned_counts[si : si + 1, sj : sj + 1].clone(),
                    sampler.pruned_locs[si : si + 1, sj : sj + 1].clone(),
                    sampler.pruned_fluxes[si : si + 1, sj : sj + 1].clone(),
                )

            # Track the final color's state for catalog assembly
            self._final_sampler = sampler
            self._final_mapping = mapping
            self._final_cond_provenance = per_tile_provenance
            self._final_prune_bounds = (
                prune_h_lower,
                prune_h_upper,
                prune_w_lower,
                prune_w_upper,
            )

        # Combine all tiles' pruned and raw results
        self._combine_results()
        self._combine_raw_results()

        print("All tiles complete!")

    def _prune_tile(self, locs, fluxes, h_lo, h_hi, w_lo, w_hi):
        """Apply spatial bounds and flux pruning to a single tile's sources."""
        lower = torch.tensor([h_lo, w_lo])
        upper = torch.tensor([h_hi, w_hi])
        while lower.dim() < locs.dim():
            lower = lower.unsqueeze(0)
        while upper.dim() < locs.dim():
            upper = upper.unsqueeze(0)
        mask = torch.all(torch.logical_and(locs > lower, locs < upper), dim=-1)
        mask = mask * (fluxes > self.prune_flux_lower)
        counts = mask.sum(-1)

        locs = mask.unsqueeze(-1) * locs
        locs_mask = (locs != 0).int()
        locs_index = torch.sort(locs_mask, dim=3, descending=True)[1]
        locs = torch.gather(locs, dim=3, index=locs_index)

        fluxes = mask * fluxes
        fluxes_mask = (fluxes != 0).int()
        fluxes_index = torch.sort(fluxes_mask, dim=3, descending=True)[1]
        fluxes = torch.gather(fluxes, dim=3, index=fluxes_index)

        return counts, locs, fluxes

    def _append_conditioning(self, si, sj, all_locs, all_fluxes, claimed):
        """Append resampled conditioning sources from the final sampler,
        skipping already-claimed global tiles."""
        provenance = self._final_cond_provenance.get((si, sj), [])
        if not provenance or self._final_sampler.locs_cond is None:
            return
        offset = 0
        for ni, nj, nsrc in provenance:
            if (ni, nj) not in claimed:
                lc = self._final_sampler.locs_cond[si, sj, :, offset : offset + nsrc, :]
                fc = self._final_sampler.fluxes_cond[si, sj, :, offset : offset + nsrc]
                all_locs.append(lc.unsqueeze(0).unsqueeze(0))
                all_fluxes.append(fc.unsqueeze(0).unsqueeze(0))
                claimed.add((ni, nj))
            offset += nsrc

    def _combine_results(self):
        """Combine pruned catalogs from the final color's joint state."""
        all_locs = []
        all_fluxes = []
        claimed = set()
        ph_lo, ph_hi, pw_lo, pw_hi = self._final_prune_bounds

        for si, sj, gi, gj in self._final_mapping:
            # Prune the final-color tile's own sources to strict tile bounds
            tile_locs = self._final_sampler.locs[si : si + 1, sj : sj + 1]
            tile_fluxes = self._final_sampler.fluxes[si : si + 1, sj : sj + 1]
            _, pruned_locs, pruned_fluxes = self._prune_tile(
                tile_locs,
                tile_fluxes,
                ph_lo[si, sj],
                ph_hi[si, sj],
                pw_lo[si, sj],
                pw_hi[si, sj],
            )
            all_locs.append(pruned_locs)
            all_fluxes.append(pruned_fluxes)
            claimed.add((gi, gj))
            # Conditioning sources are already pruned — pass through
            self._append_conditioning(si, sj, all_locs, all_fluxes, claimed)

        self.combined_locs = torch.cat(all_locs, dim=-2)
        self.combined_fluxes = torch.cat(all_fluxes, dim=-1)
        self.combined_counts = (self.combined_fluxes > 0).sum(-1)

    def _combine_raw_results(self):
        """Combine raw (unpruned) catalogs from the final color's joint state."""
        all_locs = []
        all_fluxes = []
        claimed = set()

        for si, sj, gi, gj in self._final_mapping:
            # Final-color tile's own unpruned sources
            all_locs.append(self._final_sampler.locs[si : si + 1, sj : sj + 1])
            all_fluxes.append(self._final_sampler.fluxes[si : si + 1, sj : sj + 1])
            claimed.add((gi, gj))
            # Resampled conditioning from earlier tiles
            self._append_conditioning(si, sj, all_locs, all_fluxes, claimed)

        self.combined_raw_locs = torch.cat(all_locs, dim=-2)
        self.combined_raw_fluxes = torch.cat(all_fluxes, dim=-1)
        self.combined_raw_counts = (self.combined_raw_fluxes > 0).sum(-1)
