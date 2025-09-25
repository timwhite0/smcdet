import torch
from einops import rearrange
from torch.distributions import Normal, Poisson


class ImageModel(object):
    def __init__(
        self, image_height, image_width, background, psf_radius: int, psf_stdev=None
    ):
        self.image_height = image_height
        self.image_width = image_width
        self.background = background
        self.psf_radius = psf_radius
        self.psf_stdev = psf_stdev

        if self.psf_stdev is not None:
            self.psf_density = Normal(torch.zeros(1), self.psf_stdev * torch.ones(1))

        psf_patch_seq = torch.arange(-self.psf_radius, self.psf_radius + 1)
        psf_patch_h, psf_patch_w = torch.meshgrid(
            psf_patch_seq, psf_patch_seq, indexing="ij"
        )
        self.psf_patch = torch.stack([psf_patch_h, psf_patch_w], dim=-1)

    def _compute_normalized_psf(self, r):
        return self.psf_density.log_prob(r).exp()

    def psf(self, locs):
        numH, numW, n, d, _ = locs.shape

        star_centers = rearrange(locs, "numH numW n d t -> numH numW n d 1 1 t")

        pixel_coords = torch.floor(star_centers) + rearrange(
            self.psf_patch, "dimH dimW t -> 1 1 1 1 dimH dimW t"
        )

        h_in_bounds = (pixel_coords[..., 0] >= 0) & (
            pixel_coords[..., 0] < self.image_height
        )
        w_in_bounds = (pixel_coords[..., 1] >= 0) & (
            pixel_coords[..., 1] < self.image_width
        )
        coords_in_bounds = h_in_bounds & w_in_bounds

        r = torch.norm((pixel_coords + 0.5) - star_centers, dim=-1)

        if coords_in_bounds.any():
            h_tile, w_tile, n_idx, d_idx, h_patch, w_patch = torch.where(
                coords_in_bounds
            )
            valid_r = r[coords_in_bounds]
            valid_psf_vals = self._compute_normalized_psf(valid_r)

            pixel_coords_int = pixel_coords.long()
            target_h = pixel_coords_int[
                h_tile, w_tile, n_idx, d_idx, h_patch, w_patch, 0
            ]
            target_w = pixel_coords_int[
                h_tile, w_tile, n_idx, d_idx, h_patch, w_patch, 1
            ]

            linear_indices = (
                h_tile * numW * self.image_height * self.image_width * n * d
                + w_tile * self.image_height * self.image_width * n * d
                + target_h * self.image_width * n * d
                + target_w * n * d
                + n_idx * d
                + d_idx
            )

            psf_flat = torch.zeros(
                numH * numW * self.image_height * self.image_width * n * d
            )
            psf_flat.scatter_add_(0, linear_indices, valid_psf_vals)

        return psf_flat.view(numH, numW, self.image_height, self.image_width, n, d)

    def sample(self, locs, fluxes):
        psf = self.psf(locs)
        rate = (psf * rearrange(fluxes, "numH numW n d -> numH numW 1 1 n d")).sum(
            -1
        ) + self.background
        return Poisson(rate).sample()

    def loglikelihood(self, tiled_image, locs, fluxes):
        psf = self.psf(locs)
        rate = (psf * rearrange(fluxes, "numH numW n d -> numH numW 1 1 n d")).sum(
            -1
        ) + self.background

        mask = rate > 50000

        loglik_poisson = Poisson(rate).log_prob(tiled_image.unsqueeze(-1))

        if mask.sum() > 0:
            loglik_normal = Normal(rate, rate.sqrt()).log_prob(
                tiled_image.unsqueeze(-1)
            )
            loglik = torch.where(mask, loglik_normal, loglik_poisson).sum([-2, -3])
            return loglik
        else:
            return loglik_poisson.sum([-2, -3])


class M71ImageModel(ImageModel):
    def __init__(
        self,
        *args,
        adu_per_nmgy,
        psf_params,
        noise_additive=0,
        noise_multiplicative=1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.adu_per_nmgy = adu_per_nmgy
        self.sigma1, self.sigma2, self.sigmap, self.beta, self.b, self.p0 = psf_params
        self.noise_additive = noise_additive
        self.noise_multiplicative = noise_multiplicative

        # compute PSF normalizing constant
        psf_marginal_h = torch.arange(0, 32 * self.psf_radius)
        psf_marginal_w = torch.arange(0, 32 * self.psf_radius)
        grid_h, grid_w = torch.meshgrid(psf_marginal_h, psf_marginal_w, indexing="ij")
        big_psf_grid = torch.stack([grid_h, grid_w], dim=-1)
        one_loc_in_center = torch.tensor(
            [psf_marginal_h.shape[0] / 2.0, psf_marginal_w.shape[0] / 2.0]
        )
        psf_grid_adjusted = (
            big_psf_grid - rearrange(one_loc_in_center, "t -> 1 1 t") + 0.5
        )
        self.psf_normalizing_constant = self._compute_unnormalized_psf(
            (psf_grid_adjusted**2).sum(-1).sqrt()
        ).sum()

    def _compute_unnormalized_psf(self, r):
        term1 = torch.exp(-(r**2) / (2 * self.sigma1))
        term2 = self.b * torch.exp(-(r**2) / (2 * self.sigma2))
        term3 = self.p0 * (1 + r**2 / (self.beta * self.sigmap)) ** (-self.beta / 2)
        return (term1 + term2 + term3) / (1 + self.b + self.p0)

    # override parent class
    def _compute_normalized_psf(self, r):
        return self._compute_unnormalized_psf(r) / self.psf_normalizing_constant

    def sample(self, locs, fluxes):
        psf = self.psf(locs)
        rate = (
            psf
            * rearrange(
                self.adu_per_nmgy * fluxes, "numH numW n d -> numH numW 1 1 n d"
            )
        ).sum(-1) + self.background
        return Normal(
            rate, (self.noise_additive + self.noise_multiplicative * rate).sqrt()
        ).sample()

    def loglikelihood(
        self,
        tiled_image,
        locs,
        fluxes,
        locs_cond=None,
        fluxes_cond=None,
        image_cond=None,
    ):
        if locs_cond is not None:
            locs = torch.cat((locs, locs_cond), dim=-2)
        if fluxes_cond is not None:
            fluxes = torch.cat((fluxes, fluxes_cond), dim=-1)
        if image_cond is not None:
            tiled_image = image_cond

        psf = self.psf(locs)

        rate = (
            psf
            * rearrange(
                self.adu_per_nmgy * fluxes, "numH numW n d -> numH numW 1 1 n d"
            )
        ).sum(-1) + self.background

        return (
            Normal(
                rate, (self.noise_additive + self.noise_multiplicative * rate).sqrt()
            )
            .log_prob(tiled_image.unsqueeze(-1))
            .sum([-2, -3])
        )


def generate_images(
    Prior,
    ImageModel,
    flux_threshold,
    loc_threshold_lower,
    loc_threshold_upper,
    num_images=1,
):
    catalogs = Prior.sample(num_catalogs=num_images)
    unpruned_counts, unpruned_locs, unpruned_fluxes = catalogs

    images = ImageModel.sample(unpruned_locs, unpruned_fluxes)

    mask = torch.all(
        torch.logical_and(
            unpruned_locs > loc_threshold_lower,
            unpruned_locs < torch.tensor((loc_threshold_upper, loc_threshold_upper)),
        ),
        dim=-1,
    )
    mask *= unpruned_fluxes > flux_threshold

    pruned_counts = mask.sum(-1)

    pruned_locs = mask.unsqueeze(-1) * unpruned_locs
    pruned_locs_mask = (pruned_locs != 0).int()
    pruned_locs_index = torch.sort(pruned_locs_mask, dim=3, descending=True)[1]
    pruned_locs = torch.gather(pruned_locs, dim=3, index=pruned_locs_index)

    pruned_fluxes = mask * unpruned_fluxes
    pruned_fluxes_mask = (pruned_fluxes != 0).int()
    pruned_fluxes_index = torch.sort(pruned_fluxes_mask, dim=3, descending=True)[1]
    pruned_fluxes = torch.gather(pruned_fluxes, dim=3, index=pruned_fluxes_index)

    unpruned_counts = unpruned_counts.squeeze([0, 1])
    unpruned_locs = unpruned_locs.squeeze([0, 1])
    unpruned_fluxes = unpruned_fluxes.squeeze([0, 1])
    pruned_counts = pruned_counts.squeeze([0, 1])
    pruned_locs = pruned_locs.squeeze([0, 1])
    pruned_fluxes = pruned_fluxes.squeeze([0, 1])
    images = rearrange(images.squeeze([0, 1]), "dimH dimW n -> n dimH dimW")

    return [
        unpruned_counts,
        unpruned_locs,
        unpruned_fluxes,
        pruned_counts,
        pruned_locs,
        pruned_fluxes,
        images,
    ]
