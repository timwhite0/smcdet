import torch
from einops import rearrange
from torch.distributions import Independent, Normal, Poisson


class ImageModel(object):
    def __init__(self, image_height, image_width, background, psf_stdev=None):
        self.image_height = image_height
        self.image_width = image_width
        self.background = background
        self.psf_stdev = psf_stdev

        if self.psf_stdev is not None:
            self.psf_density = Independent(
                Normal(torch.zeros(1), self.psf_stdev * torch.ones(1)), 1
            )

        self.update_psf_grid()

    def update_psf_grid(self):
        psf_marginal_h = torch.arange(0, self.image_height)
        psf_marginal_w = torch.arange(0, self.image_width)
        grid_h, grid_w = torch.meshgrid(psf_marginal_h, psf_marginal_w, indexing="ij")
        self.psf_grid = torch.stack([grid_h, grid_w], dim=-1)

    def psf(self, locs):
        psf_grid_adjusted = (
            self.psf_grid
            - rearrange(locs, "numH numW n d t -> numH numW n d 1 1 t")
            + 0.5
        )
        logpsf = self.psf_density.log_prob(psf_grid_adjusted)
        psf = logpsf.exp()
        return rearrange(psf, "numH numW n d dimH dimW -> numH numW dimH dimW n d")

    def sample(self, locs, fluxes):
        psf = self.psf(locs)
        rate = (psf * rearrange(fluxes, "numH numW n d -> numH numW 1 1 n d")).sum(
            -1
        ) + self.background
        return Poisson(rate).sample()

    def generate(self, Prior, num_images=1):
        catalogs = Prior.sample(num_catalogs=num_images)
        unpruned_counts, unpruned_locs, unpruned_fluxes = catalogs

        images = self.sample(unpruned_locs, unpruned_fluxes)

        if Prior.pad > 0:
            in_bounds = torch.all(
                torch.logical_and(
                    unpruned_locs > 0,
                    unpruned_locs < torch.tensor((self.image_height, self.image_width)),
                ),
                dim=-1,
            )

            pruned_counts = in_bounds.sum(-1)

            pruned_locs = in_bounds.unsqueeze(-1) * unpruned_locs
            pruned_locs_mask = (pruned_locs != 0).int()
            pruned_locs_index = torch.sort(pruned_locs_mask, dim=3, descending=True)[1]
            pruned_locs = torch.gather(pruned_locs, dim=3, index=pruned_locs_index)

            pruned_fluxes = in_bounds * unpruned_fluxes
            pruned_fluxes_mask = (pruned_fluxes != 0).int()
            pruned_fluxes_index = torch.sort(
                pruned_fluxes_mask, dim=3, descending=True
            )[1]
            pruned_fluxes = torch.gather(
                pruned_fluxes, dim=3, index=pruned_fluxes_index
            )

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
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.adu_per_nmgy = adu_per_nmgy
        self.sigma1, self.sigma2, self.sigmap, self.beta, self.b, self.p0 = psf_params
        self.noise_additive = noise_additive
        self.noise_multiplicative = noise_multiplicative

        # compute PSF normalizing constant
        psf_marginal_h = torch.arange(0, 32 * self.image_height)
        psf_marginal_w = torch.arange(0, 32 * self.image_width)
        grid_h, grid_w = torch.meshgrid(psf_marginal_h, psf_marginal_w, indexing="ij")
        big_psf_grid = torch.stack([grid_h, grid_w], dim=-1)
        one_loc_in_center = torch.tensor(
            [psf_marginal_h.shape[0] / 2.0, psf_marginal_w.shape[0] / 2.0]
        )
        psf_grid_adjusted = (
            big_psf_grid - rearrange(one_loc_in_center, "t -> 1 1 t") + 0.5
        )
        self.psf_normalizing_constant = self.unnormalized_psf(
            (psf_grid_adjusted**2).sum(-1).sqrt()
        ).sum()

    def unnormalized_psf(self, r):
        term1 = torch.exp(-(r**2) / (2 * self.sigma1))
        term2 = self.b * torch.exp(-(r**2) / (2 * self.sigma2))
        term3 = self.p0 * (1 + r**2 / (self.beta * self.sigmap)) ** (-self.beta / 2)
        return (term1 + term2 + term3) / (1 + self.b + self.p0)

    def psf(self, locs):
        psf_grid_adjusted = (
            self.psf_grid
            - rearrange(locs, "numH numW n d t -> numH numW n d 1 1 t")
            + 0.5
        )
        unnormalized_psf = self.unnormalized_psf((psf_grid_adjusted**2).sum(-1).sqrt())
        psf = unnormalized_psf / self.psf_normalizing_constant
        return rearrange(psf, "numH numW n d dimH dimW -> numH numW dimH dimW n d")

    def sample(self, locs, fluxes):
        psf = self.psf(locs)
        rate = (
            psf
            * rearrange(
                self.adu_per_nmgy * fluxes, "numH numW n d -> numH numW 1 1 n d"
            )
        ).sum(-1) + self.background
        return Normal(
            rate, self.noise_additive + self.noise_multiplicative * rate.sqrt()
        ).sample()

    def loglikelihood(self, tiled_image, locs, fluxes):
        psf = self.psf(locs)
        rate = (
            psf
            * rearrange(
                self.adu_per_nmgy * fluxes, "numH numW n d -> numH numW 1 1 n d"
            )
        ).sum(-1) + self.background

        return (
            Normal(rate, self.noise_additive + self.noise_multiplicative * rate.sqrt())
            .log_prob(tiled_image.unsqueeze(-1))
            .sum([-2, -3])
        )
