import torch
from einops import rearrange
from torch.distributions import Independent, Normal, Poisson


class ImageModel(object):
    def __init__(self, image_height, image_width, psf_stdev, background):
        self.image_height = image_height
        self.image_width = image_width
        self.psf_stdev = psf_stdev
        self.background = background

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
        psf_grid_adjusted = self.psf_grid - (
            rearrange(locs, "numH numW n d t -> numH numW n d 1 1 t") - 0.5
        )
        logpsf = self.psf_density.log_prob(psf_grid_adjusted)
        psf = logpsf.exp()
        psf = rearrange(psf, "numH numW n d dimH dimW -> numH numW dimH dimW n d")

        return psf

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
    def __init__(self, *args, flux_calibration, **kwargs):
        super().__init__(*args, **kwargs)
        self.flux_calibration = flux_calibration

    def sample(self, locs, fluxes):
        psf = self.psf(locs)
        rate = (
            psf
            * rearrange(
                self.flux_calibration * fluxes, "numH numW n d -> numH numW 1 1 n d"
            )
        ).sum(-1) + self.background
        return Normal(rate, rate.sqrt()).sample()

    def loglikelihood(self, tiled_image, locs, fluxes):
        psf = self.psf(locs)
        rate = (
            psf
            * rearrange(
                self.flux_calibration * fluxes, "numH numW n d -> numH numW 1 1 n d"
            )
        ).sum(-1) + self.background

        return Normal(rate, rate.sqrt()).log_prob(tiled_image.unsqueeze(-1))
