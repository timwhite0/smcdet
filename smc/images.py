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
        psf_grid_adjusted = self.psf_grid - (locs.unsqueeze(-2).unsqueeze(-3) - 0.5)
        logpsf = self.psf_density.log_prob(psf_grid_adjusted)
        psf = logpsf.exp()
        psf = rearrange(psf, "numH numW n d dimH dimW -> numH numW dimH dimW n d")

        return psf

    def generate(self, Prior, num_images=1):
        catalogs = Prior.sample(num_catalogs=num_images)
        counts, locs, fluxes = catalogs

        psf = self.psf(locs)
        rate = (psf * fluxes.unsqueeze(-3).unsqueeze(-4)).sum(-1) + self.background
        images = Poisson(rate).sample()

        if Prior.pad > 0:
            in_bounds = torch.all(
                torch.logical_and(
                    locs > 0, locs < torch.tensor((self.image_height, self.image_width))
                ),
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

        counts = counts.squeeze([0, 1])
        locs = locs.squeeze([0, 1])
        fluxes = fluxes.squeeze([0, 1])
        images = rearrange(images.squeeze([0, 1]), "dimH dimW n -> n dimH dimW")

        return [counts, locs, fluxes, images]

    def loglikelihood(self, tiled_image, locs, fluxes):
        psf = self.psf(locs)
        rate = (psf * fluxes.unsqueeze(-3).unsqueeze(-4)).sum(-1) + self.background

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
