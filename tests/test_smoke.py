"""Smoke test: verify SMCsampler runs to completion with optimized code."""

import torch

from smcdet.images import ImageModel
from smcdet.kernel import SingleComponentMH
from smcdet.prior import StarPrior
from smcdet.sampler import SMCsampler


def test_smc_sampler_smoke():
    torch.manual_seed(42)

    # Small problem: 1x1 tile, 8x8 image, few catalogs
    image_dim = 8
    num_catalogs_per_count = 5
    max_objects = 3

    prior = StarPrior(
        num_objects=max_objects,
        h_lower=0.0,
        h_upper=float(image_dim),
        w_lower=0.0,
        w_upper=float(image_dim),
        flux_mean=100.0,
        flux_stdev=20.0,
    )

    image_model = ImageModel(
        h_lower=0.0,
        h_upper=float(image_dim),
        w_lower=0.0,
        w_upper=float(image_dim),
        background=10.0,
        psf_radius=2,
        psf_stdev=1.0,
    )

    kernel = SingleComponentMH(
        num_iters=2,
        locs_stdev=[0.5, 0.5],
        fluxes_stdev=5.0,
        fluxes_min=0.1,
        fluxes_max=500.0,
    )

    # Generate a synthetic image
    cats = prior.sample(
        num_tiles_h=1,
        num_tiles_w=1,
        num_catalogs=1,
    )
    counts, locs, fluxes = cats
    image = image_model.sample(locs, fluxes).squeeze(-1)

    sampler = SMCsampler(
        image=image,
        Prior=prior,
        ImageModel=image_model,
        MutationKernel=kernel,
        num_catalogs=num_catalogs_per_count,
        ess_threshold_prop=0.5,
        resample_method="systematic",
        max_smc_iters=50,
        prune_flux_lower=1.0,
        print_every=100,
    )

    sampler.run()

    # Temperature should reach 1.0
    assert torch.all(sampler.temperature >= 1.0), (
        f"Temperature did not reach 1.0: {sampler.temperature}"
    )

    # Weights should sum to ~1
    weight_sums = sampler.weights.sum(-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), (
        f"Weights don't sum to 1: {weight_sums}"
    )

    # Counts should be non-negative
    assert torch.all(sampler.counts >= 0), "Negative counts found"
    assert torch.all(sampler.pruned_counts >= 0), "Negative pruned counts found"


def test_smc_sampler_multinomial_resample():
    torch.manual_seed(123)

    image_dim = 8
    prior = StarPrior(
        num_objects=2,
        h_lower=0.0,
        h_upper=float(image_dim),
        w_lower=0.0,
        w_upper=float(image_dim),
        flux_mean=100.0,
        flux_stdev=20.0,
    )
    image_model = ImageModel(
        h_lower=0.0,
        h_upper=float(image_dim),
        w_lower=0.0,
        w_upper=float(image_dim),
        background=10.0,
        psf_radius=2,
        psf_stdev=1.0,
    )
    kernel = SingleComponentMH(
        num_iters=1,
        locs_stdev=[0.5, 0.5],
        fluxes_stdev=5.0,
        fluxes_min=0.1,
        fluxes_max=500.0,
    )

    cats = prior.sample(
        num_tiles_h=1,
        num_tiles_w=1,
        num_catalogs=1,
    )
    _, locs, fluxes = cats
    image = image_model.sample(locs, fluxes).squeeze(-1)

    sampler = SMCsampler(
        image=image,
        Prior=prior,
        ImageModel=image_model,
        MutationKernel=kernel,
        num_catalogs=3,
        ess_threshold_prop=0.5,
        resample_method="multinomial",
        max_smc_iters=20,
        prune_flux_lower=1.0,
        print_every=100,
    )
    sampler.run()

    assert torch.all(sampler.temperature >= 1.0)
    assert torch.all(sampler.counts >= 0)


def test_vectorized_bisect_matches_scipy():
    """Verify vectorized bisection matches scipy brentq to within tolerance."""
    from scipy.optimize import brentq

    torch.manual_seed(99)

    # Directly test the bisection logic with synthetic loglik values
    # Create a minimal sampler just to access the methods
    image_dim = 8
    max_objects = 3
    prior = StarPrior(
        num_objects=max_objects,
        h_lower=0.0,
        h_upper=float(image_dim),
        w_lower=0.0,
        w_upper=float(image_dim),
        flux_mean=100.0,
        flux_stdev=20.0,
    )
    image_model = ImageModel(
        h_lower=0.0,
        h_upper=float(image_dim),
        w_lower=0.0,
        w_upper=float(image_dim),
        background=10.0,
        psf_radius=2,
        psf_stdev=1.0,
    )
    kernel = SingleComponentMH(
        num_iters=1,
        locs_stdev=[0.5, 0.5],
        fluxes_stdev=5.0,
        fluxes_min=0.1,
        fluxes_max=500.0,
    )

    # Create a dummy 1-pixel image so the sampler can be constructed
    dummy_image = torch.ones(1, 1, image_dim, image_dim) * 10.0
    sampler = SMCsampler(
        image=dummy_image,
        Prior=prior,
        ImageModel=image_model,
        MutationKernel=kernel,
        num_catalogs=10,
        ess_threshold_prop=0.5,
        resample_method="systematic",
        max_smc_iters=50,
        prune_flux_lower=1.0,
        print_every=100,
    )

    # Synthetic loglik: [2, 2, 20] with varying spreads across tiles
    N = 20
    loglik = torch.randn(2, 2, N) * 10 - 50  # negative logliks with spread
    loglik[0, 1] *= 2.0  # more spread -> needs more bisection
    loglik[1, 0] *= 0.5
    loglik[1, 1] *= 3.0

    sampler.num_tiles_h = 2
    sampler.num_tiles_w = 2
    upper = torch.ones(2, 2)

    # Compute with vectorized bisection
    obj_at_upper = sampler._batched_tempering_objective(loglik, upper)
    needs_search = obj_at_upper < 0
    delta_vec = upper.clone()
    if needs_search.any():
        searched = sampler._vectorized_bisect(loglik, upper, needs_search)
        delta_vec = torch.where(needs_search, searched, delta_vec)

    # Compute with scipy brentq
    delta_scipy = torch.zeros_like(upper)
    for h in range(2):
        for w in range(2):

            def func(d):
                return sampler.tempering_objective(loglik[h, w], d)

            if func(upper[h, w].item()) < 0:
                delta_scipy[h, w] = brentq(
                    func, 0.0, upper[h, w].item(), xtol=1e-6, rtol=1e-6
                )
            else:
                delta_scipy[h, w] = upper[h, w]

    max_diff = (delta_vec - delta_scipy).abs().max().item()
    assert max_diff < 1e-5, f"Max difference between vectorized and scipy: {max_diff}"
    print(f"  max |vectorized - scipy| = {max_diff:.2e}")


if __name__ == "__main__":
    test_smc_sampler_smoke()
    print("smoke test (systematic) passed")
    test_smc_sampler_multinomial_resample()
    print("smoke test (multinomial) passed")
    test_vectorized_bisect_matches_scipy()
    print("vectorized bisection test passed")
