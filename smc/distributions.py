import torch
from torch.distributions import Distribution, Normal


class TruncatedDiagonalMVN(Distribution):
    """A truncated diagonal multivariate normal distribution."""

    def __init__(self, mu, sigma, lb, ub):
        super().__init__(validate_args=False)

        self.dim = mu.size()

        self.lb = lb
        self.ub = ub

        self.base_dist = Normal(mu, sigma)
        prob_in_box_hw = self.base_dist.cdf(self.ub) - self.base_dist.cdf(self.lb)
        self.log_prob_in_box = prob_in_box_hw.log()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.base_dist})"

    def sample(self, shape=None):
        if shape is None:
            shape = tuple(self.dim)

        p = torch.rand(shape).clamp(min=1e-6, max=1.0 - 1e-6)
        p_tilde = self.base_dist.cdf(self.lb) + p * (self.log_prob_in_box.exp())
        x = self.base_dist.icdf(p_tilde)

        return x.clamp(min=self.lb, max=self.ub)

    def log_prob(self, value):
        assert (value >= self.lb).all() and (value <= self.ub).all()
        return self.base_dist.log_prob(value) - self.log_prob_in_box

    def cdf(self, value):
        cdf_at_val = self.base_dist.cdf(value)
        cdf_at_lb = self.base_dist.cdf(self.lb)
        log_cdf = (cdf_at_val - cdf_at_lb + 1e-9).log().sum(-1) - self.log_prob_in_box
        return log_cdf.exp()
