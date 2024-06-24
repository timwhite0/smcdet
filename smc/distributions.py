import torch
from torch.distributions import Normal, Distribution

class TruncatedDiagonalMVN(Distribution):
    """A truncated diagonal multivariate normal distribution."""

    def __init__(self, mu, sigma, a, b):
        super().__init__(validate_args=False)

        self.dim = mu.size()
        
        self.lb = a*torch.ones_like(mu)
        self.ub = b*torch.ones_like(mu)
        
        self.base_dist = Normal(mu, sigma)
        prob_in_box_hw = self.base_dist.cdf(b*torch.ones_like(mu)) - self.base_dist.cdf(a*torch.ones_like(mu))
        self.log_prob_in_box = prob_in_box_hw.log()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.base_dist})"

    def sample(self, **args):
        p = torch.rand(tuple(self.dim)).clamp(min = 1e-6, max = 1.0 - 1e-6)
        p_tilde = self.base_dist.cdf(self.lb) + p * (self.log_prob_in_box.exp())
        x = self.base_dist.icdf(p_tilde)
        
        return x

    @property
    def mode(self):
        assert (self.mean >= self.lb).all() and (self.mean <= self.ub).all()
        return self.base_dist.mode

    def log_prob(self, value):
        assert (value >= self.lb).all() and (value <= self.ub).all()
        return self.base_dist.log_prob(value) - self.log_prob_in_box

    def cdf(self, value):
        cdf_at_val = self.base_dist.cdf(value)
        cdf_at_lb = self.base_dist.cdf(self.lb*torch.ones_like(self.mean))
        log_cdf = (cdf_at_val - cdf_at_lb + 1e-9).log().sum(dim=-1) - self.log_prob_in_box
        return log_cdf.exp()
