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
        self.log_prob_in_box = prob_in_box_hw.log().nan_to_num()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.base_dist})"

    def sample(self, shape=None):
        if shape is None:
            shape = tuple(self.dim)

        p = torch.rand(shape).clamp(min=1e-6, max=1.0 - 1e-6)
        p_tilde = self.base_dist.cdf(self.lb) + p * (self.log_prob_in_box.exp())
        x = self.base_dist.icdf(p_tilde.clamp(min=1e-6, max=1.0 - 1e-6))

        return x.clamp(min=self.lb, max=self.ub)

    def log_prob(self, value):
        assert (value >= self.lb).all() and (value <= self.ub).all()
        return self.base_dist.log_prob(value) - self.log_prob_in_box

    def cdf(self, value):
        cdf_at_val = self.base_dist.cdf(value)
        cdf_at_lb = self.base_dist.cdf(self.lb)
        log_cdf = (cdf_at_val - cdf_at_lb + 1e-9).log().sum(-1) - self.log_prob_in_box
        return log_cdf.exp()


class TruncatedPareto(Distribution):
    """See https://en.wikipedia.org/wiki/Pareto_distribution#Bounded_Pareto_distribution."""

    def __init__(self, alpha, lower, upper):
        self.alpha = torch.tensor(alpha)
        self.lower = torch.tensor(lower)
        self.upper = torch.tensor(upper)

        self.logpdf_norm_const = (
            self.alpha.log()
            + alpha * self.lower.log()
            + self.alpha * self.upper.log()
            - (self.upper**self.alpha - self.lower**self.alpha).log()
        )

    def sample(self, shape=[]):
        unif = torch.rand(shape)
        numerator = (
            self.upper**self.alpha
            - unif * (self.upper**self.alpha)
            + unif * (self.lower**self.alpha)
        )
        denominator = (self.lower**self.alpha) * (self.upper**self.alpha)
        exponent = -1 / self.alpha
        return (numerator / denominator) ** exponent

    def log_prob(self, value):
        assert (value >= self.lower).all() and (value <= self.upper).all()
        return self.logpdf_norm_const - (self.alpha + 1) * value.log()
