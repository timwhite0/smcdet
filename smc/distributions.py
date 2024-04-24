import torch
from torch.distributions import Normal, Independent, Distribution

class TruncatedDiagonalMVN(Distribution):
    """A truncated diagonal multivariate normal distribution."""

    def __init__(self, mu, sigma, a, b):
        super().__init__(validate_args=False)

        self.dim = mu.size()
        
        self.lb = a*torch.ones_like(mu)
        self.ub = b*torch.ones_like(mu)
        
        base = Normal(mu, sigma)
        prob_in_box_hw = base.cdf(b*torch.ones_like(mu)) - base.cdf(a*torch.ones_like(mu))
        self.log_prob_in_box = prob_in_box_hw.log()

        self.base_dist = base

    def __repr__(self):
        return f"{self.__class__.__name__}({self.base_dist})"

    def sample(self, sample_shape=(1,), **args):
        p = torch.rand(sample_shape).clamp(min = 1e-6, max = 1.0 - 1e-6)
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
    
# class TruncatedDiagonalMVN(Distribution):
#     """A truncated diagonal multivariate normal distribution."""

#     def __init__(self, mu, sigma, a, b):
#         super().__init__(validate_args=False)

#         self.multiple_normals = Normal(mu, sigma)
#         self.base_dist = Independent(self.multiple_normals, 1)

#         self.a = a
#         self.b = b
#         self.lb = self.a * torch.ones_like(self.base_dist.mean)
#         self.ub = self.b * torch.ones_like(self.base_dist.mean)

#         prob_in_box = self.multiple_normals.cdf(self.ub) - self.multiple_normals.cdf(self.lb)
#         self.log_prob_in_box = prob_in_box.log().sum(dim=-1)

#     def __repr__(self):
#         return f"{self.__class__.__name__}({self.base_dist})"

#     def sample(self):
#         q = Independent(Normal(self.base_dist.mean, self.base_dist.stddev), 1)
#         samples = q.sample()
#         valid = (samples.min(dim=-1)[0] >= self.a) & (samples.max(dim=-1)[0] < self.b)
#         while not valid.all():
#             new_samples = q.sample()
#             samples[~valid] = new_samples[~valid]
#             valid = (samples.min(dim=-1)[0] >= self.a) & (samples.max(dim=-1)[0] < self.b)
#         return samples

#     @property
#     def mean(self):
#         mu = self.base_dist.mean
#         sigma = self.base_dist.stddev

#         offset_numerator = (
#             self.multiple_normals.log_prob(self.lb).exp()
#             - self.multiple_normals.log_prob(self.ub).exp()
#         )
#         offset_denominator = self.multiple_normals.cdf(self.ub) - self.multiple_normals.cdf(self.lb)
#         return mu + (sigma * offset_numerator / offset_denominator)

#     @property
#     def stddev(self):
#         # See https://arxiv.org/pdf/1206.5387.pdf for the formula for the variance of a truncated
#         # multivariate normal. The covariance terms simplify since our dimensions are independent,
#         # but it's still tricky to compute.
#         raise NotImplementedError("Standard deviation for truncated normal is not implemented yet")

#     @property
#     def mode(self):
#         assert (self.mean >= self.lb).all() and (self.mean <= self.ub).all()
#         return self.base_dist.mode

#     def log_prob(self, value):
#         assert (value >= self.lb).all() and (value <= self.ub).all()
#         return self.base_dist.log_prob(value) - self.log_prob_in_box

#     def cdf(self, value):
#         cdf_at_val = self.base_dist.cdf(value)
#         cdf_at_lb = self.base_dist.cdf(self.lb * torch.ones_like(self.mean))
#         log_cdf = (cdf_at_val - cdf_at_lb + 1e-9).log().sum(dim=-1) - self.log_prob_in_box
#         return log_cdf.exp()