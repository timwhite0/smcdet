import torch
from torch.distributions import Poisson, Normal, Uniform, Distribution, Categorical
from distributions import TruncatedDiagonalMVN
import numpy as np

class MCMCKernel(object):
    def __init__(self):