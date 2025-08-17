from abc import ABC

import torch

EPS = 2**-22


class CausalExperimentModel(ABC):
    """
    Basic interface for probabilistic models
    """

    def __init__(self):
        self.epsilon = torch.tensor(EPS)

    def sanity_check(self):
        assert self.var_dim > 0
        assert len(self.var_names) > 0

    def reset(self, n_parallel):
        raise NotImplementedError

    def run_experiment(self, design, theta):
        """
        Execute an experiment with given design.
        """
        # create model from sampled params
        n_samples = design.shape[-2]
        y = self.rsample(design, theta, n_samples)
        if isinstance(y, torch.Tensor):
            y = y.detach().clone()
        return y

    def get_likelihoods(self, y, design, thetas):
        lik = self.log_prob(y, design, thetas)
        return lik

    def sample_theta(self, num_theta, zero_bias):
        thetas = self.sample_prior(num_theta, zero_bias=zero_bias)
        return thetas
