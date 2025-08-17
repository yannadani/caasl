import numpy as np
import torch
import torch.distributions as dist

from .causal_experiment_model import CausalExperimentModel
from .graph_priors import ErdosRenyi, PermuteGenerate, ScaleFree
from .utils import InverseGamma, compute_rff


class LinGaussANMModel(CausalExperimentModel):
    """
    Linear Additive Noise Structural Equation Model
    Args:
        d (int): number of nodes in the graph
        n_parallel (int): number of parallel models to sample
        graph_prior (str): graph prior (erdos_renyi, scale_free)
        graph_args (dict): graph arguments for sampling the graph (degree for erdos_renyi, degree and power for scale_free)
        noise_type (str): type of noise (gauss, laplace, gumbel)
        noise_args (dict): noise arguments for sampling the noise (concentration and rate for gauss, scale for laplace, loc and scale for gumbel)
        intervention_type (str): type of intervention (shift)
        coeff_mean (float): mean of the coefficients
    """

    def __init__(
        self,
        d=2,
        n_parallel=1,
        graph_prior="erdos_renyi",
        graph_args={},
        noise_type="gauss",
        noise_args={},
        intervention_type="shift",
        coeff_mean=0.0,
    ):
        super().__init__()
        noise_types = {
            "gauss": dist.Normal,
            "laplace": dist.Laplace,
            "gumbel": dist.Gumbel,
        }
        graph_priors = {
            "erdos_renyi": ErdosRenyi,
            "scale_free": ScaleFree,
            "permute_generate": PermuteGenerate,
        }
        self.graph_prior_init = graph_priors[graph_prior]
        self.d = d
        self.var_dim = 2 * d
        self.n_parallel = n_parallel
        self.graph_args = graph_args
        self.coeff_mean = coeff_mean
        self.noise_args = noise_args
        self.noise_dist = noise_types[noise_type]
        self.noise_type = noise_type
        self.var_names = [
            "graph",
            "coefficients",
            "noise_scales",
            "bias",
        ]  # , "noise_type"]
        if intervention_type == "shift":
            self.intervention_class = LinearANSEMShiftIntervention
        else:
            self.intervention_class = LinearANSEM
        self.reset(n_parallel)
        self.sanity_check()

    def sample_prior(self, num_theta, n_parallel=None, zero_bias=False):
        if n_parallel:
            self.reset(n_parallel)
        full_graph = self.graph_prior(num_theta)
        coeff_var = full_graph.sum(-2, keepdim=True)
        coeff_var = torch.cat(full_graph.shape[-1] * [coeff_var], -2)
        coeff_var[coeff_var == 0] = 1.0
        coeff_std = 1.0 / torch.sqrt(coeff_var)
        coeff_mean = torch.zeros_like(coeff_std) + self.coeff_mean
        # Randomly make some coefficients negative
        neg_indices = torch.rand(coeff_mean.shape) > 0.5
        coeff_mean[neg_indices] = -coeff_mean[neg_indices]
        coefficients = dist.Normal(loc=coeff_mean, scale=coeff_std).sample()
        noise_scales = torch.sqrt(
            InverseGamma(
                concentration=self.noise_scale_concentration, rate=self.noise_scale_rate
            ).sample((num_theta,))
        )
        bias_flag = torch.zeros(noise_scales.shape[:-1])
        if zero_bias is None:
            bias_flag = torch.randint(2, size=noise_scales.shape[:-1])
        if not zero_bias:
            bias_flag = torch.ones(noise_scales.shape[:-1])
        bias_flag = bias_flag.to(torch.bool)
        bias_uniform = dist.Uniform(low=-1.0, high=1.0).sample(noise_scales.shape)
        bias = torch.zeros_like(noise_scales)
        bias[bias_flag] = bias_uniform[bias_flag]
        return {
            "graph": full_graph,
            "coefficients": coefficients,
            "noise_scales": noise_scales,
            "bias": bias,
            # "noise_type": noise_type,
        }

    def rsample(self, design, theta, n_samples=1):
        graph = theta["graph"]
        coefficients = theta["coefficients"]
        noise_scales = theta["noise_scales"]
        bias = theta["bias"]
        return self.intervention_class(
            graph,
            coefficients,
            self.noise_dist(loc=bias, scale=noise_scales),
            (design[..., : self.d] > 0).to(design.dtype),
            design[..., self.d :],
        ).rsample((n_samples,))

    def log_prob(self, y, design, theta):
        graph = theta["graph"]
        coefficients = theta["coefficients"]
        noise_scales = theta["noise_scales"]
        bias = theta["bias"]
        return self.intervention_class(
            graph,
            coefficients,
            self.noise_dist(loc=bias, scale=noise_scales),
            (design[..., : self.d] > 0).to(design.dtype),
            design[..., self.d :],
        ).log_prob(y)

    def reset(self, n_parallel):
        self.n_parallel = n_parallel
        self.graph_prior = self.graph_prior_init(
            **{**self.graph_args, "n_parallel": n_parallel, "num_nodes": self.d}
        )
        self.noise_scale_concentration = 10.0 * torch.ones(n_parallel, 1, self.d)
        self.noise_scale_rate = torch.ones(n_parallel, 1, self.d)


class LinGaussANMModelHeteroskedastic(LinGaussANMModel):
    """
    Linear Additive Noise Structural Equation Model with Heteroskedastic Noise
    """

    def __init__(
        self,
        d=2,
        n_parallel=1,
        graph_prior="erdos_renyi",
        graph_args={},
        noise_type="gauss",
        noise_args={},
        intervention_type="shift",
        coeff_mean=0.0,
    ):
        super().__init__(
            d=d,
            n_parallel=n_parallel,
            graph_prior=graph_prior,
            graph_args=graph_args,
            noise_type=noise_type,
            noise_args=noise_args,
            intervention_type=intervention_type,
            coeff_mean=coeff_mean,
        )
        self.var_names = [
            "graph",
            "coefficients",
            "omegas",
            "b",
            "w",
            "bias",
        ]  # , "noise_type"]
        self.num_rff = 100
        if intervention_type == "shift":
            self.intervention_class = LinearANSEMShiftInterventionHk
        else:
            self.intervention_class = LinearANSEMHeteroskedastic
        self.reset(n_parallel)
        self.sanity_check()

    def sample_prior(self, num_theta, n_parallel=None, zero_bias=False):
        if n_parallel:
            self.reset(n_parallel)
        full_graph = self.graph_prior(num_theta)
        coeff_var = full_graph.sum(-2, keepdim=True)
        coeff_var = torch.cat(full_graph.shape[-1] * [coeff_var], -2)
        coeff_var[coeff_var == 0] = 1.0
        coeff_std = 1.0 / torch.sqrt(coeff_var)
        coeff_mean = torch.zeros_like(coeff_std) + self.coeff_mean
        # Randomly make some coefficients negative
        neg_indices = torch.rand(coeff_mean.shape) > 0.5
        coeff_mean[neg_indices] = -coeff_mean[neg_indices]
        coefficients = dist.Normal(loc=coeff_mean, scale=coeff_std).sample()
        omega = dist.Normal(loc=0, scale=0.1).sample(
            (num_theta, self.d, self.num_rff, self.n_parallel, 1, self.d)
        )
        w = dist.Normal(loc=0, scale=1.0).sample(
            (num_theta, self.n_parallel, 1, self.d, self.num_rff)
        )
        b = dist.Uniform(low=0.0, high=2 * np.pi).sample(
            (num_theta, self.n_parallel, 1, self.d, self.num_rff)
        )
        bias_shape = (num_theta, self.n_parallel, 1, self.d)
        bias_flag = torch.zeros(bias_shape[:-1])
        if zero_bias is None:
            bias_flag = torch.randint(2, size=bias_shape[:-1])
        if not zero_bias:
            bias_flag = torch.ones(bias_shape[:-1])
        bias_flag = bias_flag.to(torch.bool)
        bias_uniform = dist.Uniform(low=-1.0, high=1.0).sample(bias_shape)
        bias = torch.zeros(bias_shape)
        bias[bias_flag] = bias_uniform[bias_flag]
        return {
            "graph": full_graph,
            "coefficients": coefficients,
            "bias": bias,
            "omegas": omega,
            "w": w,
            "b": b,
        }

    def rsample(self, design, theta, n_samples=1):
        graph = theta["graph"]
        coefficients = theta["coefficients"]
        bias = theta["bias"]
        omega = theta["omegas"]
        w = theta["w"]
        b = theta["b"]
        return self.intervention_class(
            graph,
            coefficients,
            self.noise_dist(loc=bias, scale=1.0),
            omega,
            w,
            b,
            (design[..., : self.d] > 0).to(design.dtype),
            design[..., self.d :],
        ).rsample((n_samples,))

    def log_prob(self, y, design, theta):
        pass

    def reset(self, n_parallel):
        self.n_parallel = n_parallel
        self.graph_prior = self.graph_prior_init(
            **{**self.graph_args, "n_parallel": n_parallel, "num_nodes": self.d}
        )


class LinearANSEM(object):
    """
    Linear Additive Noise Structural Equation Model
    Args:
        graph (torch.Tensor): graph adjacency matrix
        lin_coefficients (torch.Tensor): linear coefficients of the Additive Noise Structural Equation Model
        exogenous_noise_dist (torch.Distribution): distribution of the exogenous noise (gauss, laplace, gumbel)
        intervention_mask (torch.Tensor): mask of the nodes to intervene on (1 for intervened, 0 for not intervened)
        intervention_values (torch.Tensor): values of the intervention
    """

    has_rsample = True

    def __init__(
        self,
        graph,
        lin_coefficients,
        exogenous_noise_dist,
        intervention_mask,
        intervention_values,
    ):
        self.exogenous_noise_dist = exogenous_noise_dist
        self.graph = graph
        self.lin_coefficients = lin_coefficients
        self.num_nodes = self.graph.shape[-1]
        self.intervention_mask = intervention_mask
        self.intervention_values = intervention_values

    def predict(self, y):
        return torch.matmul(
            y.unsqueeze(-2), self.graph * self.lin_coefficients
        ).squeeze(-2)

    def rsample(self, sample_shape=torch.Size([])):
        z = self.exogenous_noise_dist.rsample(sample_shape).squeeze(-2).transpose(0, 1)
        sample = torch.ones_like(z) * self.intervention_mask * self.intervention_values
        for i in range(self.num_nodes):
            sample = self.predict(sample) + z
            sample = (
                self.intervention_mask * self.intervention_values
                + (1 - self.intervention_mask) * sample
            )
        return sample

    def log_prob(self, y):
        predict = self.intervention_mask * self.intervention_values + (
            1 - self.intervention_mask
        ) * self.predict(y)
        z = y - predict
        log_prob = self.exogenous_noise_dist.log_prob(z)
        # Do not add log prob terms for nodes which received a do-intervention
        # import pdb; pdb.set_trace()
        return (log_prob * (1 - self.intervention_mask)).sum(-1)


class LinearANSEMHeteroskedastic(object):
    """
    Linear Additive Noise Structural Equation Model with Heteroskedastic Noise
    Args:
        graph (torch.Tensor): graph adjacency matrix
        lin_coefficients (torch.Tensor): linear coefficients of the Additive Noise Structural Equation Model
        exogenous_noise_dist (torch.Distribution): distribution of the exogenous noise (gauss, laplace, gumbel)
        omega (torch.Tensor): omega parameters for the heteroskedastic noise
        w (torch.Tensor): w parameters for the heteroskedastic noise
        b (torch.Tensor): b parameters for the heteroskedastic noise
        intervention_mask (torch.Tensor): mask of the nodes to intervene on (1 for intervened, 0 for not intervened)
        intervention_values (torch.Tensor): values of the intervention
    """

    has_rsample = True

    def __init__(
        self,
        graph,
        lin_coefficients,
        exogenous_noise_dist,
        omega,
        w,
        b,
        intervention_mask,
        intervention_values,
    ):
        self.graph = graph
        self.lin_coefficients = lin_coefficients
        self.num_nodes = self.graph.shape[-1]
        self.exogenous_noise_dist = exogenous_noise_dist
        self.omega = omega
        self.w = w
        self.b = b
        if len(self.omega.shape) == 5:
            self.omega = self.omega.unsqueeze(0)
            self.w = self.w.unsqueeze(0)
            self.b = self.b.unsqueeze(0)
        self.intervention_mask = intervention_mask
        self.intervention_values = intervention_values
        self.num_rff = self.w.shape[-1]

    def predict(self, y):
        return torch.matmul(
            y.unsqueeze(-2), self.graph * self.lin_coefficients
        ).squeeze(-2)

    def rsample(self, sample_shape=torch.Size([])):
        z = self.exogenous_noise_dist.rsample(sample_shape).squeeze(-2).transpose(0, 1)
        sample = torch.ones_like(z) * self.intervention_mask * self.intervention_values
        for i in range(self.num_nodes):
            f = compute_rff(
                omega=self.omega, y=sample, graph=self.graph, b=self.b, w=self.w, c=2.0
            )
            f = torch.sqrt(torch.log(1 + torch.exp(f)))
            sample = self.predict(sample) + z * f
            sample = (
                self.intervention_mask * self.intervention_values
                + (1 - self.intervention_mask) * sample
            )
        return sample

    def log_prob(self, y):
        pass


class LinearANSEMShiftIntervention(LinearANSEM):
    """
    Linear Additive Noise Structural Equation Model with Shift Intervention
    A shift intervention on any node j modifies the conditional distribution from p(y_j | pa_j) to p(y_j + c | pa_j)
    Args:
        graph (torch.Tensor): graph adjacency matrix
        lin_coefficients (torch.Tensor): linear coefficients of the Additive Noise Structural Equation Model
        exogenous_noise_dist (torch.Distribution): distribution of the exogenous noise (gauss, laplace, gumbel)
        intervention_mask (torch.Tensor): mask of the nodes to intervene on (1 for intervened, 0 for not intervened)
        intervention_values (torch.Tensor): values of the intervention
    """

    def __init__(
        self,
        graph,
        lin_coefficients,
        exogenous_noise_dist,
        intervention_mask,
        intervention_values,
    ):
        super().__init__(
            graph,
            lin_coefficients,
            exogenous_noise_dist,
            intervention_mask,
            intervention_values,
        )

    def rsample(self, sample_shape=torch.Size([])):
        # change the mean of the noise distribution
        z = self.exogenous_noise_dist.rsample(sample_shape).squeeze(-2).transpose(0, 1)
        z = z + self.intervention_mask * self.intervention_values
        sample = torch.ones_like(z)
        for i in range(self.num_nodes):
            sample = self.predict(sample) + z
        return sample

    def log_prob(self, y):
        # change the mean of the noise distribution
        predict = self.predict(y)
        z = y - predict
        z = z - self.intervention_mask * self.intervention_values
        log_prob = self.exogenous_noise_dist.log_prob(z)
        return log_prob.sum(-1)


class LinearANSEMShiftInterventionHk(LinearANSEMHeteroskedastic):
    """
    Linear Additive Noise Structural Equation Model with Shift Intervention and Heteroskedastic Noise
    A shift intervention on any node j modifies the conditional distribution from p(y_j | pa_j) to p(y_j + c | pa_j)
    Args:
        graph (torch.Tensor): graph adjacency matrix
        lin_coefficients (torch.Tensor): linear coefficients of the Additive Noise Structural Equation Model
        exogenous_noise_dist (torch.Distribution): distribution of the exogenous noise (gauss, laplace, gumbel)
        omega (torch.Tensor): omega parameters for the heteroskedastic noise
        w (torch.Tensor): w parameters for the heteroskedastic noise
        b (torch.Tensor): b parameters for the heteroskedastic noise
        intervention_mask (torch.Tensor): mask of the nodes to intervene on (1 for intervened, 0 for not intervened)
        intervention_values (torch.Tensor): values of the intervention
    """

    def __init__(
        self,
        graph,
        lin_coefficients,
        exogenous_noise_dist,
        omega,
        w,
        b,
        intervention_mask,
        intervention_values,
    ):
        super().__init__(
            graph,
            lin_coefficients,
            exogenous_noise_dist,
            omega,
            w,
            b,
            intervention_mask,
            intervention_values,
        )

    def rsample(self, sample_shape=torch.Size([])):
        # change the mean of the noise distribution
        z = self.exogenous_noise_dist.rsample(sample_shape).squeeze(-2).transpose(0, 1)
        z = z + self.intervention_mask * self.intervention_values
        sample = torch.ones_like(z)
        for i in range(self.num_nodes):
            f = compute_rff(
                omega=self.omega, y=sample, graph=self.graph, b=self.b, w=self.w, c=2.0
            )
            f = torch.sqrt(torch.log(1 + torch.exp(f)))
            sample = self.predict(sample) + z * f
        return sample

    def log_prob(self, y):
        pass
