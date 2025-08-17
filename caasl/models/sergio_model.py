import math

import igraph as ig
import numpy as np
import torch
import torch.distributions as td
import yaml

from .causal_experiment_model import CausalExperimentModel
from .graph_priors import ErdosRenyi, PermuteGenerate, ScaleFree


def calc_hill(reg_conc, half_response, coop_state, repressive):
    """
    Calculate the Hill function for the given parameters.
    """
    coop_state = coop_state.unsqueeze(-1)
    temp = torch.div(
        torch.pow(reg_conc, coop_state),
        (
            torch.pow(half_response.unsqueeze(-1), coop_state)
            + torch.pow(reg_conc, coop_state)
        ),
    )
    temp = torch.where(reg_conc == 0, 0, temp)
    result = torch.where(repressive.unsqueeze(-1), 1 - temp, temp)
    return result


def sim_sergio_pytorch(
    *,
    graph,
    toporder,
    number_bins,
    number_sc,
    noise_params,
    decays,
    basal_rates,
    k,
    hill,
    targets=None,
    interv_type="kout",
    sampling_state=15,
    dt=0.01,
    safety_steps=1,
):
    """
    SERGIO simulator for GRNs, vectorized in pytorch.

    Args:
        graph (torch.Tensor): graph adjacency matrix
        toporder (torch.Tensor): topological order of the graph
        number_bins (int): number of bins
        number_sc (int): number of single cells
        noise_params (torch.Tensor): noise parameters
        decays (torch.Tensor): decay parameters
        basal_rates (torch.Tensor): basal rates
        k (torch.Tensor): interaction parameters
        hill (torch.Tensor): Hill function parameters
        targets (torch.Tensor): target genes
        interv_type (str): intervention type
        sampling_state (int): sampling state
        dt (float): time step
        safety_steps (int): safety steps for simulation

    Returns:
        sc_expression (torch.Tensor): expression levels of the single cells
    """
    n_parallel = graph.shape[0]
    graph = graph.bool()
    n_p_range = torch.arange(n_parallel)
    d = graph.shape[-1]
    mean_expr = -1 * torch.ones(n_parallel, d, number_bins)
    conc = torch.zeros(
        *mean_expr.shape, sampling_state * number_sc + d * safety_steps + 1
    )
    curr_conc_counter = torch.zeros(n_parallel, d).to(torch.int32)
    sc_expression = torch.zeros(n_parallel, d, number_bins, number_sc)

    assert toporder.shape[-1] == n_parallel
    for i, gene in enumerate(toporder):
        # gene is of size n_parallel
        is_mr = ~graph[n_p_range, :, gene].sum(-1).bool()
        n_req_steps = sampling_state * number_sc + (d - i) * safety_steps

        half_response = torch.zeros(*graph.shape[:-1])
        half_response[graph[n_p_range, :, gene]] = mean_expr[
            graph[n_p_range, :, gene]
        ].mean(-1)
        if interv_type == "kout":
            interv_factor = (~targets[n_p_range, gene]).float()
        else:
            interv_factor = torch.where(targets[n_p_range, gene] == 1.0, 0.5, 1.0)
        rate_ = torch.zeros(n_parallel, number_bins)
        rate_[is_mr] = basal_rates[is_mr, gene[is_mr]]
        mean_exp_pa = mean_expr * graph[n_p_range, :, gene].unsqueeze(-1)
        hill_unmasked = calc_hill(
            mean_exp_pa,
            half_response,
            hill[n_p_range, :, gene],
            k[n_p_range, :, gene] < 0,
        )
        rate_ += (
            hill_unmasked * ((graph * torch.abs(k))[n_p_range, :, gene]).unsqueeze(-1)
        ).sum(-2)
        append_conc = torch.div(
            interv_factor.unsqueeze(-1) * rate_, decays[n_p_range, gene][:, None]
        )
        append_conc = torch.where(append_conc < 0, 0.0, append_conc)
        conc[n_p_range, gene, :, curr_conc_counter[n_p_range, gene]] = append_conc
        curr_conc_counter[n_p_range, gene] += 1
        for _ in range(n_req_steps):
            curr_exp = conc[n_p_range, gene, :, curr_conc_counter[n_p_range, gene] - 1]
            # Calculate Production Rate
            rate_ = torch.zeros(n_parallel, number_bins)
            rate_[is_mr] = basal_rates[is_mr, gene[is_mr]]
            conc_parent = conc[
                n_p_range, ..., curr_conc_counter[n_p_range, gene] - 1
            ] * graph[n_p_range, :, gene].unsqueeze(-1)
            hill_unmasked = calc_hill(
                conc_parent,
                half_response,
                hill[n_p_range, :, gene],
                k[n_p_range, :, gene] < 0,
            )
            rate_ += (
                hill_unmasked
                * ((graph * torch.abs(k))[n_p_range, :, gene]).unsqueeze(-1)
            ).sum(-2)
            prod_rate = interv_factor.unsqueeze(-1) * rate_
            decay_ = decays[n_p_range, gene].unsqueeze(-1) * curr_exp
            dw_p = torch.randn_like(curr_exp)
            dw_d = torch.randn_like(curr_exp)
            amplitude_p = noise_params[n_p_range, gene].unsqueeze(-1) * torch.pow(
                prod_rate, 0.5
            )
            amplitude_d = noise_params[n_p_range, gene].unsqueeze(-1) * torch.pow(
                decay_, 0.5
            )
            noise = (amplitude_p * dw_p) + (amplitude_d * dw_d)
            dxdt = (dt * (prod_rate - decay_)) + (np.power(dt, 0.5) * noise)
            append_conc = (
                conc[n_p_range, gene, :, curr_conc_counter[n_p_range, gene] - 1] + dxdt
            )
            append_conc = torch.where(append_conc < 0, 0.0, append_conc)
            conc[n_p_range, gene, :, curr_conc_counter[n_p_range, gene]] = append_conc
            curr_conc_counter[n_p_range, gene] += 1
        select_steps = torch.randint(
            low=-sampling_state * number_sc,
            high=0,
            size=(
                n_parallel,
                number_bins,
                number_sc,
            ),
        )
        select_steps = curr_conc_counter[n_p_range, gene, None, None] + select_steps
        sampled_expr = torch.gather(conc[n_p_range, gene], -1, select_steps)
        mean_expr[n_p_range, gene] = sampled_expr.mean(-1)
        sc_expression[n_p_range, gene] = sampled_expr
    return sc_expression


def outlier_effect(scData, outlier_prob, mean, scale):
    """
    Calculate the outlier effect for the given parameters.
    """
    sc_shape = scData.shape
    d = sc_shape[-3]
    out_indicator = (td.binomial.Binomial(probs=outlier_prob).sample((d,)) == 1).t()
    out_factors = td.log_normal.LogNormal(mean, scale).sample((d,)).t()
    scData = scData.view(*sc_shape[:2], -1)
    scData[out_indicator] *= out_factors[out_indicator].unsqueeze(-1)
    return scData.view(*sc_shape)


def lib_size_effect(scData, mean, scale):
    """
    Calculate the library size effect for the given parameters.
    """
    sc_shape = scData.shape
    out_factors = (
        td.log_normal.LogNormal(mean, scale).sample(sc_shape[-2:]).permute(2, 0, 1)
    )  # n_parallel x n_bins x n_sc
    sum_factors = scData.sum(-3)
    of_normalized = out_factors / torch.where(sum_factors == 0, 1.0, sum_factors)
    ret = of_normalized.unsqueeze(-3) * scData
    return ret


def dropout_indicator(scData, shape=1, percentile=65):
    """
    Calculate the dropout indicator for the given parameters.
    """
    sc_log_data = torch.log1p(scData)
    log_mid_point = torch.quantile(
        sc_log_data.view(scData.shape[0], -1), percentile[0] / 100, dim=1
    )
    prob_ber = torch.div(
        1,
        1
        + torch.exp(
            -1
            * shape[:, None, None, None]
            * (sc_log_data - log_mid_point[:, None, None, None])
        ),
    )
    binary_ind = td.binomial.Binomial(probs=prob_ber).sample()
    return binary_ind


class GRNSergioModel(CausalExperimentModel):
    """
    SERGIO model for GRNs.

    Args:
        d (int): number of genes
        n_parallel (int): number of parallel graphs to sample
        graph_prior (str): graph prior
        graph_args (dict): graph arguments for sampling the graph
        intervention_type (str): intervention type
        cell_types (int): number of cell types
        b (torch.Distribution): distribution for sampling basic reproduction rates
        k_param (torch.Distribution): distribution for sampling interaction strengths
        k_sign_p (torch.Distribution): distribution for sampling the sign of interaction strengths
        hill (float): Hill function coefficient
        decays (float): decay rate
        noise_params (float): noise scale parameter
        add_outlier_effect (bool): whether to add outlier effect
        add_lib_size_effect (bool): whether to add library size effect
        add_dropout_effect (bool): whether to add dropout effect
        return_count_data (bool): whether to return count data
        noise_config_type (str): type of noise configuration
        noise_config_file (str): path to noise configuration file
    """

    def __init__(
        self,
        d=2,
        n_parallel=1,
        graph_prior="erdos_renyi",
        graph_args={},
        intervention_type="kout",
        cell_types=5,
        b=td.uniform.Uniform(1.0, 3.0),
        k_param=td.uniform.Uniform(1.0, 5.0),
        k_sign_p=td.beta.Beta(0.5, 0.5),
        hill=1.0,
        decays=0.8,
        noise_params=1.0,
        add_outlier_effect=True,
        add_lib_size_effect=True,
        add_dropout_effect=True,
        return_count_data=True,
        noise_config_type="10x-chromium-mini",
        noise_config_file="caasl/models/noise_config.yaml",
    ):
        super().__init__()
        graph_priors = {
            "erdos_renyi": ErdosRenyi,
            "scale_free": ScaleFree,
            "permute_generate": PermuteGenerate,
        }
        self.graph_prior_init = graph_priors[graph_prior]
        self.d = d
        self.var_dim = d
        self.n_parallel = n_parallel
        self.graph_args = graph_args
        self.var_names = [
            "graph",
            "k",
            "basal_rates",
            "hill",
            "decays",
            "noise_params",
            "outlier_prob",
            "outlier_mean",
            "outlier_scale",
            "lib_size_mean",
            "lib_size_scale",
            "dropout_shape",
            "dropout_percentile",
        ]
        self.intervention_type = intervention_type
        self.cell_types = cell_types
        self.b = b
        self.k_param = k_param
        self.k_sign_p = k_sign_p
        self.hill = hill
        self.decays = decays
        self.noise_params = noise_params
        self.add_outlier_effect = add_outlier_effect
        self.add_lib_size_effect = add_lib_size_effect
        self.add_dropout_effect = add_dropout_effect
        self.return_count_data = return_count_data
        self.noise_config_type = noise_config_type
        self.noise_config_file = noise_config_file
        self.reset(n_parallel)
        self.sanity_check()

    def sample_prior(self, num_theta, n_parallel=None, zero_bias=False):
        if n_parallel:
            self.reset(n_parallel)
        full_graph = self.graph_prior(num_theta).squeeze((-3))
        k = torch.abs(self.k_param.sample((num_theta, self.n_parallel, self.d, self.d)))
        effect_sgn = (
            td.binomial.Binomial(
                1,
                self.k_sign_p.sample((num_theta, self.n_parallel, self.d, 1)),
            ).sample()
            * 2.0
            - 1.0
        )

        k = k * effect_sgn.to(torch.float32)
        basal_rates = self.b.sample(
            (num_theta, self.n_parallel, self.d, self.cell_types)
        )  # assuming 1 cell type is simulated
        # Load noise config
        with open(self.noise_config_file, "r") as file:
            config = yaml.safe_load(file)
        assert self.noise_config_type in config.keys(), (
            f"tech_noise_config `{self.noise_config_type}` "
            f"not in config keys: `{list(config.keys())}`"
        )
        outlier_prob_ = torch.tensor(config[self.noise_config_type]["outlier_prob"])
        #  Randomly choose outlier probability from the list with replacement
        outlier_prob = outlier_prob_[
            torch.randint(
                0, len(outlier_prob_), (num_theta, self.n_parallel), dtype=torch.int64
            )
        ]

        outlier_mean_ = torch.tensor(config[self.noise_config_type]["outlier_mean"])
        outlier_mean = outlier_mean_[
            torch.randint(
                0, len(outlier_mean_), (num_theta, self.n_parallel), dtype=torch.int64
            )
        ]
        outlier_scale_ = torch.tensor(config[self.noise_config_type]["outlier_scale"])
        outlier_scale = outlier_scale_[
            torch.randint(
                0, len(outlier_scale_), (num_theta, self.n_parallel), dtype=torch.int64
            )
        ]
        lib_size_mean_ = torch.tensor(config[self.noise_config_type]["lib_size_mean"])
        lib_size_mean = lib_size_mean_[
            torch.randint(
                0, len(lib_size_mean_), (num_theta, self.n_parallel), dtype=torch.int64
            )
        ]
        lib_size_scale_ = torch.tensor(config[self.noise_config_type]["lib_size_scale"])
        lib_size_scale = lib_size_scale_[
            torch.randint(
                0, len(lib_size_scale_), (num_theta, self.n_parallel), dtype=torch.int64
            )
        ]
        dropout_shape_ = torch.tensor(config[self.noise_config_type]["dropout_shape"])
        dropout_shape = dropout_shape_[
            torch.randint(
                0, len(dropout_shape_), (num_theta, self.n_parallel), dtype=torch.int64
            )
        ]
        dropout_percentile_ = torch.tensor(
            config[self.noise_config_type]["dropout_percentile"]
        ).to(torch.int32)
        dropout_percentile = dropout_percentile_[
            torch.randint(
                0,
                len(dropout_percentile_),
                (num_theta, self.n_parallel),
                dtype=torch.int64,
            )
        ]
        return {
            "graph": full_graph,
            "k": k,
            "basal_rates": basal_rates,
            "hill": self.hill
            * torch.ones((num_theta, self.n_parallel, self.d, self.d)),
            "decays": self.decays * torch.ones((num_theta, self.n_parallel, self.d)),
            "noise_params": self.noise_params
            * torch.ones((num_theta, self.n_parallel, self.d)),
            "outlier_prob": outlier_prob,
            "outlier_mean": outlier_mean,
            "outlier_scale": outlier_scale,
            "lib_size_mean": lib_size_mean,
            "lib_size_scale": lib_size_scale,
            "dropout_shape": dropout_shape,
            "dropout_percentile": dropout_percentile,
        }

    def rsample(self, design, theta, n_samples=1):
        graph = theta["graph"]
        k = theta["k"]
        basal_rates = theta["basal_rates"]
        hill = theta["hill"]
        decays = theta["decays"]
        noise_params = theta["noise_params"]
        outlier_prob = theta["outlier_prob"]
        outlier_mean = theta["outlier_mean"]
        outlier_scale = theta["outlier_scale"]
        lib_size_mean = theta["lib_size_mean"]
        lib_size_scale = theta["lib_size_scale"]
        dropout_shape = theta["dropout_shape"]
        dropout_percentile = theta["dropout_percentile"]
        return SergioGene(
            graph,
            k,
            hill,
            decays,
            noise_params,
            self.cell_types,
            basal_rates,
            design,
            add_outlier_effect=self.add_outlier_effect,
            add_lib_size_effect=self.add_lib_size_effect,
            add_dropout_effect=self.add_dropout_effect,
            outlier_prob=outlier_prob,
            outlier_mean=outlier_mean,
            outlier_scale=outlier_scale,
            lib_size_mean=lib_size_mean,
            lib_size_scale=lib_size_scale,
            dropout_shape=dropout_shape,
            dropout_percentile=dropout_percentile,
            interv_type=self.intervention_type,
        ).rsample((n_samples,))

    def log_prob(self, y, design, theta):
        pass

    def reset(self, n_parallel):
        self.n_parallel = n_parallel
        self.graph_prior = self.graph_prior_init(
            **{**self.graph_args, "n_parallel": self.n_parallel, "num_nodes": self.d}
        )


class GRNSergioModelNoisyIntervention(GRNSergioModel):
    """
    SERGIO model for GRNs with noisy intervention.

    Args:
        d (int): number of genes
        n_parallel (int): number of parallel graphs to sample
        graph_prior (str): graph prior
        graph_args (dict): graph arguments for sampling the graph
        intervention_type (str): intervention type
        cell_types (int): number of cell types
        b (torch.Distribution): distribution for sampling basic reproduction rates
        k_param (torch.Distribution): distribution for sampling interaction strengths
        k_sign_p (torch.Distribution): distribution for sampling the sign of interaction strengths
        hill (float): Hill function coefficient
        decays (float): decay rate
        noise_params (float): noise scale parameter
        add_outlier_effect (bool): whether to add outlier effect
        add_lib_size_effect (bool): whether to add library size effect
        add_dropout_effect (bool): whether to add dropout effect
        return_count_data (bool): whether to return count data
        noise_config_type (str): type of noise configuration
        noise_config_file (str): path to noise configuration file
        intervention_noise (float): noise level for the intervention
    """

    def __init__(
        self,
        d=2,
        n_parallel=1,
        graph_prior="erdos_renyi",
        graph_args={},
        intervention_type="kout",
        cell_types=5,
        b=td.uniform.Uniform(1.0, 3.0),
        k_param=td.uniform.Uniform(1.0, 5.0),
        k_sign_p=td.beta.Beta(0.5, 0.5),
        hill=1.0,
        decays=0.8,
        noise_params=1.0,
        add_outlier_effect=True,
        add_lib_size_effect=True,
        add_dropout_effect=True,
        return_count_data=True,
        noise_config_type="10x-chromium-mini",
        noise_config_file="caasl/models/noise_config.yaml",
        intervention_noise=0.1,
    ):
        super().__init__(
            d=d,
            n_parallel=n_parallel,
            graph_prior=graph_prior,
            graph_args=graph_args,
            intervention_type=intervention_type,
            cell_types=cell_types,
            b=b,
            k_param=k_param,
            k_sign_p=k_sign_p,
            hill=hill,
            decays=decays,
            noise_params=noise_params,
            add_outlier_effect=add_outlier_effect,
            add_lib_size_effect=add_lib_size_effect,
            add_dropout_effect=add_dropout_effect,
            return_count_data=return_count_data,
            noise_config_type=noise_config_type,
            noise_config_file=noise_config_file,
        )
        self.intervention_noise = intervention_noise

    def rsample(self, design, theta, n_samples=1):
        graph = theta["graph"]
        k = theta["k"]
        basal_rates = theta["basal_rates"]
        hill = theta["hill"]
        decays = theta["decays"]
        noise_params = theta["noise_params"]
        outlier_prob = theta["outlier_prob"]
        outlier_mean = theta["outlier_mean"]
        outlier_scale = theta["outlier_scale"]
        lib_size_mean = theta["lib_size_mean"]
        lib_size_scale = theta["lib_size_scale"]
        dropout_shape = theta["dropout_shape"]
        dropout_percentile = theta["dropout_percentile"]
        design = design.clone()
        flip_indices = (
            td.bernoulli.Bernoulli(self.intervention_noise).sample(design.shape).bool()
        )
        return SergioGene(
            graph,
            k,
            hill,
            decays,
            noise_params,
            self.cell_types,
            basal_rates,
            design,
            add_outlier_effect=self.add_outlier_effect,
            add_lib_size_effect=self.add_lib_size_effect,
            add_dropout_effect=self.add_dropout_effect,
            outlier_prob=outlier_prob,
            outlier_mean=outlier_mean,
            outlier_scale=outlier_scale,
            lib_size_mean=lib_size_mean,
            lib_size_scale=lib_size_scale,
            dropout_shape=dropout_shape,
            dropout_percentile=dropout_percentile,
            interv_type=self.intervention_type,
            flip_indices=flip_indices,
        ).rsample((n_samples,))


class SergioGene(object):
    """
    SERGIO simulator for GRNs, encalpsulated as a Probabilistic Model.

    Args:
        b (Distribution): distribution for sampling basic reproduction rates. Example: `avici.synthetic.Uniform`
        k_param (Distribution): distribution for sampling (non-negative) interaction strenghts.
            Example: `avici.synthetic.Uniform`
        k_sign_p (Distribution): distribution of sampling probability for positive (vs. negative)
            interaction sign signs. Example: `avici.synthetic.Beta`
        hill (float): Hill function coefficient
        interv_mask (torch.Tensor): genes to intervene on, given a sequential design (1 for intervened, 0 for not intervened)
        decays (float): decay rate
        noise_params (float): noise scale parameter
        cell_types (Distribution): distribution for sampling integer number of cell types.
            Example: `avici.synthetic.RandInt`
        noise_type (str): noise type in SERGIO simulator. Default: `dpd`
        sampling_state (int): configuration of SERGIO sampler. Default: 15
        dt (float): dt increment in stochastic process. Default: 0.01


        * Technical noise*

        tech_noise_config (str): specification of noise elvels.
            Select one of the keys in `avici/synthetic/sergio/noise_config.yaml`
        add_outlier_effect (bool): whether to simulate outlier effects based on `tech_noise_config`
        add_lib_size_effect (bool): whether to simulate library size effects based on `tech_noise_config`
        add_dropout_effect (bool): whether to simulate dropout effects based on `tech_noise_config`
        return_count_data (bool): whether to return Poisson count data of the float mean expression levels

        * Interventions *

        n_ko_genes (int): no. unique genes knocked out in all of data collected; -1 indicates all genes
        flip_indices (torch.Tensor): indices of the genes to flip in case of a noisy intervention

        * Technical noise*

        outlier_prob (float): probability of the outlier effect
        outlier_mean (float): mean of the outlier effect
        outlier_scale (float): scale of the outlier effect
        lib_size_mean (float): mean of the library size effect
        lib_size_scale (float): scale of the library size effect
        dropout_shape (float): shape of the dropout effect
    """

    has_rsample = False

    def __init__(
        self,
        graph,
        k,
        hill,
        decays,
        noise_params,
        n_cell_types,
        basal_rates,
        interv_mask,
        noise_type="dpd",
        sampling_state=15,
        dt=0.01,
        tech_noise_config=None,
        add_outlier_effect=True,
        add_lib_size_effect=True,
        add_dropout_effect=True,
        return_count_data=True,
        outlier_prob=0.01,
        outlier_mean=3.0,
        outlier_scale=1.0,
        lib_size_mean=6.0,
        lib_size_scale=0.3,
        dropout_shape=8,
        dropout_percentile=45,
        interv_type="kout",
        flip_indices=None,
    ):

        self.graph = graph
        self.k = k
        self.hill = hill
        self.decays = decays
        self.noise_params = noise_params
        self.n_cell_types = n_cell_types
        self.basal_rates = basal_rates
        self.noise_type = noise_type
        self.sampling_state = sampling_state
        self.dt = dt
        self.noise_config_type = tech_noise_config
        self.add_outlier_effect = add_outlier_effect
        self.add_lib_size_effect = add_lib_size_effect
        self.add_dropout_effect = add_dropout_effect
        self.return_count_data = return_count_data
        interv_mask = interv_mask[..., 0, :]
        self.interv_mask = (interv_mask > 0.5).cpu().squeeze()
        if self.interv_mask.sum() > 0:
            if flip_indices is not None:
                flip_indices = flip_indices[..., 0, :]
                self.interv_mask[flip_indices] = ~self.interv_mask[flip_indices]
        self.outlier_prob = outlier_prob
        self.outlier_mean = outlier_mean
        self.outlier_scale = outlier_scale
        self.lib_size_mean = lib_size_mean
        self.lib_size_scale = lib_size_scale
        self.dropout_shape = dropout_shape
        self.dropout_percentile = dropout_percentile
        self.interv_type = interv_type

    def rsample(self, sample_shape=torch.Size([])):
        # sample interaction terms K
        toporder = []
        for i in self.graph:
            g = ig.Graph.Adjacency(i.numpy().tolist())
            toporder.append(torch.tensor(g.topological_sorting(mode="out")))
        toporder = torch.stack(toporder, dim=1)

        if self.interv_mask.sum() == 0:
            number_sc = math.ceil(sample_shape[0] / self.n_cell_types)

        else:
            number_sc = 1
        # setup simulator
        expr = sim_sergio_pytorch(
            graph=self.graph,
            toporder=toporder,
            number_bins=self.n_cell_types,
            number_sc=number_sc,
            noise_params=self.noise_params,
            decays=self.decays,
            basal_rates=self.basal_rates,
            k=self.k,
            hill=self.hill,
            targets=self.interv_mask,
            interv_type=self.interv_type,
            sampling_state=self.sampling_state,
            dt=self.dt,
            safety_steps=2,
        )

        if self.add_outlier_effect:
            expr = outlier_effect(
                expr, self.outlier_prob, self.outlier_mean, self.outlier_scale
            )

        # 2) library size
        if self.add_lib_size_effect:
            expr = lib_size_effect(expr, self.lib_size_mean, self.lib_size_scale)

        # 3) dropout
        if self.add_dropout_effect:
            binary_ind = dropout_indicator(
                expr, self.dropout_shape, self.dropout_percentile
            )
            expr *= binary_ind

        # 4) mRNA count data
        if self.return_count_data:
            expr = torch.poisson(expr)

        # expr_agg = np.concatenate(expr, axis=1)
        x = expr.reshape(*expr.shape[:2], -1)

        x = x[..., torch.randperm(x.size(-1))]
        return x[..., : sample_shape[0]].transpose(-1, -2)

    def log_prob(self, y):
        pass
