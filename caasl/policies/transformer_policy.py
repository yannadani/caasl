"""AdaptiveTanhGaussianPolicy."""

import akro
import numpy as np
import torch
from garage.torch import global_device
from garage.torch.distributions import TanhNormal
from garage.torch.policies.stochastic_policy import StochasticPolicy
from torch import nn

from ..modules import (
    AlternateAttention,
    GaussianMLPTwoHeadedModuleDoCausal,
)


class AdaptiveTransformerTanhGaussianPolicy(StochasticPolicy):
    """
    Implements a transformer policy for causal experiemntal design.

    It takes as input entire histories and maps them to a Normal
    distribution with a tanh transformation. Inputs to the network should be of
    the shape (batch_dim, history_length, obs_dim)

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        encoder_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for encoder. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        encoder_nonlinearity (callable): Activation function for intermediate
            dense layer(s) of encoder. It should return a torch.Tensor. Set it
            to None to maintain a linear activation.
        encoder_output_nonlinearity (callable): Activation function for encoder
            output dense layer. It should return a torch.Tensor. Set it to None
            to maintain a linear activation.
        encoding_dim (int): Output dimension of output dense layer for encoder.
        emitter_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for emitter.
        emitter_nonlinearity (callable): Activation function for intermediate
            dense layer(s) of emitter.
        emitter_output_nonlinearity (callable): Activation function for emitter
            output dense layer.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        layer_normalization (bool): Bool for using layer normalization or not.
        batch_size (int): Batch size of the experiments.
        device (str): Device to use.
        no_value (bool): Whether to predict the value of the intervention.

    """

    def __init__(
        self,
        env_spec,
        dropout=0.1,
        widening_factor=4,
        pooling="max",
        embedding_dim=32,
        n_attention_layers=8,
        n_attention_heads=8,
        emitter_sizes=(128, 128),
        emitter_nonlinearity=nn.ReLU,
        emitter_output_nonlinearity=None,
        hidden_w_init=nn.init.xavier_uniform_,
        hidden_b_init=nn.init.zeros_,
        output_w_init=nn.init.xavier_uniform_,
        output_b_init=nn.init.zeros_,
        init_mean=0.0,
        init_std=np.sqrt(1 / 3),
        min_std=np.exp(-20.0),
        max_std=np.exp(0.0),
        std_parameterization="exp",
        layer_normalization=False,
        batch_size=1,
        device="cpu",
        no_value=False,
    ):
        super().__init__(env_spec, name="AdaptiveTanhGaussianPolicy")

        self._pooling = pooling
        self.device = device
        self.batch_size = batch_size

        self.no_value = no_value
        self._encoder = AlternateAttention(
            dim=embedding_dim,
            feedforward_dim=widening_factor * embedding_dim,
            dropout=dropout,
            n_layers=n_attention_layers,
            num_heads=n_attention_heads,
        )
        self._encoder = nn.DataParallel(self._encoder)
        self._emitter = GaussianMLPTwoHeadedModuleDoCausal(
            input_dim=embedding_dim,
            output_dim=2 * self.batch_size if not no_value else self.batch_size,
            hidden_sizes=emitter_sizes,
            hidden_nonlinearity=emitter_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=emitter_output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            init_mean=init_mean,
            init_std=init_std,
            min_std=min_std,
            max_std=max_std,
            std_parameterization=std_parameterization,
            layer_normalization=layer_normalization,
            normal_distribution_cls=TanhNormal,
            batch_size=batch_size,
            no_value=no_value,
        )

    def get_actions(self, observations, mask=None):
        r"""Get actions given observations.

        Args:
            observations (np.ndarray): Observations from the environment.
                Shape is :math:`batch_dim \bullet env_spec.observation_space`.

        Returns:
            tuple:
                * np.ndarray: Predicted actions.
                    :math:`batch_dim \bullet env_spec.action_space`.
                * dict:
                    * np.ndarray[float]: Mean of the distribution.
                    * np.ndarray[float]: Standard deviation of logarithmic
                        values of the distribution.

        """
        if not isinstance(observations[0], np.ndarray) and not isinstance(
            observations[0], torch.Tensor
        ):
            observations = self._env_spec.observation_space.flatten_n(observations)

        # frequently users like to pass lists of torch tensors or lists of
        # numpy arrays. This handles those conversions.
        if isinstance(observations, list):
            if isinstance(observations[0], np.ndarray):
                observations = np.stack(observations)
            elif isinstance(observations[0], torch.Tensor):
                observations = torch.stack(observations)

        if isinstance(self._env_spec.observation_space, akro.Image) and len(
            observations.shape
        ) < len(self._env_spec.observation_space.shape):
            observations = self._env_spec.observation_space.unflatten_n(observations)
        with torch.no_grad():
            if not isinstance(observations, torch.Tensor):
                observations = torch.as_tensor(observations).float().to(global_device())
                if mask is not None:
                    mask = torch.as_tensor(mask).float().to(global_device())

            dist, info = self.forward(
                observations.to(self.device),
                mask.to(self.device) if mask is not None else None,
            )
            actions = dist.sample().detach().cpu()
        return actions, {k: v.detach().cpu() for (k, v) in info.items()}

    def forward(self, observations, mask=None):
        """Compute the action distributions from the observations.

        Args:
            observations (torch.Tensor): Batch of observations on default
                torch device.
            mask (torch.Tensor): a mask to account for 0-padded inputs

        Returns:
            torch.distributions.Distribution: Batch distribution of actions.
            dict[str, torch.Tensor]: Additional agent_info, as torch Tensors

        """
        encoding = self._encoder(observations)
        if self._pooling:
            if self._pooling == "max":
                if mask is not None:
                    min_value = torch.min(encoding).detach()
                    encoding[~mask.expand(*encoding.shape)] = min_value
                encoding = torch.max(encoding, -3).values
            elif self._pooling == "sum":
                if mask is not None:
                    encoding = encoding * mask
                encoding = torch.sum(encoding, -3)
            else:
                raise NotImplementedError(f"{self._pooling} not implemented")
        dist = self._emitter(encoding)
        ret_mean = dist.mean.clone()
        ret_log_std = (dist.variance.sqrt()).log().clone()
        info = dict(mean=ret_mean, log_std=ret_log_std)
        return dist, info
