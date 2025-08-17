"""This modules creates a continuous Q-function network."""

import torch
from garage.torch.modules import MLPModule
from torch import nn

from ..modules import AlternateAttention


class AdaptiveMLPQFunctionDoCausal(nn.Module):
    """
    Implements a continuous MLP Q-value network for causal experiemntal design.

    It predicts the Q-value for all actions based on the history of experiments.
    It uses a PyTorch neural network module to fit the function of Q(s, a).
    Inputs to the encoder should be of the shape
    (batch_dim, history_length, obs_dim)

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        encoding_dim (int): Output dimension of output dense layer for encoder.
        batch_size (int): Batch size of the experiments.
        encoder_widening_factor (int): Widening factor for the encoder.
        encoder_dropout (float): Dropout rate for the encoder.
        encoder_n_layers (int): Number of layers for the encoder.
        encoder_num_heads (int): Number of attention heads for the encoder.
        pooling (str): Pooling operation to use.
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
        no_value (bool): Whether to predict the value of the intervention.
        kwargs (dict): Additional arguments to pass to the MLPModule.
    """

    def __init__(
        self,
        env_spec,
        encoding_dim=32,
        batch_size=1,
        encoder_widening_factor=2,
        encoder_dropout=0.0,
        encoder_n_layers=1,
        encoder_num_heads=8,
        pooling="max",
        emitter_sizes=(128, 128),
        emitter_nonlinearity=nn.ReLU,
        emitter_output_nonlinearity=None,
        no_value=False,
        **kwargs,
    ):
        super().__init__()
        self._env_spec = env_spec
        self._obs_dim = env_spec.observation_space.shape[-1]
        self._action_dim = env_spec.action_space.flat_dim
        self._pooling = pooling

        self.no_value = no_value

        self._encoder = AlternateAttention(
            dim=encoding_dim,
            feedforward_dim=encoder_widening_factor * encoding_dim,
            dropout=encoder_dropout,
            n_layers=encoder_n_layers,
            num_heads=encoder_num_heads,
        )
        self._encoder = nn.DataParallel(self._encoder)
        # Multi-target only: use standard MLPModule
        if no_value:
            self._emitter = MLPModule(
                input_dim=encoding_dim + batch_size,
                output_dim=1,
                hidden_sizes=emitter_sizes,
                hidden_nonlinearity=emitter_nonlinearity,
                output_nonlinearity=emitter_output_nonlinearity,
                **kwargs,
            )
        else:
            self._emitter = MLPModule(
                input_dim=encoding_dim + 2 * batch_size,
                output_dim=1,
                hidden_sizes=emitter_sizes,
                hidden_nonlinearity=emitter_nonlinearity,
                output_nonlinearity=emitter_output_nonlinearity,
                **kwargs,
            )
        self._emitter = nn.DataParallel(self._emitter)

    def forward(self, observations, actions, mask):
        """Return Q-value(s)."""
        encoding = self._encoder.forward(observations)
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

        if actions is not None:
            if self.no_value:
                actions = actions.unsqueeze(-1)
            else:
                actions = actions.reshape(*actions.shape[:-1], -1, 2)
            actions = actions.transpose(-2, -3)
            actions = actions.reshape(*actions.shape[:-2], -1)

        q_vals = self._emitter.forward(torch.cat([encoding, actions], -1)).sum(-2)
        return q_vals
