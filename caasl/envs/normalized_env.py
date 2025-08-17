"""An environment wrapper that normalizes action, observation and reward."""

import gym
import gym.spaces
import gym.spaces.utils
import torch

from ..util import clip, torch_nanmin, torch_nanstd


class NormalizedEnv(gym.Wrapper):
    """An environment wrapper for normalization.

    This wrapper normalizes action, and optionally observation and reward.

    Args:
        env (garage.envs.GarageEnv): An environment instance.
        scale_reward (float): Scale of environment reward.
        normalize_obs (bool): If True, normalize observation.
        normalize_reward (bool): If True, normalize reward. scale_reward is
            applied after normalization.
        expected_action_scale (float): Assuming action falls in the range of
            [-expected_action_scale, expected_action_scale] when normalize it.
        flatten_obs (bool): Flatten observation if True.
        obs_alpha (float): Update rate of moving average when estimating the
            mean and variance of observations.
        reward_alpha (float): Update rate of moving average when estimating the
            mean and variance of rewards.

    """

    def __init__(
        self,
        env,
        scale_reward=1.0,
        normalize_obs=False,
        normalize_reward=False,
        expected_action_scale=1.0,
        flatten_obs=True,
        obs_alpha=0.001,
        reward_alpha=0.001,
    ):
        super().__init__(env)

        self._scale_reward = scale_reward
        self._normalize_obs = normalize_obs
        self._normalize_reward = normalize_reward
        self._expected_action_scale = expected_action_scale
        self._flatten_obs = flatten_obs

        self._obs_alpha = obs_alpha
        flat_obs_dim = gym.spaces.utils.flatdim(env.observation_space)
        self._obs_mean = torch.zeros(flat_obs_dim)
        self._obs_var = torch.ones(flat_obs_dim)

        self._reward_alpha = reward_alpha
        self._reward_mean = 0.0
        self._reward_var = 1.0

    def _update_obs_estimate(self, obs):
        flat_obs = gym.spaces.utils.flatten(self.env.observation_space, obs)
        self._obs_mean = (
            1 - self._obs_alpha
        ) * self._obs_mean + self._obs_alpha * flat_obs
        self._obs_var = (
            1 - self._obs_alpha
        ) * self._obs_var + self._obs_alpha * torch.square(flat_obs - self._obs_mean)

    def _update_reward_estimate(self, reward):
        self._reward_mean = (
            1 - self._reward_alpha
        ) * self._reward_mean + self._reward_alpha * reward
        self._reward_var = (
            1 - self._reward_alpha
        ) * self._reward_var + self._reward_alpha * torch.square(
            reward - self._reward_mean
        )

    # def _apply_normalize_obs(self, obs):
    #     """Compute normalized observation.
    #
    #     Args:
    #         obs (torch.Tensor): Observation.
    #
    #     Returns:
    #         torch.Tensor: Normalized observation.
    #
    #     """
    #     self._update_obs_estimate(obs)
    #     flat_obs = gym.spaces.utils.flatten(self.env.observation_space, obs)
    #     normalized_obs = (flat_obs -
    #                       self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)
    #     if not self._flatten_obs:
    #         normalized_obs = gym.spaces.utils.unflatten(
    #             self.env.observation_space, normalized_obs)
    #     return normalized_obs
    def _apply_normalize_obs(self, obs):
        """rescale observatios to range [0,1]

        Args:
            obs (torch.Tensor): Observation.

        Returns:
            torch.Tensor: Normalized observation.
        """
        lb, ub = self.observation_space.low, self.observation_space.high
        norm_obs = (obs - lb) / (ub - lb)
        return norm_obs

    def _apply_denormalize_obs(self, obs):
        """rescale observations from [0,1] to range [lb, ub]"""
        lb, ub = self.observation_space.low, self.observation_space.high
        denorm_obs = obs * (ub - lb) + lb
        return denorm_obs

    def _scale_action(self, action):
        """rescale action from [-1,1] to [lb, ub]"""
        lb, ub = self.action_space.low, self.action_space.high
        if torch.isfinite(lb).all() and torch.isfinite(ub).all():
            scaled_action = lb + (action + self._expected_action_scale) * (
                0.5 * (ub - lb) / self._expected_action_scale
            )
            return clip(scaled_action, lb, ub)
        else:
            return action

    def _unscale_action(self, action):
        """rescale action from [lb, ub] tp [-1,1]"""
        lb, ub = self.action_space.low, self.action_space.high
        scaling_factor = 0.5 * (ub - lb) / self._expected_action_scale
        return (action - lb) / scaling_factor - self._expected_action_scale

    def _apply_normalize_reward(self, reward):
        """Compute normalized reward.

        Args:
            reward (float): Reward.

        Returns:
            float: Normalized reward.

        """
        self._update_reward_estimate(reward)
        return reward / (torch.sqrt(self._reward_var) + 1e-8)

    def reset(self, **kwargs):
        """Reset environment.

        Args:
            **kwargs: Additional parameters for reset.

        Returns:
            tuple:
                * observation (torch.Tensor): The observation of the environment.
                * reward (float): The reward acquired at this time step.
                * done (boolean): Whether the environment was completed at this
                    time step.
                * infos (dict): Environment-dependent additional information.

        """
        ret = self.env.reset(**kwargs)
        if self._normalize_obs:
            return self._apply_normalize_obs(ret)
        else:
            return ret

    def step(self, action):
        """Feed environment with one step of action and get result.

        Args:
            action (torch.Tensor): An action fed to the environment.

        Returns:
            tuple:
                * observation (torch.Tensor): The observation of the environment.
                * reward (float): The reward acquired at this time step.
                * done (boolean): Whether the environment was completed at this
                    time step.
                * infos (dict): Environment-dependent additional information.

        """
        if isinstance(self.action_space, gym.spaces.Box):
            # rescale the action when the bounds are not inf
            scaled_action = self._scale_action(action)
        else:
            scaled_action = action

        next_obs, reward, done, info = self.env.step(scaled_action)

        if self._normalize_obs:
            next_obs = self._apply_normalize_obs(next_obs)
        if self._normalize_reward:
            reward = self._apply_normalize_reward(reward)

        return next_obs, reward * self._scale_reward, done, info


class NormalizedCausalEnv(NormalizedEnv):
    """An environment wrapper for normalization.

    This wrapper normalizes action, and optionally observation and reward.

    Args:
        env (garage.envs.GarageEnv): An environment instance.
        scale_reward (float): Scale of environment reward.
        normalize_obs (bool): If True, normalize observation.
        normalize_reward (bool): If True, normalize reward. scale_reward is
            applied after normalization.
        expected_action_scale (float): Assuming action falls in the range of
            [-expected_action_scale, expected_action_scale] when normalize it.
        flatten_obs (bool): Flatten observation if True.
        obs_alpha (float): Update rate of moving average when estimating the
            mean and variance of observations.
        reward_alpha (float): Update rate of moving average when estimating the
            mean and variance of rewards.

    """

    def __init__(
        self,
        env,
        scale_reward=1.0,
        normalize_obs=False,
        normalize_reward=False,
        expected_action_scale=1.0,
        flatten_obs=True,
        obs_alpha=1.0,
        reward_alpha=0.001,
        is_count_data=False,
    ):
        super().__init__(env)

        self._scale_reward = scale_reward
        self._normalize_obs = normalize_obs
        self._normalize_reward = normalize_reward
        self._expected_action_scale = expected_action_scale
        self._flatten_obs = flatten_obs

        self._obs_alpha = obs_alpha
        self._obs_mean = torch.zeros(1)
        self._obs_min = torch.zeros(1)
        self._obs_std = torch.ones(1)

        self._reward_alpha = reward_alpha
        self._reward_mean = 0.0
        self._reward_var = 1.0
        self.is_count_data = is_count_data

    def _update_obs_estimate(self, obs):
        self._obs_mean = (
            1 - self._obs_alpha
        ) * self._obs_mean + self._obs_alpha * torch.mean(obs, dim=-2, keepdim=True)
        obs_std = torch.std(obs, dim=-2, keepdim=True)
        self._obs_std = torch.where(
            obs_std == 0.0,
            1.0,
            (1 - self._obs_alpha) * self._obs_std + self._obs_alpha * obs_std,
        )

    def _update_obs_estimate_count(self, obs):
        self._obs_min = (
            1 - self._obs_alpha
        ) * self._obs_min + self._obs_alpha * torch_nanmin(
            obs, dim=(-1, -2), keepdim=True
        )
        obs_std = torch_nanstd(obs, dim=(-1, -2), keepdim=True)
        self._obs_std = torch.where(
            obs_std == 0.0,
            1.0,
            (1 - self._obs_alpha) * self._obs_std + self._obs_alpha * obs_std,
        )

    def _update_reward_estimate(self, reward):
        self._reward_mean = (
            1 - self._reward_alpha
        ) * self._reward_mean + self._reward_alpha * reward
        self._reward_var = (
            1 - self._reward_alpha
        ) * self._reward_var + self._reward_alpha * torch.square(
            reward - self._reward_mean
        )

    def _apply_normalize_obs(self, obs):
        """Compute normalized observation.

        Args:
            obs (torch.Tensor): Observation.

        Returns:
            torch.Tensor: Normalized observation.

        """
        if obs.shape[-3] < 2:  # if there is only one observation, do not normalize
            return obs
        self._update_obs_estimate(obs[..., 0])
        obs[..., 0] = (obs[..., 0] - self._obs_mean) / (self._obs_std)
        return obs

    def _apply_normalize_obs_count(self, obs):
        """Compute normalized observation.

        Args:
            obs (torch.Tensor): Observation.

        Returns:
            torch.Tensor: Normalized observation.

        """
        if obs.shape[-3] < 2:  # if there is only one observation, do not normalize
            return obs
        libsize = obs[..., 0].sum(dim=-1, keepdim=True)
        obs[..., 0] = torch.where(
            torch.isclose(obs[..., 0], torch.tensor(0.0)),
            torch.tensor(float("nan")),
            torch.log2(obs[..., 0] / (libsize * 1e-6)),
        )
        self._update_obs_estimate_count(obs[..., 0])
        obs[..., 0] = (
            obs[..., 0] - torch.where(torch.isnan(self._obs_min), 0.0, self._obs_min)
        ) / torch.where(torch.isnan(self._obs_std), 1.0, self._obs_std)
        obs[..., 0] = torch.where(torch.isnan(obs[..., 0]), 0.0, obs[..., 0])
        return obs

    def _scale_action(self, action):
        """rescale action from [-1,1] to [lb, ub]"""
        lb, ub = self.action_space.low, self.action_space.high
        if torch.isfinite(lb).all() and torch.isfinite(ub).all():
            scaled_action = lb + (action + self._expected_action_scale) * (
                0.5 * (ub - lb) / self._expected_action_scale
            )
            return clip(scaled_action, lb, ub)
        else:
            return action

    def _unscale_action(self, action):
        """rescale action from [lb, ub] tp [-1,1]"""
        lb, ub = self.action_space.low, self.action_space.high
        scaling_factor = 0.5 * (ub - lb) / self._expected_action_scale
        return (action - lb) / scaling_factor - self._expected_action_scale

    def _apply_normalize_reward(self, reward):
        """Compute normalized reward.

        Args:
            reward (float): Reward.

        Returns:
            float: Normalized reward.

        """
        self._update_reward_estimate(reward)
        return reward / (torch.sqrt(self._reward_var) + 1e-8)

    def reset(self, **kwargs):
        """Reset environment.

        Args:
            **kwargs: Additional parameters for reset.

        Returns:
            tuple:
                * observation (torch.Tensor): The observation of the environment.
                * reward (float): The reward acquired at this time step.
                * done (boolean): Whether the environment was completed at this
                    time step.
                * infos (dict): Environment-dependent additional information.

        """
        ret, reward, *other_args = self.env.reset(**kwargs)
        if self._normalize_reward:
            reward = self._apply_normalize_reward(reward)
        if self._normalize_obs:
            if self.is_count_data:
                ret = self._apply_normalize_obs_count(ret)
            else:
                ret = self._apply_normalize_obs(ret)
        return ret, reward, *other_args

    def step(self, action):
        """Feed environment with one step of action and get result.

        Args:
            action (torch.Tensor): An action fed to the environment.

        Returns:
            tuple:
                * observation (torch.Tensor): The observation of the environment.
                * reward (float): The reward acquired at this time step.
                * done (boolean): Whether the environment was completed at this
                    time step.
                * infos (dict): Environment-dependent additional information.

        """
        if isinstance(self.action_space, gym.spaces.Box):
            # rescale the action when the bounds are not inf
            scaled_action = self._scale_action(action)
        else:
            scaled_action = action

        next_obs, reward, done, info = self.env.step(scaled_action)

        if self._normalize_obs:
            if self.is_count_data:
                next_obs = self._apply_normalize_obs_count(next_obs)
            else:
                next_obs = self._apply_normalize_obs(next_obs)
        if self._normalize_reward:
            reward = self._apply_normalize_reward(reward)

        return next_obs, reward * self._scale_reward, done, info


normalize = NormalizedEnv
