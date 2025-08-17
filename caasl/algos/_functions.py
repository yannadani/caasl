"""Utility functions for NumPy-based Reinforcement learning algorithms."""

import torch

from .._dtypes import EpisodeBatch
from ..sampler.utils import rollout


def obtain_evaluation_episodes(
    policy, env, max_path_length=1000, num_eps=100, n_parallel=1, eval_mode="policy"
):
    """Sample the policy for num_eps trajectories and return average values.

    Args:
        policy (garage.Policy): Policy to use as the actor when
            gathering samples.
        env (garage.envs.GarageEnv): The environement used to obtain
            trajectories.
        max_path_length (int): Maximum path length. The episode will
            terminate when length of trajectory reaches max_path_length.
        num_eps (int): Number of trajectories.

    Returns:
        TrajectoryBatch: Evaluation trajectories, representing the best
            current performance of the algorithm.

    """
    lengths = torch.full((n_parallel,), max_path_length + 1)
    # Use a finite length rollout for evaluation.

    path = rollout(
        env,
        policy,
        max_path_length=max_path_length,
        deterministic=False,
        n_parallel=n_parallel,
        eval_mode=eval_mode,
    )
    return EpisodeBatch(
        env_spec=env.spec,
        episode_infos=dict(),
        observations=path["observations"],
        last_observations=path["last_observations"],
        masks=path["masks"],
        last_masks=None,
        actions=path["actions"],
        rewards=path["rewards"],
        step_types=path["dones"],
        env_infos={},
        agent_infos={},
        lengths=lengths,
    ), EpisodeBatch(
        env_spec=env.spec,
        episode_infos=dict(),
        observations=path["observations"],
        last_observations=path["last_observations"],
        masks=path["masks"],
        last_masks=None,
        actions=path["actions"],
        rewards=path["rewards_c"],
        step_types=path["dones"],
        env_infos={},
        agent_infos={},
        lengths=lengths,
    )
