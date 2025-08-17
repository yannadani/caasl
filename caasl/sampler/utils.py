"""Utility functions related to sampling."""

import time

import numpy as np
import torch
from garage.np import truncate_tensor_dict


def transpose_tensor(arr_list):
    arr = np.stack(arr_list)
    arr_s = list(range(len(arr.shape)))
    arr_s[:2] = [1, 0]
    return arr.transpose(*arr_s)


def transpose_list(lst):
    return np.stack(lst).transpose()


def pad_observation(obs, max_episode_length):
    pad_shape = list(obs.shape)
    pad_shape[1] = max_episode_length - pad_shape[1]
    pad = torch.zeros(pad_shape)
    padded_obs = torch.cat([obs, pad], dim=1)
    mask = torch.cat(
        [
            torch.ones_like(obs, dtype=torch.bool),
            torch.zeros_like(pad, dtype=torch.bool),
        ],
        dim=1,
    )[..., :1]
    return padded_obs, mask


def rollout(
    env,
    agent,
    *,
    max_path_length=np.inf,
    animated=False,
    speedup=1,
    deterministic=False,
    n_parallel=1,
    eval_mode="policy",
):
    """Sample a single rollout of the agent in the environment.

    Args:
        agent(Policy): Agent used to select actions.
        env(gym.Env): Environment to perform actions in.
        max_path_length(int): If the rollout reaches this many timesteps, it is
            terminated.
        animated(bool): If true, render the environment after each step.
        speedup(float): Factor by which to decrease the wait time between
            rendered steps. Only relevant, if animated == true.
        deterministic (bool): If true, use the mean action returned by the
            stochastic policy instead of sampling from the returned action
            distribution.

    Returns:
        dict[str, torch.Tensor or dict]: Dictionary, with keys:
            * observations(torch.Tensor): Flattened array of observations.
                There should be one more of these than actions. Note that
                observations[i] (for i < len(observations) - 1) was used by the
                agent to choose actions[i]. Should have shape (T + 1, S^*) (the
                unflattened state space of the current environment).
            * actions(torch.Tensor): Non-flattened array of actions. Should have
                shape (T, S^*) (the unflattened action space of the current
                environment).
            * rewards(torch.Tensor): Array of rewards of shape (T,) (1D array of
                length timesteps).
            * agent_infos(Dict[str, torch.Tensor]): Dictionary of stacked,
                non-flattened `agent_info` arrays.
            * env_infos(Dict[str, torch.Tensor]): Dictionary of stacked,
                non-flattened `env_info` arrays.
            * dones(torch.Tensor): Array of termination signals.

    """
    observations = []
    masks = []
    actions = []
    rewards = []
    rewards_c = []
    last_observations = []
    g_probs = []
    dones = []
    o, _dict = env.reset(n_parallel=n_parallel)
    o, mask = pad_observation(o, max_path_length * env.batch_size + env.num_initial_obs)
    observations.append(o)
    masks.append(mask)
    rewards.append(_dict["reward"])
    dones.append(torch.zeros_like(_dict["reward"]))
    rewards_c.append(_dict["info"]["reward_c"])
    g_probs.append(_dict["info"]["g_prob"])
    a_shape = (
        tuple(env.action_space.shape[:-3]) + (n_parallel,) + env.action_space.shape[-2:]
    )
    actions.append(torch.zeros(a_shape))
    if eval_mode == "policy":
        agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < (max_path_length or np.inf):
        if eval_mode == "multi-random-random":
            a = torch.randn(a_shape)
            print(a_shape)
        elif eval_mode == "policy":
            a, agent_info = agent.get_actions(o, mask=mask)
            if deterministic and "mean" in agent_info:
                a = agent_info["mean"]
        elif eval_mode == "multi-random-zero":
            a = torch.randn(a_shape)
            num_nodes = a_shape[-1] // 2
            a[..., num_nodes:] = 0
        elif eval_mode == "observational":
            a = torch.zeros(a_shape)
        elif eval_mode == "single-random-zero":
            num_nodes = a_shape[-1] // 2
            a_logits = torch.zeros(a_shape[:-1] + (num_nodes,))
            a = torch.distributions.OneHotCategorical(logits=a_logits).sample()
            a = torch.cat([a, torch.zeros_like(a)], dim=-1)
        elif eval_mode == "single-random-one":
            num_nodes = a_shape[-1] // 2
            a_logits = torch.zeros(a_shape[:-1] + (num_nodes,))
            a = torch.distributions.OneHotCategorical(logits=a_logits).sample()
            a = torch.cat([a, torch.ones_like(a)], dim=-1)
        elif eval_mode == "single-random-random":
            num_nodes = a_shape[-1] // 2
            a_logits = torch.zeros(a_shape[:-1] + (num_nodes,))
            a = torch.distributions.OneHotCategorical(logits=a_logits).sample()
            values = torch.randn_like(a) > 0
            values[values == 0] = -1
            a = torch.cat([a, values], dim=-1)
        else:
            raise ValueError("Invalid eval_mode: {}".format(eval_mode))
        print(a.shape, a_shape)
        env_step = env.step(a.reshape(a_shape))
        o, r = env_step.observation, env_step.reward
        d, env_info = env_step.terminal, env_step.env_info
        d = d * torch.ones_like(r)

        o, mask = pad_observation(
            o, max_path_length * env.batch_size + env.num_initial_obs
        )
        observations.append(o)
        masks.append(mask)
        rewards.append(r)
        actions.append(a)
        dones.append(d)
        rewards_c.append(env_info["reward_c"])
        g_probs.append(env_info["g_prob"])
        """
        for k, v in agent_info.items():
            agent_infos[k].append(v)
        for k, v in env_info.items():
            if hasattr(v, 'shape'):
                env_infos[k].append(v.squeeze())
            else:
                env_infos[k].append([v] * n_parallel)
        """
        path_length += 1
        if env_step.terminal:
            break
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    """
    for k, v in agent_infos.items():
        agent_infos[k] = torch.cat(torch.split(torch.stack(v, dim=1), 1), dim=1
                                   ).squeeze(0)
    for k, v in env_infos.items():
        if torch.is_tensor(v[0]):
            env_infos[k] = torch.cat(
                torch.split(torch.stack(v, dim=1), 1), dim=1).squeeze(0)
        else:
            env_infos[k] = torch.cat(
                torch.split(torch.as_tensor(v), 1), dim=1).squeeze(0)
    """
    last_observations.append(observations[-1])
    return dict(
        observations=torch.cat(
            torch.split(torch.stack(observations, dim=1), 1), dim=1
        ).squeeze(0),
        actions=torch.cat(torch.split(torch.stack(actions, dim=1), 1), dim=1).squeeze(
            0
        ),
        rewards=torch.cat(torch.split(torch.stack(rewards, dim=1), 1), dim=1).squeeze(
            0
        ),
        # agent_infos=agent_infos,
        # env_infos=env_infos,
        dones=torch.cat(torch.split(torch.stack(dones, dim=1), 1), dim=1).squeeze(0),
        last_observations=torch.cat(
            torch.split(torch.stack(last_observations, dim=1), 1), dim=1
        ).squeeze(0),
        rewards_c=torch.cat(
            torch.split(torch.stack(rewards_c, dim=1), 1), dim=1
        ).squeeze(0),
        masks=torch.cat(torch.split(torch.stack(masks, dim=1), 1), dim=1).squeeze(0),
        g_prob=torch.cat(torch.split(torch.stack(g_probs, dim=1), 1), dim=1).squeeze(0),
    )


def truncate_paths(paths, max_samples):
    """Truncate the paths so that the total number of samples is max_samples.

    This is done by removing extra paths at the end of
    the list, and make the last path shorter if necessary

    Args:
        paths (list[dict[str, np.ndarray]]): Samples, items with keys:
            * observations (np.ndarray): Enviroment observations
            * actions (np.ndarray): Agent actions
            * rewards (np.ndarray): Environment rewards
            * env_infos (dict): Environment state information
            * agent_infos (dict): Agent state information
        max_samples(int) : Maximum number of samples allowed.

    Returns:
        list[dict[str, np.ndarray]]: A list of paths, truncated so that the
            number of samples adds up to max-samples

    Raises:
        ValueError: If key a other than 'observations', 'actions', 'rewards',
            'env_infos' and 'agent_infos' is found.

    """
    # chop samples collected by extra paths
    # make a copy
    valid_keys = {"observations", "actions", "rewards", "env_infos", "agent_infos"}
    paths = list(paths)
    total_n_samples = sum(len(path["rewards"]) for path in paths)
    while paths and total_n_samples - len(paths[-1]["rewards"]) >= max_samples:
        total_n_samples -= len(paths.pop(-1)["rewards"])
    if paths:
        last_path = paths.pop(-1)
        truncated_last_path = dict()
        truncated_len = len(last_path["rewards"]) - (total_n_samples - max_samples)
        for k, v in last_path.items():
            if k in ["observations", "actions", "rewards"]:
                truncated_last_path[k] = v[:truncated_len]
            elif k in ["env_infos", "agent_infos"]:
                truncated_last_path[k] = truncate_tensor_dict(v, truncated_len)
            else:
                raise ValueError(
                    "Unexpected key {} found in path. Valid keys: {}".format(
                        k, valid_keys
                    )
                )
        paths.append(truncated_last_path)
    return paths
