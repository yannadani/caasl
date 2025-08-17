import os
import pickle as pkl

import numpy as np
import torch
from gym import Env

from .. import set_rng_seed
from ..spaces import BatchBox


class AdaptiveIntervDesignEnvLikelihoodFree(Env):
    def __init__(
        self,
        model,
        budget,
        batch_size=1,
        num_initial_obs=0,
        zero_bias=True,
        action_space=None,
        observation_space=None,
        reward_model=None,
    ):
        """
        A generic class for building a SED MDP for do causal experiments (Hard Interventions with intervention targets and values)

        args:
            design_space (gym.Space): the space of experiment designs
            model_space (gym.Space): the space of model parameterisations
            outcome_space (gym.Space): the space of experiment outcomes
            model (models.ExperimentModel): a model of experiment outcomes
            true_model (models.ExperimentModel): a ground-truth model
            M (int): number of trajectories per sample of theta
            N (int): number of samples of theta
        """
        num_nodes = model.d
        if action_space is None:
            if reward_model.expects_counts:
                self.action_space = BatchBox(
                    low=-1.0, high=1.0, shape=(1, batch_size, num_nodes)
                )
            else:
                self.action_space = BatchBox(
                    low=-1.0, high=1.0, shape=(1, batch_size, 2 * num_nodes)
                )
        else:
            self.action_space = action_space
        if observation_space is None:
            self.observation_space = BatchBox(
                low=torch.as_tensor([[-10.0] * num_nodes, [-10.0] * num_nodes]).T,
                high=torch.as_tensor([[10.0] * num_nodes, [10.0] * num_nodes]).T,
            )
        else:
            self.observation_space = observation_space
        self.model = model
        self.n_parallel = model.n_parallel
        self.num_initial_obs = num_initial_obs
        self.budget = budget
        # self.reward_model = vmap(reward_model.__call__, in_axes=(0, 0))
        self.reward_model = reward_model
        # self.M =
        self.history = []
        self.last_reward = None
        self.last_reward_c = None
        self.batch_size = batch_size
        self.theta0 = None
        self.zero_bias = zero_bias
        self.eval = False

    def reset(self, n_parallel=1):
        self.model.reset(n_parallel=n_parallel)
        self.n_parallel = n_parallel
        self.history = []
        thetas = self.model.sample_theta(1, zero_bias=self.zero_bias)
        # if self.M != 1 and self.M * self.N == n_parallel:
        #     for k, v in self.thetas.items():
        #         self.thetas[k] = v[:, :self.N].repeat_interleave(self.M, dim=1)
        # index theta correctly because it is a dict
        self.theta0 = {k: v[0] for k, v in thetas.items()}
        reward = torch.zeros(self.n_parallel)
        self.last_reward = reward
        self.last_reward_c = reward
        info = {}
        if self.num_initial_obs > 0:
            design_shape = (
                tuple(self.action_space.shape[:-3])
                + (self.n_parallel, self.num_initial_obs)
                + self.action_space.shape[-1:]
            )
            design = torch.zeros(*design_shape)
            y = self.model.run_experiment(design, self.theta0)
            if isinstance(y, np.ndarray):
                y = torch.from_numpy(y).unsqueeze(0)
            self.history.extend(
                torch.stack(
                    [
                        y,
                        design[..., : self.model.d].squeeze(dim=-2),
                    ],
                    dim=-1,
                ).unbind(1)
            )
            reward, reward_c, g_prob = self.get_reward(y, design)
            self.last_reward = reward
            self.last_reward_c = reward_c
            info = {"y": y.squeeze(-2)}
            info["reward_c"] = reward_c
            info["g_prob"] = g_prob
        print("reset")
        return self.get_obs(), reward, False, info

    def step(self, action):
        design = torch.as_tensor(action)
        # y = self.true_model(design)
        y = self.model.run_experiment(design, self.theta0)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).unsqueeze(0)
        self.history.extend(
            torch.stack(
                [
                    y,
                    (design[..., : self.model.d] > 0).to(
                        y.dtype
                    ),  # Directly append the hard intervention
                ],
                dim=-1,
            ).unbind(1)
        )
        obs = self.get_obs()
        cum_reward, cum_reward_c, g_prob = self.get_reward(y, design)
        reward = cum_reward - self.last_reward
        reward_c = cum_reward_c - self.last_reward_c
        self.last_reward = cum_reward
        self.last_reward_c = cum_reward_c
        done = self.terminal()
        done = done * torch.ones_like(reward, dtype=torch.bool)
        info = {"y": y.squeeze(-2)}
        info["reward_c"] = reward_c
        info["g_prob"] = g_prob
        return obs, reward, done, info

    def get_obs(self):
        if self.history:
            return torch.stack(self.history, dim=-3)
        else:
            return torch.zeros(
                (self.n_parallel, 0, *self.observation_space.shape[-2:]),
            )

    def terminal(self):
        return len(self.history) >= self.budget * self.batch_size + self.num_initial_obs
        # return False

    def get_reward(self, y, design):
        # Current y and design are the last y and design in the history
        history = self.get_obs()
        data_ = history[..., 0].clone().numpy()
        interv_ = history[..., 1].clone().numpy()

        if data_.shape[0] > 2000:
            count = 0
            g_probs = []
            while count < data_.shape[0]:
                g_prob = self.reward_model(
                    data_[count : count + 2000], interv=interv_[count : count + 2000]
                )
                g_probs.append(np.array(g_prob))
                count += 2000
            g_prob = np.concatenate(g_probs, axis=0)
        else:
            g_prob = self.reward_model(data_, interv=interv_)
            g_prob = np.array(g_prob)

        target = self.theta0["graph"]
        if not self.reward_model.expects_counts:
            target = target.squeeze(-3)
        reward_c = 1 - torch.nn.functional.binary_cross_entropy(
            torch.from_numpy(g_prob), target, reduction="none"
        ).mean((-1, -2))
        # reward = 1 - torch.abs((torch.from_numpy(g_prob)>0.5).to(self.theta0["graph"].dtype)-self.theta0["graph"].squeeze(-3)).mean((-1,-2))
        # Sample graph from g_prob and compute edge accuracy for each sample before averaging them
        g_samples = torch.distributions.Bernoulli(torch.from_numpy(g_prob)).sample(
            (100,)
        )
        reward = (g_samples == target).float().sum((-1, -2)).mean(0)
        return reward, reward_c, torch.from_numpy(g_prob)

    def render(self, mode="human"):
        pass


class AdaptiveIntervDesignEnvEvalLikelihoodFree(AdaptiveIntervDesignEnvLikelihoodFree):
    def __init__(
        self,
        model,
        budget,
        batch_size=1,
        num_initial_obs=0,
        zero_bias=True,
        data_seed=0,
        save_path=None,
        action_space=None,
        observation_space=None,
        reward_model=None,
    ):
        """
        A generic class for building a SED MDP for do causal experiments (Hard Interventions with intervention targets and values)

        args:
            design_space (gym.Space): the space of experiment designs
            model_space (gym.Space): the space of model parameterisations
            outcome_space (gym.Space): the space of experiment outcomes
            model (models.ExperimentModel): a model of experiment outcomes
            true_model (models.ExperimentModel): a ground-truth model
            M (int): number of trajectories per sample of theta
            N (int): number of samples of theta
        """
        super().__init__(
            model=model,
            budget=budget,
            batch_size=batch_size,
            num_initial_obs=num_initial_obs,
            zero_bias=zero_bias,
            action_space=action_space,
            observation_space=observation_space,
            reward_model=reward_model,
        )
        self.data_seed = data_seed
        self.eval = True
        set_rng_seed(data_seed)
        thetas = self.model.sample_theta(1, zero_bias=zero_bias)
        self.theta0 = {k: v[0] for k, v in thetas.items()}
        if save_path is not None:
            theta0 = {k: v[0].cpu().numpy() for k, v in thetas.items()}
            theta0["noise_type"] = self.model.noise_type
            os.makedirs(save_path, exist_ok=True)
            pkl.dump(theta0, open(f"{save_path}/theta0.pkl", "wb"))
        self.init_design = None
        if self.num_initial_obs > 0:
            design_shape = (
                tuple(self.action_space.shape[:-3])
                + (self.n_parallel, self.num_initial_obs)
                + self.action_space.shape[-1:]
            )
            self.init_design = torch.zeros(*design_shape)
            self.init_y = self.model.run_experiment(self.init_design, self.theta0)
            if isinstance(self.init_y, np.ndarray):
                self.init_y = torch.from_numpy(self.init_y).unsqueeze(0)

    def reset(self, n_parallel=1):
        self.history = []
        if self.num_initial_obs > 0:
            self.history.extend(
                torch.stack(
                    [
                        self.init_y,
                        self.init_design[..., : self.model.d].squeeze(dim=-2),
                    ],
                    dim=-1,
                ).unbind(1)
            )
        if self.init_design is not None:
            reward, reward_c, g_prob = self.get_reward(self.init_y, self.init_design)
        else:
            reward = torch.zeros(self.n_parallel)
            reward_c = reward
        self.last_reward = reward
        self.last_reward_c = reward_c
        info = {"y": self.init_y.squeeze(-2)} if self.init_design is not None else {}
        info["reward_c"] = reward_c
        info["g_prob"] = g_prob
        return self.get_obs(), reward, False, info
