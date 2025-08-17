"""
Use SAC to learn an agent that adaptively designs source location experiments
"""

import os
from functools import partial

import avici
import torch
from dowel import logger
from garage.experiment import deterministic
from garage.torch import set_gpu_mode

from caasl import set_rng_seed, wrap_experiment
from caasl.algos import SAC
from caasl.envs import (
    AdaptiveIntervDesignEnvLikelihoodFree,
    GymEnv,
    NormalizedCausalEnv,
)
from caasl.experiment import Trainer
from caasl.models.linearANM_model import LinGaussANMModel
from caasl.policies import AdaptiveTransformerTanhGaussianPolicy
from caasl.q_functions.adaptive_mlp_q_function import AdaptiveMLPQFunctionDoCausal
from caasl.replay_buffer import PathBuffer
from caasl.sampler.local_sampler import LocalSampler
from caasl.sampler.vector_worker import VectorWorker
from caasl.utils.ood_evaluation_loader import OODEvaluationLoader


def main(
    n_parallel=1,
    budget=1,
    n_rl_itr=1,
    seed=None,
    eval_save_dir=None,
    snapshot_mode=None,
    snapshot_gap=None,
    discount=1.0,
    alpha=None,
    k=None,
    d=2,
    log_info=None,
    tau=5e-3,
    pi_lr=3e-4,
    qf_lr=3e-4,
    buffer_capacity=int(1e6),
    ens_size=2,
    M=2,
    G=1,
    minibatch_size=4096,
    data_seed=None,
    num_initial_obs=50,
    batch_size=None,
    use_wandb=None,
    num_attn_layers=2,
    num_attn_layers_q_func=2,
    norm_rewards=None,
    intervention_type=None,
    shared_encoder=None,
    graph_degree=3,
    config_path="caasl/configs/linear_gaussian_train.yaml",
):
    # Load training defaults from config
    ood_loader = OODEvaluationLoader(config_path)
    defaults = ood_loader.get_training_defaults()

    # Apply defaults for None values
    seed = seed if seed is not None else defaults.get("seed", 0)
    eval_save_dir = (
        eval_save_dir if eval_save_dir is not None else defaults.get("eval_save_dir")
    )
    snapshot_mode = (
        snapshot_mode
        if snapshot_mode is not None
        else defaults.get("snapshot_mode", "gap")
    )
    snapshot_gap = (
        snapshot_gap if snapshot_gap is not None else defaults.get("snapshot_gap", 500)
    )
    k = k if k is not None else defaults.get("k", 2)
    data_seed = data_seed if data_seed is not None else defaults.get("data_seed", 1)
    batch_size = batch_size if batch_size is not None else defaults.get("batch_size", 1)
    use_wandb = use_wandb if use_wandb is not None else defaults.get("use_wandb", False)
    norm_rewards = (
        norm_rewards
        if norm_rewards is not None
        else defaults.get("norm_rewards", False)
    )
    intervention_type = (
        intervention_type
        if intervention_type is not None
        else defaults.get("intervention_type", "do")
    )
    shared_encoder = (
        shared_encoder
        if shared_encoder is not None
        else defaults.get("shared_encoder", False)
    )

    # Handle wandb initialization if use_wandb is True
    if use_wandb:
        import wandb

        wandb.init(
            project="caasl",
            entity="",
            config={
                "n_parallel": n_parallel,
                "budget": budget,
                "n_rl_itr": n_rl_itr,
                "d": d,
                "G": G,
                "ens_size": ens_size,
                "M": M,
                "minibatch_size": minibatch_size,
                "num_attn_layers": num_attn_layers,
                "num_attn_layers_q_func": num_attn_layers_q_func,
                "tau": tau,
                "pi_lr": pi_lr,
                "qf_lr": qf_lr,
                "buffer_capacity": buffer_capacity,
                "num_initial_obs": num_initial_obs,
                "discount": discount,
                "alpha": alpha,
                "intervention_type": intervention_type,
                "shared_encoder": shared_encoder,
                "graph_degree": graph_degree,
            },
            dir="wandb/",
            group=f"lin-gauss-sem-sac/d={d}",
            job_type="train",
        )

    # Handle eval_save_dir creation if provided
    if eval_save_dir is not None:
        os.makedirs(eval_save_dir, exist_ok=True)
        # Save args to pickle file
        import pickle as pkl

        args_dict = {
            "n_parallel": n_parallel,
            "budget": budget,
            "n_rl_itr": n_rl_itr,
            "d": d,
            "G": G,
            "ens_size": ens_size,
            "M": M,
            "minibatch_size": minibatch_size,
            "num_attn_layers": num_attn_layers,
            "num_attn_layers_q_func": num_attn_layers_q_func,
            "tau": tau,
            "pi_lr": pi_lr,
            "qf_lr": qf_lr,
            "buffer_capacity": buffer_capacity,
            "num_initial_obs": num_initial_obs,
            "discount": discount,
            "alpha": alpha,
            "intervention_type": intervention_type,
            "shared_encoder": shared_encoder,
            "graph_degree": graph_degree,
        }
        pkl.dump(args_dict, open(os.path.join(eval_save_dir, "args.pkl"), "wb"))

    if log_info is None:
        log_info = []

    @wrap_experiment(
        log_dir=eval_save_dir, snapshot_mode=snapshot_mode, snapshot_gap=snapshot_gap
    )
    def sac_source(
        ctxt=None,
        n_parallel=1,
        budget=1,
        n_rl_itr=1,
        eval_save_dir=None,
        seed=0,
        discount=1.0,
        alpha=None,
        k=2,
        d=2,
        tau=5e-3,
        pi_lr=3e-4,
        qf_lr=3e-4,
        buffer_capacity=int(1e6),
        ens_size=2,
        M=2,
        G=1,
        minibatch_size=1024,
        num_initial_obs=50,
        batch_size=1,
        use_wandb=False,
        num_attn_layers=2,
        num_attn_layers_q_func=2,
        norm_rewards=False,
        intervention_type="do",
        shared_encoder=False,
        graph_degree=3,
        config_path="caasl/configs/linear_gaussian_train.yaml",
    ):
        trainer = Trainer(snapshot_config=ctxt, wandb=use_wandb)
        replay_buffer = PathBuffer(capacity_in_transitions=buffer_capacity)
        if os.path.exists(os.path.join(ctxt.snapshot_dir, "params.pkl")):
            sampler = partial(
                LocalSampler,
                max_episode_length=budget,
                worker_class=VectorWorker,
                worker_args={
                    "num_init_obs": num_initial_obs,
                    "batch_size": batch_size,
                },
            )
            trainer.restore(
                from_dir=ctxt.snapshot_dir,
                from_epoch="last",
                replay_buffer=replay_buffer,
                sampler=sampler,
                alpha=alpha,
            )
            trainer.resume(n_epochs=n_rl_itr, batch_size=n_parallel * budget)
        else:
            if log_info:
                logger.log(str(log_info))
            if torch.cuda.is_available():
                set_gpu_mode(True, 0)
                device = torch.device("cuda")
                # torch.set_default_tensor_type("torch.cuda.FloatTensor")
                logger.log("GPU available")
            else:
                device = torch.device("cpu")
                set_gpu_mode(False)
                logger.log("no GPU detected")
            # if there is a saved agent to load

            logger.log("creating new policy")

            model = LinGaussANMModel(
                n_parallel=n_parallel,
                d=d,
                intervention_type=intervention_type,
                graph_args={"degree": graph_degree},
            )
            # Set up environment types and parameters
            reward_model = avici.load_pretrained(download="neurips-linear")
            kwargs = {}
            kwargs["zero_bias"] = True
            kwargs["batch_size"] = batch_size
            kwargs["num_initial_obs"] = num_initial_obs
            kwargs["reward_model"] = reward_model

            def make_q_func():
                return AdaptiveMLPQFunctionDoCausal(
                    env_spec=env.spec,
                    batch_size=batch_size,
                    encoder_n_layers=num_attn_layers_q_func,
                ).to(device)

            # Create training environment
            deterministic.set_seed(seed)
            set_rng_seed(seed)
            env = GymEnv(
                NormalizedCausalEnv(
                    AdaptiveIntervDesignEnvLikelihoodFree(
                        model,
                        budget,
                        **kwargs,
                    ),
                    normalize_obs=False,
                    normalize_reward=norm_rewards,
                )
            )

            # Create evaluation and OOD environments
            ood_loader = OODEvaluationLoader(config_path)
            eval_env = ood_loader.create_main_eval_environment(budget=budget, **kwargs)
            eval_ood_envs = ood_loader.create_ood_environments(budget=budget, **kwargs)

            # Create policy and Q-functions
            policy = AdaptiveTransformerTanhGaussianPolicy(
                env_spec=env.spec,
                n_attention_layers=num_attn_layers,
                batch_size=batch_size,
                device=device,
            ).to(device)
            qfs = [make_q_func() for _ in range(ens_size)]
            if shared_encoder:
                for qf in qfs:
                    qf._encoder = policy._encoder

            # Create sampler
            sampler = LocalSampler(
                agents=policy,
                envs=env,
                max_episode_length=budget,
                worker_class=VectorWorker,
                worker_args={
                    "num_init_obs": num_initial_obs,
                    "batch_size": batch_size,
                },
            )

            sac = SAC(
                env_spec=env.spec,
                policy=policy,
                qfs=qfs,
                replay_buffer=replay_buffer,
                sampler=sampler,
                max_episode_length_eval=budget,
                gradient_steps_per_itr=64,
                min_buffer_size=int(1e4),
                target_update_tau=tau,
                policy_lr=pi_lr,
                qf_lr=qf_lr,
                discount=discount,
                discount_delta=0.0,
                fixed_alpha=alpha,
                buffer_batch_size=minibatch_size,
                reward_scale=1.0,
                M=M,
                G=G,
                ent_anneal_rate=1 / 1.4e4,
                eval_env=eval_env,
                eval_ood_envs=eval_ood_envs,
                device=device,
                save_dir=eval_save_dir,
            )

        sac.to()
        trainer.setup(algo=sac, env=env)
        trainer.train(n_epochs=n_rl_itr, batch_size=n_parallel * budget)

    sac_source(
        n_parallel=n_parallel,
        budget=budget,
        n_rl_itr=n_rl_itr,
        seed=seed,
        eval_save_dir=eval_save_dir,
        discount=discount,
        alpha=alpha,
        k=k,
        d=d,
        tau=tau,
        pi_lr=pi_lr,
        qf_lr=qf_lr,
        buffer_capacity=buffer_capacity,
        ens_size=ens_size,
        M=M,
        G=G,
        minibatch_size=minibatch_size,
        num_initial_obs=num_initial_obs,
        batch_size=batch_size,
        use_wandb=use_wandb,
        num_attn_layers=num_attn_layers,
        num_attn_layers_q_func=num_attn_layers_q_func,
        norm_rewards=norm_rewards,
        intervention_type=intervention_type,
        shared_encoder=shared_encoder,
        graph_degree=graph_degree,
        config_path=config_path,
    )
