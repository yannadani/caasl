# Causal Amortized Structure Learning (CAASL)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/caasl.svg)](https://badge.fury.io/py/caasl)

This repository contains the implementation code for the paper **"[Amortized Active Causal Induction with Deep Reinforcement Learning](https://arxiv.org/abs/2405.16718)"**.

## Overview

CAASL is a deep reinforcement learning framework for adaptive causal structure learning. The method uses Soft Actor-Critic (SAC) to learn an agent that can design optimal interventions for discovering causal relationships in structural causal models.

### Key Features

- **Adaptive Intervention Design**: Learns to design optimal interventions for causal discovery
- **Multiple SCM Types**: Supports Synthetic (Linear Gaussian) and SERGIO simulators
- **Scalable Architecture**: Transformer-based policies for handling variable-sized graphs
- **Multi-GPU Support**: Parallel training across multiple GPUs
- **Comprehensive Evaluation**: Out-of-distribution testing and ablation studies

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yannadani/caasl.git
   cd caasl
   ```

2. **Create and activate conda environment:**
   ```bash
   conda create -n caasl python=3.10.13
   conda activate caasl
   ```

3. **Install dependencies and CAASL package:**
   ```bash
   pip install -r requirements.txt .
   ```

   This will automatically install:
   - All Python dependencies (PyTorch, gym, wandb, etc.)
   - The custom garage fork from GitHub

## Quick Start

### Using the CLI (Recommended)

```bash
# Train with SERGIO config
caasl train --config caasl/configs/sergio_train.yaml

# Train with Linear Gaussian config
caasl train --config caasl/configs/linear_gaussian_train.yaml
```

## Project Structure

```
caasl/
├── synthetic.py            # Linear Gaussian experiments
├── sergio.py               # SERGIO experiments
├── cli.py                  # Command-line interface
├── configs/                # YAML configuration files
├── algos/                  # RL algorithms (SAC)
├── envs/                   # Environment implementations
├── models/                 # Causal model definitions
├── policies/               # Neural network policies
├── q_functions/            # Q-function implementations
├── replay_buffer/          # Replay buffer implementations
├── sampler/                # Sampling utilities
├── spaces/                 # Space definitions
├── modules/                # Neural network modules
├── experiment/             # Experiment utilities
├── dowel/                  # Logging utilities
├── params/                 # Parameter management
├── ops/                    # Operations
├── optim/                  # Optimization utilities
├── ...                     # Other core modules
├── requirements.txt        # Python dependencies
├── setup.py                # Package installation
└── README.md               # This file
```

## Configuration

### Key Parameters

- `--d`: Number of variables in the causal graph for training
- `--budget`: Number of interventions allowed
- `--n-rl-itr`: Number of RL training iterations
- `--n-parallel`: Number of parallel environments for training
- `--graph-degree`: Average degree of the causal graph
- `--num-attn-layers`: Number of attention layers in policy
- `--wandb`: Enable Weights & Biases logging

### Environment Variables

- `CUDA_VISIBLE_DEVICES`: Specify which GPUs to use
- `WANDB_PROJECT`: Weights & Biases project name
- `WANDB_ENTITY`: Weights & Biases username/team

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{annadani2024amortized,
  title={Amortized Active Causal Induction with Deep Reinforcement Learning},
  author={Annadani, Yashas and Tigas, Panagiotis and Bauer, Stefan and Foster, Adam},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024},
  volume={37}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This code is based on the RL for SED implementation by Blau et al. 2022 with the [open source code](https://github.com/csiro-mlai/RL-BOED) and [MIT License](https://github.com/csiro-mlai/RL-BOED?tab=MIT-1-ov-file#readme).

