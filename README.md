# Amortized Active Causal Induction with Deep Reinforcement Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the implementation code for the paper **"[Amortized Active Causal Induction with Deep Reinforcement Learning](https://arxiv.org/abs/2405.16718)", NeurIPS 2024**.

## Overview

Causal Active Amortized Structure Learning (CAASL) is a deep reinforcement learning framework for causal structure learning with adaptive sequential intervention design. The method uses Soft Actor-Critic (SAC) to learn a policy that can design optimal interventions to perform and collect additional interventional data for discovering causal relationships in structural causal models. The data for training this policy comes from a simulator of the envionment we wish to do causal structure learning in. The reward is defined as the number of correct entries in the predicted adjacency matrix by an amortized causal structure learning framework (for ex. [AVICI](https://github.com/larslorch/avici)) due to the intervention predicted by the policy. Once the policy is trained, informative interventions for any dataset can be obtained by just a forward pass of the dataset (and the data collected so far) through the trained policy.

### Key Features

- **Adaptive Intervention Design**: Learns to design optimal interventions for causal discovery
- **Multiple SCM Types**: Supports Synthetic (Linear Gaussian) and SERGIO simulators
- **Scalable Architecture**: Transformer-based policies for handling variable-sized graphs
- **Multi-GPU Support**: Parallel training across multiple GPUs
- **Comprehensive Evaluation**: Out-of-distribution testing and ablation studies
- **Noisy Intervention Support**: OOD evaluation with intervention noise for robustness testing

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

## Out-of-Distribution (OOD) Evaluation

CAASL includes a comprehensive OOD evaluation framework that allows testing model robustness under various distribution shifts:

### SERGIO OOD Settings
- **Graph Structure Changes**: Different graph priors (Erdős-Rényi, Scale-free)
- **Dimensionality Changes**: Variable count variations
- **Intervention Type Changes**: Different intervention strategies
- **Noise Config Changes**: Various noise configurations
- **Noisy Intervention**: Intervention noise for robustness testing

### Linear Gaussian OOD Settings
- **Graph Structure Changes**: Different graph topologies
- **Dimensionality Changes**: Variable count variations
- **Intervention Type Changes**: Different intervention strategies

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

