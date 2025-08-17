#!/usr/bin/env python3
"""Command-line interface for CAASL."""

import argparse
import sys
from pathlib import Path

import yaml

from caasl.sergio import main as run_sergio_training
from caasl.synthetic import main as run_linear_gaussian_training


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_training(config_path):
    """Run training based on config file."""
    config = load_config(config_path)

    # Determine which training function to use based on config
    if "sergio" in config_path.lower():
        run_sergio_training(**config)
    else:
        run_linear_gaussian_training(**config)


def create_default_configs():
    """Create default configuration files."""
    configs_dir = Path("caasl/configs")
    configs_dir.mkdir(exist_ok=True)

    # SERGIO training config
    sergio_config = {
        "n_rl_itr": 10001,
        "d": 10,
        "budget": 10,
        "discount": 0.95,
        "buffer_capacity": 1000000,
        "tau": 0.001,
        "pi_lr": 0.01,
        "qf_lr": 0.000003,
        "ens_size": 5,
        "n_parallel": 1000,
        "G": 1,
        "num_attn_layers": 3,
        "num_attn_layers_q_func": 3,
        "graph_degree": 3.0,
    }

    # Linear Gaussian training config
    linear_config = {
        "n_rl_itr": 10001,
        "d": 10,
        "budget": 10,
        "discount": 0.9,
        "buffer_capacity": 10000000,
        "tau": 0.001,
        "pi_lr": 0.001,
        "qf_lr": 0.00003,
        "ens_size": 5,
        "n_parallel": 1000,
        "G": 1,
        "num_attn_layers": 4,
        "num_attn_layers_q_func": 4,
        "graph_degree": 3.0,
    }

    # Write config files
    with open(configs_dir / "sergio_train.yaml", "w") as f:
        yaml.dump(sergio_config, f, default_flow_style=False)

    with open(configs_dir / "linear_gaussian_train.yaml", "w") as f:
        yaml.dump(linear_config, f, default_flow_style=False)

    print(f"Created default configs in {configs_dir}/")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CAASL: Causal Amortized Structure Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with SERGIO config
  caasl train --config caasl/configs/sergio_train.yaml
  
  # Train with Linear Gaussian config
  caasl train --config caasl/configs/linear_gaussian_train.yaml
  
  # Create default config files
  caasl init-configs
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--config", required=True, help="Path to YAML configuration file"
    )

    # Init configs command
    init_parser = subparsers.add_parser(
        "init-configs", help="Create default config files"
    )

    args = parser.parse_args()

    if args.command == "train":
        if not Path(args.config).exists():
            print(f"Error: Config file '{args.config}' not found.")
            sys.exit(1)
        run_training(args.config)
    elif args.command == "init-configs":
        create_default_configs()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    import torch

    torch.set_default_dtype(torch.float32)
    main()
