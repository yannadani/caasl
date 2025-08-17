"""
Simplified OOD evaluation loader that creates environments based on training configs.
This loader only requires specifying what's different from the training environment.
"""

import os
from typing import Any, Dict

import yaml

from caasl.envs import AdaptiveIntervDesignEnvEvalLikelihoodFree, NormalizedCausalEnv
from caasl.envs.gym_env import GymEnv
from caasl.models.linearANM_model import LinGaussANMModel
from caasl.models.sergio_model import GRNSergioModel


class OODEvaluationLoader:
    """Loads training configs and creates OOD evaluation environments with minimal specification."""

    def __init__(self, training_config_path: str):
        """Initialize the loader with a training configuration file path.

        Args:
            training_config_path: Path to the training YAML configuration file
        """
        self.training_config_path = training_config_path
        self.training_config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load the training YAML configuration file.

        Returns:
            Dictionary containing the training configuration
        """
        if not os.path.exists(self.training_config_path):
            raise FileNotFoundError(
                f"Training config not found: {self.training_config_path}"
            )

        with open(self.training_config_path, "r") as f:
            config = yaml.safe_load(f)

        return config

    def _apply_changes(
        self, base_config: Dict[str, Any], changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply changes to the base configuration.

        Args:
            base_config: Base configuration (training or eval model)
            changes: Dictionary of changes to apply

        Returns:
            Modified configuration
        """
        result = base_config.copy()

        for key, value in changes.items():
            if key == "d_offset":
                result["d"] = result["d"] + value
            elif key == "graph_degree_offset":
                result["graph_args"]["degree"] = result["graph_args"]["degree"] + value
            else:
                # Direct replacement
                result[key] = value

        return result

    def get_training_defaults(self) -> Dict[str, Any]:
        """Get the default parameters from the training configuration.

        Returns:
            Dictionary containing default parameter values
        """
        if "defaults" in self.training_config:
            return self.training_config["defaults"].copy()
        else:
            # Fallback defaults if not specified in config
            return {}

    def create_ood_models(
        self, ood_config_path: str = None, use_eval_model: bool = True
    ) -> Dict[str, Any]:
        """Create OOD evaluation models based on the training configuration.

        Args:
            ood_config_path: Path to the OOD evaluation configuration file
            use_eval_model: Whether to use eval_model (True) or model (False) as base

        Returns:
            Dictionary mapping OOD environment names to model instances
        """
        # Auto-determine OOD config path if not provided
        if ood_config_path is None:
            if "model" in self.training_config:
                model_type = self.training_config["model"]["type"]
                # Get the directory of the training config
                training_dir = os.path.dirname(self.training_config_path)
                if "LinGaussANMModel" in model_type:
                    ood_config_path = os.path.join(
                        training_dir, "ood_evaluation_linear_gaussian.yaml"
                    )
                elif "GRNSergioModel" in model_type:
                    ood_config_path = os.path.join(
                        training_dir, "ood_evaluation_sergio.yaml"
                    )
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
            else:
                raise ValueError("Training config must contain 'model' section")

        if not os.path.exists(ood_config_path):
            raise FileNotFoundError(
                f"OOD evaluation config not found: {ood_config_path}"
            )

        with open(ood_config_path, "r") as f:
            ood_config = yaml.safe_load(f)

        # Determine which model type we're working with
        if "model" in self.training_config:
            model_type = self.training_config["model"]["type"]
        else:
            raise ValueError("Training config must contain 'model' section")

        # Choose base configuration
        base_config = (
            self.training_config["eval_model"]
            if use_eval_model
            else self.training_config["model"]
        )

        # Determine which OOD configs to use
        if "LinGaussANMModel" in model_type:
            ood_specs = ood_config.get("linear_gaussian", {})
        elif "GRNSergioModel" in model_type:
            ood_specs = ood_config.get("sergio", {})
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        eval_ood_models = {}

        for ood_name, spec in ood_specs.items():
            # Apply changes to base configuration
            modified_config = self._apply_changes(base_config, spec["changes"])

            # Create model instance (filter out the 'type' field if present)
            model_kwargs = {k: v for k, v in modified_config.items() if k != "type"}

            if "LinGaussANMModel" in model_type:
                model = LinGaussANMModel(**model_kwargs)
            elif "GRNSergioModel" in model_type:
                if "intervention_noise" in model_kwargs:
                    from caasl.models.sergio_model import (
                        GRNSergioModelNoisyIntervention,
                    )

                    model = GRNSergioModelNoisyIntervention(**model_kwargs)
                else:
                    model = GRNSergioModel(**model_kwargs)
            else:
                raise ValueError(f"Unknown model type in config: {model_type}")

            eval_ood_models[ood_name] = model

        return eval_ood_models

    def create_ood_environments(
        self,
        ood_config_path: str = None,
        use_eval_model: bool = True,
        budget: int = None,
        **env_kwargs,
    ) -> Dict[str, Any]:
        """Create complete OOD evaluation environments based on the training configuration.

        Args:
            ood_config_path: Path to the OOD evaluation configuration file
            use_eval_model: Whether to use eval_model (True) or model (False) as base
            budget: Budget for the environments
            **env_kwargs: Additional keyword arguments for environment creation

        Returns:
            Dictionary mapping OOD environment names to environment instances
        """
        # Get the models first
        ood_models = self.create_ood_models(ood_config_path, use_eval_model)

        # Create environments from models
        eval_ood_envs = {}
        for env_name, model in ood_models.items():
            # Determine environment type based on model type
            if "LinGaussANMModel" in str(type(model)):
                env_type = AdaptiveIntervDesignEnvEvalLikelihoodFree
                normalize_obs = False
                is_count_data = False
            elif "GRNSergioModel" in str(
                type(model)
            ) or "GRNSergioModelNoisyIntervention" in str(type(model)):
                env_type = AdaptiveIntervDesignEnvEvalLikelihoodFree
                normalize_obs = True
                is_count_data = True
            else:
                raise ValueError(f"Unknown model type: {type(model)}")

            # Create the environment
            env = GymEnv(
                NormalizedCausalEnv(
                    env_type(
                        model,
                        budget,
                        **env_kwargs,
                    ),
                    normalize_obs=normalize_obs,
                    is_count_data=is_count_data,
                )
            )
            eval_ood_envs[env_name] = env

        return eval_ood_envs

    def create_main_eval_environment(self, budget: int = None, **env_kwargs) -> Any:
        """Create the main evaluation environment using the eval_model config.

        Args:
            budget: Budget for the environment
            **env_kwargs: Additional keyword arguments for environment creation

        Returns:
            Main evaluation environment instance
        """
        if "eval_model" not in self.training_config:
            raise ValueError("Training config must contain 'eval_model' section")

        # Get the eval model config
        eval_model_config = self.training_config["eval_model"]
        model_type = eval_model_config["type"]

        # Create the model instance (filter out the 'type' field)
        model_kwargs = {k: v for k, v in eval_model_config.items() if k != "type"}

        if "LinGaussANMModel" in model_type:
            model = LinGaussANMModel(**model_kwargs)
            normalize_obs = False
            is_count_data = False
        elif "GRNSergioModel" in model_type:
            if "intervention_noise" in model_kwargs:
                from caasl.models.sergio_model import GRNSergioModelNoisyIntervention

                model = GRNSergioModelNoisyIntervention(**model_kwargs)
            else:
                model = GRNSergioModel(**model_kwargs)
            normalize_obs = True
            is_count_data = True
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Create the environment
        env = GymEnv(
            NormalizedCausalEnv(
                AdaptiveIntervDesignEnvEvalLikelihoodFree(
                    model,
                    budget,
                    **env_kwargs,
                ),
                normalize_obs=normalize_obs,
                is_count_data=is_count_data,
            )
        )

        return env

    def get_ood_environment_names(self, ood_config_path: str = None) -> list:
        """Get the list of OOD environment names.

        Args:
            ood_config_path: Path to the OOD evaluation configuration file

        Returns:
            List of OOD environment names
        """
        # Auto-determine OOD config path if not provided
        if ood_config_path is None:
            if "model" in self.training_config:
                model_type = self.training_config["model"]["type"]
                # Get the directory of the training config
                training_dir = os.path.dirname(self.training_config_path)
                if "LinGaussANMModel" in model_type:
                    ood_config_path = os.path.join(
                        training_dir, "ood_evaluation_linear_gaussian.yaml"
                    )
                elif "GRNSergioModel" in model_type:
                    ood_config_path = os.path.join(
                        training_dir, "ood_evaluation_sergio.yaml"
                    )
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
            else:
                raise ValueError("Training config must contain 'model' section")

        if not os.path.exists(ood_config_path):
            raise FileNotFoundError(
                f"OOD evaluation config not found: {ood_config_path}"
            )

        with open(ood_config_path, "r") as f:
            ood_config = yaml.safe_load(f)

        # Determine which model type we're working with
        if "model" in self.training_config:
            model_type = self.training_config["model"]["type"]
        else:
            raise ValueError("Training config must contain 'model' section")

        if "LinGaussANMModel" in model_type:
            return list(ood_config.get("linear_gaussian", {}).keys())
        elif "GRNSergioModel" in model_type:
            return list(ood_config.get("sergio", {}).keys())
        else:
            raise ValueError(f"Unknown model type: {model_type}")


# Example usage:
if __name__ == "__main__":
    # Example for linear Gaussian using class directly
    ood_loader_linear = OODEvaluationLoader("caasl/configs/linear_gaussian_train.yaml")

    # Get OOD models
    ood_models = ood_loader_linear.create_ood_models()
    print("Linear Gaussian OOD models:", list(ood_models.keys()))

    # Get OOD environments
    ood_envs = ood_loader_linear.create_ood_environments(budget=10)
    print("Linear Gaussian OOD environments:", list(ood_envs.keys()))

    # Get main evaluation environment
    main_eval_env = ood_loader_linear.create_main_eval_environment(budget=10)
    print("Main evaluation environment created successfully")

    # Example for SERGIO using class directly
    ood_loader_sergio = OODEvaluationLoader("caasl/configs/sergio_train.yaml")

    # Get OOD environments with custom parameters
    ood_envs_sergio = ood_loader_sergio.create_ood_environments(budget=10, data_seed=42)
    print("SERGIO OOD environments:", list(ood_envs_sergio.keys()))

    # Example with explicit config path (override default)
    ood_loader_custom = OODEvaluationLoader("caasl/configs/linear_gaussian_train.yaml")
    ood_envs_custom = ood_loader_custom.create_ood_environments(
        "caasl/configs/ood_evaluation_linear_gaussian.yaml", budget=10
    )
    print("Custom config OOD environments:", list(ood_envs_custom.keys()))
