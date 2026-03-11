"""
Weights & Biases Integration Utilities
========================================

This module provides utilities for integrating Weights & Biases (W&B) with the
Forecast-Augmented Reinforcement Learning for Trading project.

Features:
    - Easy initialization of W&B runs
    - Logging helper functions
    - Configuration tracking
    - Artifact management
"""

import wandb
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch


class WandBManager:
    """Manages Weights & Biases integration for trading RL experiments."""

    def __init__(self, project: str = "trading-rl-forecast",
                 entity: Optional[str] = None,
                 enabled: bool = True):
        """
        Initialize WandB manager.

        Args:
            project: W&B project name
            entity: W&B entity (team/user) name
            enabled: Whether to enable W&B logging
        """
        self.project = project
        self.entity = entity
        self.enabled = enabled
        self.run = None

    def init_forecasting_run(self, config: Dict[str, Any], **kwargs):
        """Initialize W&B run for forecasting phase."""
        if not self.enabled:
            return

        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            name="LSTM-Forecasting",
            config=config,
            tags=["forecasting", "lstm"],
            **kwargs
        )
        print(f"✓ W&B Forecasting run initialized: {self._get_run_url()}")

    def init_training_run(self, config: Dict[str, Any], **kwargs):
        """Initialize W&B run for PPO training phase."""
        if not self.enabled:
            return

        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            name="PPO-Trading",
            config=config,
            tags=["trading", "ppo", "reinforcement-learning"],
            **kwargs
        )
        print(f"✓ W&B Training run initialized: {self._get_run_url()}")

    def log_forecast_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log forecasting epoch metrics."""
        if not self.enabled or self.run is None:
            return

        log_dict = {f"forecast/{k}": v for k, v in metrics.items()}
        log_dict["forecast/epoch"] = epoch
        wandb.log(log_dict)

    def log_forecast_test(self, metrics: Dict[str, float]):
        """Log forecasting test metrics."""
        if not self.enabled or self.run is None:
            return

        log_dict = {f"forecast/{k}": v for k, v in metrics.items()}
        wandb.log(log_dict)

    def log_training_step(self, update: int, metrics: Dict[str, float]):
        """Log PPO training step metrics."""
        if not self.enabled or self.run is None:
            return

        log_dict = {f"training/{k}": v for k, v in metrics.items()}
        log_dict["training/update"] = update
        wandb.log(log_dict)

    def log_evaluation(self, metrics: Dict[str, float]):
        """Log evaluation metrics."""
        if not self.enabled or self.run is None:
            return

        log_dict = {f"evaluation/{k}": v for k, v in metrics.items()}
        wandb.log(log_dict)

    def log_diagnostics(self, metrics: Dict[str, float]):
        """Log training diagnostics."""
        if not self.enabled or self.run is None:
            return

        log_dict = {f"diagnostics/{k}": v for k, v in metrics.items()}
        wandb.log(log_dict)

    def log_figure(self, name: str, figure, step: Optional[int] = None):
        """Log matplotlib figure."""
        if not self.enabled or self.run is None:
            return

        wandb.log({name: wandb.Image(figure)})

    def log_chart(self, name: str, x: np.ndarray, y: np.ndarray,
                  xlabel: str = "x", ylabel: str = "y", title: str = ""):
        """Log a simple line chart."""
        if not self.enabled or self.run is None:
            return

        data = [[xi, yi] for xi, yi in zip(x, y)]
        table = wandb.Table(data=data, columns=[xlabel, ylabel])
        wandb.log({
            name: wandb.plot.line(
                table, xlabel, ylabel, title=title or name
            )
        })

    def save_artifact(self, artifact_path: Path, artifact_type: str = "model"):
        """Save artifact to W&B."""
        if not self.enabled or self.run is None:
            return

        artifact = wandb.Artifact(
            name=artifact_path.stem,
            type=artifact_type
        )
        artifact.add_file(str(artifact_path))
        wandb.log_artifact(artifact)
        print(f"✓ Artifact logged: {artifact_path.name}")

    def finish(self):
        """Finish the W&B run."""
        if not self.enabled or self.run is None:
            return

        wandb.finish()
        print("✓ W&B run finished")

    def _get_run_url(self) -> str:
        """Get the URL of the current run."""
        if self.run is None:
            return "Not initialized"
        return f"https://wandb.ai/{self.run.entity}/{self.run.project}/runs/{self.run.id}"


def create_config_summary(config_dict: Dict[str, Any]) -> str:
    """Create a formatted summary of configuration."""
    summary = "Configuration Summary\n" + "="*50 + "\n"
    for key, value in config_dict.items():
        if isinstance(value, (int, float)):
            summary += f"{key:.<40} {value}\n"
        else:
            summary += f"{key:.<40} {value}\n"
    return summary


def log_model_architecture(model: torch.nn.Module, name: str = "model"):
    """Log model architecture to W&B."""
    if wandb.run is None:
        return

    model_str = str(model)
    wandb.log({
        f"{name}_architecture": wandb.Html(f"<pre>{model_str}</pre>")
    })


def create_hyperparameter_summary(config: Dict[str, Any]) -> str:
    """Create a markdown table of hyperparameters."""
    md = "| Parameter | Value |\n"
    md += "|-----------|-------|\n"
    for key, value in config.items():
        md += f"| {key} | {value} |\n"
    return md


if __name__ == "__main__":
    # Example usage
    manager = WandBManager(project="test-project", enabled=True)

    # Initialize a test run
    test_config = {
        "model": "LSTM",
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001
    }

    manager.init_forecasting_run(test_config)

    # Log some test metrics
    for epoch in range(5):
        manager.log_forecast_epoch(epoch, {
            "train_loss": 0.5 - epoch * 0.1,
            "val_loss": 0.52 - epoch * 0.09,
            "val_accuracy": 0.6 + epoch * 0.05
        })

    manager.finish()
    print("✓ WandB utilities test completed")

