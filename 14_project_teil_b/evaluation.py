"""
Evaluation and Metrics Module

Computes financial metrics for RL trading agents:
- Cumulative return
- Sharpe ratio
- Max drawdown
- Volatility
- Win rate
- Turnover
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple


class TradingMetrics:
    """Compute financial metrics for trading strategies."""

    @staticmethod
    def cumulative_return(equity_curve: np.ndarray) -> float:
        """
        Compute total return.

        Parameters
        ----------
        equity_curve : np.ndarray
            Equity values over time

        Returns
        -------
        float
            Total return (e.g., 0.25 for 25%)
        """
        if len(equity_curve) < 2:
            return 0.0
        return (equity_curve[-1] - equity_curve[0]) / equity_curve[0]

    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.04) -> float:
        """
        Compute Sharpe ratio (annualized).

        Parameters
        ----------
        returns : np.ndarray
            Daily returns
        risk_free_rate : float
            Annual risk-free rate

        Returns
        -------
        float
            Annualized Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0

        excess_return = np.mean(returns) - risk_free_rate / 252.0
        volatility = np.std(returns)

        if volatility == 0:
            return 0.0

        return (excess_return / volatility) * np.sqrt(252)

    @staticmethod
    def max_drawdown(equity_curve: np.ndarray) -> float:
        """
        Compute maximum drawdown.

        Parameters
        ----------
        equity_curve : np.ndarray

        Returns
        -------
        float
            Max drawdown (e.g., -0.15 for 15% decline)
        """
        if len(equity_curve) < 2:
            return 0.0

        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return np.min(drawdown)

    @staticmethod
    def volatility(returns: np.ndarray) -> float:
        """
        Compute annualized volatility.

        Parameters
        ----------
        returns : np.ndarray
            Daily returns

        Returns
        -------
        float
            Annualized volatility
        """
        if len(returns) < 2:
            return 0.0
        return np.std(returns) * np.sqrt(252)

    @staticmethod
    def win_rate(returns: np.ndarray) -> float:
        """
        Compute proportion of positive returns.

        Parameters
        ----------
        returns : np.ndarray

        Returns
        -------
        float
            Win rate (0-1)
        """
        if len(returns) == 0:
            return 0.0
        return np.sum(returns > 0) / len(returns)

    @staticmethod
    def profit_factor(returns: np.ndarray) -> float:
        """
        Compute profit factor (sum of gains / sum of losses).

        Parameters
        ----------
        returns : np.ndarray

        Returns
        -------
        float
            Profit factor
        """
        gains = np.sum(returns[returns > 0])
        losses = np.abs(np.sum(returns[returns < 0]))

        if losses == 0:
            return np.inf if gains > 0 else 1.0

        return gains / losses

    @staticmethod
    def turnover(positions: np.ndarray) -> float:
        """
        Compute average turnover (position change).

        Parameters
        ----------
        positions : np.ndarray
            Position values over time

        Returns
        -------
        float
            Average absolute position change
        """
        if len(positions) < 2:
            return 0.0
        return np.mean(np.abs(np.diff(positions)))

    @staticmethod
    def calmar_ratio(returns: np.ndarray, equity_curve: np.ndarray) -> float:
        """
        Compute Calmar ratio (return / max drawdown).

        Parameters
        ----------
        returns : np.ndarray
        equity_curve : np.ndarray

        Returns
        -------
        float
        """
        annual_return = np.mean(returns) * 252
        max_dd = TradingMetrics.max_drawdown(equity_curve)

        if max_dd == 0:
            return 0.0

        return annual_return / np.abs(max_dd)

    @staticmethod
    def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.04) -> float:
        """
        Compute Sortino ratio (uses downside deviation).

        Parameters
        ----------
        returns : np.ndarray
        risk_free_rate : float

        Returns
        -------
        float
        """
        excess_return = np.mean(returns) - risk_free_rate / 252.0

        downside = returns[returns < 0]
        if len(downside) == 0:
            return 0.0

        downside_dev = np.std(downside)

        if downside_dev == 0:
            return 0.0

        return (excess_return / downside_dev) * np.sqrt(252)


class StrategyEvaluator:
    """Evaluate and compare trading strategies."""

    def __init__(self, risk_free_rate: float = 0.04):
        self.risk_free_rate = risk_free_rate

    def evaluate_episode(
        self,
        env,
        model,
        device,
        deterministic: bool = True,
        action_bounds: Tuple = (-1.0, 1.0),
    ) -> Dict:
        """
        Evaluate policy on a single episode.

        Parameters
        ----------
        env : gym.Env
            Trading environment
        model : nn.Module
            Policy network
        device : torch.device
        deterministic : bool
            Whether to use deterministic actions (mean)
        action_bounds : tuple

        Returns
        -------
        dict
            Metrics and trajectory data
        """
        obs, _ = env.reset()
        done = False

        equity_history = [env.equity]
        position_history = [env.position]
        reward_history = []

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                dist, _ = model(obs_t)
                if deterministic:
                    u = dist.mean
                else:
                    u = dist.sample()
                from ppo_trainer import squash_action
                a = squash_action(u, action_bounds[0], action_bounds[1])

            obs, reward, terminated, truncated, info = env.step(a.cpu().numpy()[0])
            done = terminated or truncated

            equity_history.append(info["equity"])
            position_history.append(info["position"])
            reward_history.append(reward)

        # Convert to arrays
        equity_curve = np.array(equity_history)
        positions = np.array(position_history)
        rewards = np.array(reward_history)

        # Compute returns
        returns = np.diff(equity_curve) / (equity_curve[:-1] + 1e-8)

        # Compute metrics
        metrics = {
            "cumulative_return": TradingMetrics.cumulative_return(equity_curve),
            "sharpe_ratio": TradingMetrics.sharpe_ratio(returns, self.risk_free_rate),
            "max_drawdown": TradingMetrics.max_drawdown(equity_curve),
            "volatility": TradingMetrics.volatility(returns),
            "win_rate": TradingMetrics.win_rate(returns),
            "profit_factor": TradingMetrics.profit_factor(returns),
            "calmar_ratio": TradingMetrics.calmar_ratio(returns, equity_curve),
            "sortino_ratio": TradingMetrics.sortino_ratio(returns, self.risk_free_rate),
            "turnover": TradingMetrics.turnover(positions),
            "final_equity": float(equity_curve[-1]),
            "mean_reward": float(np.mean(rewards)),
        }

        # Store trajectory
        trajectory = {
            "equity_curve": equity_curve,
            "positions": positions,
            "returns": returns,
            "rewards": rewards,
        }

        return metrics, trajectory

    def compare_strategies(
        self,
        strategies: Dict,
        env_fn,
        num_episodes: int = 10,
        deterministic: bool = True,
    ) -> pd.DataFrame:
        """
        Compare multiple strategies.

        Parameters
        ----------
        strategies : dict
            {name: (model, device)}
        env_fn : callable
            Function that returns environment
        num_episodes : int
        deterministic : bool

        Returns
        -------
        pd.DataFrame
            Comparison table
        """
        results = {name: [] for name in strategies.keys()}

        for name, (model, device) in strategies.items():
            print(f"\nEvaluating {name}...")

            for episode in range(num_episodes):
                env = env_fn()
                metrics, _ = self.evaluate_episode(
                    env, model, device, deterministic=deterministic
                )
                results[name].append(metrics)

                if (episode + 1) % 5 == 0:
                    print(f"  Episode {episode + 1}/{num_episodes}")

        # Create summary
        summary = {}
        for name in strategies.keys():
            metrics_list = results[name]

            summary[name] = {
                "Return": f"{np.mean([m['cumulative_return'] for m in metrics_list]):.4f}",
                "Sharpe": f"{np.mean([m['sharpe_ratio'] for m in metrics_list]):.4f}",
                "MaxDD": f"{np.mean([m['max_drawdown'] for m in metrics_list]):.4f}",
                "Volatility": f"{np.mean([m['volatility'] for m in metrics_list]):.4f}",
                "WinRate": f"{np.mean([m['win_rate'] for m in metrics_list]):.4f}",
                "Turnover": f"{np.mean([m['turnover'] for m in metrics_list]):.4f}",
                "Calmar": f"{np.mean([m['calmar_ratio'] for m in metrics_list]):.4f}",
                "Sortino": f"{np.mean([m['sortino_ratio'] for m in metrics_list]):.4f}",
            }

        return pd.DataFrame(summary).T


if __name__ == "__main__":
    print("Evaluation module imported successfully!")

