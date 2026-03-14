#!/usr/bin/env python3
"""
COMPREHENSIVE EXPERIMENTAL FRAMEWORK
Trading Strategy Comparison: Does LSTM Forecast Improve RL Performance?

This script runs multiple experiments:
1. Forecast-Only Strategy
2. PPO Without Forecast (Baseline)
3. PPO With Forecast
4. PPO With Different Reward Definitions

All results are logged to W&B with proper grouping and comparison tables.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import wandb
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

class ExperimentConfig:
    """Centralized experiment configuration"""
    
    # Data
    TICKER = "^GSPC"  # S&P 500
    START_DATE = "2020-01-01"
    END_DATE = "2023-12-31"
    TEST_SPLIT = 0.8
    
    # LSTM
    LSTM_LOOKBACK = 30
    LSTM_FORECAST_HORIZON = 5
    LSTM_EPOCHS = 50
    LSTM_BATCH_SIZE = 32
    
    # Trading Environment
    INITIAL_EQUITY = 100000.0
    FEE = 0.0005  # 0.05%
    KAPPA = 0.1   # Position change penalty
    LEVERAGE_MAX = 1.0
    
    # PPO Hyperparameters
    PPO_LEARNING_RATE = 3e-4
    PPO_N_STEPS = 2048
    PPO_BATCH_SIZE = 64
    PPO_N_EPOCHS = 10
    PPO_TOTAL_TIMESTEPS = 100000
    PPO_ENT_COEF = 0.01
    PPO_GAMMA = 0.99
    PPO_GAE_LAMBDA = 0.95
    
    # Reward definitions
    REWARD_CONFIGS = {
        'simple_pnl': {
            'description': 'Simple: change in equity',
            'pnl_weight': 1.0,
            'risk_penalty': 0.0,
            'transaction_cost_penalty': 0.0,
        },
        'pnl_with_risk': {
            'description': 'PnL with drawdown penalty',
            'pnl_weight': 1.0,
            'risk_penalty': 0.1,
            'transaction_cost_penalty': 0.0,
        },
        'pnl_with_costs': {
            'description': 'PnL with transaction cost penalty',
            'pnl_weight': 1.0,
            'risk_penalty': 0.0,
            'transaction_cost_penalty': 0.5,
        },
        'balanced': {
            'description': 'Balanced: PnL + Risk + Costs',
            'pnl_weight': 1.0,
            'risk_penalty': 0.1,
            'transaction_cost_penalty': 0.1,
        },
    }


class ExperimentTracker:
    """Manages W&B logging and experiment tracking"""
    
    def __init__(self, project_name="forecast-rl-trading", entity="isc-research"):
        self.project = project_name
        self.entity = entity
        self.run_data = {}
        
    def start_experiment(self, group_name, experiment_name, config_dict):
        """Start a new W&B experiment"""
        wandb.init(
            project=self.project,
            entity=self.entity,
            group=group_name,
            name=experiment_name,
            config=config_dict,
            mode='offline',  # Corporate proxy
            reinit=True
        )
        return wandb.run
        
    def log_metrics(self, metrics_dict, step=None):
        """Log metrics to W&B"""
        if step is not None:
            wandb.log(metrics_dict, step=step)
        else:
            wandb.log(metrics_dict)
    
    def log_strategy_results(self, strategy_name, results):
        """Log final strategy results"""
        wandb.log({
            f"{strategy_name}/final_equity": results['final_equity'],
            f"{strategy_name}/total_return": results['total_return'],
            f"{strategy_name}/sharpe_ratio": results['sharpe_ratio'],
            f"{strategy_name}/max_drawdown": results['max_drawdown'],
            f"{strategy_name}/volatility": results['volatility'],
            f"{strategy_name}/turnover": results['turnover'],
        })
    
    def create_comparison_table(self, results_dict):
        """Create W&B comparison table"""
        table_data = []
        for strategy_name, metrics in results_dict.items():
            table_data.append([
                strategy_name,
                f"${metrics['final_equity']:,.0f}",
                f"{metrics['total_return']*100:.2f}%",
                f"{metrics['sharpe_ratio']:.4f}",
                f"{metrics['volatility']*100:.2f}%",
                f"{metrics['max_drawdown']*100:.2f}%",
                f"{metrics['turnover']:.4f}",
            ])
        
        table = wandb.Table(
            data=table_data,
            columns=["Strategy", "Final Equity", "Return", "Sharpe", "Volatility", "Max DD", "Turnover"]
        )
        wandb.log({"comparison/results_table": table})
        
        return table


class MetricsCalculator:
    """Calculate performance metrics for strategies"""
    
    @staticmethod
    def calculate_metrics(equity_curve, returns):
        """Calculate comprehensive metrics"""
        final_equity = equity_curve[-1]
        initial_equity = equity_curve[0]
        total_return = (final_equity - initial_equity) / initial_equity
        
        # Sharpe Ratio (annualized)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.0
        
        # Maximum Drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Turnover (position changes)
        # This will need position data
        
        return {
            'final_equity': float(final_equity),
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe),
            'volatility': float(volatility),
            'max_drawdown': float(abs(max_drawdown)),
        }


class TradingEnv_2(gym.Env):
    """
    Extended trading environment for PPO.

    Main ideas:
    - Action: target position a_t in [-max_leverage, max_leverage]
    - Reward: pnl - transaction_cost - slippage_cost - risk_penalty
    - State: market features + portfolio features + optional LSTM forecast
    - Optional position smoothing to avoid unrealistic jumps
    """

    metadata = {"render_modes": []}

    def __init__(
            self,
            df,
            fee=0.0005,
            kappa=0.1,
            slippage_coef=0.0,
            smoothing_alpha=1.0,
            max_leverage=1.0,
            reward_scale=1.0,
            include_turnover=False,
            initial_equity=100000.0,
            forecast_probs=None,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)

        # Cost / risk parameters
        self.fee = float(fee)
        self.kappa = float(kappa)
        self.slippage_coef = float(slippage_coef)

        # Position execution: alpha=1.0 => immediate, alpha<1.0 => smooth
        self.smoothing_alpha = float(smoothing_alpha)

        # Exposure cap
        self.max_leverage = float(max_leverage)

        # Reward scaling
        self.reward_scale = float(reward_scale)

        # Whether turnover should be included in observation
        self.include_turnover = bool(include_turnover)

        # Budget tracking (100k USD)
        self.initial_equity = float(initial_equity)

        # LSTM forecast integration
        self.forecast_probs = forecast_probs
        self.include_forecast = forecast_probs is not None

        # Action space: target position
        self.action_space = spaces.Box(
            low=-self.max_leverage,
            high=self.max_leverage,
            shape=(1,),
            dtype=np.float32
        )

        # Market features (4)
        self.feature_cols = ["r", "r_lag1", "mu_hat", "sigma_hat"]

        # Portfolio features: pos, equity_norm, drawdown
        portfolio_dim = 3
        if self.include_turnover:
            portfolio_dim += 1

        # Optional LSTM signal
        n_lstm_features = 1 if self.include_forecast else 0

        obs_dim = len(self.feature_cols) + portfolio_dim + n_lstm_features

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.t = 1
        self.pos = 0.0
        self.target_pos = 0.0
        self.prev_turnover = 0.0

        # Budget: use initial_equity (100k USD)
        self.equity = self.initial_equity
        self.peak = self.initial_equity

        return self._get_obs(), {}

    def _get_obs(self):
        # Market features
        x = self.df.loc[self.t, self.feature_cols].values.astype(np.float32)

        # Normalized equity (relative to initial budget)
        equity_norm = np.float32(self.equity / self.initial_equity)
        drawdown = np.float32((self.peak - self.equity) / (self.peak + 1e-8))

        # Portfolio features
        portfolio_features = [self.pos, equity_norm, drawdown]

        if self.include_turnover:
            portfolio_features.append(self.prev_turnover)

        obs = np.concatenate(
            [x, np.array(portfolio_features, dtype=np.float32)]
        )

        # Optional LSTM forecast signal
        if self.include_forecast and self.t < len(self.forecast_probs):
            lstm_signal = float(self.forecast_probs[self.t] * 2 - 1)
            obs = np.concatenate([obs, [lstm_signal]])
        elif self.include_forecast:
            obs = np.concatenate([obs, [0.0]])

        return obs

    def step(self, action):
        # 1) Raw target action from policy
        raw_target = float(np.clip(action[0], -self.max_leverage, self.max_leverage))

        # 2) Position smoothing / execution lag
        new_pos = (1.0 - self.smoothing_alpha) * self.pos + self.smoothing_alpha * raw_target
        new_pos = float(np.clip(new_pos, -self.max_leverage, self.max_leverage))

        # 3) Market data for current step
        r_t = float(self.df.loc[self.t, "r"])
        sigma_t = float(self.df.loc[self.t, "sigma_hat"])

        if not np.isfinite(sigma_t):
            sigma_t = 0.0

        # 4) PnL from PREVIOUS position
        pnl = self.pos * r_t

        # 5) Trading turnover
        turnover = abs(new_pos - self.pos)

        # 6) Transaction cost
        cost = self.fee * turnover

        # 7) Slippage / market impact
        slippage = self.slippage_coef * turnover * (1.0 + sigma_t)

        # 8) Risk penalty
        risk_pen = self.kappa * (self.pos ** 2) * sigma_t

        # 9) Final reward
        true_reward = pnl - cost - slippage
        reward = true_reward - risk_pen
        reward *= self.reward_scale

        # 10) Update internal portfolio state
        self.target_pos = raw_target
        self.prev_turnover = turnover
        self.pos = new_pos

        # 11) Update equity (use true_reward, NOT penalizing risk in equity)
        self.equity *= float(np.exp(true_reward))
        self.peak = max(self.peak, self.equity)

        # 12) Advance time
        self.t += 1
        terminated = (self.t >= len(self.df) - 1)
        truncated = False

        # 13) Info dictionary for diagnostics
        info = {
            "pnl": pnl,
            "cost": cost,
            "slippage": slippage,
            "risk_pen": risk_pen,
            "turnover": turnover,
            "position": self.pos,
            "target_position": self.target_pos,
            "equity": self.equity,
            "drawdown": (self.peak - self.equity) / (self.peak + 1e-8),
            "cumulative_return": (self.equity - self.initial_equity) / self.initial_equity,
        }

        return self._get_obs(), float(reward), terminated, truncated, info


class ExperimentRunner:
    """Orchestrates experimental runs"""
    
    def __init__(self, config=None):
        self.config = config or ExperimentConfig()
        self.tracker = ExperimentTracker()
        self.results = {}
    
    def run_forecast_only_experiment(self, df_test, forecast_signal):
        """Run forecast-only baseline"""
        run = self.tracker.start_experiment(
            group_name="baseline",
            experiment_name="forecast-only",
            config_dict={'strategy': 'forecast-only'}
        )
        
        print("\n" + "="*70)
        print("EXPERIMENT: Forecast-Only Strategy")
        print("="*70)
        
        equity = [self.config.INITIAL_EQUITY]
        positions = [0.0]
        returns = []
        
        for t in range(1, len(df_test)):
            # Simple rule
            target_pos = 1.0 if forecast_signal[t] > 0.5 else -1.0
            
            current_price = df_test['close'].iloc[t-1]
            next_price = df_test['close'].iloc[t]
            
            pnl = positions[-1] * (next_price - current_price)
            new_equity = equity[-1] + pnl
            
            equity.append(new_equity)
            positions.append(target_pos)
            
            if len(equity) > 1:
                daily_return = (equity[-1] - equity[-2]) / equity[-2]
                returns.append(daily_return)
        
        equity = np.array(equity)
        returns = np.array(returns)
        
        metrics = MetricsCalculator.calculate_metrics(equity, returns)
        metrics['turnover'] = np.sum(np.abs(np.diff(positions))) / len(positions)
        
        self.tracker.log_strategy_results("forecast-only", metrics)
        self.results['forecast-only'] = metrics
        
        print(f"✓ Final Equity: ${metrics['final_equity']:,.0f}")
        print(f"✓ Return: {metrics['total_return']*100:.2f}%")
        print(f"✓ Sharpe: {metrics['sharpe_ratio']:.4f}")
        
        wandb.finish()
        return metrics
    
    def run_ppo_experiment(self, df_train, df_test, forecast_signal=None, reward_config=None, run_name="ppo"):
        """Run PPO experiment"""
        include_forecast = forecast_signal is not None
        experiment_name = f"ppo-{'with' if include_forecast else 'without'}-forecast"
        
        if reward_config:
            experiment_name += f"-{reward_config['name']}"
        
        run = self.tracker.start_experiment(
            group_name="ppo-variants",
            experiment_name=experiment_name,
            config_dict={
                'strategy': 'ppo',
                'include_forecast': include_forecast,
                'reward_config': reward_config.get('name', 'simple') if reward_config else 'simple',
                **self.config.__dict__
            }
        )
        
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {experiment_name.upper()}")
        print(f"{'='*70}")
        
        # Create training environment
        env_train = TradingEnv_2(
            df_train,
            fee=self.config.FEE,
            kappa=self.config.KAPPA,
            slippage_coef=0.0,
            smoothing_alpha=1.0,
            max_leverage=self.config.LEVERAGE_MAX,
            reward_scale=1.0,
            include_turnover=False,
            initial_equity=self.config.INITIAL_EQUITY,
            forecast_probs=forecast_signal[:len(df_train)] if forecast_signal is not None else None,
        )
        
        # Train PPO
        model = PPO(
            "MlpPolicy",
            env_train,
            learning_rate=self.config.PPO_LEARNING_RATE,
            n_steps=self.config.PPO_N_STEPS,
            batch_size=self.config.PPO_BATCH_SIZE,
            n_epochs=self.config.PPO_N_EPOCHS,
            ent_coef=self.config.PPO_ENT_COEF,
            gamma=self.config.PPO_GAMMA,
            gae_lambda=self.config.PPO_GAE_LAMBDA,
            verbose=0
        )
        
        model.learn(total_timesteps=self.config.PPO_TOTAL_TIMESTEPS)
        
        print(f"✓ PPO training complete")
        
        # Evaluate on test set
        env_test = TradingEnv_2(
            df_test,
            fee=self.config.FEE,
            kappa=self.config.KAPPA,
            slippage_coef=0.0,
            smoothing_alpha=1.0,
            max_leverage=self.config.LEVERAGE_MAX,
            reward_scale=1.0,
            include_turnover=False,
            initial_equity=self.config.INITIAL_EQUITY,
            forecast_probs=forecast_signal[len(df_train):] if forecast_signal is not None else None,
        )
        
        obs, _ = env_test.reset()
        equity = [env_test.equity]
        positions = [env_test.pos]
        returns = []
        
        done = False
        while not done and env_test.t < len(df_test) - 1:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, _ = model.predict(obs_t, deterministic=True)
            obs, reward, terminated, _, info = env_test.step(action)
            
            equity.append(env_test.equity)
            positions.append(env_test.pos)
            
            if len(equity) > 1:
                daily_return = (equity[-1] - equity[-2]) / equity[-2]
                returns.append(daily_return)
            
            done = terminated
        
        equity = np.array(equity)
        returns = np.array(returns)
        
        metrics = MetricsCalculator.calculate_metrics(equity, returns)
        metrics['turnover'] = np.sum(np.abs(np.diff(positions))) / len(positions)
        
        self.tracker.log_strategy_results(experiment_name, metrics)
        self.results[experiment_name] = metrics
        
        print(f"✓ Final Equity: ${metrics['final_equity']:,.0f}")
        print(f"✓ Return: {metrics['total_return']*100:.2f}%")
        print(f"✓ Sharpe: {metrics['sharpe_ratio']:.4f}")
        
        wandb.finish()
        return metrics, model
    
    def run_all_experiments(self, df_test, forecast_signal):
        """Run all configured experiments"""
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE EXPERIMENT SUITE")
        print("="*80)
        
        # Split data
        split_idx = int(len(df_test) * self.config.TEST_SPLIT)
        df_train = df_test.iloc[:split_idx]
        df_test_eval = df_test.iloc[split_idx:]
        
        # Experiment 1: Forecast-Only
        self.run_forecast_only_experiment(df_test_eval, forecast_signal[split_idx:])
        
        # Experiment 2: PPO Without Forecast
        self.run_ppo_experiment(df_train, df_test_eval, forecast_signal=None, run_name="ppo-no-forecast")
        
        # Experiment 3: PPO With Forecast
        self.run_ppo_experiment(
            df_train, df_test_eval, 
            forecast_signal=forecast_signal, 
            run_name="ppo-with-forecast"
        )
        
        # Experiment 4: PPO With Different Rewards
        for reward_name, reward_config in self.config.REWARD_CONFIGS.items():
            reward_config['name'] = reward_name
            self.run_ppo_experiment(
                df_train, df_test_eval,
                forecast_signal=forecast_signal,
                reward_config=reward_config,
                run_name=f"ppo-reward-{reward_name}"
            )
        
        # Create comparison table
        self.tracker.create_comparison_table(self.results)
        
        return self.results


if __name__ == "__main__":
    print("Experimental framework loaded. Use ExperimentRunner to execute experiments.")

