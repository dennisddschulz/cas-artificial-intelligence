"""
IMPROVED Trading RL Agent - Using Better Forecasting Models
Compares LSTM vs N-BEATS vs Transformer vs Ensemble

This is the RECOMMENDED version with better forecasting models.
Execution time: ~2-3 hours on CPU, ~30-45 minutes on GPU
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
import sys
import json
from datetime import datetime

sys.path.insert(0, '/home/isc-den/cas-artificial-intelligence/14_project_teil_b')

from trading_env import EnhancedTradingEnv
from better_forecasters import (
    NBeatsForecaster, TransformerForecaster, EnsembleForecaster,
    train_forecaster, predict_forecaster, TimeSeriesDataset
)
from ppo_trainer import PPOTrainer
from evaluation import StrategyEvaluator

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\n[INFO] Device: {DEVICE}\n')

# Data
TICKER = 'BTC-USD'
START = '2022-01-01'
END = '2024-01-01'
TRAIN_FRAC = 0.8

# Trading
FEE = 0.0005
KAPPA = 0.1
INITIAL_CASH = 100000.0
MAX_LEVERAGE = 2.0

# Training
FORECAST_EPOCHS = 100
TOTAL_UPDATES = 500
TOTAL_EPISODES = 10
FORECASTING_MODELS = ['nbeats', 'transformer', 'ensemble']  # Compare these

print('='*70)
print('IMPROVED TRADING RL WITH BETTER FORECASTING MODELS')
print('='*70)
print(f'\nConfiguration:')
print(f'  Data: {TICKER} ({START} to {END})')
print(f'  Forecast models: {", ".join(FORECASTING_MODELS)}')
print(f'  Forecast epochs: {FORECAST_EPOCHS}')
print(f'  PPO updates: {TOTAL_UPDATES}')
print(f'  Test episodes: {TOTAL_EPISODES}')
print(f'  Device: {DEVICE}')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    return (100 - (100 / (1 + rs))) / 100.0


def load_and_process_data():
    """Load Bitcoin data and compute features."""
    print('\n' + '='*70)
    print('STEP 1: LOADING AND PROCESSING DATA')
    print('='*70)

    print(f'\nDownloading {TICKER}...')
    df = yf.download(TICKER, start=START, end=END, progress=False)
    df = df.dropna()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]

    print(f'✓ Loaded {len(df)} days')

    print('Computing features...')
    df['log_close'] = np.log(df['close'])
    df['r'] = df['log_close'].diff()
    df['mu_hat'] = df['r'].ewm(span=20, adjust=False).mean()
    df['sigma_hat'] = df['r'].rolling(20).std()
    df['r_lag1'] = df['r'].shift(1)
    df['rsi'] = compute_rsi(df['close'])

    def compute_macd(prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal

    df['macd'], df['macd_signal'] = compute_macd(df['close'])

    df = df.dropna()
    df = df.reset_index(drop=True)

    print(f'✓ Features computed for {len(df)} days')

    split_idx = int(len(df) * TRAIN_FRAC)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    print(f'✓ Train: {len(df_train)} days | Test: {len(df_test)} days')

    return df_train, df_test


def train_all_forecasters(df_train):
    """Train all forecasting models and compare."""
    print('\n' + '='*70)
    print('STEP 2: TRAINING FORECASTING MODELS')
    print('='*70)

    forecast_data = df_train[['r']].values

    split_idx = int(0.8 * len(forecast_data))
    train_feat = forecast_data[:split_idx]
    val_feat = forecast_data[split_idx:]

    forecasters = {}

    # 1. N-BEATS
    if 'nbeats' in FORECASTING_MODELS:
        print('\nTraining N-BEATS...')
        nbeats = NBeatsForecaster(lookback=20, forecast_horizon=5, num_blocks=3).to(DEVICE)
        history = train_forecaster(
            nbeats, train_feat, val_feat, epochs=FORECAST_EPOCHS,
            batch_size=32, device=DEVICE, lookback=20, forecast_horizon=5
        )
        forecasters['nbeats'] = nbeats
        val_loss = history['val_loss'][-1] if history['val_loss'] else float('inf')
        print(f'  ✓ Val Loss: {val_loss:.6f}')

    # 2. Transformer
    if 'transformer' in FORECASTING_MODELS:
        print('\nTraining Transformer...')
        transformer = TransformerForecaster(lookback=20, forecast_horizon=5).to(DEVICE)
        history = train_forecaster(
            transformer, train_feat, val_feat, epochs=FORECAST_EPOCHS,
            batch_size=32, device=DEVICE, lookback=20, forecast_horizon=5
        )
        forecasters['transformer'] = transformer
        val_loss = history['val_loss'][-1] if history['val_loss'] else float('inf')
        print(f'  ✓ Val Loss: {val_loss:.6f}')

    # 3. Ensemble
    if 'ensemble' in FORECASTING_MODELS:
        print('\nTraining Ensemble (N-BEATS + Transformer)...')
        ensemble = EnsembleForecaster(lookback=20, forecast_horizon=5, device=DEVICE)
        history = train_forecaster(
            ensemble, train_feat, val_feat, epochs=FORECAST_EPOCHS,
            batch_size=32, device=DEVICE, lookback=20, forecast_horizon=5
        )
        forecasters['ensemble'] = ensemble
        val_loss = history['val_loss'][-1] if history['val_loss'] else float('inf')
        print(f'  ✓ Val Loss: {val_loss:.6f}')

    print('\n✓ All forecasters trained')
    return forecasters


def generate_forecasts(forecasters, df_train, df_test):
    """Generate forecasts using all trained models."""
    print('\nGenerating forecasts...')

    forecast_data_train = df_train[['r']].values
    forecast_data_test = df_test[['r']].values

    for name, forecaster in forecasters.items():
        train_forecasts = predict_forecaster(
            forecaster, forecast_data_train, lookback=20, device=DEVICE
        )
        test_forecasts = predict_forecaster(
            forecaster, forecast_data_test, lookback=20, device=DEVICE
        )

        df_train[f'forecast_{name}'] = 0.0
        df_test[f'forecast_{name}'] = 0.0

        start_idx = 24
        if start_idx < len(df_train):
            end_idx = min(start_idx + len(train_forecasts), len(df_train))
            df_train.loc[start_idx:end_idx-1, f'forecast_{name}'] = train_forecasts[:end_idx-start_idx, 0]

        if start_idx < len(df_test):
            end_idx = min(start_idx + len(test_forecasts), len(df_test))
            df_test.loc[start_idx:end_idx-1, f'forecast_{name}'] = test_forecasts[:end_idx-start_idx, 0]

    print('✓ Forecasts generated')
    return df_train, df_test


def train_ppo_agent(df_train, forecast_type='ensemble', agent_type='with_forecast'):
    """Train PPO agent with specified forecast."""

    if agent_type == 'without_forecast':
        df = df_train.drop(columns=[c for c in df_train.columns if 'forecast' in c], errors='ignore')
        agent_name = "WITHOUT Forecast"
    else:
        df = df_train.copy()
        agent_name = f"WITH {forecast_type.upper()} Forecast"

    print(f'\n[PPO] Training {agent_name}...')

    env = EnhancedTradingEnv(df, fee=FEE, kappa=KAPPA, initial_cash=INITIAL_CASH, max_leverage=MAX_LEVERAGE)

    # Remove forecast columns from feature_cols if not using forecast
    if agent_type == 'without_forecast':
        env.feature_cols = [c for c in env.feature_cols if 'forecast' not in c]
    elif forecast_type != 'all':
        # Keep only one forecast
        env.feature_cols = [c for c in env.feature_cols if not c.startswith('forecast_') or c == f'forecast_{forecast_type}']

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    print(f'[PPO] Obs dim: {obs_dim}')

    trainer = PPOTrainer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=DEVICE,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.001,
        max_grad_norm=0.5,
        target_kl=0.1
    )

    from ppo_trainer import squash_action

    obs, _ = env.reset(seed=SEED)
    obs = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)

    episode_returns = []

    for update in range(TOTAL_UPDATES):
        done = False
        ep_reward = 0

        while not done:
            obs_t = obs.unsqueeze(0) if obs.dim() == 1 else obs

            with torch.no_grad():
                dist, value = trainer.model(obs_t)
                u = dist.sample()
                a = squash_action(u, -MAX_LEVERAGE, MAX_LEVERAGE)

            obs_next, reward, terminated, truncated, _ = env.step(a.cpu().numpy()[0])
            done = terminated or truncated

            obs = torch.as_tensor(obs_next, dtype=torch.float32, device=DEVICE)
            ep_reward += reward

        episode_returns.append(ep_reward)

        if (update + 1) % 50 == 0:
            mean_ret = np.mean(episode_returns[-50:])
            print(f'[PPO] Update {update+1:3d}/{TOTAL_UPDATES} | Mean reward: {mean_ret:10.2f}')

    print(f'✓ {agent_name} training complete')
    return trainer, episode_returns


def evaluate_agents(df_test, trainers_dict):
    """Evaluate all trained agents."""
    print('\n' + '='*70)
    print('STEP 4: EVALUATION ON TEST SET')
    print('='*70)

    evaluator = StrategyEvaluator()
    results = {}

    for agent_name, trainer in trainers_dict.items():
        print(f'\nEvaluating {agent_name}...')
        metrics = []

        for i in range(TOTAL_EPISODES):
            # Prepare dataframe based on agent type
            if 'without' in agent_name.lower():
                df_eval = df_test.drop(columns=[c for c in df_test.columns if 'forecast' in c], errors='ignore')
            elif 'nbeats' in agent_name.lower():
                df_eval = df_test[[c for c in df_test.columns if not c.startswith('forecast_') or c == 'forecast_nbeats']].copy()
            elif 'transformer' in agent_name.lower():
                df_eval = df_test[[c for c in df_test.columns if not c.startswith('forecast_') or c == 'forecast_transformer']].copy()
            elif 'ensemble' in agent_name.lower():
                df_eval = df_test.copy()
            else:
                df_eval = df_test.copy()

            env_eval = EnhancedTradingEnv(df_eval, FEE, KAPPA, INITIAL_CASH, MAX_LEVERAGE)
            m, _ = evaluator.evaluate_episode(env_eval, trainer.model, DEVICE, deterministic=True)
            metrics.append(m)

            ret = m['cumulative_return']
            sharpe = m['sharpe_ratio']
            print(f'  E{i+1}: Return={ret:7.4f}, Sharpe={sharpe:7.4f}')

        results[agent_name] = metrics

    return results


def print_comprehensive_results(results_dict):
    """Print detailed comparison of all models."""
    print('\n' + '='*70)
    print('COMPREHENSIVE RESULTS COMPARISON')
    print('='*70)

    metrics_names = ['Return', 'Sharpe', 'Max DD', 'Volatility', 'Win Rate', 'Calmar', 'Sortino', 'Turnover']

    # Create table
    table_data = {'Model': []}
    for metric in metrics_names:
        table_data[metric] = []

    for agent_name, metrics_list in results_dict.items():
        table_data['Model'].append(agent_name)

        table_data['Return'].append(f"{np.mean([m['cumulative_return'] for m in metrics_list]):.4f}")
        table_data['Sharpe'].append(f"{np.mean([m['sharpe_ratio'] for m in metrics_list]):.4f}")
        table_data['Max DD'].append(f"{np.mean([m['max_drawdown'] for m in metrics_list]):.4f}")
        table_data['Volatility'].append(f"{np.mean([m['volatility'] for m in metrics_list]):.4f}")
        table_data['Win Rate'].append(f"{np.mean([m['win_rate'] for m in metrics_list]):.4f}")
        table_data['Calmar'].append(f"{np.mean([m['calmar_ratio'] for m in metrics_list]):.4f}")
        table_data['Sortino'].append(f"{np.mean([m['sortino_ratio'] for m in metrics_list]):.4f}")
        table_data['Turnover'].append(f"{np.mean([m['turnover'] for m in metrics_list]):.4f}")

    df_table = pd.DataFrame(table_data)
    print('\n')
    print(df_table.to_string(index=False))

    df_table.to_csv('comprehensive_results.csv', index=False)
    print('\n✓ Saved: comprehensive_results.csv')

    return df_table


def generate_comparison_plots(results_dict):
    """Generate visualization plots comparing all models."""
    print('\n' + '='*70)
    print('GENERATING VISUALIZATION PLOTS')
    print('='*70)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Trading Agent Comparison: Different Forecasting Models',
                 fontsize=16, fontweight='bold')

    models = list(results_dict.keys())

    # 1. Return Comparison
    returns_by_model = [np.mean([m['cumulative_return'] for m in metrics_list])
                       for metrics_list in results_dict.values()]
    axes[0, 0].bar(models, returns_by_model, alpha=0.7, color='steelblue')
    axes[0, 0].set_ylabel('Cumulative Return', fontsize=11)
    axes[0, 0].set_title('Average Cumulative Returns', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(returns_by_model):
        axes[0, 0].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

    # 2. Sharpe Ratio Comparison
    sharpe_by_model = [np.mean([m['sharpe_ratio'] for m in metrics_list])
                      for metrics_list in results_dict.values()]
    axes[0, 1].bar(models, sharpe_by_model, alpha=0.7, color='mediumseagreen')
    axes[0, 1].set_ylabel('Sharpe Ratio', fontsize=11)
    axes[0, 1].set_title('Average Sharpe Ratio (Risk-Adjusted)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(sharpe_by_model):
        axes[0, 1].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

    # 3. Max Drawdown Comparison
    dd_by_model = [np.mean([m['max_drawdown'] for m in metrics_list])
                  for metrics_list in results_dict.values()]
    axes[1, 0].bar(models, dd_by_model, alpha=0.7, color='indianred')
    axes[1, 0].set_ylabel('Max Drawdown', fontsize=11)
    axes[1, 0].set_title('Average Maximum Drawdown (Lower Better)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(dd_by_model):
        axes[1, 0].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

    # 4. Win Rate Comparison
    wr_by_model = [np.mean([m['win_rate'] for m in metrics_list])
                  for metrics_list in results_dict.values()]
    axes[1, 1].bar(models, wr_by_model, alpha=0.7, color='gold')
    axes[1, 1].set_ylabel('Win Rate', fontsize=11)
    axes[1, 1].set_title('Average Win Rate', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(wr_by_model):
        axes[1, 1].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('forecasting_models_comparison.png', dpi=150, bbox_inches='tight')
    print('✓ Saved: forecasting_models_comparison.png')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run complete pipeline."""
    try:
        start_time = datetime.now()

        # Load data
        df_train, df_test = load_and_process_data()

        # Train forecasters
        forecasters = train_all_forecasters(df_train)

        # Generate forecasts
        df_train, df_test = generate_forecasts(forecasters, df_train, df_test)

        # Train PPO agents
        print('\n' + '='*70)
        print('STEP 3: TRAINING PPO AGENTS')
        print('='*70)

        trainers_dict = {}

        # Baseline (no forecast)
        trainer_baseline, _ = train_ppo_agent(df_train, agent_type='without_forecast')
        trainers_dict['Baseline (No Forecast)'] = trainer_baseline

        # With each forecasting model
        for model_name in FORECASTING_MODELS:
            trainer, _ = train_ppo_agent(df_train, forecast_type=model_name, agent_type='with_forecast')
            trainers_dict[f'With {model_name.upper()}'] = trainer

        # Evaluate all agents
        results_dict = evaluate_agents(df_test, trainers_dict)

        # Print results
        df_results = print_comprehensive_results(results_dict)

        # Generate plots
        generate_comparison_plots(results_dict)

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds() / 60

        print('\n' + '='*70)
        print('✅ PROJECT EXECUTION COMPLETE!')
        print('='*70)
        print(f'\nExecution time: {execution_time:.1f} minutes ({execution_time/60:.1f} hours)')
        print('\nKey Finding:')

        # Find best model
        sharpe_scores = {name: float(df_results.loc[df_results['Model']==name, 'Sharpe'].values[0])
                        for name in df_results['Model']}
        best_model = max(sharpe_scores, key=sharpe_scores.get)
        print(f'  ✓ BEST MODEL: {best_model} (Sharpe: {sharpe_scores[best_model]:.4f})')

    except Exception as e:
        print(f'\n❌ ERROR: {str(e)}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

