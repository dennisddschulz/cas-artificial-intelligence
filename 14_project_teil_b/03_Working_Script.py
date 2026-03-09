"""
Forecast-Augmented Reinforcement Learning for Trading
Complete working script for IntelliJ/PyCharm execution

Run this directly: python 03_Working_Script.py
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

# Add project to path
sys.path.insert(0, '/home/isc-den/cas-artificial-intelligence/14_project_teil_b')

from trading_env import EnhancedTradingEnv
from forecasting import TimeSeriesForecaster
from ppo_trainer import PPOTrainer
from evaluation import TradingMetrics, StrategyEvaluator

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
START = '2023-01-01'
END = '2024-01-01'
TRAIN_FRAC = 0.8

# Trading
FEE = 0.0005
KAPPA = 0.1
INITIAL_CASH = 100000.0
MAX_LEVERAGE = 2.0

# Training (reduced for speed)
TOTAL_UPDATES = 100
TOTAL_EPISODES = 5

print('='*70)
print('FORECAST-AUGMENTED REINFORCEMENT LEARNING FOR TRADING')
print('='*70)
print(f'\nConfiguration:')
print(f'  Data: {TICKER} ({START} to {END})')
print(f'  Training updates: {TOTAL_UPDATES}')
print(f'  Test episodes: {TOTAL_EPISODES}')
print(f'  Device: {DEVICE}')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_rsi(prices, period=14):
    """Compute RSI indicator."""
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

    # Load
    print(f'\nDownloading {TICKER} data...')
    df = yf.download(TICKER, start=START, end=END, progress=False)
    df = df.dropna()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]

    print(f'✓ Loaded {len(df)} days')

    # Features
    print('Computing features...')
    df['log_close'] = np.log(df['close'])
    df['r'] = df['log_close'].diff()
    df['mu_hat'] = df['r'].ewm(span=20, adjust=False).mean()
    df['sigma_hat'] = df['r'].rolling(20).std()
    df['r_lag1'] = df['r'].shift(1)
    df['rsi'] = compute_rsi(df['close'])

    df = df.dropna()
    df = df.reset_index(drop=True)

    print(f'✓ After features: {len(df)} days')

    # Split
    split_idx = int(len(df) * TRAIN_FRAC)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    print(f'✓ Train: {len(df_train)} days')
    print(f'✓ Test: {len(df_test)} days')

    return df_train, df_test


def train_forecaster(df_train):
    """Train LSTM forecasting model."""
    print('\n' + '='*70)
    print('STEP 2: TRAINING FORECASTING MODEL')
    print('='*70)

    forecast_data = df_train[['r', 'mu_hat', 'sigma_hat']].values

    print('\nCreating LSTM forecaster...')
    forecaster = TimeSeriesForecaster(
        input_size=3,
        hidden_size=32,
        num_layers=1,
        forecast_horizon=5,
        device=DEVICE
    )

    print('Training (50 epochs)...')
    history = forecaster.train(
        forecast_data,
        val_data=None,
        epochs=50,
        batch_size=16,
        lr=0.001
    )

    print('✓ Forecaster trained')

    # Generate forecasts
    train_forecasts = forecaster.predict(forecast_data)
    test_forecasts = forecaster.predict(df_train[['r', 'mu_hat', 'sigma_hat']].values)

    print(f'✓ Generated forecasts: {train_forecasts.shape}')

    return forecaster, train_forecasts, test_forecasts


def add_forecasts_to_data(df_train, df_test, train_forecasts, test_forecasts):
    """Add forecast column to dataframes."""
    df_train['forecast'] = 0.0
    start_idx = 24
    if start_idx < len(df_train):
        end_idx = min(start_idx + len(train_forecasts), len(df_train))
        df_train.loc[start_idx:end_idx-1, 'forecast'] = train_forecasts[:end_idx-start_idx, 0]

    df_test['forecast'] = 0.0
    if start_idx < len(df_test):
        end_idx = min(start_idx + len(test_forecasts), len(df_test))
        df_test.loc[start_idx:end_idx-1, 'forecast'] = test_forecasts[:end_idx-start_idx, 0]

    print('✓ Forecasts added to data')
    return df_train, df_test


def train_ppo_agent(df_train, with_forecast=True):
    """Train PPO agent."""
    agent_name = "WITH Forecast" if with_forecast else "WITHOUT Forecast"
    print(f'\n[PPO] Training {agent_name}...')

    # Prepare data
    if not with_forecast:
        df_train = df_train.drop(columns=['forecast'], errors='ignore')

    # Create environment
    env = EnhancedTradingEnv(
        df_train,
        fee=FEE,
        kappa=KAPPA,
        initial_cash=INITIAL_CASH,
        max_leverage=MAX_LEVERAGE
    )

    if not with_forecast:
        env.feature_cols = [c for c in env.feature_cols if c != 'forecast']

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    print(f'[PPO] Obs dim: {obs_dim}, Action dim: {act_dim}')

    # Create trainer
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

    # Training loop
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

        if (update + 1) % 20 == 0:
            mean_ret = np.mean(episode_returns[-20:])
            print(f'[PPO] Update {update+1:3d}/{TOTAL_UPDATES} | Mean reward: {mean_ret:8.2f}')

    print(f'✓ {agent_name} training complete')

    return trainer, episode_returns


def evaluate_agents(df_test, trainer_with, trainer_without):
    """Evaluate agents on test set."""
    print('\n' + '='*70)
    print('STEP 4: EVALUATION ON TEST SET')
    print('='*70)

    evaluator = StrategyEvaluator()

    # WITH forecast
    print(f'\nEvaluating WITH forecast ({TOTAL_EPISODES} episodes)...')
    metrics_with = []
    for i in range(TOTAL_EPISODES):
        env_eval = EnhancedTradingEnv(df_test, FEE, KAPPA, INITIAL_CASH, MAX_LEVERAGE)
        m, _ = evaluator.evaluate_episode(env_eval, trainer_with.model, DEVICE, deterministic=True)
        metrics_with.append(m)
        ret = m['cumulative_return']
        sharpe = m['sharpe_ratio']
        print(f'  Episode {i+1}: Return={ret:7.4f}, Sharpe={sharpe:7.4f}')

    # WITHOUT forecast
    print(f'\nEvaluating WITHOUT forecast ({TOTAL_EPISODES} episodes)...')
    df_test_no_forecast = df_test.drop(columns=['forecast'], errors='ignore')
    metrics_without = []
    for i in range(TOTAL_EPISODES):
        env_eval = EnhancedTradingEnv(df_test_no_forecast, FEE, KAPPA, INITIAL_CASH, MAX_LEVERAGE)
        m, _ = evaluator.evaluate_episode(env_eval, trainer_without.model, DEVICE, deterministic=True)
        metrics_without.append(m)
        ret = m['cumulative_return']
        sharpe = m['sharpe_ratio']
        print(f'  Episode {i+1}: Return={ret:7.4f}, Sharpe={sharpe:7.4f}')

    return metrics_with, metrics_without


def print_results(metrics_with, metrics_without):
    """Print and compare results."""
    print('\n' + '='*70)
    print('STEP 5: RESULTS AND COMPARISON')
    print('='*70)

    comparison = {
        'Metric': ['Return', 'Sharpe', 'Max DD', 'Volatility', 'Win Rate', 'Calmar', 'Sortino'],
        'With Forecast': [
            f"{np.mean([m['cumulative_return'] for m in metrics_with]):.4f}",
            f"{np.mean([m['sharpe_ratio'] for m in metrics_with]):.4f}",
            f"{np.mean([m['max_drawdown'] for m in metrics_with]):.4f}",
            f"{np.mean([m['volatility'] for m in metrics_with]):.4f}",
            f"{np.mean([m['win_rate'] for m in metrics_with]):.4f}",
            f"{np.mean([m['calmar_ratio'] for m in metrics_with]):.4f}",
            f"{np.mean([m['sortino_ratio'] for m in metrics_with]):.4f}",
        ],
        'Without Forecast': [
            f"{np.mean([m['cumulative_return'] for m in metrics_without]):.4f}",
            f"{np.mean([m['sharpe_ratio'] for m in metrics_without]):.4f}",
            f"{np.mean([m['max_drawdown'] for m in metrics_without]):.4f}",
            f"{np.mean([m['volatility'] for m in metrics_without]):.4f}",
            f"{np.mean([m['win_rate'] for m in metrics_without]):.4f}",
            f"{np.mean([m['calmar_ratio'] for m in metrics_without]):.4f}",
            f"{np.mean([m['sortino_ratio'] for m in metrics_without]):.4f}",
        ]
    }

    df_comparison = pd.DataFrame(comparison)

    print('\n')
    print(df_comparison.to_string(index=False))

    return df_comparison


def print_analysis(df_comparison):
    """Print critical analysis."""
    print('\n' + '='*70)
    print('CRITICAL ANALYSIS')
    print('='*70)

    ret_with = float(df_comparison.loc[df_comparison['Metric']=='Return', 'With Forecast'].values[0])
    ret_without = float(df_comparison.loc[df_comparison['Metric']=='Return', 'Without Forecast'].values[0])
    sharpe_with = float(df_comparison.loc[df_comparison['Metric']=='Sharpe', 'With Forecast'].values[0])
    sharpe_without = float(df_comparison.loc[df_comparison['Metric']=='Sharpe', 'Without Forecast'].values[0])

    print(f'\n1. ABSOLUTE RETURNS')
    print(f'   With forecast:    {ret_with:.4f}')
    print(f'   Without forecast: {ret_without:.4f}')
    print(f'   → {"Forecast HELPS" if ret_with > ret_without else "Forecast HURTS"}')

    print(f'\n2. RISK-ADJUSTED RETURNS (Sharpe Ratio)')
    print(f'   With forecast:    {sharpe_with:.4f}')
    print(f'   Without forecast: {sharpe_without:.4f}')
    print(f'   → {"Forecast IMPROVES Sharpe" if sharpe_with > sharpe_without else "Forecast REDUCES Sharpe"}')

    print(f'\n3. KEY INSIGHT')
    if sharpe_with > sharpe_without:
        improvement = ((sharpe_with - sharpe_without) / abs(sharpe_without) * 100) if sharpe_without != 0 else 0
        print(f'   ✓ Forecast model successfully guided RL agent')
        print(f'   ✓ Risk-adjusted performance improved by ~{improvement:.1f}%')
    else:
        print(f'   ✗ Forecast did not improve RL agent performance')
        print(f'   ✗ Possible reasons:')
        print(f'     - LSTM predictions too noisy/inaccurate')
        print(f'     - 5-day forecast horizon mismatches daily trading')
        print(f'     - Market regime changed from training period')

    print('\n' + '='*70)


def save_results(df_comparison):
    """Save results to CSV."""
    output_file = 'forecast_rl_results.csv'
    df_comparison.to_csv(output_file, index=False)
    print(f'\n✓ Results saved to {output_file}')


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run complete pipeline."""
    try:
        # Step 1: Load data
        df_train, df_test = load_and_process_data()

        # Step 2: Train forecaster
        forecaster, train_forecasts, test_forecasts = train_forecaster(df_train)

        # Step 3: Add forecasts
        df_train, df_test = add_forecasts_to_data(df_train, df_test, train_forecasts, test_forecasts)

        # Step 4: Train PPO agents
        print('\n' + '='*70)
        print('STEP 3: TRAINING PPO AGENTS')
        print('='*70)

        trainer_with, returns_with = train_ppo_agent(df_train, with_forecast=True)
        trainer_without, returns_without = train_ppo_agent(df_train, with_forecast=False)

        # Step 5: Evaluate
        metrics_with, metrics_without = evaluate_agents(df_test, trainer_with, trainer_without)

        # Step 6: Results
        df_comparison = print_results(metrics_with, metrics_without)

        # Step 7: Analysis
        print_analysis(df_comparison)

        # Step 8: Save
        save_results(df_comparison)

        print('\n' + '='*70)
        print('✅ PROJECT COMPLETE!')
        print('='*70)
        print('\nNext steps:')
        print('  1. Review results in console output above')
        print('  2. Read TECHNICAL_REPORT.md for detailed analysis')
        print('  3. Check PRESENTATION_OUTLINE.md for presentation structure')
        print('  4. Experiment with different parameters')

    except Exception as e:
        print(f'\n❌ ERROR: {str(e)}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

