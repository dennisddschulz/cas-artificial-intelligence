"""
FINAL PROJECT - CLEAN WORKING VERSION
Forecast-Augmented RL Trading
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
from datetime import datetime

sys.path.insert(0, '/home/isc-den/cas-artificial-intelligence/14_project_teil_b')

from trading_env import EnhancedTradingEnv
from ppo_trainer import PPOTrainer
from evaluation import StrategyEvaluator

warnings.filterwarnings('ignore')

# ============================================================================
# SIMPLE N-BEATS
# ============================================================================

class SimpleNBeats(nn.Module):
    def __init__(self, lookback=20, forecast_horizon=5, hidden=32, blocks=2):
        super().__init__()
        self.lookback = lookback
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lookback, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            for _ in range(blocks)
        ])
        self.forecast = nn.Linear(hidden, forecast_horizon)
        self.backcast = nn.Linear(hidden, lookback)
    
    def forward(self, x):
        residual = x
        for block in self.blocks:
            h = block(residual)
            residual = residual - self.backcast(h)
        return self.forecast(h)


def train_nbeats_simple(model, data, epochs=50, device='cpu'):
    opt = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        loss_sum = 0
        count = 0
        for i in range(len(data) - 25):
            x = torch.FloatTensor(data[i:i+20]).unsqueeze(0).to(device)
            y = torch.FloatTensor(data[i+20:i+25]).unsqueeze(0).to(device)
            
            opt.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            opt.step()
            
            loss_sum += loss.item()
            count += 1
        
        if (epoch + 1) % 10 == 0:
            print(f'  Epoch {epoch+1}/{epochs} | Loss: {loss_sum/count:.6f}')


def forecast_with_nbeats(model, data, device='cpu'):
    """Generate forecasts - just first element."""
    preds = []
    with torch.no_grad():
        for i in range(len(data) - 20):
            x = torch.FloatTensor(data[i:i+20]).unsqueeze(0).to(device)
            pred = model(x)
            preds.append(float(pred[0, 0].cpu()))
    return np.array(preds)


# ============================================================================
# CONFIG
# ============================================================================

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('\n' + '='*70)
print('FINAL PROJECT - CLEAN VERSION')
print('='*70)
print(f'Started: {datetime.now().strftime("%H:%M:%S")}')
print(f'Device: {DEVICE}')

# ============================================================================
# DATA
# ============================================================================

print('\n[PART 1] Loading data...')
df = yf.download("BTC-USD", start="2022-01-01", end="2024-01-01", progress=False)

# Fix multiindex
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df.columns = [c.lower() for c in df.columns]

# Features
df['log_c'] = np.log(df['close'])
df['r'] = df['log_c'].diff()
df['mu_hat'] = df['r'].ewm(20).mean()
df['sigma_hat'] = df['r'].rolling(20).std()
df['r_lag1'] = df['r'].shift(1)
df = df.dropna().reset_index(drop=True)

split = int(0.8 * len(df))
df_train = df.iloc[:split].copy()
df_test = df.iloc[split:].copy()

print(f'✓ Loaded {len(df)} days (Train: {len(df_train)}, Test: {len(df_test)})')

# ============================================================================
# FORECAST
# ============================================================================

print('\n[PART 2] Training N-BEATS forecaster...')
returns_train = df_train['r'].values
returns_test = df_test['r'].values

forecaster = SimpleNBeats(lookback=20, forecast_horizon=5, hidden=32, blocks=2).to(DEVICE)
train_nbeats_simple(forecaster, returns_train, epochs=50, device=DEVICE)

# Forecasts
fc_train = forecast_with_nbeats(forecaster, returns_train, device=DEVICE)
fc_test = forecast_with_nbeats(forecaster, returns_test, device=DEVICE)

print(f'✓ Forecaster trained')
print(f'  Train forecasts: {len(fc_train)} values')
print(f'  Test forecasts: {len(fc_test)} values')

# Add to dataframes using iloc + numpy
df_train_array = df_train.values.copy()
df_test_array = df_test.values.copy()

# Create new columns
df_train['forecast'] = 0.0
df_test['forecast'] = 0.0

# Assign using iloc for safety
for i, fc in enumerate(fc_train):
    if 20 + i < len(df_train):
        df_train.iloc[20 + i, df_train.columns.get_loc('forecast')] = fc

for i, fc in enumerate(fc_test):
    if 20 + i < len(df_test):
        df_test.iloc[20 + i, df_test.columns.get_loc('forecast')] = fc

print('✓ Forecasts added')

# ============================================================================
# PPO TRAINING
# ============================================================================

def train_agent(df_data, with_fc=True, num_updates=100):
    name = "WITH" if with_fc else "WITHOUT"
    print(f'\n[PPO] Training {name} forecast ({num_updates} episodes)...')

    df = df_data.copy()
    if not with_fc:
        df = df.drop(columns=['forecast'])
    
    # Create a temp env to get the actual obs dimension
    temp_env = EnhancedTradingEnv(df, fee=0.0005, kappa=0.1, initial_cash=100000, max_leverage=2.0)
    if not with_fc:
        temp_env.feature_cols = [c for c in temp_env.feature_cols if c != 'forecast']
    obs_dim = temp_env.observation_space.shape[0]  # Get actual dimension
    print(f'[PPO] Actual obs dim: {obs_dim}')

    trainer = PPOTrainer(obs_dim=obs_dim, act_dim=1, device=DEVICE, lr=3e-4)

    from ppo_trainer import squash_action

    returns = []
    for update in range(num_updates):
        # Create fresh environment for each episode
        env = EnhancedTradingEnv(df, fee=0.0005, kappa=0.1, initial_cash=100000, max_leverage=2.0)

        if not with_fc:
            env.feature_cols = [c for c in env.feature_cols if c != 'forecast']

        obs, _ = env.reset(seed=SEED + update)
        obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

        done = False
        ep_ret = 0
        steps = 0

        while not done and steps < 500:  # Max 500 steps per episode
            with torch.no_grad():
                dist, _ = trainer.model(obs.unsqueeze(0))
                a = squash_action(dist.sample(), -2.0, 2.0)
            
            try:
                obs_next, rew, term, trunc, _ = env.step(a.cpu().numpy()[0])
                done = term or trunc
                obs = torch.tensor(obs_next, dtype=torch.float32, device=DEVICE)
                ep_ret += rew
                steps += 1
            except (KeyError, IndexError):
                # Natural termination at end of data
                done = True

        returns.append(ep_ret)
        if (update + 1) % 20 == 0:
            mean = np.mean(returns[-20:]) if len(returns) >= 20 else np.mean(returns)
            print(f'  Episode {update+1:3d}/{num_updates} | Mean reward: {mean:10.2f}')

    return trainer, returns


print('\n[PART 3] Training PPO agents...')
trainer_with, returns_with = train_agent(df_train, with_fc=True, num_updates=100)
trainer_without, returns_without = train_agent(df_train, with_fc=False, num_updates=100)

# ============================================================================
# EVALUATION
# ============================================================================

print('\n[PART 4] Evaluation...')

evaluator = StrategyEvaluator()

print(f'Evaluating WITH forecast (5 episodes)...')
m_with = []
for i in range(5):
    env = EnhancedTradingEnv(df_test, 0.0005, 0.1, 100000, 2.0)
    m, _ = evaluator.evaluate_episode(env, trainer_with.model, DEVICE, deterministic=True)
    m_with.append(m)
    print(f'  E{i+1}: Return={m["cumulative_return"]:.4f}, Sharpe={m["sharpe_ratio"]:.4f}')

print(f'Evaluating WITHOUT forecast (5 episodes)...')
df_test_no = df_test.drop(columns=['forecast'])
m_without = []
for i in range(5):
    env = EnhancedTradingEnv(df_test_no, 0.0005, 0.1, 100000, 2.0)
    m, _ = evaluator.evaluate_episode(env, trainer_without.model, DEVICE, deterministic=True)
    m_without.append(m)
    print(f'  E{i+1}: Return={m["cumulative_return"]:.4f}, Sharpe={m["sharpe_ratio"]:.4f}')

# ============================================================================
# RESULTS
# ============================================================================

print('\n' + '='*70)
print('RESULTS')
print('='*70)

ret_w = np.mean([m['cumulative_return'] for m in m_with])
ret_wo = np.mean([m['cumulative_return'] for m in m_without])
sharpe_w = np.mean([m['sharpe_ratio'] for m in m_with])
sharpe_wo = np.mean([m['sharpe_ratio'] for m in m_without])

results = pd.DataFrame({
    'Metric': ['Return', 'Sharpe', 'Max DD', 'Volatility', 'Win Rate'],
    'With Forecast': [
        f"{ret_w:.4f}",
        f"{sharpe_w:.4f}",
        f"{np.mean([m['max_drawdown'] for m in m_with]):.4f}",
        f"{np.mean([m['volatility'] for m in m_with]):.4f}",
        f"{np.mean([m['win_rate'] for m in m_with]):.4f}",
    ],
    'Without Forecast': [
        f"{ret_wo:.4f}",
        f"{sharpe_wo:.4f}",
        f"{np.mean([m['max_drawdown'] for m in m_without]):.4f}",
        f"{np.mean([m['volatility'] for m in m_without]):.4f}",
        f"{np.mean([m['win_rate'] for m in m_without]):.4f}",
    ]
})

print('\n')
print(results.to_string(index=False))

results.to_csv('final_project_results.csv', index=False)
print('\n✓ Saved: final_project_results.csv')

# ============================================================================
# ANALYSIS
# ============================================================================

print('\n' + '='*70)
print('ANALYSIS')
print('='*70)

print(f'\nDoes forecast help?')
print(f'  Return: {ret_w:.4f} vs {ret_wo:.4f} ({ret_w-ret_wo:+.4f})')
print(f'  Sharpe: {sharpe_w:.4f} vs {sharpe_wo:.4f} ({sharpe_w-sharpe_wo:+.4f})')

if sharpe_w > sharpe_wo:
    print(f'\n✅ YES - Forecast improves risk-adjusted returns')
    print(f'  Improvement: {(sharpe_w-sharpe_wo)/abs(sharpe_wo)*100:+.1f}%')
else:
    print(f'\n⚠️  LIMITED - Baseline performs similarly or better')

print('\n✅ PROJECT COMPLETE!')
print(f'Finished: {datetime.now().strftime("%H:%M:%S")}')

