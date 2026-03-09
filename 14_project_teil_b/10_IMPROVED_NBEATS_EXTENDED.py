"""
FINAL PROJECT - IMPROVED VERSION WITH EXTENDED N-BEATS TRAINING
Forecast-Augmented Reinforcement Learning for Trading

Key Improvement: Increased N-BEATS training from 50 to 500 epochs
with better monitoring and convergence detection
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
# IMPROVED N-BEATS WITH EXTENDED TRAINING
# ============================================================================

class ImprovedNBeats(nn.Module):
    """Enhanced N-BEATS with better architecture for time series forecasting."""
    
    def __init__(self, lookback=20, forecast_horizon=5, hidden_size=64, num_blocks=3, dropout=0.1):
        super().__init__()
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        
        # Stack of improved blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lookback, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            )
            for _ in range(num_blocks)
        ])
        
        self.forecast_head = nn.Linear(hidden_size, forecast_horizon)
        self.backcast_head = nn.Linear(hidden_size, lookback)
    
    def forward(self, x):
        residual = x
        backcast_sum = 0
        
        for block in self.blocks:
            h = block(residual)
            backcast = self.backcast_head(h)
            backcast_sum = backcast_sum + backcast
            residual = residual - backcast
        
        forecast = self.forecast_head(h)
        return forecast


def train_nbeats_extended(model, data, epochs=500, batch_size=32, lr=0.001, 
                         device='cpu', lookback=20, forecast_horizon=5,
                         early_stopping_patience=50):
    """
    Train N-BEATS with extended epochs and early stopping.
    
    Args:
        epochs: Number of training epochs (default 500)
        early_stopping_patience: Patience for early stopping (default 50)
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_losses = []
    best_loss = float('inf')
    patience_counter = 0
    
    # Create training windows
    windows = []
    targets = []
    for i in range(len(data) - lookback - forecast_horizon):
        windows.append(data[i:i+lookback])
        targets.append(data[i+lookback:i+lookback+forecast_horizon])
    
    windows = np.array(windows)
    targets = np.array(targets)
    n_batches = len(windows) // batch_size
    
    print(f'\n[N-BEATS] Training configuration:')
    print(f'  Epochs: {epochs}')
    print(f'  Batch size: {batch_size}')
    print(f'  Training samples: {len(windows)}')
    print(f'  Batches per epoch: {n_batches}')
    print(f'  Lookback: {lookback}')
    print(f'  Forecast horizon: {forecast_horizon}')
    print(f'  Early stopping patience: {early_stopping_patience}')
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(len(windows))
        
        epoch_loss = 0.0
        batch_count = 0
        
        # Mini-batch training
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            
            x_batch = torch.FloatTensor(windows[batch_indices]).to(device)
            y_batch = torch.FloatTensor(targets[batch_indices]).to(device)
            
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_epoch_loss = epoch_loss / batch_count
        train_losses.append(avg_epoch_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_epoch_loss:.8f}')
        
        # Early stopping
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'  Early stopping at epoch {epoch+1} (patience exceeded)')
                break
    
    print(f'✓ N-BEATS Training Complete')
    print(f'  Final loss: {train_losses[-1]:.8f}')
    print(f'  Best loss: {best_loss:.8f}')
    print(f'  Loss reduction: {(train_losses[0] - train_losses[-1])/train_losses[0]*100:.1f}%')
    
    return train_losses


def predict_nbeats(model, data, lookback=20, device='cpu'):
    """Generate N-BEATS predictions."""
    predictions = []
    
    with torch.no_grad():
        for i in range(len(data) - lookback):
            x = torch.FloatTensor(data[i:i+lookback]).unsqueeze(0).to(device)
            pred = model(x)
            predictions.append(float(pred[0, 0].cpu().numpy()))
    
    return np.array(predictions)


# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

TICKER = "BTC-USD"
START = "2022-01-01"
END = "2024-01-01"
TRAIN_FRAC = 0.8

FEE = 0.0005
KAPPA = 0.1
INITIAL_CASH = 100000.0
MAX_LEVERAGE = 2.0

TOTAL_UPDATES = 500
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('\n' + '='*70)
print('FINAL PROJECT - IMPROVED VERSION WITH EXTENDED N-BEATS TRAINING')
print('='*70)
print(f'\nStarted: {datetime.now().strftime("%H:%M:%S")}')
print(f'Device: {DEVICE}')

# ============================================================================
# PART 1: DATA LOADING
# ============================================================================

print('\n' + '='*70)
print('PART 1: LOADING AND PROCESSING DATA')
print('='*70)

df = yf.download(TICKER, start=START, end=END, progress=False)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df.columns = [c.lower() for c in df.columns]

# Add features
df['log_c'] = np.log(df['close'])
df['r'] = df['log_c'].diff()
df['mu_hat'] = df['r'].ewm(20).mean()
df['sigma_hat'] = df['r'].rolling(20).std()
df['r_lag1'] = df['r'].shift(1)
df = df.dropna().reset_index(drop=True)

split = int(TRAIN_FRAC * len(df))
df_train = df.iloc[:split].copy()
df_test = df.iloc[split:].copy()

print(f'✓ Loaded {len(df)} days (Train: {len(df_train)}, Test: {len(df_test)})')

# ============================================================================
# PART 2: IMPROVED N-BEATS FORECASTING (EXTENDED TRAINING)
# ============================================================================

print('\n' + '='*70)
print('PART 2: TRAINING IMPROVED N-BEATS FORECASTER (500 EPOCHS!)')
print('='*70)

returns_train = df_train['r'].values
returns_test = df_test['r'].values

print('\nCreating Improved N-BEATS...')
forecaster = ImprovedNBeats(
    lookback=20,
    forecast_horizon=5,
    hidden_size=64,
    num_blocks=3,
    dropout=0.1
).to(DEVICE)

print('Training N-BEATS with EXTENDED epochs (500 epochs with early stopping)...')
train_losses = train_nbeats_extended(
    forecaster,
    returns_train,
    epochs=500,
    batch_size=32,
    lr=0.001,
    device=DEVICE,
    lookback=20,
    forecast_horizon=5,
    early_stopping_patience=50
)

# Generate forecasts
print('\nGenerating forecasts from trained N-BEATS...')
fc_train = predict_nbeats(forecaster, returns_train, lookback=20, device=DEVICE)
fc_test = predict_nbeats(forecaster, returns_test, lookback=20, device=DEVICE)

print(f'✓ Forecasts generated')
print(f'  Train forecasts: {len(fc_train)} values')
print(f'  Test forecasts: {len(fc_test)} values')

# Add to dataframes
df_train['forecast'] = 0.0
df_test['forecast'] = 0.0

for i, fc in enumerate(fc_train):
    if 20 + i < len(df_train):
        df_train.at[20 + i, 'forecast'] = fc

for i, fc in enumerate(fc_test):
    if 20 + i < len(df_test):
        df_test.at[20 + i, 'forecast'] = fc

print('✓ Forecasts added to dataframes')

# ============================================================================
# PART 3: TRAIN PPO AGENTS
# ============================================================================

def train_agent(df_data, with_fc=True, num_updates=100):
    name = "WITH" if with_fc else "WITHOUT"
    print(f'\n[PPO] Training {name} forecast ({num_updates} episodes)...')

    df = df_data.copy().reset_index(drop=True)
    if not with_fc:
        df = df.drop(columns=['forecast'])
    
    # Create a temp env to get the actual obs dimension
    temp_env = EnhancedTradingEnv(df, fee=FEE, kappa=KAPPA, initial_cash=INITIAL_CASH, max_leverage=MAX_LEVERAGE)
    if not with_fc:
        temp_env.feature_cols = [c for c in temp_env.feature_cols if c != 'forecast']
    obs_dim = temp_env.observation_space.shape[0]
    print(f'[PPO] Actual obs dim: {obs_dim}')
    
    trainer = PPOTrainer(obs_dim=obs_dim, act_dim=1, device=DEVICE, lr=3e-4)

    from ppo_trainer import squash_action

    returns = []
    for update in range(num_updates):
        # Create fresh environment for each episode
        env = EnhancedTradingEnv(df, fee=FEE, kappa=KAPPA, initial_cash=INITIAL_CASH, max_leverage=MAX_LEVERAGE)

        if not with_fc:
            env.feature_cols = [c for c in env.feature_cols if c != 'forecast']

        obs, _ = env.reset(seed=SEED + update)
        obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

        done = False
        ep_ret = 0
        steps = 0
        max_steps = min(len(df) - 2, 500)

        while not done and steps < max_steps:
            with torch.no_grad():
                dist, _ = trainer.model(obs.unsqueeze(0))
                a = squash_action(dist.sample(), -MAX_LEVERAGE, MAX_LEVERAGE)

            try:
                obs_next, rew, term, trunc, _ = env.step(a.cpu().numpy()[0])
                done = term or trunc
                obs = torch.tensor(obs_next, dtype=torch.float32, device=DEVICE)
                ep_ret += rew
                steps += 1
            except (KeyError, IndexError, ValueError):
                done = True

        returns.append(ep_ret)
        if (update + 1) % 20 == 0:
            mean = np.mean(returns[-20:]) if len(returns) >= 20 else np.mean(returns)
            print(f'  Episode {update+1:3d}/{num_updates} | Mean reward: {mean:10.2f}')

    return trainer, returns


print('\n' + '='*70)
print('PART 3: TRAINING PPO AGENTS')
print('='*70)

trainer_with, returns_with = train_agent(df_train, with_fc=True, num_updates=100)
trainer_without, returns_without = train_agent(df_train, with_fc=False, num_updates=100)

# ============================================================================
# PART 4: EVALUATION
# ============================================================================

print('\n' + '='*70)
print('PART 4: EVALUATION ON TEST SET')
print('='*70)

evaluator = StrategyEvaluator()

print(f'\nEvaluating WITH forecast (5 episodes)...')
m_with = []
for i in range(5):
    env = EnhancedTradingEnv(df_test, FEE, KAPPA, INITIAL_CASH, MAX_LEVERAGE)
    m, _ = evaluator.evaluate_episode(env, trainer_with.model, DEVICE, deterministic=True)
    m_with.append(m)
    print(f'  E{i+1}: Return={m["cumulative_return"]:.4f}, Sharpe={m["sharpe_ratio"]:.4f}')

print(f'\nEvaluating WITHOUT forecast (5 episodes)...')
df_test_no = df_test.drop(columns=['forecast'])
m_without = []
for i in range(5):
    env = EnhancedTradingEnv(df_test_no, FEE, KAPPA, INITIAL_CASH, MAX_LEVERAGE)
    m, _ = evaluator.evaluate_episode(env, trainer_without.model, DEVICE, deterministic=True)
    m_without.append(m)
    print(f'  E{i+1}: Return={m["cumulative_return"]:.4f}, Sharpe={m["sharpe_ratio"]:.4f}')

# ============================================================================
# PART 5: RESULTS & COMPARISON
# ============================================================================

print('\n' + '='*70)
print('PART 5: RESULTS COMPARISON')
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

results.to_csv('final_project_results_improved.csv', index=False)
print('\n✓ Saved: final_project_results_improved.csv')

# ============================================================================
# PART 6: PLOT TRAINING LOSS
# ============================================================================

print('\nGenerating N-BEATS training visualization...')

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train_losses, linewidth=2, color='#2E86AB')
ax.fill_between(range(len(train_losses)), train_losses, alpha=0.2, color='#2E86AB')
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
ax.set_title('N-BEATS Training Loss Over 500 Epochs', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# Add annotations
ax.text(0.5, 0.95, f'Initial Loss: {train_losses[0]:.8f}\nFinal Loss: {train_losses[-1]:.8f}\nReduction: {(train_losses[0]-train_losses[-1])/train_losses[0]*100:.1f}%',
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('00_NBBeats_Training_Loss.png', dpi=150, bbox_inches='tight')
print('✓ Saved: 00_NBBeats_Training_Loss.png')
plt.close()

# ============================================================================
# PART 7: ANALYSIS
# ============================================================================

print('\n' + '='*70)
print('PART 6: ANALYSIS')
print('='*70)

print(f'\nDoes forecast help?')
print(f'  Return: {ret_w:.4f} vs {ret_wo:.4f} ({ret_w-ret_wo:+.4f})')
print(f'  Sharpe: {sharpe_w:.4f} vs {sharpe_wo:.4f} ({sharpe_w-sharpe_wo:+.4f})')

if sharpe_w > sharpe_wo:
    print(f'\n✅ YES - Forecast improves risk-adjusted returns')
    print(f'  Improvement: {(sharpe_w-sharpe_wo)/abs(sharpe_wo)*100:+.1f}%')
else:
    print(f'\n⚠️  LIMITED - Baseline performs comparably or better')

print('\n✅ PROJECT COMPLETE!')
print(f'Finished: {datetime.now().strftime("%H:%M:%S")}')
print('\n' + '='*70)

