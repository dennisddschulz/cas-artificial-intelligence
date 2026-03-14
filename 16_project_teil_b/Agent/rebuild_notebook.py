#!/usr/bin/env python3
"""
Convert run_experiments.py to notebook cells and add to notebook
"""
import json

notebook_path = '/home/isc-den/cas-artificial-intelligence/16_project_teil_b/Agent/Project_Part_2_Final_Architecture.ipynb'

# The working code split into functional cells
cells_to_add = [
    {
        'title': 'Setup & Imports',
        'code': '''import os
os.environ['WANDB_MODE'] = 'offline'
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

import yfinance as yf
import wandb
from datetime import datetime

print("✓ All imports successful")
'''
    },
    {
        'title': 'Data Loading',
        'code': '''def load_and_prepare_data(ticker="^GSPC", start="2020-01-01", end="2023-12-31"):
    """Load market data"""
    print("\\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df.rename(columns={'Close': 'close', 'High': 'high', 'Low': 'low', 'Volume': 'volume'})
    df['close'] = df['close'].astype(float)
    
    print(f"✓ Loaded {len(df)} days of {ticker}")
    print(f"  Period: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    return df.reset_index(drop=True)

# Load data
df = load_and_prepare_data()
print(f"\\n✓ Data shape: {df.shape}")
print(f"✓ Columns: {df.columns.tolist()}")
'''
    },
    {
        'title': 'LSTM Forecaster Model',
        'code': '''class LSTMForecaster(nn.Module):
    """LSTM for price prediction"""
    def __init__(self, lookback=30):
        super().__init__()
        self.lookback = lookback
        
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.dense1 = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(0.1)
        self.output = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = x[:, -1, :]
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x

print("✓ LSTMForecaster class defined")
'''
    },
    {
        'title': 'Train LSTM',
        'code': '''def train_lstm_forecaster(df, lookback=30, forecast_horizon=5, epochs=30):
    """Train LSTM forecaster"""
    print("\\n" + "="*70)
    print("TRAINING LSTM FORECASTER")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    scaler = StandardScaler()
    prices = df['close'].values.reshape(-1, 1)
    prices_scaled = scaler.fit_transform(prices)
    
    # Create sequences
    X, y = [], []
    for i in range(len(prices_scaled) - lookback - forecast_horizon):
        X.append(prices_scaled[i:i+lookback])
        future_price = prices_scaled[i+lookback+forecast_horizon]
        current_price = prices_scaled[i+lookback]
        y.append(1.0 if future_price > current_price else 0.0)
    
    X, y = np.array(X), np.array(y)
    print(f"✓ Created {len(X)} sequences")
    
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    model = LSTMForecaster(lookback=lookback).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    batch_size = 32
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_X = torch.from_numpy(X_train[i:i+batch_size]).float().to(device)
            batch_y = torch.from_numpy(y_train[i:i+batch_size]).float().unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.from_numpy(X_val).float().to(device))
            val_loss = criterion(val_pred, torch.from_numpy(y_val).float().unsqueeze(1).to(device))
            val_acc = ((val_pred.cpu().numpy().squeeze() > 0.5) == y_val).mean()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss/len(X_train):.4f}, Val Acc={val_acc:.2%}")
    
    print(f"✓ LSTM training complete")
    return model, scaler, device

# Train LSTM
model, scaler, device = train_lstm_forecaster(df, lookback=30, forecast_horizon=5, epochs=30)
'''
    },
    {
        'title': 'Generate Forecasts',
        'code': '''def generate_forecasts(model, df, scaler, lookback=30, device='cpu'):
    """Generate forecast probabilities"""
    prices_scaled = scaler.transform(df['close'].values.reshape(-1, 1))
    forecasts = np.zeros(len(df))
    
    model.eval()
    with torch.no_grad():
        for i in range(lookback, len(df)):
            seq = torch.from_numpy(prices_scaled[i-lookback:i]).float().unsqueeze(0).to(device)
            pred = model(seq).cpu().numpy()[0, 0]
            forecasts[i] = pred
    
    forecasts[:lookback] = np.mean(forecasts[lookback:lookback+10])
    return forecasts

forecast_signal = generate_forecasts(model, df, scaler, device=device)
print(f"\\n✓ Generated {len(forecast_signal)} forecast signals")
print(f"  Mean forecast: {np.mean(forecast_signal):.3f}")
print(f"  Std forecast: {np.std(forecast_signal):.3f}")

# Split data
split_idx = int(0.8 * len(df))
df_train = df.iloc[:split_idx].reset_index(drop=True)
df_test = df.iloc[split_idx:].reset_index(drop=True)
print(f"\\n✓ Data split: {len(df_train)} train, {len(df_test)} test")
'''
    },
    {
        'title': 'Trading Environment',
        'code': '''class TradingEnv(gym.Env):
    """Trading environment"""
    
    def __init__(self, df, initial_equity=100000, fee=0.0005, kappa=0.1, 
                 max_leverage=1.0, forecast_signal=None, reward_config=None):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_equity = initial_equity
        self.fee = fee
        self.kappa = kappa
        self.max_leverage = max_leverage
        self.forecast_signal = forecast_signal
        self.reward_config = reward_config or {'pnl_w': 1.0, 'risk_w': 0.0, 'cost_w': 0.0}
        
        include_forecast = forecast_signal is not None
        obs_size = 5 if include_forecast else 4
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Box(low=-max_leverage, high=max_leverage, shape=(1,), dtype=np.float32)
        self.reset()
    
    def reset(self):
        self.t = 0
        self.equity = self.initial_equity
        self.cash = self.initial_equity
        self.position = 0.0
        self.peak_equity = self.initial_equity
        return self._get_obs(), {}
    
    def _get_obs(self):
        if self.t >= len(self.df):
            self.t = len(self.df) - 1
        
        price = self.df['close'].iloc[self.t]
        momentum = (price - self.df['close'].iloc[self.t-5]) / self.df['close'].iloc[self.t-5] if self.t >= 5 else 0.0
        
        if self.t >= 20:
            returns = self.df['close'].iloc[self.t-20:self.t].pct_change().dropna()
            volatility = np.std(returns) if len(returns) > 0 else 0.01
        else:
            volatility = 0.01
        
        obs = [self.position, self.cash/self.equity if self.equity > 0 else 0.5, momentum, volatility]
        if self.forecast_signal is not None and self.t < len(self.forecast_signal):
            obs.append(float(self.forecast_signal[self.t]))
        
        return np.array(obs, dtype=np.float32)
    
    def step(self, action):
        target_pos = float(np.clip(action[0], -self.max_leverage, self.max_leverage))
        if self.t >= len(self.df) - 1:
            return self._get_obs(), 0.0, True, False, {}
        
        current_price = self.df['close'].iloc[self.t]
        pos_change = target_pos - self.position
        cost = abs(pos_change) * current_price * (self.fee + self.kappa)
        
        self.cash -= cost
        self.position = target_pos
        self.t += 1
        
        if self.t < len(self.df):
            next_price = self.df['close'].iloc[self.t]
            pnl = self.position * (next_price - current_price)
            self.cash += pnl
        
        current_price = self.df['close'].iloc[self.t] if self.t < len(self.df) else self.df['close'].iloc[-1]
        self.equity = self.cash + self.position * current_price
        
        pnl_reward = (self.equity / self.initial_equity - 1.0) * self.reward_config['pnl_w']
        drawdown = 1.0 - self.equity / self.peak_equity if self.peak_equity > 0 else 0.0
        risk_reward = -drawdown * self.reward_config['risk_w']
        cost_reward = -cost / self.initial_equity * self.reward_config['cost_w']
        
        reward = pnl_reward + risk_reward + cost_reward
        self.peak_equity = max(self.peak_equity, self.equity)
        
        done = self.equity <= 0 or self.t >= len(self.df) - 1
        return self._get_obs(), float(reward), done, False, {}

print("✓ TradingEnv class defined")
'''
    },
    {
        'title': 'Experiment 1: Forecast-Only Baseline',
        'code': '''def run_forecast_only(df_test, forecast_signal):
    """Forecast-only baseline strategy"""
    print("\\n" + "="*70)
    print("EXPERIMENT 1: FORECAST-ONLY BASELINE")
    print("="*70)
    
    wandb.init(project="forecast-rl-trading", group="baseline", name="forecast-only", 
               mode='offline', reinit=True)
    
    equity = [100000]
    positions = [0.0]
    returns = []
    
    for t in range(1, len(df_test)):
        pos = 1.0 if forecast_signal[t] > 0.5 else -1.0
        price_move = df_test['close'].iloc[t] - df_test['close'].iloc[t-1]
        pnl = positions[-1] * price_move
        new_equity = equity[-1] + pnl
        
        equity.append(new_equity)
        positions.append(pos)
        if len(equity) > 1:
            returns.append((equity[-1] - equity[-2]) / equity[-2])
    
    equity = np.array(equity)
    returns = np.array(returns)
    
    metrics = {
        'return': (equity[-1] - equity[0]) / equity[0],
        'sharpe': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0.0,
        'volatility': np.std(returns) * np.sqrt(252),
        'maxdd': float(np.min((equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity))),
        'turnover': np.sum(np.abs(np.diff(positions))) / len(positions),
    }
    
    print(f"✓ Return: {metrics['return']*100:.2f}%")
    print(f"✓ Sharpe: {metrics['sharpe']:.4f}")
    print(f"✓ Max DD: {metrics['maxdd']*100:.2f}%")
    
    wandb.log(metrics)
    wandb.finish()
    return metrics

forecast_only_metrics = run_forecast_only(df_test, forecast_signal[split_idx:])
'''
    },
    {
        'title': 'Experiment 2 & 3: PPO Experiments',
        'code': '''def run_ppo_experiment(df_train, df_test, name, with_forecast=False, 
                        forecast_signal=None, reward_cfg=None):
    """Train and evaluate PPO agent"""
    print("\\n" + "="*70)
    print(f"EXPERIMENT: {name}")
    print("="*70)
    
    wandb.init(project="forecast-rl-trading", group="ppo-variants", name=name, 
               mode='offline', reinit=True)
    
    env = TradingEnv(
        df_train,
        forecast_signal=forecast_signal[:len(df_train)] if with_forecast else None,
        reward_config=reward_cfg or {'pnl_w': 1.0, 'risk_w': 0.0, 'cost_w': 0.0}
    )
    
    model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=512, batch_size=32, 
                n_epochs=5, ent_coef=0.01, verbose=0)
    model.learn(total_timesteps=20000)
    print(f"✓ Training complete")
    
    env_test = TradingEnv(
        df_test,
        forecast_signal=forecast_signal[len(df_train):] if with_forecast else None,
        reward_config=reward_cfg or {'pnl_w': 1.0, 'risk_w': 0.0, 'cost_w': 0.0}
    )
    
    obs, _ = env_test.reset()
    equity = [env_test.equity]
    positions = [env_test.position]
    returns = []
    
    done = False
    while not done and env_test.t < len(df_test) - 1:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, _, _ = env_test.step(action)
        equity.append(env_test.equity)
        positions.append(env_test.position)
        if len(equity) > 1:
            returns.append((equity[-1] - equity[-2]) / equity[-2])
        done = terminated
    
    equity = np.array(equity)
    returns = np.array(returns)
    
    metrics = {
        'return': (equity[-1] - equity[0]) / equity[0],
        'sharpe': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0.0,
        'volatility': np.std(returns) * np.sqrt(252),
        'maxdd': float(np.min((equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity))),
        'turnover': np.sum(np.abs(np.diff(positions))) / len(positions),
    }
    
    print(f"✓ Return: {metrics['return']*100:.2f}%")
    print(f"✓ Sharpe: {metrics['sharpe']:.4f}")
    print(f"✓ Max DD: {metrics['maxdd']*100:.2f}%")
    
    wandb.log(metrics)
    wandb.finish()
    return metrics

# Run PPO experiments
ppo_without = run_ppo_experiment(df_train, df_test, "PPO-Without-Forecast", 
                                 with_forecast=False)

ppo_with = run_ppo_experiment(df_train, df_test, "PPO-With-Forecast", 
                              with_forecast=True, forecast_signal=forecast_signal)
'''
    },
    {
        'title': 'Experiment 4: Reward Variations',
        'code': '''reward_configs = {
    'pnl-risk': {'pnl_w': 1.0, 'risk_w': 0.1, 'cost_w': 0.0},
    'pnl-cost': {'pnl_w': 1.0, 'risk_w': 0.0, 'cost_w': 0.1},
    'balanced': {'pnl_w': 1.0, 'risk_w': 0.1, 'cost_w': 0.1},
}

reward_results = {}
for reward_name, reward_cfg in reward_configs.items():
    metrics = run_ppo_experiment(
        df_train, df_test,
        f"PPO-With-Forecast-{reward_name}",
        with_forecast=True,
        forecast_signal=forecast_signal,
        reward_cfg=reward_cfg
    )
    reward_results[reward_name] = metrics

print("\\n✓ All reward experiments complete")
'''
    },
    {
        'title': 'Results Summary & Analysis',
        'code': '''print("\\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)

# Collect all results
all_results = {
    'Forecast-Only': forecast_only_metrics,
    'PPO-Without-Forecast': ppo_without,
    'PPO-With-Forecast': ppo_with,
}
all_results.update({f'PPO-{k}': v for k, v in reward_results.items()})

# Create results table
results_df = pd.DataFrame(all_results).T
results_df['return_pct'] = results_df['return'].apply(lambda x: f"{x*100:.2f}%")
results_df['sharpe_fmt'] = results_df['sharpe'].apply(lambda x: f"{x:.4f}")
results_df['volatility_pct'] = results_df['volatility'].apply(lambda x: f"{x*100:.2f}%")
results_df['maxdd_pct'] = results_df['maxdd'].apply(lambda x: f"{x*100:.2f}%")
results_df['turnover_fmt'] = results_df['turnover'].apply(lambda x: f"{x:.4f}")

display_df = results_df[['return_pct', 'sharpe_fmt', 'volatility_pct', 'maxdd_pct', 'turnover_fmt']]
display_df.columns = ['Return', 'Sharpe', 'Volatility', 'Max DD', 'Turnover']

print("\\n" + display_df.to_string())

# Save results
results_df.to_csv('results_comparison.csv')
print("\\n✓ Results saved to results_comparison.csv")

# Answer research question
print("\\n" + "="*80)
print("RESEARCH QUESTION: Does forecast improve RL performance?")
print("="*80)

ppo_with_ret = all_results['PPO-With-Forecast']['return']
ppo_without_ret = all_results['PPO-Without-Forecast']['return']
improvement = (ppo_with_ret - ppo_without_ret) / abs(ppo_without_ret) * 100

print(f"\\nPPO with Forecast:    {ppo_with_ret*100:>8.2f}%")
print(f"PPO without Forecast: {ppo_without_ret*100:>8.2f}%")
print(f"Improvement:          {improvement:>8.1f}%")

if improvement > 5:
    answer = "✓ YES - Forecast HELPS"
    explanation = "RL agent benefits from price movement predictions"
elif improvement < -5:
    answer = "✗ NO - Forecast HURTS"
    explanation = "RL agent performs worse with forecast signal"
else:
    answer = "~ NEUTRAL"
    explanation = "Forecast signal has minimal impact"

print(f"\\n{answer}")
print(f"Explanation: {explanation}")
print("\\n" + "="*80)
print("✓ EXPERIMENT SUITE COMPLETE")
print("="*80)
'''
    }
]

# Load existing notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Remove all existing cells (start fresh)
nb['cells'] = []

# Add working cells
for cell_data in cells_to_add:
    cell = {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': cell_data['code'].split('\n')
    }
    nb['cells'].append(cell)

# Save notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✓ Notebook rebuilt with working code")
print(f"✓ Added {len(cells_to_add)} functional cells")
print(f"✓ Total cells: {len(nb['cells'])}")

