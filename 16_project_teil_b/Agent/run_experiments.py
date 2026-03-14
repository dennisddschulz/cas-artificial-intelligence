#!/usr/bin/env python3
"""
COMPLETE WORKING SCRIPT - Forecast-Augmented RL Trading Experiments
Run this directly: python3 run_experiments.py
"""

import os
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

# ============================================================
# 1. DATA LOADING & PREPARATION
# ============================================================

def load_and_prepare_data(ticker="^GSPC", start="2020-01-01", end="2023-12-31"):
    """Load market data and prepare features"""
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df.rename(columns={'Close': 'close', 'High': 'high', 'Low': 'low', 'Volume': 'volume'})
    df['close'] = df['close'].astype(float)
    
    print(f"✓ Loaded {len(df)} days of {ticker}")
    print(f"  Period: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    return df.reset_index(drop=True)

# ============================================================
# 2. LSTM FORECASTER
# ============================================================

class LSTMForecaster(nn.Module):
    """LSTM model for binary price movement prediction"""
    
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
        
        x = x[:, -1, :]  # Take last timestep
        
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        x = self.output(x)
        x = self.sigmoid(x)
        
        return x

def train_lstm_forecaster(df, lookback=30, forecast_horizon=5, epochs=50):
    """Train LSTM to predict price movements"""
    print("\n" + "="*70)
    print("TRAINING LSTM FORECASTER")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Normalize prices
    scaler = StandardScaler()
    prices = df['close'].values.reshape(-1, 1)
    prices_scaled = scaler.fit_transform(prices)
    
    # Create sequences
    X, y = [], []
    for i in range(len(prices_scaled) - lookback - forecast_horizon):
        X.append(prices_scaled[i:i+lookback])
        # Target: 1 if price goes up, 0 if goes down
        future_price = prices_scaled[i+lookback+forecast_horizon]
        current_price = prices_scaled[i+lookback]
        y.append(1.0 if future_price > current_price else 0.0)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✓ Created {len(X)} training sequences (lookback={lookback}, horizon={forecast_horizon})")
    
    # Train/val split
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Model
    model = LSTMForecaster(lookback=lookback).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # Training loop
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
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.from_numpy(X_val).float().to(device))
            val_loss = criterion(val_pred, torch.from_numpy(y_val).float().unsqueeze(1).to(device))
            val_acc = ((val_pred.cpu().numpy().squeeze() > 0.5) == y_val).mean()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss/len(X_train):.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
    
    print(f"✓ LSTM training complete. Final validation accuracy: {val_acc:.2%}")
    
    return model, scaler, device

def generate_forecasts(model, df, scaler, lookback=30, device='cpu'):
    """Generate forecast probabilities for entire dataset"""
    prices_scaled = scaler.transform(df['close'].values.reshape(-1, 1))
    
    forecasts = np.zeros(len(df))
    
    model.eval()
    with torch.no_grad():
        for i in range(lookback, len(df)):
            seq = torch.from_numpy(prices_scaled[i-lookback:i]).float().unsqueeze(0).to(device)
            pred = model(seq).cpu().numpy()[0, 0]
            forecasts[i] = pred
    
    # Fill first lookback values with mean
    forecasts[:lookback] = np.mean(forecasts[lookback:lookback+10])
    
    return forecasts

# ============================================================
# 3. TRADING ENVIRONMENT
# ============================================================

class TradingEnv(gym.Env):
    """Trading environment with configurable state"""
    
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
        
        # State space
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
        
        # Momentum
        if self.t >= 5:
            momentum = (price - self.df['close'].iloc[self.t-5]) / self.df['close'].iloc[self.t-5]
        else:
            momentum = 0.0
        
        # Volatility
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
        
        # Transaction costs
        pos_change = target_pos - self.position
        cost = abs(pos_change) * current_price * (self.fee + self.kappa)
        
        self.cash -= cost
        self.position = target_pos
        self.t += 1
        
        # P&L
        if self.t < len(self.df):
            next_price = self.df['close'].iloc[self.t]
            price_move = next_price - current_price
            pnl = self.position * price_move
            self.cash += pnl
        
        current_price = self.df['close'].iloc[self.t] if self.t < len(self.df) else self.df['close'].iloc[-1]
        self.equity = self.cash + self.position * current_price
        
        # Reward
        pnl_reward = (self.equity / self.initial_equity - 1.0) * self.reward_config['pnl_w']
        drawdown = 1.0 - self.equity / self.peak_equity if self.peak_equity > 0 else 0.0
        risk_reward = -drawdown * self.reward_config['risk_w']
        cost_reward = -cost / self.initial_equity * self.reward_config['cost_w']
        
        reward = pnl_reward + risk_reward + cost_reward
        self.peak_equity = max(self.peak_equity, self.equity)
        
        done = self.equity <= 0 or self.t >= len(self.df) - 1
        
        return self._get_obs(), float(reward), done, False, {}

# ============================================================
# 4. EXPERIMENTS
# ============================================================

def run_forecast_only(df_test, forecast_signal):
    """Baseline: Simple rule-based forecast strategy"""
    print("\n" + "="*70)
    print("EXPERIMENT 1: FORECAST-ONLY BASELINE")
    print("="*70)
    
    wandb.init(project="forecast-rl-trading", group="baseline", name="forecast-only", mode='offline', reinit=True)
    
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
    
    return metrics, equity

def run_ppo_experiment(df_train, df_test, name, with_forecast=False, forecast_signal=None, reward_cfg=None):
    """Train and evaluate PPO agent"""
    print("\n" + "="*70)
    print(f"EXPERIMENT: {name}")
    print("="*70)
    
    wandb.init(project="forecast-rl-trading", group="ppo-variants", name=name, mode='offline', reinit=True)
    
    # Create environment and train
    env = TradingEnv(
        df_train,
        forecast_signal=forecast_signal[:len(df_train)] if with_forecast else None,
        reward_config=reward_cfg or {'pnl_w': 1.0, 'risk_w': 0.0, 'cost_w': 0.0}
    )
    
    model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=512, batch_size=32, 
                n_epochs=5, ent_coef=0.01, verbose=0)
    model.learn(total_timesteps=20000)  # Reduced for quick testing
    
    print(f"✓ Training complete")
    
    # Evaluate
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
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
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

# ============================================================
# 5. MAIN EXECUTION
# ============================================================

def main():
    print("\n" + "="*80)
    print("FORECAST-AUGMENTED RL TRADING SYSTEM - COMPLETE EXPERIMENT SUITE")
    print("="*80)
    
    # Load data
    df = load_and_prepare_data()
    
    # Train LSTM
    model, scaler, device = train_lstm_forecaster(df, lookback=30, forecast_horizon=5, epochs=30)
    forecast_signal = generate_forecasts(model, df, scaler, device=device)
    
    # Split data
    split_idx = int(0.8 * len(df))
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_test = df.iloc[split_idx:].reset_index(drop=True)
    
    print(f"\n✓ Data split: {len(df_train)} train, {len(df_test)} test")
    
    # Run experiments
    results = {}
    
    # 1. Forecast-Only
    metrics_fo, equity_fo = run_forecast_only(df_test, forecast_signal[split_idx:])
    results['Forecast-Only'] = metrics_fo
    
    # 2. PPO Without Forecast
    metrics_ppono = run_ppo_experiment(
        df_train, df_test, 
        "PPO-Without-Forecast",
        with_forecast=False
    )
    results['PPO-Without-Forecast'] = metrics_ppono
    
    # 3. PPO With Forecast
    metrics_ppowf = run_ppo_experiment(
        df_train, df_test,
        "PPO-With-Forecast",
        with_forecast=True,
        forecast_signal=forecast_signal
    )
    results['PPO-With-Forecast'] = metrics_ppowf
    
    # 4. PPO With Different Rewards
    reward_configs = {
        'pnl-risk': {'pnl_w': 1.0, 'risk_w': 0.1, 'cost_w': 0.0},
        'pnl-cost': {'pnl_w': 1.0, 'risk_w': 0.0, 'cost_w': 0.1},
        'balanced': {'pnl_w': 1.0, 'risk_w': 0.1, 'cost_w': 0.1},
    }
    
    for reward_name, reward_cfg in reward_configs.items():
        metrics_reward = run_ppo_experiment(
            df_train, df_test,
            f"PPO-With-Forecast-{reward_name}",
            with_forecast=True,
            forecast_signal=forecast_signal,
            reward_cfg=reward_cfg
        )
        results[f"PPO-With-Forecast-{reward_name}"] = metrics_reward
    
    # Print results
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    results_df = pd.DataFrame(results).T
    results_df['return'] = results_df['return'].apply(lambda x: f"{x*100:.2f}%")
    results_df['sharpe'] = results_df['sharpe'].apply(lambda x: f"{x:.4f}")
    results_df['volatility'] = results_df['volatility'].apply(lambda x: f"{x*100:.2f}%")
    results_df['maxdd'] = results_df['maxdd'].apply(lambda x: f"{x*100:.2f}%")
    results_df['turnover'] = results_df['turnover'].apply(lambda x: f"{x:.4f}")
    
    print("\n" + results_df.to_string())
    
    # Save results
    results_df.to_csv('results_comparison.csv')
    print("\n✓ Results saved to results_comparison.csv")
    
    # Analysis
    print("\n" + "="*80)
    print("ANSWER TO RESEARCH QUESTION")
    print("="*80)
    
    ppo_with = results['PPO-With-Forecast']['return']
    ppo_without = results['PPO-Without-Forecast']['return']
    improvement = (ppo_with - ppo_without) / abs(ppo_without) * 100
    
    print(f"\nQ: Does forecast improve RL performance?")
    print(f"A: {('YES' if improvement > 5 else 'NO' if improvement < -5 else 'NEUTRAL')}")
    print(f"\n   PPO with Forecast:    {ppo_with*100:.2f}%")
    print(f"   PPO without Forecast: {ppo_without*100:.2f}%")
    print(f"   Improvement:          {improvement:+.1f}%")
    
    if improvement > 5:
        print(f"\n✓ FORECAST HELPS! RL agent benefits from price predictions.")
    elif improvement < -5:
        print(f"\n✗ FORECAST HURTS! RL agent performs worse with forecast signal.")
    else:
        print(f"\n~ FORECAST NEUTRAL! Signal has minimal impact on RL performance.")
    
    print("\n" + "="*80)
    print("✓ EXPERIMENT SUITE COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

