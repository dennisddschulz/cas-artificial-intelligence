#!/usr/bin/env python3
"""
COMPLETE EXPERIMENTAL FRAMEWORK
Forecast-Augmented RL Trading System
Does LSTM Forecast Improve PPO Performance?

Run this script to execute all experiments:
    python3 main.py
"""

import os
os.environ['WANDB_MODE'] = 'offline'
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import yfinance as yf
import wandb
import json
import pickle
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    'data': {
        'ticker': 'BTC-USD',        # Bitcoin (from original notebook)
        'start': '2018-01-01',      # Longer history for Bitcoin
        'end': None,
        'train_split': 0.60,        # 60% train (from notebook)
        'val_split': 0.20,          # 20% val
        'test_split': 0.20,         # 20% test
    },
    'lstm': {
        'lookback': 20,             # 20-day window (from notebook)
        'forecast_horizon': 5,      # 5-day forecast (from notebook)
        'epochs': 100,              # Increased epochs (from notebook)
        'batch_size': 32,
        'learning_rate': 1e-3,      # Learning rate (from notebook)
        'hidden_dim': 64,           # LSTM hidden dim
        'num_layers': 2,            # 2 LSTM layers
        'dropout': 0.2,
    },
    'trading': {
        'initial_equity': 100000.0,
        'fee': 0.0005,
        'kappa': 0.1,
        'max_leverage': 1.0,
    },
    'ppo': {
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'total_timesteps': 100000,
        'ent_coef': 0.01,
        'gamma': 0.99,
        'gae_lambda': 0.95,
    },
    'rewards': {
        'simple': {'pnl_w': 1.0, 'risk_w': 0.0, 'cost_w': 0.0},
        'with_risk': {'pnl_w': 1.0, 'risk_w': 0.1, 'cost_w': 0.0},
        'with_cost': {'pnl_w': 1.0, 'risk_w': 0.0, 'cost_w': 0.1},
        'balanced': {'pnl_w': 1.0, 'risk_w': 0.1, 'cost_w': 0.1},
    }
}

# ============================================================
# SECTION 1: DATA LOADING
# ============================================================

def load_data():
    """Load market data from Yahoo Finance"""
    print("\n" + "="*80)
    print("SECTION 1: LOADING DATA")
    print("="*80)
    
    ticker = CONFIG['data']['ticker']
    start = CONFIG['data']['start']
    end = CONFIG['data']['end']
    
    print(f"\nDownloading {ticker} ({start} to {end})...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    
    # Handle MultiIndex columns (yfinance sometimes returns this)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()
    
    # Ensure close column exists
    if 'close' not in df.columns:
        if 'adj close' in df.columns:
            df['close'] = df['adj close']
        else:
            raise ValueError("No 'close' or 'adj close' column found in downloaded data")
    
    # Ensure numeric type
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    
    # Remove any NaN rows
    df = df.dropna(subset=['close'])
    
    print(f"✓ Loaded {len(df)} trading days")
    min_price = float(df['close'].min())
    max_price = float(df['close'].max())
    print(f"  Price range: ${min_price:.2f} - ${max_price:.2f}")
    
    return df.reset_index(drop=True)

# ============================================================
# SECTION 2: LSTM FORECASTER
# ============================================================

class LSTMForecaster(nn.Module):
    """LSTM model for binary price prediction"""
    
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

def train_lstm(df):
    """Train LSTM forecaster"""
    print("\n" + "="*80)
    print("SECTION 2: TRAINING LSTM FORECASTER")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    lookback = CONFIG['lstm']['lookback']
    horizon = CONFIG['lstm']['forecast_horizon']
    
    # Prepare data
    scaler = StandardScaler()
    prices = df['close'].values.reshape(-1, 1)
    prices_scaled = scaler.fit_transform(prices)
    
    # Create sequences
    X, y = [], []
    for i in range(len(prices_scaled) - lookback - horizon):
        X.append(prices_scaled[i:i+lookback])
        future_price = prices_scaled[i+lookback+horizon]
        current_price = prices_scaled[i+lookback]
        y.append(1.0 if future_price > current_price else 0.0)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Created {len(X)} training sequences")
    
    # Split
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Model
    model = LSTMForecaster(lookback=lookback).to(device)
    optimizer = Adam(model.parameters(), lr=CONFIG['lstm']['learning_rate'])
    criterion = nn.BCELoss()
    
    # Training
    batch_size = CONFIG['lstm']['batch_size']
    epochs = CONFIG['lstm']['epochs']
    
    best_val_acc = 0
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
        
        best_val_acc = max(best_val_acc, val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss/len(X_train):.4f}, Val Acc={val_acc:.2%}")
    
    print(f"✓ LSTM training complete. Best val accuracy: {best_val_acc:.2%}")
    
    return model, scaler, device

def generate_forecasts(model, df, scaler, device):
    """Generate forecast probabilities"""
    lookback = CONFIG['lstm']['lookback']
    prices_scaled = scaler.transform(df['close'].values.reshape(-1, 1))
    
    forecasts = np.zeros(len(df))
    model.eval()
    
    with torch.no_grad():
        for i in range(lookback, len(df)):
            seq = torch.from_numpy(prices_scaled[i-lookback:i]).float().unsqueeze(0).to(device)
            pred = model(seq).cpu().numpy()[0, 0]
            forecasts[i] = pred
    
    # Fill beginning with mean
    forecasts[:lookback] = np.mean(forecasts[lookback:lookback+10])
    
    return forecasts

# ============================================================
# SECTION 3: TRADING ENVIRONMENT
# ============================================================

class TradingEnv(gym.Env):
    """Trading environment for RL"""
    
    def __init__(self, df, forecast_signal=None, reward_config=None, is_training=True):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.forecast_signal = forecast_signal
        self.reward_config = reward_config or CONFIG['rewards']['simple']
        self.is_training = is_training
        
        config = CONFIG['trading']
        self.initial_equity = config['initial_equity']
        self.fee = config['fee']
        self.kappa = config['kappa']
        self.max_leverage = config['max_leverage']
        
        # State space
        include_forecast = forecast_signal is not None
        obs_size = 5 if include_forecast else 4
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Box(low=-config['max_leverage'], high=config['max_leverage'], 
                                      shape=(1,), dtype=np.float32)
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
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
            pnl = self.position * (next_price - current_price)
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
# SECTION 4: METRICS CALCULATION
# ============================================================

def calculate_metrics(equity_curve, positions):
    """Calculate performance metrics"""
    equity = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1]
    
    # Return
    total_return = (equity[-1] - equity[0]) / equity[0]
    
    # Sharpe
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Volatility
    volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.0
    
    # Max Drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = np.min(drawdown)
    
    # Turnover
    turnover = np.sum(np.abs(np.diff(positions))) / len(positions) if len(positions) > 0 else 0.0
    
    return {
        'return': float(total_return),
        'sharpe': float(sharpe),
        'volatility': float(volatility),
        'maxdd': float(abs(max_dd)),
        'turnover': float(turnover),
        'final_equity': float(equity[-1]),
    }

# ============================================================
# SECTION 5: EXPERIMENTS
# ============================================================

def experiment_forecast_only(df_test, forecast_signal):
    """Experiment 1: Forecast-Only Baseline"""
    print("\n" + "="*80)
    print("EXPERIMENT 1: FORECAST-ONLY BASELINE")
    print("="*80)
    
    wandb.init(
        project="forecast-rl-trading",
        group="baseline",
        name="forecast-only",
        config=CONFIG,
        mode='offline',
        reinit=True
    )
    
    equity = [CONFIG['trading']['initial_equity']]
    positions = [0.0]
    
    for t in range(1, len(df_test)):
        pos = 1.0 if forecast_signal[t] > 0.5 else -1.0
        price_move = df_test['close'].iloc[t] - df_test['close'].iloc[t-1]
        pnl = positions[-1] * price_move
        new_equity = equity[-1] + pnl
        
        equity.append(new_equity)
        positions.append(pos)
    
    metrics = calculate_metrics(equity, positions)
    
    print(f"✓ Return: {metrics['return']*100:.2f}%")
    print(f"✓ Sharpe: {metrics['sharpe']:.4f}")
    print(f"✓ Max DD: {metrics['maxdd']*100:.2f}%")
    print(f"✓ Turnover: {metrics['turnover']:.4f}")
    
    wandb.log({
        'return': metrics['return'],
        'sharpe': metrics['sharpe'],
        'volatility': metrics['volatility'],
        'maxdd': metrics['maxdd'],
        'turnover': metrics['turnover'],
        'final_equity': metrics['final_equity'],
    })
    
    wandb.finish()
    return metrics

def experiment_ppo(df_train, df_test, name, with_forecast=False, forecast_signal=None, reward_name='simple'):
    """Run PPO experiment"""
    print("\n" + "="*80)
    print(f"EXPERIMENT: {name}")
    print("="*80)
    
    wandb.init(
        project="forecast-rl-trading",
        group="ppo-variants",
        name=name,
        config={**CONFIG, 'with_forecast': with_forecast, 'reward': reward_name},
        mode='offline',
        reinit=True
    )
    
    reward_config = CONFIG['rewards'][reward_name]
    
    # Train
    print("\nTraining...")
    env_train = TradingEnv(
        df_train,
        forecast_signal=forecast_signal[:len(df_train)] if with_forecast else None,
        reward_config=reward_config,
        is_training=True
    )
    
    model = PPO(
        "MlpPolicy",
        env_train,
        learning_rate=CONFIG['ppo']['learning_rate'],
        n_steps=CONFIG['ppo']['n_steps'],
        batch_size=CONFIG['ppo']['batch_size'],
        n_epochs=CONFIG['ppo']['n_epochs'],
        ent_coef=CONFIG['ppo']['ent_coef'],
        gamma=CONFIG['ppo']['gamma'],
        gae_lambda=CONFIG['ppo']['gae_lambda'],
        verbose=0
    )
    
    model.learn(total_timesteps=CONFIG['ppo']['total_timesteps'])
    print(f"✓ Training complete")
    
    # Evaluate
    print("Evaluating...")
    env_test = TradingEnv(
        df_test,
        forecast_signal=forecast_signal[len(df_train):] if with_forecast else None,
        reward_config=reward_config,
        is_training=False
    )
    
    obs, _ = env_test.reset()
    equity = [env_test.equity]
    positions = [env_test.position]
    
    done = False
    while not done and env_test.t < len(df_test) - 1:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, _, _ = env_test.step(action)
        equity.append(env_test.equity)
        positions.append(env_test.position)
        done = terminated
    
    metrics = calculate_metrics(equity, positions)
    
    print(f"✓ Return: {metrics['return']*100:.2f}%")
    print(f"✓ Sharpe: {metrics['sharpe']:.4f}")
    print(f"✓ Max DD: {metrics['maxdd']*100:.2f}%")
    print(f"✓ Turnover: {metrics['turnover']:.4f}")
    
    wandb.log({
        'return': metrics['return'],
        'sharpe': metrics['sharpe'],
        'volatility': metrics['volatility'],
        'maxdd': metrics['maxdd'],
        'turnover': metrics['turnover'],
        'final_equity': metrics['final_equity'],
    })
    
    wandb.finish()
    return metrics

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("\n" + "="*80)
    print("FORECAST-AUGMENTED RL TRADING SYSTEM")
    print("Complete Experimental Suite")
    print("="*80)
    
    # Section 1: Load data
    df = load_data()
    
    # Section 2: Train LSTM
    model, scaler, device = train_lstm(df)
    forecast_signal = generate_forecasts(model, df, scaler, device)
    
    # Split data
    split_idx = int(CONFIG['data']['train_split'] * len(df))
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_test = df.iloc[split_idx:].reset_index(drop=True)
    
    print(f"\n✓ Data split: {len(df_train)} train, {len(df_test)} test")
    
    # Run experiments
    results = {}
    
    # Experiment 1: Forecast-Only
    results['Forecast-Only'] = experiment_forecast_only(df_test, forecast_signal[split_idx:])
    
    # Experiment 2: PPO Without Forecast
    results['PPO-Without-Forecast'] = experiment_ppo(
        df_train, df_test,
        "PPO-Without-Forecast",
        with_forecast=False
    )
    
    # Experiment 3: PPO With Forecast
    results['PPO-With-Forecast'] = experiment_ppo(
        df_train, df_test,
        "PPO-With-Forecast",
        with_forecast=True,
        forecast_signal=forecast_signal
    )
    
    # Experiment 4: Reward Variations
    for reward_name in ['with_risk', 'with_cost', 'balanced']:
        results[f'PPO-With-Forecast-{reward_name}'] = experiment_ppo(
            df_train, df_test,
            f"PPO-With-Forecast-{reward_name}",
            with_forecast=True,
            forecast_signal=forecast_signal,
            reward_name=reward_name
        )
    
    # Print results summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    results_df = pd.DataFrame(results).T
    
    print("\n" + results_df.to_string())
    
    # Save to pickle (primary) and CSV (secondary)
    with open('metrics.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✓ Results saved to metrics.pkl")
    
    results_df.to_csv('results_comparison.csv')
    print(f"✓ Results also saved to results_comparison.csv")
    
    # Answer research question
    print("\n" + "="*80)
    print("RESEARCH QUESTION: Does forecast improve RL performance?")
    print("="*80)
    
    ppo_with = results['PPO-With-Forecast']['return']
    ppo_without = results['PPO-Without-Forecast']['return']
    improvement = (ppo_with - ppo_without) / abs(ppo_without) * 100 if ppo_without != 0 else 0
    
    print(f"\nPPO with Forecast:    {ppo_with*100:>8.2f}%")
    print(f"PPO without Forecast: {ppo_without*100:>8.2f}%")
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
    
    print(f"\n{answer}")
    print(f"Explanation: {explanation}")
    
    # Save results JSON
    results_json = {
        'research_question': 'Does forecast improve RL performance?',
        'answer': answer,
        'explanation': explanation,
        'improvement_pct': improvement,
        'results': {k: v for k, v in results.items()},
        'config': CONFIG,
    }
    
    with open('experiment_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"✓ Results also saved to experiment_results.json")
    
    print("\n" + "="*80)
    print("✓ EXPERIMENT SUITE COMPLETE")
    print("="*80)
    print(f"\nAll experiments logged to W&B (offline mode)")
    print(f"Results saved to:")
    print(f"  - metrics.pkl (primary - for visualizations)")
    print(f"  - results_comparison.csv (backup)")
    print(f"  - experiment_results.json (backup)")
    print(f"  - ./wandb/ directory (W&B logs)")

if __name__ == "__main__":
    main()

