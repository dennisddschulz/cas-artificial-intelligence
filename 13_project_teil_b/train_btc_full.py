#!/usr/bin/env python3
"""
Complete BTCUSDT Trading Agent Training Script
With full PPO training, detailed plots, and profit tracking
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from binance.client import Client
from datetime import datetime, timezone
import time
import os
import matplotlib.pyplot as plt

print("=" * 80)
print("BTCUSDT TRADING AGENT - COMPLETE TRAINING PIPELINE")
print("=" * 80)

# CONFIG
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

TRAIN_FRAC = 0.8
FEE = 0.0005
KAPPA = 0.1
INITIAL_BUDGET = 100000.0

# PPO hyperparameters (IDENTICAL to original notebook)
num_envs = 16  # SAME as original
n_steps = 128  # SAME as original
total_updates = 2000  # SAME as original

gamma = 0.99
gae_lambda = 0.95
lr = 3e-4
vf_coef = 0.5
ent_coef = 0.001
max_grad_norm = 0.5
clip_eps = 0.2
ppo_epochs = 10  # FULL EPOCHS!
minibatch_size = 64
target_kl = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")
print(f"Config: {num_envs} envs, {n_steps} steps, {total_updates} updates, {ppo_epochs} epochs")
print()

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("STEP 1: Loading data...")
start_time = time.time()

cache_file = "btcusdt_cache.csv"

if os.path.exists(cache_file):
    print(f"✅ Loading from cache: {cache_file}")
    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    print(f"   Loaded {len(df)} rows in {time.time() - start_time:.1f}s")
else:
    print(f"⏳ Fetching from Binance API (this takes 30-60s)...")
    client = Client()
    klines = client.get_historical_klines(
        "BTCUSDT", Client.KLINE_INTERVAL_1DAY, "1 Jan, 2023",
        datetime.now(timezone.utc).strftime("%d %b, %Y")
    )
    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("open_time")
    df = df[["open", "high", "low", "close", "volume"]].astype(float).dropna()
    df.to_csv(cache_file)
    print(f"✅ Loaded {len(df)} rows in {time.time() - start_time:.1f}s and cached")

print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
print()

# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================
print("STEP 2: Feature engineering...")

def add_features(df, ewma_span=20, vol_window=20):
    """Add Bitcoin-optimized technical features"""
    df = df.copy()

    # Original features
    df["log_close"] = np.log(df["close"])
    df["r"] = df["log_close"].diff()
    df["mu_hat"] = df["r"].ewm(span=ewma_span, adjust=False).mean()
    df["sigma_hat"] = df["r"].rolling(vol_window).std()
    df["r_lag1"] = df["r"].shift(1)

    # ============================================================================
    # BITCOIN-SPECIFIC FEATURES
    # ============================================================================

    # 1. RSI (Relative Strength Index) - 14 period
    def compute_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi / 100.0  # Normalize to 0-1

    df["rsi"] = compute_rsi(df["close"], period=14)

    # 2. MACD (Moving Average Convergence Divergence)
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["macd_norm"] = df["macd"] / (df["close"] + 1e-8)  # Normalize by price

    # 3. Bollinger Bands (20-day, 2-std)
    sma_20 = df["close"].rolling(window=20).mean()
    std_20 = df["close"].rolling(window=20).std()
    df["bb_upper"] = sma_20 + (2 * std_20)
    df["bb_lower"] = sma_20 - (2 * std_20)
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-8)
    df["bb_position"] = df["bb_position"].clip(0, 1)  # Normalize 0-1

    # 4. SMA 20 & 200 (Kurz- und Langtrend)
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["sma_200"] = df["close"].rolling(window=200).mean()
    df["sma_ratio"] = df["sma_20"] / (df["sma_200"] + 1e-8) - 1.0  # -1 to +inf
    df["sma_ratio"] = df["sma_ratio"].clip(-0.5, 0.5)  # Clip extremes

    # 5. Price Position relative to SMA20
    df["price_sma20_dist"] = (df["close"] - df["sma_20"]) / (df["sma_20"] + 1e-8)
    df["price_sma20_dist"] = df["price_sma20_dist"].clip(-0.1, 0.1)

    # 6. ATR (Average True Range) - Volatility in Price Units
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(14).mean()
    df["atr_ratio"] = df["atr"] / (df["close"] + 1e-8)  # ATR as % of price
    df["atr_ratio"] = df["atr_ratio"].clip(0, 0.05)  # Limit to 0-5%

    # 7. Volume Features
    df["volume_ma_20"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_ma_20"] + 1e-8)
    df["volume_ratio"] = np.log1p(df["volume_ratio"].clip(0, 5))  # Log-scale, max 5x

    # 8. Golden Cross / Death Cross Signal
    df["golden_cross"] = (df["sma_20"] > df["sma_200"]).astype(float)

    # 9. Momentum (Rate of Change)
    df["momentum_5"] = df["close"].pct_change(5)
    df["momentum_5"] = df["momentum_5"].clip(-0.1, 0.1)  # Clip to -10% to +10%

    # 10. Trend Strength (Close position in High-Low range)
    df["trend_strength"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-8)

    # Drop NaN rows
    return df.dropna()

df_feat = add_features(df)
print(f"✅ Features created: shape {df_feat.shape}")
print(f"   Available features: {list(df_feat.columns)}")
print()

# ============================================================================
# STEP 3: TRAIN/TEST SPLIT
# ============================================================================
print("STEP 3: Train/Test split...")
n = len(df_feat)
split = int(TRAIN_FRAC * n)
df_train = df_feat.iloc[:split].reset_index(drop=True)
df_test = df_feat.iloc[split:].reset_index(drop=True)
print(f"✅ Train: {len(df_train)} | Test: {len(df_test)}")
print()

# ============================================================================
# STEP 4: TRADING ENVIRONMENT
# ============================================================================
print("STEP 4: Creating trading environment...")

class TradingEnv(gym.Env):
    """Enhanced trading environment with liquidity tracking"""
    metadata = {"render_modes": []}

    def __init__(self, df, fee=0.0005, kappa=0.1, initial_budget=100000.0, max_leverage=2.0):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.fee = float(fee)
        self.kappa = float(kappa)
        self.initial_budget = float(initial_budget)
        self.max_leverage = float(max_leverage)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Use Bitcoin-optimized features
        self.feature_cols = [
            "r", "r_lag1", "mu_hat", "sigma_hat",  # Original 4
            "rsi", "macd_norm", "bb_position",      # TA Indicators 3
            "sma_ratio", "price_sma20_dist",        # Trend Features 2
            "atr_ratio", "volume_ratio",            # Volatility & Volume 2
            "golden_cross", "momentum_5"            # Signal Features 2
        ]

        # Enhanced observation space: 14 market features + 11 portfolio features = 25 total
        obs_dim = len(self.feature_cols) + 11
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 1
        self.pos = 0.0
        self.equity = self.initial_budget
        self.peak = self.initial_budget
        self.cumulative_profit = 0.0
        self.cost_accumulated = 0.0

        # Additional tracking for enhanced state
        self.position_entry_price = None
        self.position_entry_time = None
        self.trades_made = 0
        self.winning_trades = 0
        self.last_position = 0.0
        self.cumulative_returns = 0.0

        return self._get_obs(), {}

    def _get_obs(self):
        """
        Enhanced observation with 15 features:
        - 4 market features: r, r_lag1, mu_hat, sigma_hat
        - 11 portfolio features: pos, equity_norm, drawdown, liquidity_ratio, leverage,
                                unrealized_pnl, cumulative_returns, momentum,
                                win_rate, time_in_position, costs_norm
        """
        x = self.df.loc[self.t, self.feature_cols].values.astype(np.float32)

        # Portfolio metrics
        equity_norm = np.float32(self.equity / self.initial_budget)
        drawdown = np.float32((self.peak - self.equity) / (self.peak + 1e-8))
        position_value = abs(self.pos) * self.equity
        free_capital = max(0.0, self.equity - position_value)
        liquidity_ratio = np.float32(free_capital / (self.equity + 1e-8))
        leverage = np.float32(abs(self.pos) * self.max_leverage)

        # Unrealized PnL from current position
        r_t = float(self.df.loc[self.t, "r"])
        unrealized_pnl = np.float32(self.pos * r_t)

        # Cumulative returns since start
        cumulative_returns = np.float32(self.cumulative_returns)

        # Momentum (simple: mean of last 5 returns if available)
        momentum = np.float32(0.0)
        if self.t >= 5:
            recent_returns = self.df.loc[max(1, self.t-5):self.t, "r"].mean()
            momentum = np.float32(recent_returns)

        # Win rate (% of profitable trades)
        win_rate = np.float32(0.0)
        if self.trades_made > 0:
            win_rate = np.float32(self.winning_trades / (self.trades_made + 1e-8))

        # Time in current position (normalized: 0-1 for first 20 days)
        time_in_position = np.float32(0.0)
        if self.position_entry_time is not None:
            time_in_pos = min(20, self.t - self.position_entry_time) / 20.0
            time_in_position = np.float32(time_in_pos)

        # Normalized costs (0-1)
        costs_norm = np.float32(min(1.0, self.cost_accumulated / (self.initial_budget + 1e-8)))

        # Concatenate all features
        obs = np.concatenate([
            x,  # 4 market features
            np.array([
                self.pos,              # 1
                equity_norm,           # 2
                drawdown,              # 3
                liquidity_ratio,       # 4
                leverage,              # 5
                unrealized_pnl,        # 6
                cumulative_returns,    # 7
                momentum,              # 8
                win_rate,              # 9
                time_in_position,      # 10
                costs_norm             # 11
            ], dtype=np.float32)
        ])
        return obs

    def step(self, action):
        a = float(np.clip(action[0], -1.0, 1.0))
        r_t = float(self.df.loc[self.t, "r"])
        sigma_t = float(self.df.loc[self.t, "sigma_hat"])
        if not np.isfinite(sigma_t):
            sigma_t = 0.01

        # STABLE REWARD
        pnl_reward = self.pos * r_t
        position_change = abs(a - self.pos)
        cost_penalty = self.fee * position_change
        risk_penalty = 0.01 * (a ** 2) * sigma_t
        mu_hat = float(self.df.loc[self.t, "mu_hat"])
        stability_bonus = 0.001 * (1.0 - abs(a)) if abs(mu_hat) < 0.0001 else 0.0

        reward = pnl_reward - cost_penalty - risk_penalty + stability_bonus

        # UPDATE EQUITY
        pnl_dollars = self.pos * self.equity * r_t
        transaction_cost_dollars = cost_penalty * self.equity
        self.equity = self.equity * (1.0 + pnl_reward) - transaction_cost_dollars
        self.cumulative_profit += (pnl_dollars - transaction_cost_dollars)
        self.cost_accumulated += transaction_cost_dollars
        self.cumulative_returns += pnl_reward  # Track cumulative returns
        self.equity = max(self.equity, self.initial_budget * 0.0001)

        # TRACK TRADE STATISTICS
        if position_change > 0.01:  # Position changed significantly
            self.trades_made += 1
            if pnl_reward > 0:  # Previous position was profitable
                self.winning_trades += 1
            self.position_entry_time = self.t

        self.last_position = self.pos
        self.pos = a
        self.peak = max(self.peak, self.equity)

        self.t += 1
        terminated = (self.t >= len(self.df) - 1) or (self.equity <= 0)

        return self._get_obs(), float(reward), terminated, False, {
            "cumulative_profit": self.cumulative_profit,
            "equity": self.equity,
            "position": self.pos,
            "costs": self.cost_accumulated
        }

def make_env(df):
    def thunk():
        return TradingEnv(df, fee=FEE, kappa=KAPPA, initial_budget=INITIAL_BUDGET, max_leverage=2.0)
    return thunk

env = gym.vector.SyncVectorEnv([make_env(df_train) for _ in range(num_envs)])
obs_dim = env.single_observation_space.shape[0]
act_dim = env.single_action_space.shape[0]
print(f"✅ Environment created: obs_dim={obs_dim}, act_dim={act_dim}")
print()

# ============================================================================
# STEP 5: PPO MODEL & HELPERS
# ============================================================================
print("STEP 5: Creating PPO model...")

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh()
        )
        self.mu = nn.Linear(128, act_dim)
        self.log_std = nn.Parameter(torch.ones(act_dim) * -1.0)
        self.v = nn.Linear(128, 1)

    def forward(self, obs):
        x = self.net(obs)
        mu = self.mu(x)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        value = self.v(x).squeeze(-1)
        return dist, value

def squash(u):
    return torch.tanh(u)

def logprob_squashed(dist, u):
    logp_u = dist.log_prob(u).sum(-1)
    eps = 1e-6
    log_det = torch.log(1.0 - torch.tanh(u).pow(2) + eps).sum(-1)
    return logp_u - log_det

def compute_gae(rewards, dones, values, last_value, gamma=0.99, lam=0.95):
    T, N = rewards.shape
    adv = torch.zeros(T, N, device=values.device)
    gae = torch.zeros(N, device=values.device)
    for t in reversed(range(T)):
        not_done = 1.0 - dones[t]
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        gae = delta + gamma * lam * not_done * gae
        adv[t] = gae
    returns = adv + values
    return returns, adv

model = ActorCritic(obs_dim, act_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
print(f"✅ Model created and moved to {device}")
print()

# ============================================================================
# STEP 6: TRAINING LOOP
# ============================================================================
print("STEP 6: STARTING TRAINING (2000 updates)")
print("=" * 80)

obs, _ = env.reset(seed=SEED)
obs = torch.as_tensor(obs, dtype=torch.float32, device=device)

ep_returns = np.zeros(num_envs, dtype=np.float32)
ep_history = []

train_start = time.time()

for update in range(total_updates):
    obs_buf = torch.zeros(n_steps, num_envs, obs_dim, device=device)
    u_buf = torch.zeros(n_steps, num_envs, act_dim, device=device)
    logp_buf = torch.zeros(n_steps, num_envs, device=device)
    rew_buf = torch.zeros(n_steps, num_envs, device=device)
    done_buf = torch.zeros(n_steps, num_envs, device=device)
    val_buf = torch.zeros(n_steps, num_envs, device=device)

    # Rollout
    for t in range(n_steps):
        obs_buf[t] = obs
        with torch.no_grad():
            dist, value = model(obs)
            u = dist.sample()
            a = squash(u)
            logp = logprob_squashed(dist, u)

        u_buf[t] = u
        logp_buf[t] = logp.detach()
        val_buf[t] = value.detach()

        next_obs, reward, terminated, truncated, infos = env.step(a.detach().cpu().numpy())
        done_env = np.logical_or(terminated, truncated)
        done_boot = terminated

        rew_buf[t] = torch.as_tensor(reward, dtype=torch.float32, device=device)
        done_buf[t] = torch.as_tensor(done_boot, dtype=torch.float32, device=device)

        ep_returns += reward
        if done_env.any():
            finished = np.where(done_env)[0]
            ep_history.extend(ep_returns[finished].tolist())
            ep_returns[finished] = 0.0

        obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)

    # GAE & Update
    with torch.no_grad():
        _, last_value = model(obs)

    returns, adv = compute_gae(rew_buf, done_buf, val_buf, last_value, gamma=gamma, lam=gae_lambda)

    B = n_steps * num_envs
    obs_batch = obs_buf.reshape(B, obs_dim)
    u_batch = u_buf.reshape(B, act_dim)
    old_logp = logp_buf.reshape(B)
    ret_batch = returns.reshape(B).detach()
    adv_batch = adv.reshape(B).detach()
    adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)

    idx = torch.arange(B, device=device)
    stop = False

    for epoch in range(ppo_epochs):
        perm = idx[torch.randperm(B)]
        for start in range(0, B, minibatch_size):
            mb = perm[start:start + minibatch_size]

            dist, value = model(obs_batch[mb])
            logp = logprob_squashed(dist, u_batch[mb])
            entropy = dist.entropy().sum(-1)

            approx_kl = (old_logp[mb] - logp).mean().detach()
            if approx_kl.item() > target_kl:
                stop = True
                break

            ratio = torch.exp(logp - old_logp[mb])
            unclipped = ratio * adv_batch[mb]
            clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_batch[mb]
            policy_loss = -torch.min(unclipped, clipped).mean()

            value_loss = (ret_batch[mb] - value).pow(2).mean()
            entropy_loss = -entropy.mean()
            loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        if stop:
            break

        with torch.no_grad():
            model.log_std.clamp_(-2.0, -0.5)

    # Logging every 100 updates (SAME as original)
    if update % 100 == 0:
        mean_ret = np.mean(ep_history[-100:]) if len(ep_history) >= 100 else np.nan
        elapsed = time.time() - train_start
        print(f"Update {update:4d} | mean_return(last100) {mean_ret:8.1f} | log_std {model.log_std.data.cpu().numpy()}")

print("=" * 80)
total_time = time.time() - train_start
print(f"✅ TRAINING COMPLETE! Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
print("=" * 80)
print()

# ============================================================================
# STEP 7: EVALUATION ON TEST SET
# ============================================================================
print("STEP 7: Evaluating on test set...")

def run_equity_curve(model, df_eval):
    """Run model on test set and track all metrics"""
    env_eval = TradingEnv(df_eval, fee=FEE, kappa=KAPPA, initial_budget=INITIAL_BUDGET, max_leverage=2.0)
    obs, _ = env_eval.reset()
    done = False

    equity = [env_eval.equity]
    pos_hist = [env_eval.pos]
    profit_hist = [0.0]
    cost_hist = [0.0]

    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            dist, _ = model(obs_t)
            u = dist.mean
            a = squash(u).cpu().numpy()[0]

        obs, reward, terminated, truncated, info = env_eval.step(a)
        done = terminated or truncated
        equity.append(env_eval.equity)
        pos_hist.append(env_eval.pos)
        profit_hist.append(info["cumulative_profit"])
        cost_hist.append(info["costs"])

    return np.array(equity), np.array(pos_hist), np.array(profit_hist), np.array(cost_hist)

equity, pos_hist, profit_hist, cost_hist = run_equity_curve(model, df_test)
print("✅ Evaluation complete!")
print()

# ============================================================================
# STEP 8: PLOTTING
# ============================================================================
print("STEP 8: Creating detailed plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Equity Curve
axes[0, 0].plot(equity, linewidth=2.5, color='green', label='Equity')
axes[0, 0].fill_between(range(len(equity)), INITIAL_BUDGET, equity, alpha=0.3, color='green')
axes[0, 0].axhline(y=INITIAL_BUDGET, color='red', linestyle='--', linewidth=2, label='Initial Budget')
axes[0, 0].set_title("Equity Curve (Test Set)", fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel("Trading Days", fontsize=12)
axes[0, 0].set_ylabel("Equity (USDT)", fontsize=12)
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Plot 2: Cumulative Profit
axes[0, 1].plot(profit_hist, linewidth=2.5, color='blue', label='Profit')
axes[0, 1].fill_between(range(len(profit_hist)), 0, profit_hist, alpha=0.3, color='blue')
axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[0, 1].set_title("Cumulative Profit (Test Set)", fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel("Trading Days", fontsize=12)
axes[0, 1].set_ylabel("Profit (USDT)", fontsize=12)
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Plot 3: Position History
colors = ['red' if p < 0 else 'green' for p in pos_hist]
axes[1, 0].bar(range(len(pos_hist)), pos_hist, color=colors, alpha=0.6, width=1.0)
axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[1, 0].set_title("Position History (Test Set)", fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel("Trading Days", fontsize=12)
axes[1, 0].set_ylabel("Position [-1 Short, +1 Long]", fontsize=12)
axes[1, 0].set_ylim([-1.2, 1.2])
axes[1, 0].grid(True, alpha=0.3, axis='y')
axes[1, 0].legend(['Short', 'Long'], fontsize=11)

# Plot 4: Cumulative Costs
axes[1, 1].plot(cost_hist, linewidth=2.5, color='red', label='Transaction Costs')
axes[1, 1].fill_between(range(len(cost_hist)), 0, cost_hist, alpha=0.3, color='red')
axes[1, 1].set_title("Cumulative Transaction Costs (Test Set)", fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel("Trading Days", fontsize=12)
axes[1, 1].set_ylabel("Costs (USDT)", fontsize=12)
axes[1, 1].legend(fontsize=11)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

plt.tight_layout()
plt.savefig('trading_results.png', dpi=300, bbox_inches='tight')
print("✅ Plots saved to 'trading_results.png'")
plt.show()
print()

# ============================================================================
# STEP 9: PERFORMANCE SUMMARY
# ============================================================================
print("=" * 80)
print("TRADING PERFORMANCE SUMMARY")
print("=" * 80)

final_equity = equity[-1]
total_profit = profit_hist[-1]
total_costs = cost_hist[-1]
return_pct = (final_equity - INITIAL_BUDGET) / INITIAL_BUDGET * 100
max_drawdown = (1 - (equity.min() / INITIAL_BUDGET)) * 100
sharpe = np.std(np.diff(equity)) / (np.mean(np.diff(equity)) + 1e-8) if np.mean(np.diff(equity)) != 0 else 0
win_rate = np.sum(np.diff(equity) > 0) / len(np.diff(equity)) * 100 if len(np.diff(equity)) > 0 else 0

print(f"Initial Budget:         ${INITIAL_BUDGET:>15,.2f}")
print(f"Final Equity:           ${final_equity:>15,.2f}")
print(f"Total Profit:           ${total_profit:>15,.2f}")
print(f"Total Costs:            ${total_costs:>15,.2f}")
print(f"Return:                 {return_pct:>15.2f}%")
print(f"Max Drawdown:           {max_drawdown:>15.2f}%")
print(f"Sharpe Ratio:           {sharpe:>15.4f}")
print(f"Win Rate:               {win_rate:>15.2f}%")
print(f"Trading Days:           {len(equity):>15}")
print()
print(f"Training Time:          {total_time:>15.1f}s ({total_time/60:.1f} min)")
print("=" * 80)
print()

# Training history plot
print("STEP 10: Creating training history plot...")
fig, ax = plt.subplots(figsize=(14, 6))

# Plot mean returns during training
window = 100
if len(ep_history) > window:
    moving_avg = pd.Series(ep_history).rolling(window=window).mean()
    ax.plot(moving_avg, linewidth=2.5, color='blue', label=f'{window}-episode moving average')
    ax.fill_between(range(len(moving_avg)), moving_avg, alpha=0.3, color='blue')

ax.plot(ep_history, linewidth=0.5, color='gray', alpha=0.5, label='Episode returns')
ax.set_title("Training History - Episode Returns", fontsize=14, fontweight='bold')
ax.set_xlabel("Episode", fontsize=12)
ax.set_ylabel("Episode Return", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("✅ Training history saved to 'training_history.png'")
plt.show()
print()

print("=" * 80)
print("✅ ALL DONE! Results saved:")
print("   - trading_results.png (4 detailed plots)")
print("   - training_history.png (training curves)")
print("=" * 80)

