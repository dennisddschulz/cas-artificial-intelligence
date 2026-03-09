#!/usr/bin/env python3
"""
Optimized Training Script for BTCUSDT Trading Agent
This version includes detailed logging and faster training
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

print("=" * 70)
print("BTCUSDT TRADING AGENT - TRAINING SCRIPT")
print("=" * 70)

# CONFIG
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

TRAIN_FRAC = 0.8
FEE = 0.0005
KAPPA = 0.1
INITIAL_BUDGET = 100000.0

# PPO hyperparameters (OPTIMIZED)
num_envs = 4  # Reduced even further for faster training
n_steps = 32  # Reduced for faster iteration
total_updates = 100  # Very quick test

gamma = 0.99
gae_lambda = 0.95
lr = 3e-4
vf_coef = 0.5
ent_coef = 0.001
max_grad_norm = 0.5
clip_eps = 0.2
ppo_epochs = 2  # Minimal epochs
minibatch_size = 16
target_kl = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")
print(f"Config: {num_envs} envs, {n_steps} steps, {total_updates} updates")
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
    df = df.copy()
    df["log_close"] = np.log(df["close"])
    df["r"] = df["log_close"].diff()
    df["mu_hat"] = df["r"].ewm(span=ewma_span, adjust=False).mean()
    df["sigma_hat"] = df["r"].rolling(vol_window).std()
    df["r_lag1"] = df["r"].shift(1)
    return df.dropna()

df_feat = add_features(df)
print(f"✅ Features created: shape {df_feat.shape}")
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
    """Enhanced trading environment"""
    metadata = {"render_modes": []}

    def __init__(self, df, fee=0.0005, kappa=0.1, initial_budget=100000.0, max_leverage=2.0):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.fee = float(fee)
        self.kappa = float(kappa)
        self.initial_budget = float(initial_budget)
        self.max_leverage = float(max_leverage)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.feature_cols = ["r", "r_lag1", "mu_hat", "sigma_hat"]
        obs_dim = len(self.feature_cols) + 5
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
        return self._get_obs(), {}

    def _get_obs(self):
        x = self.df.loc[self.t, self.feature_cols].values.astype(np.float32)
        equity_norm = np.float32(self.equity / self.initial_budget)
        drawdown = np.float32((self.peak - self.equity) / (self.peak + 1e-8))
        position_value = abs(self.pos) * self.equity
        free_capital = max(0.0, self.equity - position_value)
        liquidity_ratio = np.float32(free_capital / (self.equity + 1e-8))
        leverage = np.float32(abs(self.pos) * self.max_leverage)

        obs = np.concatenate([x, np.array([self.pos, equity_norm, drawdown, liquidity_ratio, leverage], dtype=np.float32)])
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
        self.equity = max(self.equity, self.initial_budget * 0.0001)

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
print("STEP 6: STARTING TRAINING")
print("=" * 70)

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

    # Logging
    if update % 10 == 0:
        mean_ret = np.mean(ep_history[-100:]) if len(ep_history) >= 100 else np.nan
        elapsed = time.time() - train_start
        print(f"Update {update:3d} | mean_ret {mean_ret:8.2f} | log_std {model.log_std.data.cpu().numpy()[0]:.4f} | time {elapsed:.1f}s")

print("=" * 70)
print(f"✅ TRAINING COMPLETE! Total time: {time.time() - train_start:.1f}s")
print("=" * 70)
print()

