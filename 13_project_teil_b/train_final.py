#!/usr/bin/env python3
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import time

print("=" * 80)
print("BTCUSDT TRADING AGENT - TRAINING START")
print("=" * 80)

# Config
SEED, TRAIN_FRAC, FEE, KAPPA, INITIAL_BUDGET = 0, 0.8, 0.0005, 0.1, 100000.0
np.random.seed(SEED)
torch.manual_seed(SEED)

num_envs, n_steps, total_updates = 16, 128, 2000
gamma, gae_lambda, lr, ppo_epochs, minibatch_size = 0.99, 0.95, 3e-4, 10, 64
device = torch.device("cpu")

print(f"Device: {device}, Config: {num_envs} envs, {total_updates} updates\n")

# Load data
print("Step 1: Loading data...")
df = pd.read_csv("btcusdt_cache.csv", index_col=0, parse_dates=True)
print(f"  ✅ {len(df)} rows loaded\n")

# Features
print("Step 2: Computing features...")
df = df.copy()
df["log_close"] = np.log(df["close"])
df["r"] = df["log_close"].diff()
df["mu_hat"] = df["r"].ewm(span=20, adjust=False).mean()
df["sigma_hat"] = df["r"].rolling(20).std()
df["r_lag1"] = df["r"].shift(1)

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    return (100 - (100 / (1 + rs))) / 100.0

df["rsi"] = compute_rsi(df["close"], period=14)

ema_12 = df["close"].ewm(span=12, adjust=False).mean()
ema_26 = df["close"].ewm(span=26, adjust=False).mean()
df["macd"] = ema_12 - ema_26
df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
df["macd_norm"] = df["macd"] / (df["close"] + 1e-8)

sma_20 = df["close"].rolling(window=20).mean()
std_20 = df["close"].rolling(window=20).std()
df["bb_position"] = (df["close"] - (sma_20 - 2*std_20)) / (4*std_20 + 1e-8)
df["bb_position"] = df["bb_position"].clip(0, 1)

df["sma_20"] = sma_20
df["sma_200"] = df["close"].rolling(window=200).mean()
df["sma_ratio"] = (df["sma_20"] / (df["sma_200"] + 1e-8) - 1.0).clip(-0.5, 0.5)
df["price_sma20_dist"] = ((df["close"] - sma_20) / (sma_20 + 1e-8)).clip(-0.1, 0.1)

high_low = df["high"] - df["low"]
high_close = abs(df["high"] - df["close"].shift())
low_close = abs(df["low"] - df["close"].shift())
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df["atr_ratio"] = (true_range.rolling(14).mean() / (df["close"] + 1e-8)).clip(0, 0.05)

df["volume_ma_20"] = df["volume"].rolling(window=20).mean()
df["volume_ratio"] = np.log1p((df["volume"] / (df["volume_ma_20"] + 1e-8)).clip(0, 5))
df["golden_cross"] = (df["sma_20"] > df["sma_200"]).astype(float)
df["momentum_5"] = df["close"].pct_change(5).clip(-0.1, 0.1)

df = df.dropna()
print(f"  ✅ {len(df)} rows, {len(df.columns)} features\n")

# Train/Test split
n = len(df)
split = int(TRAIN_FRAC * n)
df_train = df.iloc[:split].reset_index(drop=True)
df_test = df.iloc[split:].reset_index(drop=True)
print(f"Step 3: Train/Test - {len(df_train)} / {len(df_test)}\n")

# Environment
print("Step 4: Creating environment...")

class TradingEnv(gym.Env):
    def __init__(self, df, fee=0.0005, kappa=0.1, initial_budget=100000.0):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.fee, self.kappa, self.initial_budget = fee, kappa, initial_budget
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.feature_cols = ["r", "r_lag1", "mu_hat", "sigma_hat", "rsi", "macd_norm", "bb_position",
                            "sma_ratio", "price_sma20_dist", "atr_ratio", "volume_ratio", "golden_cross", "momentum_5"]
        obs_dim = len(self.feature_cols) + 11
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t, self.pos, self.equity, self.peak = 1, 0.0, self.initial_budget, self.initial_budget
        self.cumulative_profit, self.cost_accumulated, self.trades_made, self.winning_trades, self.cumulative_returns = 0.0, 0.0, 0, 0, 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        x = self.df.loc[self.t, self.feature_cols].values.astype(np.float32)
        equity_norm = np.float32(self.equity / self.initial_budget)
        drawdown = np.float32((self.peak - self.equity) / (self.peak + 1e-8))
        liquidity_ratio = np.float32(max(0, self.equity - abs(self.pos) * self.equity) / (self.equity + 1e-8))
        leverage = np.float32(abs(self.pos) * 2.0)

        r_t = float(self.df.loc[self.t, "r"])
        unrealized_pnl = np.float32(self.pos * r_t)
        cumulative_returns = np.float32(self.cumulative_returns)
        momentum = np.float32(self.df.loc[max(1, self.t-5):self.t, "r"].mean() if self.t >= 5 else 0.0)
        win_rate = np.float32(self.winning_trades / (self.trades_made + 1e-8) if self.trades_made > 0 else 0.0)
        time_in_position = np.float32(0.0)
        costs_norm = np.float32(min(1.0, self.cost_accumulated / (self.initial_budget + 1e-8)))

        obs = np.concatenate([x, np.array([self.pos, equity_norm, drawdown, liquidity_ratio, leverage,
                                           unrealized_pnl, cumulative_returns, momentum, win_rate, time_in_position, costs_norm], dtype=np.float32)])
        return obs

    def step(self, action):
        a = float(np.clip(action[0], -1.0, 1.0))
        r_t = float(self.df.loc[self.t, "r"])
        sigma_t = float(self.df.loc[self.t, "sigma_hat"])
        if not np.isfinite(sigma_t):
            sigma_t = 0.01

        pnl_reward = self.pos * r_t
        position_change = abs(a - self.pos)
        cost_penalty = self.fee * position_change
        risk_penalty = 0.01 * (a ** 2) * sigma_t
        mu_hat = float(self.df.loc[self.t, "mu_hat"])
        stability_bonus = 0.001 * (1.0 - abs(a)) if abs(mu_hat) < 0.0001 else 0.0

        reward = pnl_reward - cost_penalty - risk_penalty + stability_bonus

        pnl_dollars = self.pos * self.equity * r_t
        transaction_cost_dollars = cost_penalty * self.equity
        self.equity = self.equity * (1.0 + pnl_reward) - transaction_cost_dollars
        self.cumulative_profit += (pnl_dollars - transaction_cost_dollars)
        self.cost_accumulated += transaction_cost_dollars
        self.cumulative_returns += pnl_reward
        self.equity = max(self.equity, self.initial_budget * 0.0001)

        if position_change > 0.01:
            self.trades_made += 1
            if pnl_reward > 0:
                self.winning_trades += 1

        self.pos, self.peak = a, max(self.peak, self.equity)
        self.t += 1
        terminated = (self.t >= len(self.df) - 1) or (self.equity <= 0)

        return self._get_obs(), float(reward), terminated, False, {
            "cumulative_profit": self.cumulative_profit, "equity": self.equity, "position": self.pos, "costs": self.cost_accumulated
        }

def make_env(df):
    def thunk():
        return TradingEnv(df, fee=FEE, kappa=KAPPA, initial_budget=INITIAL_BUDGET)
    return thunk

env = gym.vector.SyncVectorEnv([make_env(df_train) for _ in range(num_envs)])
obs_dim = env.single_observation_space.shape[0]
act_dim = env.single_action_space.shape[0]
print(f"  ✅ obs_dim={obs_dim}, act_dim={act_dim}\n")

# Model
print("Step 5: Creating model...")
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())
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
    log_det = torch.log(1.0 - torch.tanh(u).pow(2) + 1e-6).sum(-1)
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
    return adv + values, adv

model = ActorCritic(obs_dim, act_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
print(f"  ✅ Model created\n")

# Training
print("Step 6: TRAINING (2000 updates)")
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
    for epoch in range(ppo_epochs):
        perm = idx[torch.randperm(B)]
        for start in range(0, B, minibatch_size):
            mb = perm[start:start + minibatch_size]
            dist, value = model(obs_batch[mb])
            logp = logprob_squashed(dist, u_batch[mb])
            entropy = dist.entropy().sum(-1)

            ratio = torch.exp(logp - old_logp[mb])
            unclipped = ratio * adv_batch[mb]
            clipped = torch.clamp(ratio, 0.8, 1.2) * adv_batch[mb]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = (ret_batch[mb] - value).pow(2).mean()
            entropy_loss = -entropy.mean()
            loss = policy_loss + 0.5 * value_loss + 0.001 * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        with torch.no_grad():
            model.log_std.clamp_(-2.0, -0.5)

    if update % 100 == 0:
        mean_ret = np.mean(ep_history[-100:]) if len(ep_history) >= 100 else np.nan
        print(f"Update {update:4d} | mean_return(last100) {mean_ret:8.1f} | log_std {model.log_std.data.cpu().numpy()}")

print("=" * 80)
total_time = time.time() - train_start
print(f"✅ TRAINING COMPLETE! {total_time/60:.1f} min\n")

# Evaluation
print("Step 7: Evaluating...")

def run_equity_curve(model, df_eval):
    env_eval = TradingEnv(df_eval, fee=FEE, kappa=KAPPA, initial_budget=INITIAL_BUDGET)
    obs, _ = env_eval.reset()
    done = False
    equity, pos_hist, profit_hist, cost_hist = [env_eval.equity], [env_eval.pos], [0.0], [0.0]

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
print("✅ Evaluation complete!\n")

# Plotting
print("Step 8: Creating plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].plot(equity, linewidth=2.5, color='green')
axes[0, 0].fill_between(range(len(equity)), INITIAL_BUDGET, equity, alpha=0.3, color='green')
axes[0, 0].axhline(y=INITIAL_BUDGET, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_title("Equity Curve", fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel("Days")
axes[0, 0].set_ylabel("Equity (USDT)")
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(profit_hist, linewidth=2.5, color='blue')
axes[0, 1].fill_between(range(len(profit_hist)), 0, profit_hist, alpha=0.3, color='blue')
axes[0, 1].set_title("Cumulative Profit", fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel("Days")
axes[0, 1].set_ylabel("Profit (USDT)")
axes[0, 1].grid(True, alpha=0.3)

colors = ['red' if p < 0 else 'green' for p in pos_hist]
axes[1, 0].bar(range(len(pos_hist)), pos_hist, color=colors, alpha=0.6, width=1.0)
axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[1, 0].set_title("Position History", fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel("Days")
axes[1, 0].set_ylabel("Position")
axes[1, 0].set_ylim([-1.2, 1.2])
axes[1, 0].grid(True, alpha=0.3, axis='y')

axes[1, 1].plot(cost_hist, linewidth=2.5, color='red')
axes[1, 1].fill_between(range(len(cost_hist)), 0, cost_hist, alpha=0.3, color='red')
axes[1, 1].set_title("Cumulative Costs", fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel("Days")
axes[1, 1].set_ylabel("Costs (USDT)")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('trading_results.png', dpi=300, bbox_inches='tight')
print("✅ Saved: trading_results.png")
plt.close()

fig, ax = plt.subplots(figsize=(14, 6))
if len(ep_history) > 100:
    moving_avg = pd.Series(ep_history).rolling(window=100).mean()
    ax.plot(moving_avg, linewidth=2.5, color='blue', label='100-ep MA')
    ax.fill_between(range(len(moving_avg)), moving_avg, alpha=0.3, color='blue')
ax.plot(ep_history, linewidth=0.5, color='gray', alpha=0.5, label='Raw')
ax.set_title("Training History", fontsize=14, fontweight='bold')
ax.set_xlabel("Episode")
ax.set_ylabel("Return")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("✅ Saved: training_history.png")
plt.close()
print()

# Summary
print("=" * 80)
print("PERFORMANCE SUMMARY")
print("=" * 80)
final_equity = equity[-1]
total_profit = profit_hist[-1]
total_costs = cost_hist[-1]
ret_pct = (final_equity - INITIAL_BUDGET) / INITIAL_BUDGET * 100
dd = (1 - (equity.min() / INITIAL_BUDGET)) * 100
sharpe = np.std(np.diff(equity)) / (np.mean(np.diff(equity)) + 1e-8) if np.mean(np.diff(equity)) != 0 else 0
wr = np.sum(np.diff(equity) > 0) / len(np.diff(equity)) * 100

print(f"Initial:        ${INITIAL_BUDGET:>15,.0f}")
print(f"Final:          ${final_equity:>15,.0f}")
print(f"Profit:         ${total_profit:>15,.0f}")
print(f"Costs:          ${total_costs:>15,.0f}")
print(f"Return:         {ret_pct:>15.2f}%")
print(f"Max Drawdown:   {dd:>15.2f}%")
print(f"Sharpe:         {sharpe:>15.4f}")
print(f"Win Rate:       {wr:>15.2f}%")
print(f"Duration:       {total_time:>15.1f}s")
print("=" * 80)
print("\n✅ ALL DONE!")

