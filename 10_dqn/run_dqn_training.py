#!/usr/bin/env python3
"""
DQN vs Double DQN Training auf CartPole mit Plots
"""

import math, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import gymnasium as gym

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

OUTPUT_DIR = "/home/isc-den/cas-artificial-intelligence/10_dqn/plots"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# HILFSFUNKTIONEN
# ============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def moving_average(x, window=50):
    x = np.asarray(x, dtype=np.float32)
    if len(x) < window:
        return x
    c = np.cumsum(np.insert(x, 0, 0))
    return (c[window:] - c[:-window]) / window

def make_env(seed=0, render_mode=None):
    env = gym.make("CartPole-v1", render_mode=render_mode)
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    return env

# ============================================================================
# REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r.astype(np.float32), s2, d.astype(np.float32)

    def __len__(self):
        return len(self.buf)

# ============================================================================
# Q-NETWORK
# ============================================================================

class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# ============================================================================
# ACTION SELECTION
# ============================================================================

@torch.no_grad()
def select_action(q_net, obs, eps, n_actions):
    if random.random() < eps:
        return random.randrange(n_actions)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    q = q_net(obs_t)
    return int(torch.argmax(q, dim=1).item())

# ============================================================================
# TRAINING STEP
# ============================================================================

def train_step(q_online, q_target, optimizer, buffer,
               batch_size=64, gamma=0.99, double_dqn=False, grad_clip=10.0):
    if len(buffer) < batch_size:
        return None

    s, a, r, s2, done = buffer.sample(batch_size)

    s_t = torch.tensor(s, dtype=torch.float32, device=DEVICE)
    a_t = torch.tensor(a, dtype=torch.int64, device=DEVICE)
    r_t = torch.tensor(r, dtype=torch.float32, device=DEVICE)
    s2_t = torch.tensor(s2, dtype=torch.float32, device=DEVICE)
    d_t = torch.tensor(done, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        if double_dqn:
            # Double DQN: argmax with q_online, evaluate with q_target
            a_best = torch.argmax(q_online(s2_t), dim=1)
            max_q_s2 = q_target(s2_t).gather(1, a_best.unsqueeze(1)).squeeze(1)
        else:
            # Standard DQN
            max_q_s2 = q_target(s2_t).max(dim=1)[0]

        target = r_t + gamma * max_q_s2 * (1 - d_t)

    q_s_a = q_online(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
    loss = (q_s_a - target).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    if grad_clip:
        nn.utils.clip_grad_norm_(q_online.parameters(), grad_clip)
    optimizer.step()

    return loss.item()

# ============================================================================
# DIAGNOSTICS
# ============================================================================

@torch.no_grad()
def compute_bias_metrics(q_online, q_target, buffer, batch_size=256):
    """Berechne Q-value Bias Metriken"""
    if len(buffer) < batch_size:
        return None, None

    s, a, r, s2, done = buffer.sample(batch_size)
    s_t = torch.tensor(s, dtype=torch.float32, device=DEVICE)
    s2_t = torch.tensor(s2, dtype=torch.float32, device=DEVICE)
    a_t = torch.tensor(a, dtype=torch.int64, device=DEVICE)

    q_online_vals = q_online(s_t)
    q_target_vals = q_target(s2_t)

    # Mean max Q
    mean_max_q = q_target_vals.max(dim=1)[0].mean().item()

    # Gap: max_a Q(s') - Q(s', a_taken)
    max_q = q_target_vals.max(dim=1)[0]
    taken_q = q_target_vals.gather(1, a_t.unsqueeze(1)).squeeze(1)
    gap = (max_q - taken_q).mean().item()

    return mean_max_q, gap

# ============================================================================
# HAUPTTRAINING
# ============================================================================

def train_agent(double_dqn=False, episodes=500, seed=0):
    """Trainiere DQN oder Double DQN"""
    set_seed(seed)

    env = make_env(seed=seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_online = QNet(obs_dim, n_actions).to(DEVICE)
    q_target = QNet(obs_dim, n_actions).to(DEVICE)
    q_target.load_state_dict(q_online.state_dict())

    optimizer = optim.Adam(q_online.parameters(), lr=0.001)
    buffer = ReplayBuffer(capacity=100_000)

    eps_start, eps_end, eps_decay = 1.0, 0.05, 0.995
    gamma = 0.99
    target_update_freq = 1000

    episode_returns = []
    episode_lengths = []
    losses = []
    mean_max_qs = []
    gaps = []

    step_count = 0
    alg_name = "Double DQN" if double_dqn else "DQN"

    print(f"\n{'='*60}")
    print(f"Training {alg_name} (Seed {seed})")
    print(f"{'='*60}")

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_length = 0

        eps = eps_end + (eps_start - eps_end) * (eps_decay ** ep)

        while not done:
            # Select action
            a = select_action(q_online, obs, eps, n_actions)

            # Step environment
            obs2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            step_count += 1
            ep_return += r
            ep_length += 1

            # Store in buffer
            buffer.push(obs, a, r, obs2, done)

            # Train
            loss = train_step(q_online, q_target, optimizer, buffer,
                            batch_size=64, gamma=gamma, double_dqn=double_dqn)
            if loss is not None:
                losses.append(loss)

            # Update target network
            if step_count % target_update_freq == 0:
                q_target.load_state_dict(q_online.state_dict())

            obs = obs2

        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)

        # Compute bias metrics
        mean_max_q, gap = compute_bias_metrics(q_online, q_target, buffer)
        if mean_max_q is not None:
            mean_max_qs.append(mean_max_q)
            gaps.append(gap)

        if (ep + 1) % 50 == 0:
            avg_return = np.mean(episode_returns[-50:])
            print(f"Episode {ep+1:3d} | Return: {avg_return:7.2f} | Eps: {eps:.3f}")

    env.close()

    return {
        "name": alg_name,
        "episode_returns": np.array(episode_returns),
        "episode_lengths": np.array(episode_lengths),
        "mean_max_qs": np.array(mean_max_qs),
        "gaps": np.array(gaps),
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("\n" + "="*60)
print("DQN vs Double DQN auf CartPole-v1")
print("="*60)

# Train both algorithms
dqn_results = train_agent(double_dqn=False, episodes=500, seed=42)
ddqn_results = train_agent(double_dqn=True, episodes=500, seed=42)

# ============================================================================
# PLOT 1: Episode Returns
# ============================================================================

print("\nGeneriere Plots...", end=" ", flush=True)

fig, ax = plt.subplots(figsize=(12, 6))

window = 50

# DQN
returns_ma_dqn = moving_average(dqn_results["episode_returns"], window=window)
x_dqn = np.arange(len(returns_ma_dqn))
ax.plot(x_dqn, returns_ma_dqn, label="DQN", linewidth=2.5, color="#d62728", alpha=0.8)

# Double DQN
returns_ma_ddqn = moving_average(ddqn_results["episode_returns"], window=window)
x_ddqn = np.arange(len(returns_ma_ddqn))
ax.plot(x_ddqn, returns_ma_ddqn, label="Double DQN", linewidth=2.5, color="#2ca02c", alpha=0.8)

ax.set_xlabel("Episode", fontsize=13, fontweight="bold")
ax.set_ylabel("Return (Moving Avg, window=50)", fontsize=13, fontweight="bold")
ax.set_title("DQN vs Double DQN: Episode Returns", fontsize=14, fontweight="bold")
ax.legend(fontsize=12)
ax.grid(alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_episode_returns.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================================
# PLOT 2: Mean Max Q Values
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(dqn_results["mean_max_qs"]))
ax.plot(x, dqn_results["mean_max_qs"], label="DQN", linewidth=2, color="#d62728", alpha=0.8)
ax.plot(x, ddqn_results["mean_max_qs"], label="Double DQN", linewidth=2, color="#2ca02c", alpha=0.8)

ax.set_xlabel("Update Step", fontsize=13, fontweight="bold")
ax.set_ylabel("Mean Max Q-Value", fontsize=13, fontweight="bold")
ax.set_title("Q-Value Schätzungen über Zeit", fontsize=14, fontweight="bold")
ax.legend(fontsize=12)
ax.grid(alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_mean_max_q.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================================
# PLOT 3: Gap (Overestimation)
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(dqn_results["gaps"]))
ax.plot(x, dqn_results["gaps"], label="DQN", linewidth=2, color="#d62728", alpha=0.8)
ax.plot(x, ddqn_results["gaps"], label="Double DQN", linewidth=2, color="#2ca02c", alpha=0.8)

ax.set_xlabel("Update Step", fontsize=13, fontweight="bold")
ax.set_ylabel("Gap (max Q - Q_taken)", fontsize=13, fontweight="bold")
ax.set_title("Overestimation Gap: DQN vs Double DQN", fontsize=14, fontweight="bold")
ax.legend(fontsize=12)
ax.grid(alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_gap.png", dpi=150, bbox_inches="tight")
plt.close()

print("✓")

print("\n" + "="*60)
print("PLOTS GESPEICHERT")
print("="*60)
print(f"✓ 01_episode_returns.png - Episode Returns Vergleich")
print(f"✓ 02_mean_max_q.png - Mean Max Q-Values")
print(f"✓ 03_gap.png - Overestimation Gap")
print(f"\nOuput Directory: {OUTPUT_DIR}")
print("="*60)

