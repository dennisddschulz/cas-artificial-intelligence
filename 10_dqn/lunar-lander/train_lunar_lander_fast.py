#!/usr/bin/env python3
"""
DQN vs Double DQN Training auf LunarLander - SCHNELLE VERSION (für Demo)
Trainiert mit weniger Steps für schnellere Ergebnisse
"""

import math, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import gymnasium as gym
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

OUTPUT_DIR = "/home/isc-den/cas-artificial-intelligence/10_dqn/lunar-lander/plots"
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
    """Erstelle LunarLander Umgebung"""
    env = gym.make("LunarLander-v3", render_mode=render_mode)
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
    def __init__(self, obs_dim, n_actions, hidden=256):
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

    q_sa = q_online(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        if double_dqn:
            # Double DQN: argmax with q_online, evaluate with q_target
            a_best = torch.argmax(q_online(s2_t), dim=1)
            max_q_s2 = q_target(s2_t).gather(1, a_best.unsqueeze(1)).squeeze(1)
        else:
            # Standard DQN
            max_q_s2 = q_target(s2_t).max(dim=1)[0]

        target = r_t + gamma * max_q_s2 * (1 - d_t)

    loss = (q_sa - target).pow(2).mean()

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

def train_agent(double_dqn=False, total_steps=200_000, seed=0):
    """Trainiere DQN oder Double DQN auf LunarLander (schnelle Version)"""
    set_seed(seed)

    env = make_env(seed=seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    print(f"\n{'='*60}")
    print(f"LunarLander-v3: obs_dim={obs_dim}, n_actions={n_actions}")
    print(f"{'='*60}")

    q_online = QNet(obs_dim, n_actions, hidden=256).to(DEVICE)
    q_target = QNet(obs_dim, n_actions, hidden=256).to(DEVICE)
    q_target.load_state_dict(q_online.state_dict())

    optimizer = optim.Adam(q_online.parameters(), lr=0.0005)
    buffer = ReplayBuffer(capacity=500_000)

    eps_start, eps_end, eps_decay = 1.0, 0.01, 100_000
    gamma = 0.99
    target_update_freq = 5_000
    start_learning = 5_000

    episode_returns = []
    episode_lengths = []
    losses = []
    mean_max_qs = []
    gaps = []

    step_count = 0
    alg_name = "Double DQN" if double_dqn else "DQN"

    print(f"Training {alg_name} (Seed {seed})")
    print(f"Total steps: {total_steps:,} | Target update every {target_update_freq:,} steps")

    obs, _ = env.reset()
    ep_return = 0.0
    ep_length = 0
    ep_count = 0

    for step in range(1, total_steps + 1):
        # Epsilon decay
        eps = eps_end + (eps_start - eps_end) * np.exp(-step / eps_decay)

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
        if step >= start_learning:
            loss = train_step(q_online, q_target, optimizer, buffer,
                            batch_size=64, gamma=gamma, double_dqn=double_dqn)
            if loss is not None:
                losses.append(loss)

            # Update target network
            if step % target_update_freq == 0:
                q_target.load_state_dict(q_online.state_dict())

                # Compute bias metrics
                mean_max_q, gap = compute_bias_metrics(q_online, q_target, buffer)
                if mean_max_q is not None:
                    mean_max_qs.append(mean_max_q)
                    gaps.append(gap)

        obs = obs2

        # Episode end
        if done:
            episode_returns.append(ep_return)
            episode_lengths.append(ep_length)
            ep_count += 1

            obs, _ = env.reset()
            ep_return = 0.0
            ep_length = 0

            if ep_count % 50 == 0:
                avg_return = np.mean(episode_returns[-50:])
                print(f"Episode {ep_count:4d} | Avg Return (last 50): {avg_return:8.2f} | Eps: {eps:.4f}")

    env.close()

    return {
        "name": alg_name,
        "episode_returns": np.array(episode_returns),
        "episode_lengths": np.array(episode_lengths),
        "mean_max_qs": np.array(mean_max_qs),
        "gaps": np.array(gaps),
        "q_online": q_online,
        "q_target": q_target,
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("\n" + "="*60)
print("DQN vs Double DQN auf LunarLander-v3 (SCHNELLE VERSION)")
print("="*60)

# Train both algorithms (schneller: 200k statt 500k steps)
dqn_results = train_agent(double_dqn=False, total_steps=200_000, seed=42)
ddqn_results = train_agent(double_dqn=True, total_steps=200_000, seed=43)

# ============================================================================
# PLOT 1: Episode Returns
# ============================================================================

print("\nGeneriere Plots...", end=" ", flush=True)

fig, ax = plt.subplots(figsize=(14, 7))

window = 50
x_dqn = np.arange(len(dqn_results["episode_returns"]))
x_ddqn = np.arange(len(ddqn_results["episode_returns"]))

# DQN
returns_ma_dqn = moving_average(dqn_results["episode_returns"], window=window)
ax.plot(x_dqn[:len(returns_ma_dqn)], returns_ma_dqn, label="DQN", linewidth=2.5, color="#d62728", alpha=0.8)

# Double DQN
returns_ma_ddqn = moving_average(ddqn_results["episode_returns"], window=window)
ax.plot(x_ddqn[:len(returns_ma_ddqn)], returns_ma_ddqn, label="Double DQN", linewidth=2.5, color="#2ca02c", alpha=0.8)

ax.axhline(y=200, color="gray", linestyle="--", linewidth=2, alpha=0.5, label="Success Threshold (200)")
ax.set_xlabel("Episode", fontsize=14, fontweight="bold")
ax.set_ylabel("Return (Moving Avg, window=50)", fontsize=14, fontweight="bold")
ax.set_title("LunarLander: DQN vs Double DQN - Episode Returns", fontsize=15, fontweight="bold")
ax.legend(fontsize=12, loc="lower right")
ax.grid(alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_episode_returns.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================================
# PLOT 2: Mean Max Q Values
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 7))

x_dqn = np.arange(len(dqn_results["mean_max_qs"]))
x_ddqn = np.arange(len(ddqn_results["mean_max_qs"]))

ax.plot(x_dqn * 5000, dqn_results["mean_max_qs"], label="DQN", linewidth=2.5, color="#d62728", alpha=0.8, marker='o', markersize=4)
ax.plot(x_ddqn * 5000, ddqn_results["mean_max_qs"], label="Double DQN", linewidth=2.5, color="#2ca02c", alpha=0.8, marker='s', markersize=4)

ax.set_xlabel("Training Steps", fontsize=14, fontweight="bold")
ax.set_ylabel("Mean Max Q-Value", fontsize=14, fontweight="bold")
ax.set_title("Q-Value Schätzungen: DQN vs Double DQN", fontsize=15, fontweight="bold")
ax.legend(fontsize=12)
ax.grid(alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_mean_max_q.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================================
# PLOT 3: Gap (Overestimation)
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 7))

x_dqn = np.arange(len(dqn_results["gaps"]))
x_ddqn = np.arange(len(ddqn_results["gaps"]))

ax.plot(x_dqn * 5000, dqn_results["gaps"], label="DQN", linewidth=2.5, color="#d62728", alpha=0.8, marker='o', markersize=4)
ax.plot(x_ddqn * 5000, ddqn_results["gaps"], label="Double DQN", linewidth=2.5, color="#2ca02c", alpha=0.8, marker='s', markersize=4)

ax.set_xlabel("Training Steps", fontsize=14, fontweight="bold")
ax.set_ylabel("Gap (max Q - Q_taken)", fontsize=14, fontweight="bold")
ax.set_title("Overestimation Gap: DQN vs Double DQN", fontsize=15, fontweight="bold")
ax.legend(fontsize=12)
ax.grid(alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_gap.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================================
# PLOT 4: Episode Length
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 7))

x_dqn = np.arange(len(dqn_results["episode_lengths"]))
x_ddqn = np.arange(len(ddqn_results["episode_lengths"]))

lengths_ma_dqn = moving_average(dqn_results["episode_lengths"], window=50)
lengths_ma_ddqn = moving_average(ddqn_results["episode_lengths"], window=50)

ax.plot(x_dqn[:len(lengths_ma_dqn)], lengths_ma_dqn, label="DQN", linewidth=2.5, color="#d62728", alpha=0.8)
ax.plot(x_ddqn[:len(lengths_ma_ddqn)], lengths_ma_ddqn, label="Double DQN", linewidth=2.5, color="#2ca02c", alpha=0.8)

ax.axhline(y=1000, color="gray", linestyle="--", linewidth=2, alpha=0.5, label="Max Steps (1000)")
ax.set_xlabel("Episode", fontsize=14, fontweight="bold")
ax.set_ylabel("Episode Length (Moving Avg, window=50)", fontsize=14, fontweight="bold")
ax.set_title("LunarLander: Episode Längen", fontsize=15, fontweight="bold")
ax.legend(fontsize=12, loc="upper right")
ax.grid(alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_episode_lengths.png", dpi=150, bbox_inches="tight")
plt.close()

print("✓")

# ============================================================================
# EVALUATION
# ============================================================================

print("\nEvaluation...")

@torch.no_grad()
def evaluate(q_net, n_episodes=20, seed=999):
    env = make_env(seed=seed)
    returns = []
    lengths = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_r = 0.0
        steps = 0

        while not done and steps < 1000:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            a = int(torch.argmax(q_net(obs_t), dim=1).item())
            obs, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total_r += r
            steps += 1

        returns.append(total_r)
        lengths.append(steps)

    env.close()
    return np.mean(returns), np.std(returns), np.mean(lengths), np.std(lengths)

mean_dqn, std_dqn, len_dqn, len_std_dqn = evaluate(dqn_results["q_online"], n_episodes=20)
mean_ddqn, std_ddqn, len_ddqn, len_std_ddqn = evaluate(ddqn_results["q_online"], n_episodes=20)

print("\n" + "="*60)
print("EVALUATION RESULTS (20 Episodes, Greedy Policy)")
print("="*60)
print(f"\nDQN:")
print(f"  Mean Return: {mean_dqn:8.2f} ± {std_dqn:6.2f}")
print(f"  Mean Length: {len_dqn:8.2f} ± {len_std_dqn:6.2f}")

print(f"\nDouble DQN:")
print(f"  Mean Return: {mean_ddqn:8.2f} ± {std_ddqn:6.2f}")
print(f"  Mean Length: {len_ddqn:8.2f} ± {len_std_ddqn:6.2f}")

winner = "DQN" if mean_dqn > mean_ddqn else "Double DQN"
diff = abs(mean_dqn - mean_ddqn)
print(f"\nWinner: {winner} (+{diff:.2f} Return)")
print("="*60)

# ============================================================================
# SAVE RESULTS SUMMARY
# ============================================================================

summary = f"""
DQN vs Double DQN auf LunarLander-v3 (Schnelle Trainingsversion)
================================================================

Training Configuration:
- Total Steps: 200,000 (schnell für Demo)
- Batch Size: 64
- Learning Rate: 0.0005
- Gamma: 0.99
- Target Update Frequency: 5,000 steps
- Epsilon Decay: 100,000 steps

Results Summary:
================

DQN:
  Training Episodes: {len(dqn_results['episode_returns'])}
  Mean Return (last 50 eps): {np.mean(dqn_results['episode_returns'][-50:]):.2f}
  Eval Mean Return: {mean_dqn:.2f} ± {std_dqn:.2f}
  Eval Mean Length: {len_dqn:.2f} ± {len_std_dqn:.2f}
  Final Mean Max Q: {dqn_results['mean_max_qs'][-1]:.2f}
  Final Gap: {dqn_results['gaps'][-1]:.2f}

Double DQN:
  Training Episodes: {len(ddqn_results['episode_returns'])}
  Mean Return (last 50 eps): {np.mean(ddqn_results['episode_returns'][-50:]):.2f}
  Eval Mean Return: {mean_ddqn:.2f} ± {std_ddqn:.2f}
  Eval Mean Length: {len_ddqn:.2f} ± {len_std_ddqn:.2f}
  Final Mean Max Q: {ddqn_results['mean_max_qs'][-1]:.2f}
  Final Gap: {ddqn_results['gaps'][-1]:.2f}

Winner: {winner}
Difference: {diff:.2f} Return Points

Success Threshold: 200 (LunarLander official success criterion)
"""

with open(f"{OUTPUT_DIR}/results_summary.txt", "w") as f:
    f.write(summary)

print(f"\n✓ Plots gespeichert in: {OUTPUT_DIR}/")
print(f"✓ Zusammenfassung: {OUTPUT_DIR}/results_summary.txt")
