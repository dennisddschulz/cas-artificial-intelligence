#!/usr/bin/env python3
"""
DQN vs Double DQN Training auf LunarLander - HOCHOPTIMIERTE VERSION
Fokus: Speed statt Detail-Metriken
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import gymnasium as gym
import os
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

OUTPUT_DIR = "/home/isc-den/cas-artificial-intelligence/10_dqn/lunar-lander/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# OPTIMIERTE REPLAY BUFFER (pre-allocate arrays)
# ============================================================================

class FastReplayBuffer:
    def __init__(self, capacity=100_000, obs_dim=8):
        self.capacity = capacity
        self.obs_dim = obs_dim

        # Pre-allocate numpy arrays
        self.s = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)

        self.idx = 0
        self.size = 0

    def push(self, s, a, r, s2, done):
        self.s[self.idx] = s
        self.a[self.idx] = a
        self.r[self.idx] = r
        self.s2[self.idx] = s2
        self.d[self.idx] = float(done)

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, batch_size)
        return (
            torch.from_numpy(self.s[indices]).to(DEVICE),
            torch.from_numpy(self.a[indices]).to(DEVICE),
            torch.from_numpy(self.r[indices]).to(DEVICE),
            torch.from_numpy(self.s2[indices]).to(DEVICE),
            torch.from_numpy(self.d[indices]).to(DEVICE),
        )

    def __len__(self):
        return self.size

# ============================================================================
# OPTIMIERTES Q-NETWORK (kleinere Architektur)
# ============================================================================

class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),  # Reduziert von 256
            nn.ReLU(),
            nn.Linear(128, 128),      # Reduziert von 256
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# ============================================================================
# OPTIMIERTE TRAININGS-SCHLEIFE
# ============================================================================

def make_env(seed=0):
    env = gym.make("LunarLander-v3")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env

def train_fast(double_dqn=False, total_steps=100_000, seed=0):
    """Trainiere mit Fokus auf Speed"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(seed=seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_online = QNet(obs_dim, n_actions).to(DEVICE)
    q_target = QNet(obs_dim, n_actions).to(DEVICE)
    q_target.load_state_dict(q_online.state_dict())

    optimizer = optim.Adam(q_online.parameters(), lr=0.001)
    buffer = FastReplayBuffer(capacity=100_000, obs_dim=obs_dim)

    # Schnellere Parameter
    eps_start, eps_end = 1.0, 0.05
    eps_decay = total_steps // 2
    gamma = 0.99
    target_update_freq = 2_000  # Reduziert von 5000
    start_learning = 1_000      # Reduziert von 5000
    batch_size = 64

    episode_returns = []
    episode_lengths = []

    alg_name = "Double DQN" if double_dqn else "DQN"

    print(f"\nTraining {alg_name} (Seed {seed})")
    print(f"Total steps: {total_steps:,} | Updates: ~{total_steps // batch_size:,}")

    obs, _ = env.reset()
    ep_return = 0.0
    ep_length = 0
    ep_count = 0
    start_time = time.time()

    for step in range(1, total_steps + 1):
        # Epsilon decay
        eps = eps_end + (eps_start - eps_end) * np.exp(-step / eps_decay)

        # Select action (vectorized, keine Tensor-Konvertierung in jedem Step)
        if random.random() < eps:
            a = random.randrange(n_actions)
        else:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(DEVICE)
                a = int(q_online(obs_t).argmax(dim=1).item())

        # Step environment
        obs2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        ep_return += r
        ep_length += 1

        # Push to buffer (numpy operations, schnell)
        buffer.push(obs, a, r, obs2, done)

        # Training step (nur nach genug Samples)
        if step >= start_learning and len(buffer) >= batch_size:
            s_t, a_t, r_t, s2_t, d_t = buffer.sample(batch_size)

            with torch.no_grad():
                if double_dqn:
                    a_best = q_online(s2_t).argmax(dim=1)
                    max_q_s2 = q_target(s2_t).gather(1, a_best.unsqueeze(1)).squeeze(1)
                else:
                    max_q_s2 = q_target(s2_t).max(dim=1)[0]
                target = r_t + gamma * max_q_s2 * (1 - d_t)

            q_sa = q_online(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
            loss = (q_sa - target).pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update target network (seltener)
            if step % target_update_freq == 0:
                q_target.load_state_dict(q_online.state_dict())

        obs = obs2

        # Episode end
        if done:
            episode_returns.append(ep_return)
            episode_lengths.append(ep_length)
            ep_count += 1

            obs, _ = env.reset()
            ep_return = 0.0
            ep_length = 0

            # Progress update (nur alle 50 Episoden)
            if ep_count % 50 == 0:
                elapsed = time.time() - start_time
                avg_return = np.mean(episode_returns[-50:])
                steps_per_sec = step / elapsed
                print(f"Episode {ep_count:4d} | Avg Return: {avg_return:8.2f} | "
                      f"Speed: {steps_per_sec:.0f} steps/sec | Eps: {eps:.3f}")

    env.close()
    elapsed = time.time() - start_time
    print(f"✓ Fertig in {elapsed:.1f} Sekunden")

    return {
        "name": alg_name,
        "episode_returns": np.array(episode_returns),
        "episode_lengths": np.array(episode_lengths),
        "q_online": q_online,
    }

# ============================================================================
# MOVING AVERAGE
# ============================================================================

def moving_average(x, window=50):
    x = np.asarray(x, dtype=np.float32)
    if len(x) < window:
        return x
    c = np.cumsum(np.insert(x, 0, 0))
    return (c[window:] - c[:-window]) / window

# ============================================================================
# MAIN
# ============================================================================

print("\n" + "="*60)
print("DQN vs Double DQN auf LunarLander-v3 (OPTIMIERT)")
print("="*60)

start_total = time.time()

# Schnell trainieren mit weniger Steps (50k statt 200k)
dqn_results = train_fast(double_dqn=False, total_steps=50_000, seed=42)
ddqn_results = train_fast(double_dqn=True, total_steps=50_000, seed=43)

total_elapsed = time.time() - start_total
print(f"\n✓ Gesamttraining: {total_elapsed:.1f} Sekunden ({total_elapsed/60:.1f} Min)")

# ============================================================================
# PLOTS (MINIMAL)
# ============================================================================

print("\nGeneriere Plots...", end=" ", flush=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Returns
for results, label, color in [
    (dqn_results, "DQN", "#d62728"),
    (ddqn_results, "Double DQN", "#2ca02c")
]:
    ma = moving_average(results["episode_returns"], window=50)
    ax1.plot(ma, label=label, linewidth=2.5, color=color, alpha=0.8)

ax1.axhline(y=200, color="gray", linestyle="--", alpha=0.5, label="Success (200)")
ax1.set_xlabel("Episode", fontsize=12, fontweight="bold")
ax1.set_ylabel("Return (Moving Avg)", fontsize=12, fontweight="bold")
ax1.set_title("LunarLander: DQN vs Double DQN", fontsize=13, fontweight="bold")
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

# Episode Length
for results, label, color in [
    (dqn_results, "DQN", "#d62728"),
    (ddqn_results, "Double DQN", "#2ca02c")
]:
    ma = moving_average(results["episode_lengths"], window=50)
    ax2.plot(ma, label=label, linewidth=2.5, color=color, alpha=0.8)

ax2.set_xlabel("Episode", fontsize=12, fontweight="bold")
ax2.set_ylabel("Episode Length", fontsize=12, fontweight="bold")
ax2.set_title("Episode Längen", fontsize=13, fontweight="bold")
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/lunar_lander_results.png", dpi=150, bbox_inches="tight")
plt.close()

print("✓")

# ============================================================================
# EVALUATION
# ============================================================================

print("\nEvaluation...", end=" ", flush=True)

@torch.no_grad()
def evaluate(q_net, n_episodes=10):
    env = make_env(seed=999)
    returns = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=999 + ep)
        done = False
        total_r = 0.0

        while not done:
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(DEVICE)
            a = int(q_net(obs_t).argmax(dim=1).item())
            obs, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total_r += r

        returns.append(total_r)

    env.close()
    return np.mean(returns), np.std(returns)

mean_dqn, std_dqn = evaluate(dqn_results["q_online"], n_episodes=10)
mean_ddqn, std_ddqn = evaluate(ddqn_results["q_online"], n_episodes=10)

print("✓")

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "="*60)
print("ERGEBNISSE")
print("="*60)
print(f"\nDQN:")
print(f"  Training Episodes: {len(dqn_results['episode_returns'])}")
print(f"  Mean Return (last 50): {np.mean(dqn_results['episode_returns'][-50:]):.2f}")
print(f"  Eval Mean Return: {mean_dqn:.2f} ± {std_dqn:.2f}")

print(f"\nDouble DQN:")
print(f"  Training Episodes: {len(ddqn_results['episode_returns'])}")
print(f"  Mean Return (last 50): {np.mean(ddqn_results['episode_returns'][-50:]):.2f}")
print(f"  Eval Mean Return: {mean_ddqn:.2f} ± {std_ddqn:.2f}")

winner = "Double DQN" if mean_ddqn > mean_dqn else "DQN"
print(f"\n🏆 Gewinner: {winner}")
print(f"   Unterschied: {abs(mean_dqn - mean_ddqn):.2f} Return Points")

print(f"\n⏱️  Gesamtzeit: {total_elapsed:.1f} Sekunden ({total_elapsed/60:.1f} Min)")
print(f"📊 Plot gespeichert: {OUTPUT_DIR}/lunar_lander_results.png")
print("="*60)

