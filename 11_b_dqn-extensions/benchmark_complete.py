#!/usr/bin/env python3
"""
DQN Extensions: Complete Benchmark & Comparison Study
All 8 variants trained end-to-end with robust error handling
"""

import math
import random
from collections import deque, namedtuple
from typing import Dict
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    import gymnasium as gym
except:
    import gym

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

print("="*80)
print("DQN BENCHMARK & COMPARISON STUDY - STARTING")
print("="*80)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Device: {device}")

# Set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================================
# CONFIGURATION
# ============================================================================

ENV_ID = "LunarLander-v3"

# Training config
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 128
BUFFER_SIZE = 100_000
LEARNING_STARTS = 2_000
TARGET_UPDATE_EVERY = 1000
TRAIN_FREQ = 1
GRAD_CLIP_NORM = 10.0

# Reduced for faster benchmarking
TOTAL_STEPS = 100_000
EVAL_EVERY = 5_000
EVAL_EPISODES = 5

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 50_000

PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_END = 1.0
PER_BETA_STEPS = 100_000
PER_EPS = 1e-6

print(f"✓ Configuration loaded")
print(f"  - Environment: {ENV_ID}")
print(f"  - Total steps: {TOTAL_STEPS:,}")
print(f"  - Eval frequency: Every {EVAL_EVERY:,} steps\n")

# ============================================================================
# HELPERS
# ============================================================================

def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env

def epsilon_by_step(step):
    if step >= EPS_DECAY_STEPS:
        return EPS_END
    frac = step / EPS_DECAY_STEPS
    return EPS_START + frac * (EPS_END - EPS_START)

def compute_beta(step):
    if step >= PER_BETA_STEPS:
        return PER_BETA_END
    frac = step / PER_BETA_STEPS
    return PER_BETA_START + frac * (PER_BETA_END - PER_BETA_START)

# ============================================================================
# TRANSITIONS & BUFFERS
# ============================================================================

Transition = namedtuple("Transition", ["s", "a", "r", "s2", "done"])

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buf)

    def add(self, s, a, r, s2, done):
        self.buf.append(Transition(s, a, r, s2, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        return self._to_tensors(batch), None, None

    @staticmethod
    def _to_tensors(batch):
        s  = torch.tensor(np.array([b.s for b in batch]), dtype=torch.float32, device=device)
        a  = torch.tensor(np.array([b.a for b in batch]), dtype=torch.int64, device=device).unsqueeze(-1)
        r  = torch.tensor(np.array([b.r for b in batch]), dtype=torch.float32, device=device).unsqueeze(-1)
        s2 = torch.tensor(np.array([b.s2 for b in batch]), dtype=torch.float32, device=device)
        d  = torch.tensor(np.array([b.done for b in batch]), dtype=torch.float32, device=device).unsqueeze(-1)
        return s, a, r, s2, d

class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = np.empty(capacity, dtype=object)
        self.write = 0
        self.size = 0

    def total(self) -> float:
        return float(self.tree[0])

    def add(self, p: float, data):
        leaf_idx = self.write + (self.capacity - 1)
        self.data[self.write] = data
        self.update(leaf_idx, p)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx: int, p: float):
        change = p - self.tree[idx]
        self.tree[idx] = p
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, s: float):
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - (self.capacity - 1)
        return idx, float(self.tree[idx]), self.data[data_idx]

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float):
        self.alpha = alpha
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

    def __len__(self):
        return self.tree.size

    def add(self, s, a, r, s2, done):
        data = Transition(s, a, r, s2, done)
        p = (self.max_priority + PER_EPS) ** self.alpha
        self.tree.add(p, data)

    def sample(self, batch_size: int, beta: float):
        total = self.tree.total()
        if total <= 0:
            raise RuntimeError("SumTree total priority is zero")

        batch = []
        idxs = []
        priorities = []
        segment = total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            if data is None:
                s = random.uniform(0.0, total)
                idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        probs = np.array(priorities, dtype=np.float32) / (total + 1e-8)
        probs = np.clip(probs, 1e-8, None)
        weights = (len(self) * probs) ** (-beta)
        weights = weights / (weights.max() + 1e-8)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(-1)

        return ReplayBuffer._to_tensors(batch), idxs, weights_t

    def update_priorities(self, idxs, priorities):
        for idx, p in zip(idxs, priorities):
            p = max(float(p), PER_EPS)
            self.max_priority = max(self.max_priority, p)
            self.tree.update(idx, (p ** self.alpha))

# ============================================================================
# NETWORKS
# ============================================================================

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_eps", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _f(x):
        return torch.sign(x) * torch.sqrt(torch.abs(x))

    def reset_noise(self):
        eps_in = torch.randn(self.in_features, device=self.weight_mu.device)
        eps_out = torch.randn(self.out_features, device=self.weight_mu.device)
        eps_in = self._f(eps_in)
        eps_out = self._f(eps_out)
        self.weight_eps.copy_(eps_out.unsqueeze(1) * eps_in.unsqueeze(0))
        self.bias_eps.copy_(eps_out)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)

class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, dueling=False, noisy=False, hidden=256):
        super().__init__()
        self.dueling = dueling
        self.noisy = noisy

        Linear = NoisyLinear if noisy else nn.Linear

        self.feature = nn.Sequential(
            Linear(obs_dim, hidden),
            nn.ReLU(),
            Linear(hidden, hidden),
            nn.ReLU(),
        )

        if dueling:
            self.value = nn.Sequential(
                Linear(hidden, hidden),
                nn.ReLU(),
                Linear(hidden, 1)
            )
            self.adv = nn.Sequential(
                Linear(hidden, hidden),
                nn.ReLU(),
                Linear(hidden, n_actions)
            )
        else:
            self.head = nn.Sequential(
                Linear(hidden, hidden),
                nn.ReLU(),
                Linear(hidden, n_actions)
            )

    def reset_noise(self):
        if not self.noisy:
            return
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, x):
        z = self.feature(x)
        if self.dueling:
            v = self.value(z)
            a = self.adv(z)
            a = a - a.mean(dim=1, keepdim=True)
            return v + a
        else:
            return self.head(z)

# ============================================================================
# AGENT
# ============================================================================

class DQNAgent:
    def __init__(self, obs_dim, n_actions, config: Dict):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.config = config

        self.use_double = config.get('use_double', False)
        self.use_dueling = config.get('use_dueling', False)
        self.use_noisy = config.get('use_noisy', False)
        self.use_per = config.get('use_per', False)
        self.lr = config.get('lr', LR)

        self.q_online = QNetwork(obs_dim, n_actions, dueling=self.use_dueling, noisy=self.use_noisy).to(device)
        self.q_target = QNetwork(obs_dim, n_actions, dueling=self.use_dueling, noisy=self.use_noisy).to(device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()

        self.optimizer = optim.Adam(self.q_online.parameters(), lr=self.lr)

        if self.use_per:
            self.buffer = PrioritizedReplayBuffer(BUFFER_SIZE, alpha=PER_ALPHA)
        else:
            self.buffer = ReplayBuffer(BUFFER_SIZE)

        self.losses = []
        self.eval_returns = []
        self.eval_stds = []
        self.eval_steps = []

    def select_action(self, obs, step):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        if self.use_noisy:
            self.q_online.reset_noise()
            qvals = self.q_online(obs_t)
            return int(qvals.argmax(dim=1).item())

        eps = epsilon_by_step(step)
        if random.random() < eps:
            return env.action_space.sample()
        qvals = self.q_online(obs_t)
        return int(qvals.argmax(dim=1).item())

    def train_step(self, step):
        if len(self.buffer) < max(LEARNING_STARTS, BATCH_SIZE):
            return None

        if self.use_per:
            beta = compute_beta(step)
            (s, a, r, s2, done), idxs, weights = self.buffer.sample(BATCH_SIZE, beta=beta)
        else:
            (s, a, r, s2, done), idxs, weights = self.buffer.sample(BATCH_SIZE)
            weights = torch.ones((BATCH_SIZE, 1), device=device)

        if self.use_noisy:
            self.q_online.reset_noise()
            self.q_target.reset_noise()

        q = self.q_online(s).gather(1, a)

        with torch.no_grad():
            if self.use_double:
                a2 = self.q_online(s2).argmax(dim=1, keepdim=True)
                q2 = self.q_target(s2).gather(1, a2)
            else:
                q2 = self.q_target(s2).max(dim=1, keepdim=True).values
            y = r + GAMMA * (1.0 - done) * q2

        td_err = (y - q)
        loss = (weights * td_err.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_online.parameters(), GRAD_CLIP_NORM)
        self.optimizer.step()

        if self.use_per:
            prios = td_err.detach().abs().squeeze(-1).cpu().numpy() + PER_EPS
            self.buffer.update_priorities(idxs, prios)

        if step % TARGET_UPDATE_EVERY == 0:
            self.q_target.load_state_dict(self.q_online.state_dict())

        return float(loss.item())

    @torch.no_grad()
    def evaluate(self, env_id, episodes=5):
        e = make_env(env_id, SEED + 999)
        returns = []
        for _ in range(episodes):
            obs, _ = e.reset()
            done = False
            total = 0.0
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                qvals = self.q_online(obs_t)
                a = int(qvals.argmax(dim=1).item())
                obs, r, terminated, truncated, _ = e.step(a)
                done = terminated or truncated
                total += r
            returns.append(total)
        e.close()
        return float(np.mean(returns)), float(np.std(returns))

    def train(self, env_id, total_steps=TOTAL_STEPS, eval_every=EVAL_EVERY, eval_episodes=EVAL_EPISODES):
        env = make_env(env_id, SEED)
        obs, _ = env.reset()

        start_time = time.time()

        for step in range(1, total_steps + 1):
            a = self.select_action(obs, step)
            obs2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            self.buffer.add(obs, a, r, obs2, done)
            obs = obs2

            if step % TRAIN_FREQ == 0:
                loss = self.train_step(step)
                if loss is not None:
                    self.losses.append(loss)

            if done:
                obs, _ = env.reset()

            if step % eval_every == 0:
                mean_r, std_r = self.evaluate(env_id, episodes=eval_episodes)
                self.eval_steps.append(step)
                self.eval_returns.append(mean_r)
                self.eval_stds.append(std_r)
                elapsed = time.time() - start_time
                print(f"  [{self.config['name']:25s}] step {step:7d} | return {mean_r:7.1f}±{std_r:5.2f} | time {elapsed:6.0f}s")

        env.close()
        total_time = time.time() - start_time
        print(f"  ✓ Completed in {total_time:.0f}s\n")

# ============================================================================
# MAIN BENCHMARK
# ============================================================================

if __name__ == "__main__":

    # Define all configurations
    configs = [
        {'name': 'Vanilla DQN', 'use_double': False, 'use_dueling': False, 'use_noisy': False, 'use_per': False},
        {'name': 'Double DQN', 'use_double': True, 'use_dueling': False, 'use_noisy': False, 'use_per': False},
        {'name': 'Dueling DQN', 'use_double': False, 'use_dueling': True, 'use_noisy': False, 'use_per': False},
        {'name': 'Double + Dueling', 'use_double': True, 'use_dueling': True, 'use_noisy': False, 'use_per': False},
        {'name': 'Noisy DQN', 'use_double': False, 'use_dueling': False, 'use_noisy': True, 'use_per': False},
        {'name': 'PER (Vanilla)', 'use_double': False, 'use_dueling': False, 'use_noisy': False, 'use_per': True},
        {'name': 'Double + Dueling + PER', 'use_double': True, 'use_dueling': True, 'use_noisy': False, 'use_per': True},
        {'name': 'All Extensions', 'use_double': True, 'use_dueling': True, 'use_noisy': True, 'use_per': True},
    ]

    # Initialize environment
    try:
        env = make_env(ENV_ID, SEED)
        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n
        env.close()
        print(f"✓ Environment initialized: obs_dim={obs_dim}, n_actions={n_actions}\n")
    except Exception as e:
        print(f"ERROR: Failed to initialize environment: {e}")
        traceback.print_exc()
        exit(1)

    # Train all agents
    agents = {}
    print("STARTING TRAINING OF ALL 8 VARIANTS")
    print("="*80 + "\n")

    for i, config in enumerate(configs, 1):
        try:
            print(f"[{i}/8] Training: {config['name']}")
            agent = DQNAgent(obs_dim, n_actions, config)
            agent.train(ENV_ID, total_steps=TOTAL_STEPS, eval_every=EVAL_EVERY, eval_episodes=EVAL_EPISODES)
            agents[config['name']] = agent
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            traceback.print_exc()
            continue

    print("="*80)
    print("TRAINING COMPLETED - GENERATING RESULTS")
    print("="*80 + "\n")

    if not agents:
        print("ERROR: No agents trained successfully!")
        exit(1)

    # ========================================================================
    # ANALYSIS
    # ========================================================================

    # 1. Metrics Table
    metrics_summary = []
    for name, agent in agents.items():
        returns = np.array(agent.eval_returns)
        steps = np.array(agent.eval_steps)

        max_perf = returns.max()
        final_perf = returns[-1]
        final_std = agent.eval_stds[-1]
        avg_perf = returns.mean()
        stability = np.mean(agent.eval_stds[-3:] if len(agent.eval_stds) >= 3 else agent.eval_stds)

        threshold_80 = final_perf * 0.8
        idx_80 = np.where(returns >= threshold_80)[0]
        steps_to_80 = steps[idx_80[0]] if len(idx_80) > 0 else TOTAL_STEPS

        metrics_summary.append({
            'Variant': name,
            'Final Return': f"{final_perf:.1f}",
            'Final Std': f"{final_std:.2f}",
            'Max Return': f"{max_perf:.1f}",
            'Avg Return': f"{avg_perf:.1f}",
            'Stability': f"{stability:.2f}",
            'Steps to 80%': f"{int(steps_to_80)}",
            'Improvement': f"{returns[-1] - returns[0]:.1f}",
        })

    metrics_df = pd.DataFrame(metrics_summary)

    print("\n" + "="*120)
    print("PERFORMANCE METRICS TABLE")
    print("="*120)
    print(metrics_df.to_string(index=False))
    print("="*120 + "\n")

    # Save metrics
    metrics_df.to_csv('/home/isc-den/cas-artificial-intelligence/11_b_dqn-extensions/benchmark_metrics.csv', index=False)
    print("✓ Saved: benchmark_metrics.csv")

    # 2. Visualization: Learning Curves
    print("Generating: Learning Curves Visualization...")
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    ax = axes[0]
    for name, agent in agents.items():
        steps = np.array(agent.eval_steps)
        returns = np.array(agent.eval_returns)
        stds = np.array(agent.eval_stds)
        ax.plot(steps, returns, marker='o', label=name, linewidth=2, markersize=5)
        ax.fill_between(steps, returns - stds, returns + stds, alpha=0.15)

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Evaluation Return', fontsize=12)
    ax.set_title('DQN Variants: Learning Curves (with Std Dev)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name, agent in agents.items():
        steps = np.array(agent.eval_steps)
        returns = np.array(agent.eval_returns)
        max_val = returns.max()
        min_val = returns.min()
        if max_val != min_val:
            normalized = (returns - min_val) / (max_val - min_val)
        else:
            normalized = returns / max_val if max_val != 0 else returns
        ax.plot(steps, normalized, marker='s', label=name, linewidth=2.5, markersize=6)

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Normalized Performance', fontsize=12)
    ax.set_title('Normalized Learning Curves (0-1 scale)', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/isc-den/cas-artificial-intelligence/11_b_dqn-extensions/benchmark_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: benchmark_learning_curves.png")

    # 3. Visualization: Final Performance
    print("Generating: Final Performance Comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    variants = list(agents.keys())
    final_returns = [float(agents[v].eval_returns[-1]) for v in variants]
    final_stds = [float(agents[v].eval_stds[-1]) for v in variants]
    colors_palette = sns.color_palette("husl", len(agents))

    ax = axes[0]
    x_pos = np.arange(len(variants))
    bars = ax.bar(x_pos, final_returns, yerr=final_stds, capsize=5, alpha=0.8, color=colors_palette, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('DQN Variant', fontsize=12)
    ax.set_ylabel('Final Evaluation Return', fontsize=12)
    ax.set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(variants, rotation=45, ha='right')
    ax.grid(True, axis='y', alpha=0.3)

    for i, (bar, val, std) in enumerate(zip(bars, final_returns, final_stds)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax = axes[1]
    bars = ax.bar(x_pos, final_stds, alpha=0.8, color=colors_palette, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('DQN Variant', fontsize=12)
    ax.set_ylabel('Evaluation Std Dev', fontsize=12)
    ax.set_title('Stability Comparison (Lower = Better)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(variants, rotation=45, ha='right')
    ax.grid(True, axis='y', alpha=0.3)

    for bar, val in zip(bars, final_stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('/home/isc-den/cas-artificial-intelligence/11_b_dqn-extensions/benchmark_final_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: benchmark_final_performance.png")

    # 4. Visualization: Learning Speed
    print("Generating: Learning Speed Comparison...")
    fig, ax = plt.subplots(figsize=(14, 8))

    speed_data = []
    for name, agent in agents.items():
        returns = np.array(agent.eval_returns)
        steps = np.array(agent.eval_steps)
        final_perf = returns[-1]
        threshold_50 = returns[0] + (final_perf - returns[0]) * 0.5
        threshold_80 = returns[0] + (final_perf - returns[0]) * 0.8

        idx_50 = np.where(returns >= threshold_50)[0]
        idx_80 = np.where(returns >= threshold_80)[0]

        steps_50 = steps[idx_50[0]] if len(idx_50) > 0 else TOTAL_STEPS
        steps_80 = steps[idx_80[0]] if len(idx_80) > 0 else TOTAL_STEPS

        speed_data.append({'Variant': name, 'Steps to 50%': steps_50, 'Steps to 80%': steps_80})

    speed_df = pd.DataFrame(speed_data)
    x_pos = np.arange(len(speed_df))
    width = 0.35

    bars1 = ax.bar(x_pos - width/2, speed_df['Steps to 50%'], width, label='Steps to 50%', alpha=0.8, color='steelblue', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x_pos + width/2, speed_df['Steps to 80%'], width, label='Steps to 80%', alpha=0.8, color='coral', edgecolor='black', linewidth=1.5)

    ax.set_xlabel('DQN Variant', fontsize=12)
    ax.set_ylabel('Training Steps', fontsize=12)
    ax.set_title('Learning Speed: Steps to Reach Performance Thresholds', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(speed_df['Variant'], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('/home/isc-den/cas-artificial-intelligence/11_b_dqn-extensions/benchmark_learning_speed.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: benchmark_learning_speed.png")

    # Save detailed results
    print("Saving detailed results...")
    results_data = []
    for name, agent in agents.items():
        for step, ret, std in zip(agent.eval_steps, agent.eval_returns, agent.eval_stds):
            results_data.append({'Variant': name, 'Step': step, 'Return': ret, 'Std': std})

    results_df = pd.DataFrame(results_data)
    results_df.to_csv('/home/isc-den/cas-artificial-intelligence/11_b_dqn-extensions/benchmark_detailed_results.csv', index=False)
    print("✓ Saved: benchmark_detailed_results.csv")

    print("\n" + "="*80)
    print("BENCHMARK COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated Files:")
    print("  1. benchmark_metrics.csv")
    print("  2. benchmark_detailed_results.csv")
    print("  3. benchmark_learning_curves.png")
    print("  4. benchmark_final_performance.png")
    print("  5. benchmark_learning_speed.png")
    print("\nAll 8 variants have been trained and analyzed!")
    print("="*80)

