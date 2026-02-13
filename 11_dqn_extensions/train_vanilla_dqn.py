#!/usr/bin/env python3
"""
Vanilla DQN Training - Exact copy from DQN_Extensions.ipynb
NO Extensions: USE_DOUBLE=False, USE_DUELING=False, USE_NOISY=False
Compare: USE_PER=False vs USE_PER=True
"""

import math
import random
from collections import deque, namedtuple
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium as gym
import matplotlib.pyplot as plt

# ============================================================================
# SETUP
# ============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================================
# FLAGS - VANILLA DQN ONLY
# ============================================================================

USE_DOUBLE  = False
USE_DUELING = False
USE_NOISY   = False
# USE_PER will be set per training run

ENV_ID = "LunarLander-v3"

GAMMA = 0.99
LR = 2e-3
BATCH_SIZE = 256
BUFFER_SIZE = 200_000
LEARNING_STARTS = 2_000

TARGET_UPDATE_EVERY = 2000
SOFT_UPDATE = True
TAU = 0.005

TRAIN_FREQ = 1
GRAD_CLIP_NORM = 10.0

TOTAL_STEPS = 300_000
EVAL_EVERY = 15_000
EVAL_EPISODES = 5

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 30_000

# PER params
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_END = 1.0
PER_BETA_STEPS = 60_000
PER_EPS = 1e-6

# ============================================================================
# CLASSES
# ============================================================================

Transition = namedtuple("Transition", ["s", "a", "r", "s2", "done"])

def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env

# ============================================================================
# REPLAY BUFFERS
# ============================================================================

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

            # Safety loop: keep trying if data is None
            max_retries = 10
            retries = 0
            while data is None and retries < max_retries:
                s = random.uniform(0.0, total)
                idx, p, data = self.tree.get(s)
                retries += 1

            # If still None after retries, skip this sample
            if data is None:
                # Fallback: sample uniformly from buffer
                if len(self) > 0:
                    data = list(self.tree.data[:self.tree.size])[random.randint(0, self.tree.size - 1)]
                else:
                    continue

            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        # Handle case where we got fewer samples than batch_size
        if len(batch) < batch_size:
            # Pad with random samples from buffer
            while len(batch) < batch_size:
                idx = random.randint(0, self.tree.size - 1)
                data = self.tree.data[idx]
                if data is not None:
                    batch.append(data)
                    idxs.append(idx + (self.tree.capacity - 1))
                    priorities.append(self.tree.tree[idx + (self.tree.capacity - 1)])

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
# NETWORK - VANILLA DQN
# ============================================================================

class QNetwork(nn.Module):
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

    def reset_noise(self):
        pass  # No-op for vanilla network

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

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

@torch.no_grad()
def td_target_double(s2, r, done, q_online, q_target):
    a2 = q_online(s2).argmax(dim=1, keepdim=True)
    q2 = q_target(s2).gather(1, a2)
    return r + GAMMA * (1.0 - done) * q2

@torch.no_grad()
def td_target_vanilla(s2, r, done, q_online, q_target):
    q2 = q_target(s2).max(dim=1, keepdim=True).values
    return r + GAMMA * (1.0 - done) * q2

def train_step(step, q_online, q_target, optimizer, buffer, use_per):
    if len(buffer) < max(LEARNING_STARTS, BATCH_SIZE):
        return None

    if use_per:
        beta = compute_beta(step)
        (s, a, r, s2, done), idxs, weights = buffer.sample(BATCH_SIZE, beta=beta)
    else:
        (s, a, r, s2, done), idxs, weights = buffer.sample(BATCH_SIZE)
        weights = torch.ones((BATCH_SIZE, 1), device=device)

    q_online.reset_noise()
    q_target.reset_noise()

    q = q_online(s).gather(1, a)

    with torch.no_grad():
        if USE_DOUBLE:
            y = td_target_double(s2, r, done, q_online, q_target)
        else:
            y = td_target_vanilla(s2, r, done, q_online, q_target)

    td_err = (y - q)

    loss = (weights * td_err.pow(2)).mean()

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_online.parameters(), GRAD_CLIP_NORM)
    optimizer.step()

    # PER priority update
    if use_per:
        prios = td_err.detach().abs().squeeze(-1).cpu().numpy() + PER_EPS
        buffer.update_priorities(idxs, prios)

    # Target network update
    if SOFT_UPDATE:
        with torch.no_grad():
            for p_t, p in zip(q_target.parameters(), q_online.parameters()):
                p_t.data.mul_(1 - TAU).add_(TAU * p.data)
    else:
        if step % TARGET_UPDATE_EVERY == 0:
            q_target.load_state_dict(q_online.state_dict())

    return float(loss.item())

@torch.no_grad()
def select_action(obs, step, q_online, env):
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    if USE_NOISY:
        q_online.reset_noise()
        qvals = q_online(obs_t)
        return int(qvals.argmax(dim=1).item())

    eps = epsilon_by_step(step)
    if random.random() < eps:
        return env.action_space.sample()
    qvals = q_online(obs_t)
    return int(qvals.argmax(dim=1).item())

@torch.no_grad()
def evaluate(q_online, env_id, episodes=5):
    e = make_env(env_id, SEED + 999)
    returns = []
    for _ in range(episodes):
        obs, _ = e.reset()
        done = False
        total = 0.0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            qvals = q_online(obs_t)
            a = int(qvals.argmax(dim=1).item())
            obs, r, terminated, truncated, _ = e.step(a)
            done = terminated or truncated
            total += r
        returns.append(total)
    e.close()
    return float(np.mean(returns)), float(np.std(returns))

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def run_training(use_per, config_name):
    print("\n" + "="*80)
    print(f"Starting Training: {config_name}")
    print("="*80)
    print(f"Device: {device}")
    print(f"USE_PER: {use_per}")
    print(f"USE_DOUBLE: {USE_DOUBLE}, USE_DUELING: {USE_DUELING}, USE_NOISY: {USE_NOISY}")

    # Reset seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Create environment
    env = make_env(ENV_ID, SEED)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Create networks
    q_online = QNetwork(obs_dim, n_actions).to(device)
    q_target = QNetwork(obs_dim, n_actions).to(device)
    q_target.load_state_dict(q_online.state_dict())
    q_target.eval()

    print(f"Network architecture:")
    print(q_online)

    # Create optimizer
    optimizer = optim.Adam(q_online.parameters(), lr=LR)

    # Create buffer
    if use_per:
        buffer = PrioritizedReplayBuffer(BUFFER_SIZE, alpha=PER_ALPHA)
    else:
        buffer = ReplayBuffer(BUFFER_SIZE)

    # Training state
    obs, _ = env.reset()
    episode_return = 0.0
    episode = 0

    losses = []
    eval_steps = []
    eval_means = []
    eval_stds = []

    q_online.train()
    q_target.eval()

    for step in range(1, TOTAL_STEPS + 1):
        # Select and execute action
        a = select_action(obs, step, q_online, env)
        obs2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        # Store transition
        buffer.add(obs, a, r, obs2, done)

        obs = obs2
        episode_return += r

        # Train
        if step % TRAIN_FREQ == 0 and len(buffer) >= max(LEARNING_STARTS, BATCH_SIZE):
            loss = train_step(step, q_online, q_target, optimizer, buffer, use_per)
            if loss is not None:
                losses.append(loss)

        # Reset episode
        if done:
            obs, _ = env.reset()
            episode += 1
            episode_return = 0.0

        # Evaluate
        if step % EVAL_EVERY == 0:
            mean_r, std_r = evaluate(q_online, ENV_ID, episodes=EVAL_EPISODES)
            eval_steps.append(step)
            eval_means.append(mean_r)
            eval_stds.append(std_r)
            print(f"[Step {step:7d}] Return: {mean_r:7.1f} ± {std_r:6.1f} | Buffer: {len(buffer):7d} | Loss: {losses[-1] if losses else 0:.4f}")

    env.close()

    print(f"\n{config_name} - FINAL RESULTS:")
    print(f"  Final Return: {eval_means[-1]:.1f} ± {eval_stds[-1]:.1f}")
    print(f"  Max Return:   {np.max(eval_means):.1f}")
    print(f"  Mean Return:  {np.mean(eval_means):.1f} ± {np.std(eval_means):.1f}")

    return {
        'eval_steps': eval_steps,
        'eval_means': eval_means,
        'eval_stds': eval_stds,
        'losses': losses
    }

# ============================================================================
# EXECUTE
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("VANILLA DQN COMPARISON: Uniform Replay vs. PER")
    print("="*80)
    print(f"Configuration:")
    print(f"  Total Steps: {TOTAL_STEPS:,}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Buffer Size: {BUFFER_SIZE:,}")
    print(f"  Learning Rate: {LR}")
    print(f"  Extensions: NONE (USE_DOUBLE={USE_DOUBLE}, USE_DUELING={USE_DUELING}, USE_NOISY={USE_NOISY})")
    print("="*80)

    # Train without PER
    results_uniform = run_training(use_per=False, config_name="Uniform Replay (Baseline)")

    # Train with PER
    results_per = run_training(use_per=True, config_name="Prioritized Experience Replay")

    # Save results
    results = {
        'uniform': {
            'config_name': 'Uniform Replay (Baseline)',
            'eval_steps': results_uniform['eval_steps'],
            'eval_means': results_uniform['eval_means'],
            'eval_stds': results_uniform['eval_stds'],
        },
        'per': {
            'config_name': 'Prioritized Experience Replay',
            'eval_steps': results_per['eval_steps'],
            'eval_means': results_per['eval_means'],
            'eval_stds': results_per['eval_stds'],
        }
    }

    output_path = Path('/home/isc-den/cas-artificial-intelligence/11_dqn_extensions/results_vanilla_dqn.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Analysis
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)

    u_means = np.array(results_uniform['eval_means'])
    u_stds = np.array(results_uniform['eval_stds'])
    p_means = np.array(results_per['eval_means'])
    p_stds = np.array(results_per['eval_stds'])

    print(f"\nFinal Performance:")
    print(f"  Uniform: {u_means[-1]:8.1f} ± {u_stds[-1]:6.1f}")
    print(f"  PER:     {p_means[-1]:8.1f} ± {p_stds[-1]:6.1f}")
    print(f"  Difference: {p_means[-1] - u_means[-1]:+8.1f}")

    print(f"\nMean Performance (all evaluations):")
    print(f"  Uniform: {np.mean(u_means):8.1f} ± {np.std(u_means):6.1f}")
    print(f"  PER:     {np.mean(p_means):8.1f} ± {np.std(p_means):6.1f}")

    print(f"\nMax Performance:")
    print(f"  Uniform: {np.max(u_means):8.1f}")
    print(f"  PER:     {np.max(p_means):8.1f}")

    # Statistical test
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(p_means, u_means)
    print(f"\nT-Test (PER vs Uniform):")
    print(f"  t-statistic: {t_stat:8.4f}")
    print(f"  p-value:     {p_value:8.6f}")
    print(f"  Significant? {p_value < 0.05}")

    improvement = ((p_means[-1] - u_means[-1]) / abs(u_means[-1])) * 100 if u_means[-1] != 0 else 0
    print(f"\nPerformance Improvement: {improvement:+.1f}%")

    print(f"\n✓ Results saved to: {output_path}")
    print("="*80)

