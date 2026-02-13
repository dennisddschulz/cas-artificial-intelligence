#!/usr/bin/env python3
"""
MINIMALES & KORREKTES DQN Training - Ohne alle Extensions
Basiert auf DQN_Extensions.ipynb aber SIMPLIFIED
"""

import math
import random
from collections import deque, namedtuple
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Hyperparameters - STANDARD DQN
GAMMA = 0.99
LR = 1e-3  # REDUCED from 2e-3
BATCH_SIZE = 32  # REDUCED from 256
BUFFER_SIZE = 100_000
LEARNING_STARTS = 2_000
TARGET_UPDATE_EVERY = 1000  # Update more frequently
SOFT_UPDATE = True
TAU = 0.005

TOTAL_STEPS = 500_000  # INCREASED
EVAL_EVERY = 10_000
EVAL_EPISODES = 5

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 50_000

# PER params
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_END = 1.0
PER_BETA_STEPS = 100_000
PER_EPS = 1e-6

Transition = namedtuple("Transition", ["s", "a", "r", "s2", "done"])

# ============================================================================
# SIMPLE QNetwork - NO DUELING, NO NOISY
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


# ============================================================================
# UNIFORM REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buf)

    def add(self, s, a, r, s2, done):
        self.buf.append(Transition(s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        return self._to_tensors(batch), None, None

    @staticmethod
    def _to_tensors(batch):
        s = torch.tensor(np.array([b.s for b in batch]), dtype=torch.float32, device=device)
        a = torch.tensor(np.array([b.a for b in batch]), dtype=torch.int64, device=device).unsqueeze(-1)
        r = torch.tensor(np.array([b.r for b in batch]), dtype=torch.float32, device=device).unsqueeze(-1)
        s2 = torch.tensor(np.array([b.s2 for b in batch]), dtype=torch.float32, device=device)
        d = torch.tensor(np.array([b.done for b in batch]), dtype=torch.float32, device=device).unsqueeze(-1)
        return s, a, r, s2, d


# ============================================================================
# SUMTREE (from DQN_Extensions - exakt gleich)
# ============================================================================

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = np.empty(capacity, dtype=object)
        self.write = 0
        self.size = 0

    def total(self):
        return float(self.tree[0])

    def add(self, p, data):
        leaf_idx = self.write + (self.capacity - 1)
        self.data[self.write] = data
        self.update(leaf_idx, p)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, s):
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


# ============================================================================
# PER BUFFER
# ============================================================================

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.alpha = alpha
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

    def __len__(self):
        return self.tree.size

    def add(self, s, a, r, s2, done):
        data = Transition(s, a, r, s2, done)
        p = (self.max_priority + PER_EPS) ** self.alpha
        self.tree.add(p, data)

    def sample(self, batch_size, beta):
        total = self.tree.total()
        if total <= 0:
            raise RuntimeError("SumTree total is zero")

        batch, idxs, priorities = [], [], []
        segment = total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)

            if data is None:
                s = random.uniform(0, total)
                idx, p, data = self.tree.get(s)

            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        probs = np.array(priorities) / (total + 1e-8)
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
# TRAINING FUNCTIONS
# ============================================================================

def make_env():
    env = gym.make("LunarLander-v3")
    env.reset(seed=SEED)
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

@torch.no_grad()
def evaluate(q_net, episodes=5):
    env = make_env()
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            qvals = q_net(obs_t)
            a = int(qvals.argmax(dim=1).item())
            obs, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total += r
        returns.append(total)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))

def train_dqn(use_per, config_name):
    print("\n" + "="*80)
    print(f"Training: {config_name}")
    print("="*80)

    # Create environment and get dims
    env = make_env()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Create networks
    q_online = QNetwork(obs_dim, n_actions).to(device)
    q_target = QNetwork(obs_dim, n_actions).to(device)
    q_target.load_state_dict(q_online.state_dict())
    q_target.eval()

    # Create optimizer
    optimizer = optim.Adam(q_online.parameters(), lr=LR)

    # Create buffer
    if use_per:
        buffer = PrioritizedReplayBuffer(BUFFER_SIZE, alpha=PER_ALPHA)
    else:
        buffer = ReplayBuffer(BUFFER_SIZE)

    # Training loop
    obs, _ = env.reset()
    episode_return = 0.0

    losses = []
    eval_steps = []
    eval_means = []
    eval_stds = []

    q_online.train()
    q_target.eval()

    for step in range(1, TOTAL_STEPS + 1):
        # Select action
        eps = epsilon_by_step(step)
        if random.random() < eps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                a = int(q_online(obs_t).argmax(dim=1).item())

        # Step environment
        obs2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        buffer.add(obs, a, r, obs2, done)
        obs = obs2
        episode_return += r

        if done:
            obs, _ = env.reset()
            episode_return = 0.0

        # Training
        if len(buffer) >= max(LEARNING_STARTS, BATCH_SIZE):
            if use_per:
                beta = compute_beta(step)
                (s, a_t, r_t, s2, d), idxs, weights = buffer.sample(BATCH_SIZE, beta=beta)
            else:
                (s, a_t, r_t, s2, d), idxs, weights = buffer.sample(BATCH_SIZE)
                weights = torch.ones((BATCH_SIZE, 1), device=device)

            # Double DQN
            q_pred = q_online(s).gather(1, a_t)

            with torch.no_grad():
                a_best = q_online(s2).argmax(dim=1, keepdim=True)
                q_target_vals = q_target(s2).gather(1, a_best)
                q_target_vals = r_t + GAMMA * (1.0 - d) * q_target_vals

            td_error = (q_target_vals - q_pred)
            loss = (weights * td_error.pow(2)).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_online.parameters(), 10.0)
            optimizer.step()

            losses.append(float(loss.item()))

            # PER priority update
            if use_per:
                prios = td_error.detach().abs().squeeze(-1).cpu().numpy() + PER_EPS
                buffer.update_priorities(idxs, prios)

            # Soft target update
            if SOFT_UPDATE:
                with torch.no_grad():
                    for p_t, p in zip(q_target.parameters(), q_online.parameters()):
                        p_t.data.mul_(1 - TAU).add_(TAU * p.data)
            else:
                if step % TARGET_UPDATE_EVERY == 0:
                    q_target.load_state_dict(q_online.state_dict())

        # Evaluation
        if step % EVAL_EVERY == 0:
            mean_r, std_r = evaluate(q_online, episodes=EVAL_EPISODES)
            eval_steps.append(step)
            eval_means.append(mean_r)
            eval_stds.append(std_r)
            print(f"[Step {step:7d}] Return: {mean_r:7.1f} ± {std_r:6.1f} | Buffer: {len(buffer):7d} | Loss: {losses[-1] if losses else 0:.4f}")

    env.close()

    print(f"\n{config_name} - FINAL:")
    print(f"  Final: {eval_means[-1]:.1f} ± {eval_stds[-1]:.1f}")
    print(f"  Best:  {np.max(eval_means):.1f}")
    print(f"  Mean:  {np.mean(eval_means):.1f} ± {np.std(eval_means):.1f}")

    return {
        'eval_steps': eval_steps,
        'eval_means': eval_means,
        'eval_stds': eval_stds,
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("SIMPLIFIED DQN VERIFICATION")
    print("="*80)
    print(f"Configuration:")
    print(f"  LR: {LR}, Batch: {BATCH_SIZE}, Buffer: {BUFFER_SIZE}")
    print(f"  Total Steps: {TOTAL_STEPS}")
    print(f"  Device: {device}")

    # Train
    results_uniform = train_dqn(use_per=False, config_name="Uniform Replay")
    results_per = train_dqn(use_per=True, config_name="Prioritized Experience Replay")

    # Save
    results = {
        'uniform': {
            'eval_steps': results_uniform['eval_steps'],
            'eval_means': results_uniform['eval_means'],
            'eval_stds': results_uniform['eval_stds'],
        },
        'per': {
            'eval_steps': results_per['eval_steps'],
            'eval_means': results_per['eval_means'],
            'eval_stds': results_per['eval_stds'],
        }
    }

    output_path = Path('/home/isc-den/cas-artificial-intelligence/11_dqn_extensions/results_simplified.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)

    u_means = np.array(results_uniform['eval_means'])
    p_means = np.array(results_per['eval_means'])

    print(f"Final: Uniform={u_means[-1]:.1f}, PER={p_means[-1]:.1f}")
    print(f"Best: Uniform={np.max(u_means):.1f}, PER={np.max(p_means):.1f}")
    print(f"Mean: Uniform={np.mean(u_means):.1f}, PER={np.mean(p_means):.1f}")

    improvement = ((p_means[-1] - u_means[-1]) / abs(u_means[-1])) * 100 if u_means[-1] != 0 else 0
    print(f"Improvement: {improvement:+.1f}%")

    print(f"\n✓ Saved: {output_path}")

