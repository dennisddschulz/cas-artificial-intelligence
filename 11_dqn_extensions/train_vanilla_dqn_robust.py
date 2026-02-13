#!/usr/bin/env python3
"""
Vanilla DQN Training - ROBUST VERSION
Clean SumTree und PER implementation ohne None-Handling Bugs
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================================
# CONFIGURATION
# ============================================================================

USE_DOUBLE  = False
USE_DUELING = False
USE_NOISY   = False

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

PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_END = 1.0
PER_BETA_STEPS = 60_000
PER_EPS = 1e-6

# ============================================================================
# DATA STRUCTURES
# ============================================================================

Transition = namedtuple("Transition", ["s", "a", "r", "s2", "done"])

def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env

# ============================================================================
# UNIFORM REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    """Einfacher Uniform Replay Buffer - keine None-Probleme möglich"""
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

# ============================================================================
# SUMTREE - KORRIGIERT
# ============================================================================

class SumTree:
    """
    SumTree für PER - ROBUST IMPLEMENTIERUNG
    Keine None-Werte möglich wenn richtig implementiert
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = [None] * capacity  # Initialisiere mit None
        self.write_idx = 0
        self.size = 0

    def total(self) -> float:
        """Rückgabe der Gesamt-Priorität (Root)"""
        return float(self.tree[0])

    def add(self, p: float, data):
        """Füge neue Experience mit Priorität p hinzu"""
        # Calculate leaf index
        leaf_idx = self.write_idx + (self.capacity - 1)

        # Store data
        self.data[self.write_idx] = data

        # Update tree
        self._update_node(leaf_idx, p)

        # Move write pointer
        self.write_idx = (self.write_idx + 1) % self.capacity

        # Update size
        self.size = min(self.size + 1, self.capacity)

    def _update_node(self, idx: int, new_p: float):
        """Update node and propagate changes upward"""
        # Calculate change
        old_p = self.tree[idx]
        delta = new_p - old_p

        # Update node
        self.tree[idx] = new_p

        # Propagate change to parents
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def get(self, priority: float) -> tuple:
        """
        Finde Leaf der diese Priorität abdeckt
        priority: Wert zwischen 0 und total()
        Returns: (leaf_idx, priority_value, data)
        """
        idx = 0  # Start at root

        # Traverse down to leaf
        while idx < self.capacity - 1:  # While not a leaf
            left_child = 2 * idx + 1
            right_child = 2 * idx + 2

            left_sum = self.tree[left_child]

            if priority <= left_sum:
                idx = left_child
            else:
                priority -= left_sum
                idx = right_child

        # Convert leaf index to data index
        data_idx = idx - (self.capacity - 1)

        return idx, self.tree[idx], self.data[data_idx]

    def update(self, leaf_idx: int, new_p: float):
        """Update priority at leaf_idx"""
        self._update_node(leaf_idx, new_p)

# ============================================================================
# PER BUFFER
# ============================================================================

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay mit SumTree"""
    def __init__(self, capacity: int, alpha: float):
        self.alpha = alpha
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

    def __len__(self):
        return self.tree.size

    def add(self, s, a, r, s2, done):
        data = Transition(s, a, r, s2, done)
        # Neue Transitions bekommen max_priority
        p = (self.max_priority + PER_EPS) ** self.alpha
        self.tree.add(p, data)

    def sample(self, batch_size: int, beta: float):
        """
        Sample batch mit Importance-Sampling Correction
        beta: annealing parameter für IS weights
        """
        total = self.tree.total()

        if total <= 0:
            raise RuntimeError("SumTree has no priority!")

        batch = []
        leaf_idxs = []
        priorities = []

        # Stratified sampling
        segment = total / batch_size

        for i in range(batch_size):
            # Sample uniform in segment [i*segment, (i+1)*segment)
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)

            # Get from tree
            leaf_idx, p, data = self.tree.get(s)

            # KEIN None-Check nötig wenn SumTree korrekt!
            # Aber safeguard trotzdem
            if data is None:
                print(f"WARNING: Got None from SumTree at leaf_idx={leaf_idx}")
                # Resample
                s = random.uniform(0, total)
                leaf_idx, p, data = self.tree.get(s)
                if data is None:
                    raise RuntimeError("SumTree returned None twice!")

            batch.append(data)
            leaf_idxs.append(leaf_idx)
            priorities.append(p)

        # Compute IS weights
        probs = np.array(priorities, dtype=np.float32) / total
        max_weight = (self.tree.size * np.min(probs)) ** (-beta)

        weights = ((self.tree.size * probs) ** (-beta)) / max_weight
        weights_t = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(-1)

        # Convert batch to tensors
        batch_tensors = ReplayBuffer._to_tensors(batch)

        return batch_tensors, leaf_idxs, weights_t

    def update_priorities(self, leaf_idxs, td_errors):
        """Update priorities based on TD-errors"""
        for leaf_idx, td_err in zip(leaf_idxs, td_errors):
            # Priority = (|TD| + eps)^alpha
            p = (abs(float(td_err)) + PER_EPS) ** self.alpha
            self.max_priority = max(self.max_priority, p)
            self.tree.update(leaf_idx, p)

# ============================================================================
# NETWORK
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
        pass

# ============================================================================
# TRAINING
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
def td_target_vanilla(s2, r, done, q_online, q_target):
    q2 = q_target(s2).max(dim=1, keepdim=True).values
    return r + GAMMA * (1.0 - done) * q2

def train_step(step, q_online, q_target, optimizer, buffer, use_per):
    if len(buffer) < max(LEARNING_STARTS, BATCH_SIZE):
        return None

    if use_per:
        beta = compute_beta(step)
        (s, a, r, s2, done), leaf_idxs, weights = buffer.sample(BATCH_SIZE, beta=beta)
    else:
        (s, a, r, s2, done), _, _ = buffer.sample(BATCH_SIZE)
        leaf_idxs = None
        weights = torch.ones((BATCH_SIZE, 1), device=device)

    q = q_online(s).gather(1, a)

    with torch.no_grad():
        y = td_target_vanilla(s2, r, done, q_online, q_target)

    td_err = (y - q)
    loss = (weights * td_err.pow(2)).mean()

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_online.parameters(), GRAD_CLIP_NORM)
    optimizer.step()

    # Update PER priorities
    if use_per:
        td_errors = td_err.detach().squeeze(-1).cpu().numpy()
        buffer.update_priorities(leaf_idxs, td_errors)

    # Soft target update
    if SOFT_UPDATE:
        with torch.no_grad():
            for p_t, p in zip(q_target.parameters(), q_online.parameters()):
                p_t.data.mul_(1 - TAU).add_(TAU * p.data)

    return float(loss.item())

@torch.no_grad()
def select_action(obs, step, q_online, env):
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
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

def run_training(use_per, config_name):
    print("\n" + "="*80)
    print(f"Starting: {config_name}")
    print("="*80)
    print(f"USE_PER: {use_per}, Device: {device}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = make_env(ENV_ID, SEED)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_online = QNetwork(obs_dim, n_actions).to(device)
    q_target = QNetwork(obs_dim, n_actions).to(device)
    q_target.load_state_dict(q_online.state_dict())
    q_target.eval()

    optimizer = optim.Adam(q_online.parameters(), lr=LR)

    if use_per:
        buffer = PrioritizedReplayBuffer(BUFFER_SIZE, alpha=PER_ALPHA)
    else:
        buffer = ReplayBuffer(BUFFER_SIZE)

    obs, _ = env.reset()
    episode_return = 0.0

    losses = []
    eval_steps = []
    eval_means = []
    eval_stds = []

    q_online.train()
    q_target.eval()

    for step in range(1, TOTAL_STEPS + 1):
        a = select_action(obs, step, q_online, env)
        obs2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        buffer.add(obs, a, r, obs2, done)
        obs = obs2
        episode_return += r

        if step % TRAIN_FREQ == 0 and len(buffer) >= max(LEARNING_STARTS, BATCH_SIZE):
            loss = train_step(step, q_online, q_target, optimizer, buffer, use_per)
            if loss is not None:
                losses.append(loss)

        if done:
            obs, _ = env.reset()
            episode_return = 0.0

        if step % EVAL_EVERY == 0:
            mean_r, std_r = evaluate(q_online, ENV_ID, episodes=EVAL_EPISODES)
            eval_steps.append(step)
            eval_means.append(mean_r)
            eval_stds.append(std_r)
            loss_val = losses[-1] if losses else 0.0
            print(f"[{step:7d}] R={mean_r:7.1f}±{std_r:5.1f} Buf={len(buffer):7d} Loss={loss_val:.4f}")

    env.close()
    print(f"\n{config_name} FINAL: {eval_means[-1]:.1f} ± {eval_stds[-1]:.1f}")

    return {'eval_steps': eval_steps, 'eval_means': eval_means, 'eval_stds': eval_stds}

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("VANILLA DQN: Uniform vs PER (ROBUST)")
    print("="*80)

    results_uniform = run_training(use_per=False, config_name="Uniform Replay")
    results_per = run_training(use_per=True, config_name="Prioritized Experience Replay")

    results = {
        'uniform': {'eval_steps': results_uniform['eval_steps'],
                   'eval_means': results_uniform['eval_means'],
                   'eval_stds': results_uniform['eval_stds']},
        'per': {'eval_steps': results_per['eval_steps'],
               'eval_means': results_per['eval_means'],
               'eval_stds': results_per['eval_stds']}
    }

    output_path = Path('/home/isc-den/cas-artificial-intelligence/11_dqn_extensions/results_vanilla_dqn_robust.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    u_means = np.array(results_uniform['eval_means'])
    p_means = np.array(results_per['eval_means'])

    print(f"\nFinal:  Uniform={u_means[-1]:.1f}  PER={p_means[-1]:.1f}  Diff={p_means[-1]-u_means[-1]:+.1f}")
    print(f"Mean:   Uniform={np.mean(u_means):.1f}  PER={np.mean(p_means):.1f}  Diff={np.mean(p_means)-np.mean(u_means):+.1f}")
    print(f"Max:    Uniform={np.max(u_means):.1f}  PER={np.max(p_means):.1f}")

    from scipy import stats
    t_stat, p_value = stats.ttest_ind(p_means, u_means)
    print(f"\nT-Test: t={t_stat:.3f}, p={p_value:.6f}, Significant={p_value < 0.05}")
    print(f"\n✓ Saved: {output_path}")

