#!/usr/bin/env python3
"""
PER Training Script - CORRECTED VERSION
Fixed the critical SumTree index bug
"""

import math
import random
import time
import json
from collections import deque, namedtuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================
# CONFIGURATION
# ============================================

SEED = 42
ENV_ID = "LunarLander-v3"

GAMMA = 0.99
LR = 2e-3
BATCH_SIZE = 256
BUFFER_SIZE = 200000
LEARNING_STARTS = 2000
TARGET_UPDATE_EVERY = 2000
TAU = 0.005
TRAIN_FREQ = 1
GRAD_CLIP_NORM = 10.0

TOTAL_STEPS = 300000
EVAL_EVERY = 15000
EVAL_EPISODES = 5

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 30000

# ⭐ REDUCED PER ALPHA FOR STABILITY
PER_ALPHA = 0.4  # Changed from 0.6 (less aggressive prioritization)
PER_BETA_START = 0.4
PER_BETA_END = 1.0
PER_BETA_STEPS = 60000
PER_EPS = 1e-6

CONFIG = {
    'uniform': {
        'name': 'Uniform Replay (Baseline)',
        'use_per': False,
        'color': '#1f77b4',
    },
    'per': {
        'name': 'Prioritized Experience Replay (PER)',
        'use_per': True,
        'color': '#ff7f0e',
    }
}

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_env(env_id, seed):
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env

seed_everything(SEED)

env_temp = make_env(ENV_ID, SEED)
obs_dim = env_temp.observation_space.shape[0]
n_actions = env_temp.action_space.n
env_temp.close()

print(f"Observation dimension: {obs_dim}")
print(f"Number of actions: {n_actions}")

# ============================================
# BUFFERS
# ============================================

Transition = namedtuple("Transition", ["s", "a", "r", "s2", "done"])

class UniformReplayBuffer:
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
        s  = torch.tensor(np.array([b.s for b in batch]), dtype=torch.float32, device=device)
        a  = torch.tensor(np.array([b.a for b in batch]), dtype=torch.int64, device=device).unsqueeze(-1)
        r  = torch.tensor(np.array([b.r for b in batch]), dtype=torch.float32, device=device).unsqueeze(-1)
        s2 = torch.tensor(np.array([b.s2 for b in batch]), dtype=torch.float32, device=device)
        d  = torch.tensor(np.array([b.done for b in batch]), dtype=torch.float32, device=device).unsqueeze(-1)
        return s, a, r, s2, d

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

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.alpha = alpha
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        self.capacity = capacity
        # ⭐ CRITICAL FIX: Track leaf indices for each data position
        self.leaf_indices = {}

    def __len__(self):
        return self.tree.size

    def add(self, s, a, r, s2, done):
        data = Transition(s, a, r, s2, done)
        p = (self.max_priority + PER_EPS) ** self.alpha

        # ⭐ Calculate leaf index BEFORE adding
        leaf_idx = self.tree.write + (self.capacity - 1)

        self.tree.add(p, data)

        # ⭐ Store mapping: data_index -> leaf_index for later priority updates
        self.leaf_indices[self.tree.write - 1 if self.tree.write > 0 else self.capacity - 1] = leaf_idx

    def sample(self, batch_size, beta):
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

            leaf_idx, p, data = self.tree.get(s)

            # ⭐ Retry if data is None
            retry_count = 0
            while (data is None) and retry_count < 5:
                s = random.uniform(a, b)
                leaf_idx, p, data = self.tree.get(s)
                retry_count += 1

            if data is None:
                continue

            batch.append(data)
            idxs.append(leaf_idx)  # ⭐ Store LEAF index, not data index
            priorities.append(p)

        if len(batch) == 0:
            raise RuntimeError("Could not sample any valid transitions")

        probs = np.array(priorities, dtype=np.float32) / (total + 1e-8)
        probs = np.clip(probs, 1e-8, None)

        weights = (len(self) * probs) ** (-beta)
        weights = weights / (weights.max() + 1e-8)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(-1)

        return UniformReplayBuffer._to_tensors(batch), idxs, weights_t

    def update_priorities(self, idxs, priorities):
        """
        ⭐ CRITICAL FIX: idxs are now LEAF indices from SumTree, not data indices
        """
        for leaf_idx, p in zip(idxs, priorities):
            p = max(float(p), PER_EPS)
            self.max_priority = max(self.max_priority, p)
            # ⭐ Update with LEAF index directly
            self.tree.update(leaf_idx, (p ** self.alpha))

# ============================================
# NETWORK
# ============================================

class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)

# ============================================
# TRAINING
# ============================================

def epsilon_by_step(step):
    if step >= EPS_DECAY_STEPS:
        return EPS_END
    return EPS_START + (step / EPS_DECAY_STEPS) * (EPS_END - EPS_START)

def compute_beta(step):
    if step >= PER_BETA_STEPS:
        return PER_BETA_END
    return PER_BETA_START + (step / PER_BETA_STEPS) * (PER_BETA_END - PER_BETA_START)

@torch.no_grad()
def select_action(obs, step, q_network):
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    eps = epsilon_by_step(step)
    if random.random() < eps:
        return np.random.randint(n_actions)
    return int(q_network(obs_t).argmax(dim=1).item())

@torch.no_grad()
def compute_target(s2, r, done, q_online, q_target):
    a2 = q_online(s2).argmax(dim=1, keepdim=True)
    q2 = q_target(s2).gather(1, a2)
    return r + GAMMA * (1.0 - done) * q2

def train_step(q_online, q_target, optimizer, buffer, step, use_per):
    if len(buffer) < max(LEARNING_STARTS, BATCH_SIZE):
        return None, None

    if use_per:
        beta = compute_beta(step)
        (s, a, r, s2, done), idxs, weights = buffer.sample(BATCH_SIZE, beta=beta)
    else:
        (s, a, r, s2, done), idxs, weights = buffer.sample(BATCH_SIZE)
        weights = torch.ones((BATCH_SIZE, 1), device=device)

    q = q_online(s).gather(1, a)

    with torch.no_grad():
        y = compute_target(s2, r, done, q_online, q_target)

    td_err = (y - q)
    loss = (weights * td_err.pow(2)).mean()

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_online.parameters(), GRAD_CLIP_NORM)
    optimizer.step()

    if use_per:
        # ⭐ CRITICAL FIX: TD-Error are now used to update priorities correctly
        prios = td_err.detach().abs().squeeze(-1).cpu().numpy() + PER_EPS
        buffer.update_priorities(idxs, prios)

    with torch.no_grad():
        for p_t, p in zip(q_target.parameters(), q_online.parameters()):
            p_t.data.mul_(1 - TAU).add_(TAU * p.data)

    return float(loss.item()), td_err.detach().abs().mean().item()

@torch.no_grad()
def evaluate(q_network, env_id, episodes=5):
    e = make_env(env_id, SEED + 999)
    returns = []
    for _ in range(episodes):
        obs, _ = e.reset()
        done = False
        total = 0.0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            a = int(q_network(obs_t).argmax(dim=1).item())
            obs, r, terminated, truncated, _ = e.step(a)
            done = terminated or truncated
            total += r
        returns.append(total)
    e.close()
    return float(np.mean(returns)), float(np.std(returns))

def run_training(config_key, config):
    print(f"\n{'='*70}")
    print(f"Starting Training: {config['name']}")
    print(f"{'='*70}")

    use_per = config['use_per']
    seed_everything(SEED)

    q_online = QNetwork(obs_dim, n_actions).to(device)
    q_target = QNetwork(obs_dim, n_actions).to(device)
    q_target.load_state_dict(q_online.state_dict())
    q_target.eval()

    optimizer = optim.Adam(q_online.parameters(), lr=LR)

    if use_per:
        buffer = PrioritizedReplayBuffer(BUFFER_SIZE, alpha=PER_ALPHA)
    else:
        buffer = UniformReplayBuffer(BUFFER_SIZE)

    env = make_env(ENV_ID, SEED)
    q_online.train()

    losses = []
    td_errors = []
    eval_steps = []
    eval_means = []
    eval_stds = []
    episode_returns = []
    episode_lengths = []

    obs, _ = env.reset()
    episode_return = 0.0
    episode_len = 0
    start_time = time.time()

    for step in tqdm(range(1, TOTAL_STEPS + 1), desc=config['name']):
        a = select_action(obs, step, q_online)
        obs2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        buffer.add(obs, a, r, obs2, done)
        obs = obs2
        episode_return += r
        episode_len += 1

        if step % TRAIN_FREQ == 0 and len(buffer) >= max(LEARNING_STARTS, BATCH_SIZE):
            try:
                loss, td_err = train_step(q_online, q_target, optimizer, buffer, step, use_per)
                if loss is not None:
                    losses.append(loss)
                    td_errors.append(td_err)
            except Exception as e:
                print(f"⚠️ Warning in train_step: {e}")
                continue

        if done:
            episode_returns.append(episode_return)
            episode_lengths.append(episode_len)
            obs, _ = env.reset()
            episode_return = 0.0
            episode_len = 0

        if step % EVAL_EVERY == 0:
            try:
                mean_r, std_r = evaluate(q_online, ENV_ID, episodes=EVAL_EPISODES)
                eval_steps.append(step)
                eval_means.append(mean_r)
                eval_stds.append(std_r)
                print(f"[step {step:7d}] eval return: {mean_r:.1f} ± {std_r:.1f}")
            except Exception as e:
                print(f"⚠️ Warning in evaluate: {e}")

    env.close()
    elapsed = time.time() - start_time

    print(f"\nTraining completed in {elapsed:.1f}s")
    if eval_means:
        print(f"Final eval return: {eval_means[-1]:.1f} ± {eval_stds[-1]:.1f}")

    return {
        'config_key': config_key,
        'config_name': config['name'],
        'use_per': use_per,
        'color': config['color'],
        'losses': losses,
        'td_errors': td_errors,
        'eval_steps': eval_steps,
        'eval_means': eval_means,
        'eval_stds': eval_stds,
        'episode_returns': episode_returns,
        'episode_lengths': episode_lengths,
        'elapsed_time': elapsed,
    }

# ============================================
# MAIN
# ============================================

print("\n" + "="*70)
print("STARTING EXPERIMENTS - CORRECTED PER")
print("="*70)

try:
    results_uniform = run_training('uniform', CONFIG['uniform'])
    results_per = run_training('per', CONFIG['per'])
    all_results = [results_uniform, results_per]

    output_dir = Path("/home/isc-den/cas-artificial-intelligence/11_dqn_extensions")

    results_summary = {
        'uniform': {
            'config_name': results_uniform['config_name'],
            'eval_means': [float(x) for x in results_uniform['eval_means']],
            'eval_stds': [float(x) for x in results_uniform['eval_stds']],
            'eval_steps': [int(x) for x in results_uniform['eval_steps']],
            'color': results_uniform['color'],
        },
        'per': {
            'config_name': results_per['config_name'],
            'eval_means': [float(x) for x in results_per['eval_means']],
            'eval_stds': [float(x) for x in results_per['eval_stds']],
            'eval_steps': [int(x) for x in results_per['eval_steps']],
            'color': results_per['color'],
        },
    }

    if len(results_uniform['eval_means']) > 0 and len(results_per['eval_means']) > 0:
        uniform_evals = np.array(results_uniform['eval_means'])
        per_evals = np.array(results_per['eval_means'])
        t_stat, p_value = stats.ttest_ind(per_evals, uniform_evals)
        results_summary['p_value'] = float(p_value)
        results_summary['improvement_percent'] = float(
            ((results_per['eval_means'][-1] - results_uniform['eval_means'][-1]) /
             abs(results_uniform['eval_means'][-1]) * 100)
        )

        print(f"\nT-Test p-value: {p_value:.6f}")
        print(f"PER improvement: {results_summary['improvement_percent']:.1f}%")

    with open(output_dir / 'results_summary_CORRECTED.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    print("✓ Saved: results_summary_CORRECTED.json")

    print("\n" + "="*70)
    print("VERGLEICH: Uniform Replay vs. PER (CORRECTED)")
    print("="*70)

    for res in all_results:
        print(f"\n{res['config_name']}:")
        print(f"  Final Return:        {res['eval_means'][-1]:.2f} ± {res['eval_stds'][-1]:.2f}")
        print(f"  Max Return:          {max(res['eval_means']):.2f}")
        print(f"  Mean Return:         {np.mean(res['eval_means']):.2f}")
        print(f"  Training Time:       {res['elapsed_time']:.1f}s")

    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE - CORRECTED VERSION")
    print("="*70)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

