#!/usr/bin/env python3
"""
DQN Extensions: Hyperparameter Sensitivity Analysis
Tests Robustheit gegen verschiedene Hyperparameter-Einstellungen
"""

import math
import random
from collections import deque, namedtuple
from typing import Dict, List
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SEED = 42

# ============================================================================
# CONFIGURATION FOR HYPERPARAMETER TESTING
# ============================================================================

ENV_ID = "LunarLander-v3"

# SHORTER training for faster sensitivity testing
TOTAL_STEPS = 50_000
EVAL_EVERY = 2_500
EVAL_EPISODES = 3

# Standard hyperparams
GAMMA = 0.99
BATCH_SIZE = 128
BUFFER_SIZE = 100_000
LEARNING_STARTS = 2_000
TARGET_UPDATE_EVERY = 1000
TRAIN_FREQ = 1
GRAD_CLIP_NORM = 10.0

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 50_000

PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_END = 1.0
PER_BETA_STEPS = 100_000
PER_EPS = 1e-6

# ============================================================================
# HYPERPARAMETER SWEEP CONFIGURATIONS
# ============================================================================

SENSITIVITY_TESTS = {
    'learning_rate': {
        'parameter': 'lr',
        'values': [5e-4, 1e-3, 5e-3],
        'agent_config': {
            'use_double': True,
            'use_dueling': True,
            'use_noisy': False,
            'use_per': True,
        },
        'description': 'Learning Rate Sensitivity (Double + Dueling + PER)',
    },
    'batch_size': {
        'parameter': 'batch_size',
        'values': [64, 128, 256],
        'agent_config': {
            'use_double': True,
            'use_dueling': True,
            'use_noisy': False,
            'use_per': True,
        },
        'description': 'Batch Size Sensitivity (Double + Dueling + PER)',
    },
    'target_update': {
        'parameter': 'target_update_freq',
        'values': [500, 1000, 2000],
        'agent_config': {
            'use_double': True,
            'use_dueling': True,
            'use_noisy': False,
            'use_per': True,
        },
        'description': 'Target Update Frequency Sensitivity (Double + Dueling + PER)',
    },
    'gamma': {
        'parameter': 'gamma',
        'values': [0.95, 0.99, 0.999],
        'agent_config': {
            'use_double': True,
            'use_dueling': True,
            'use_noisy': False,
            'use_per': True,
        },
        'description': 'Gamma Discount Factor Sensitivity (Double + Dueling + PER)',
    },
}

# ============================================================================
# HELPER & NETWORKS (SAME AS BENCHMARK)
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
# AGENT WITH CONFIGURABLE HYPERPARAMS
# ============================================================================

class DQNAgent:
    def __init__(self, obs_dim, n_actions, config: Dict, param_overrides: Dict = None):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.config = config

        # Apply parameter overrides
        self.use_double = config.get('use_double', False)
        self.use_dueling = config.get('use_dueling', False)
        self.use_noisy = config.get('use_noisy', False)
        self.use_per = config.get('use_per', False)
        self.lr = config.get('lr', 1e-3)
        self.batch_size = config.get('batch_size', BATCH_SIZE)
        self.target_update_freq = config.get('target_update_freq', TARGET_UPDATE_EVERY)
        self.gamma = config.get('gamma', GAMMA)

        if param_overrides:
            for key, value in param_overrides.items():
                setattr(self, key, value)

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
        if len(self.buffer) < max(LEARNING_STARTS, self.batch_size):
            return None

        if self.use_per:
            beta = compute_beta(step)
            (s, a, r, s2, done), idxs, weights = self.buffer.sample(self.batch_size, beta=beta)
        else:
            (s, a, r, s2, done), idxs, weights = self.buffer.sample(self.batch_size)
            weights = torch.ones((self.batch_size, 1), device=device)

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
            y = r + self.gamma * (1.0 - done) * q2

        td_err = (y - q)
        loss = (weights * td_err.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_online.parameters(), GRAD_CLIP_NORM)
        self.optimizer.step()

        if self.use_per:
            prios = td_err.detach().abs().squeeze(-1).cpu().numpy() + PER_EPS
            self.buffer.update_priorities(idxs, prios)

        if step % self.target_update_freq == 0:
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

        env.close()

# ============================================================================
# MAIN SENSITIVITY TESTS
# ============================================================================

if __name__ == "__main__":

    print("="*80)
    print("DQN HYPERPARAMETER SENSITIVITY ANALYSIS")
    print("="*80)

    env = make_env(ENV_ID, SEED)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    env.close()

    all_results = {}

    for test_name, test_config in SENSITIVITY_TESTS.items():
        print(f"\n{'='*80}")
        print(f"TEST: {test_config['description']}")
        print(f"{'='*80}\n")

        test_results = {}
        parameter = test_config['parameter']
        base_config = test_config['agent_config'].copy()
        base_config['lr'] = 1e-3
        base_config['batch_size'] = BATCH_SIZE
        base_config['target_update_freq'] = TARGET_UPDATE_EVERY
        base_config['gamma'] = GAMMA

        for value in test_config['values']:
            print(f"Testing {parameter} = {value}")

            # Create override dict
            overrides = {parameter: value}

            # Train agent
            agent = DQNAgent(obs_dim, n_actions, base_config, param_overrides=overrides)
            agent.train(ENV_ID, total_steps=TOTAL_STEPS, eval_every=EVAL_EVERY, eval_episodes=EVAL_EPISODES)

            # Store results
            test_results[value] = {
                'returns': np.array(agent.eval_returns),
                'stds': np.array(agent.eval_stds),
                'steps': np.array(agent.eval_steps),
            }

            print(f"  Final Return: {agent.eval_returns[-1]:.1f} ± {agent.eval_stds[-1]:.2f}\n")

        all_results[test_name] = test_results

        # Visualize test
        fig, ax = plt.subplots(figsize=(12, 6))

        for value, result in test_results.items():
            steps = result['steps']
            returns = result['returns']
            stds = result['stds']
            ax.plot(steps, returns, marker='o', label=f"{parameter}={value}", linewidth=2)
            ax.fill_between(steps, returns - stds, returns + stds, alpha=0.15)

        ax.set_xlabel('Training Steps', fontsize=11)
        ax.set_ylabel('Evaluation Return', fontsize=11)
        ax.set_title(f"Sensitivity: {test_config['description']}", fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        filename = f'/home/isc-den/cas-artificial-intelligence/11_b_dqn-extensions/sensitivity_{test_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: sensitivity_{test_name}.png")
        plt.close()

    # Summary table
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("="*80 + "\n")

    for test_name, test_results in all_results.items():
        print(f"\n{test_name.upper().replace('_', ' ')}:")
        print("-" * 60)

        for value, result in test_results.items():
            final_return = result['returns'][-1]
            final_std = result['stds'][-1]
            max_return = result['returns'].max()
            avg_return = result['returns'].mean()

            print(f"  {value:8} → Final: {final_return:7.1f} ± {final_std:5.2f} | Max: {max_return:7.1f} | Avg: {avg_return:7.1f}")

    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("="*80)

