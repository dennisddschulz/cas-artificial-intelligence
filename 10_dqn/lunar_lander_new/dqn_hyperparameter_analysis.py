"""
Comprehensive DQN vs Double DQN Hyperparameter Analysis on Lunar Lander
=========================================================================
This script performs systematic analysis of:
1. Experience Replay Buffer (size, min_buffer_size)
2. Exploration Strategy (epsilon decay)
3. Target Network Update Strategy (hard vs soft updates)
4. Optional Extensions (learning rate, batch size, etc.)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from pathlib import Path
import json
from datetime import datetime
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Output directory
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Experience tuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """Experience replay buffer for DQN"""
    def __init__(self, buffer_size, device=DEVICE):
        self.buffer_size = buffer_size
        self.device = device
        self.memory = deque(maxlen=buffer_size)

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.memory.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample random batch from buffer"""
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.FloatTensor([float(e.done) for e in batch]).to(self.device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

    def is_ready(self, min_size):
        return len(self) >= min_size


class DQNNetwork(nn.Module):
    """Neural network for Q-value estimation"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, state):
        return self.network(state)


def soft_update(target_net, online_net, tau):
    """Soft update of target network using Polyak averaging"""
    for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
        target_param.data.mul_(1.0 - tau)
        target_param.data.add_(tau * online_param.data)


def hard_update(target_net, online_net):
    """Hard update of target network"""
    target_net.load_state_dict(online_net.state_dict())


class DQNAgent:
    """DQN Agent"""
    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config

        # Networks
        self.q_network = DQNNetwork(state_size, action_size, hidden_size=config['hidden_size']).to(DEVICE)
        self.target_network = DQNNetwork(state_size, action_size, hidden_size=config['hidden_size']).to(DEVICE)
        hard_update(self.target_network, self.q_network)

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config['learning_rate'])

        # Replay buffer
        self.memory = ReplayBuffer(config['buffer_size'], DEVICE)

        # Training tracking
        self.training_step = 0
        self.losses = []
        self.q_values = []

    def act(self, state, epsilon):
        """Epsilon-greedy action selection"""
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.max(1)[1].item()

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)

    def replay(self, batch_size, gamma, use_double_dqn=False):
        """Train on a batch of experiences"""
        if not self.memory.is_ready(batch_size):
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        # Current Q-values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            if use_double_dqn:
                # Double DQN: use online network to select actions, target network to evaluate
                next_actions = self.q_network(next_states).max(1)[1]
                next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # DQN: use target network for both selection and evaluation
                next_q_values = self.target_network(next_states).max(1)[0]

            target_q_values = rewards + (1 - dones) * gamma * next_q_values

        # Loss and backprop
        loss = nn.functional.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (optional extension)
        if self.config.get('gradient_clipping', True):
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)

        self.optimizer.step()

        # Update target network
        self.training_step += 1
        if self.config['update_type'] == 'hard':
            if self.training_step % self.config['update_frequency'] == 0:
                hard_update(self.target_network, self.q_network)
        elif self.config['update_type'] == 'soft':
            soft_update(self.target_network, self.q_network, self.config['tau'])

        self.losses.append(loss.item())
        self.q_values.append(q_values.mean().item())

        return loss.item()


class DoubleDQNAgent(DQNAgent):
    """Double DQN Agent - inherits from DQN but uses double Q-learning in replay"""
    pass


def train_agent(env, agent, config, use_double_dqn=False):
    """Train an agent on the environment"""
    max_episodes = config['max_episodes']
    batch_size = config['batch_size']
    gamma = config['gamma']
    epsilon_start = config['epsilon_start']
    epsilon_end = config['epsilon_end']
    epsilon_decay_steps = config['epsilon_decay_steps']
    min_buffer_size = config['min_buffer_size']

    # Epsilon schedule
    epsilon_decay_type = config.get('epsilon_decay_type', 'linear')

    episode_returns = []
    episode_lengths = []
    moving_avg_returns = []

    print(f"\n{'='*60}")
    print(f"Training {'Double DQN' if use_double_dqn else 'DQN'}")
    print(f"Config: {json.dumps({k: v for k, v in config.items() if k not in ['hidden_size', 'gradient_clipping']}, indent=2)}")
    print(f"{'='*60}")

    for episode in tqdm(range(max_episodes), desc=f"{'Double DQN' if use_double_dqn else 'DQN'} Training"):
        state, _ = env.reset()
        episode_return = 0
        episode_length = 0

        # Epsilon schedule
        if epsilon_decay_type == 'linear':
            epsilon = epsilon_start - (epsilon_start - epsilon_end) * (episode / epsilon_decay_steps)
        elif epsilon_decay_type == 'exponential':
            epsilon = max(epsilon_end, epsilon_start * np.exp(-3.0 * episode / epsilon_decay_steps))
        else:  # constant
            epsilon = epsilon_start

        epsilon = max(epsilon_end, epsilon)

        for step in range(500):  # Max steps per episode
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.remember(state, action, reward, next_state, done)

            if agent.memory.is_ready(min_buffer_size):
                agent.replay(batch_size, gamma, use_double_dqn=use_double_dqn)

            episode_return += reward
            episode_length += 1
            state = next_state

            if done:
                break

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

        # Moving average
        if len(episode_returns) >= 100:
            moving_avg = np.mean(episode_returns[-100:])
            moving_avg_returns.append(moving_avg)
        else:
            moving_avg_returns.append(np.mean(episode_returns))

    env.close()

    return {
        'episode_returns': episode_returns,
        'episode_lengths': episode_lengths,
        'moving_avg_returns': moving_avg_returns,
        'losses': agent.losses,
        'q_values': agent.q_values,
    }


def run_experiment(config_template, param_name, param_values, use_double_dqn=False):
    """Run multiple experiments with different hyperparameter values"""
    results = {}

    for param_value in param_values:
        config = config_template.copy()
        config[param_name] = param_value

        env = gym.make('LunarLander-v3')
        env.action_space.seed(RANDOM_SEED)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        agent = DoubleDQNAgent(state_size, action_size, config) if use_double_dqn else DQNAgent(state_size, action_size, config)

        result = train_agent(env, agent, config, use_double_dqn=use_double_dqn)
        results[param_value] = result

    return results


def plot_buffer_analysis(results_dqn, results_ddqn, param_values):
    """Plot buffer size analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Episode returns
    ax = axes[0, 0]
    for param_val in param_values:
        ax.plot(results_dqn[param_val]['moving_avg_returns'], label=f'DQN: {param_val}', alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Moving Avg Return (100-ep)')
    ax.set_title('Buffer Size Impact - DQN Episode Returns')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Double DQN
    ax = axes[0, 1]
    for param_val in param_values:
        ax.plot(results_ddqn[param_val]['moving_avg_returns'], label=f'Double DQN: {param_val}', alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Moving Avg Return (100-ep)')
    ax.set_title('Buffer Size Impact - Double DQN Episode Returns')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Final performance comparison
    ax = axes[1, 0]
    dqn_final = [np.mean(results_dqn[pv]['episode_returns'][-100:]) for pv in param_values]
    ddqn_final = [np.mean(results_ddqn[pv]['episode_returns'][-100:]) for pv in param_values]
    x = np.arange(len(param_values))
    width = 0.35
    ax.bar(x - width/2, dqn_final, width, label='DQN', alpha=0.8)
    ax.bar(x + width/2, ddqn_final, width, label='Double DQN', alpha=0.8)
    ax.set_xlabel('Buffer Size')
    ax.set_ylabel('Final Avg Return (last 100 episodes)')
    ax.set_title('Final Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{pv//1000}k' if pv >= 1000 else str(pv) for pv in param_values])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Training stability (std of last 100 episodes)
    ax = axes[1, 1]
    dqn_std = [np.std(results_dqn[pv]['episode_returns'][-100:]) for pv in param_values]
    ddqn_std = [np.std(results_ddqn[pv]['episode_returns'][-100:]) for pv in param_values]
    ax.bar(x - width/2, dqn_std, width, label='DQN', alpha=0.8)
    ax.bar(x + width/2, ddqn_std, width, label='Double DQN', alpha=0.8)
    ax.set_xlabel('Buffer Size')
    ax.set_ylabel('Std Dev (last 100 episodes)')
    ax.set_title('Training Stability (Lower is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{pv//1000}k' if pv >= 1000 else str(pv) for pv in param_values])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_epsilon_analysis(results_dqn, results_ddqn, param_values, param_name):
    """Plot epsilon decay analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Episode returns - DQN
    ax = axes[0, 0]
    for param_val in param_values:
        ax.plot(results_dqn[param_val]['moving_avg_returns'], label=f'DQN: {param_val}', alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Moving Avg Return (100-ep)')
    ax.set_title('Epsilon Decay Impact - DQN Episode Returns')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Episode returns - Double DQN
    ax = axes[0, 1]
    for param_val in param_values:
        ax.plot(results_ddqn[param_val]['moving_avg_returns'], label=f'Double DQN: {param_val}', alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Moving Avg Return (100-ep)')
    ax.set_title('Epsilon Decay Impact - Double DQN Episode Returns')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Final performance
    ax = axes[1, 0]
    dqn_final = [np.mean(results_dqn[pv]['episode_returns'][-100:]) for pv in param_values]
    ddqn_final = [np.mean(results_ddqn[pv]['episode_returns'][-100:]) for pv in param_values]
    x = np.arange(len(param_values))
    width = 0.35
    ax.bar(x - width/2, dqn_final, width, label='DQN', alpha=0.8)
    ax.bar(x + width/2, ddqn_final, width, label='Double DQN', alpha=0.8)
    ax.set_xlabel('Epsilon Configuration')
    ax.set_ylabel('Final Avg Return (last 100 episodes)')
    ax.set_title('Final Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([str(pv) for pv in param_values])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Training stability
    ax = axes[1, 1]
    dqn_std = [np.std(results_dqn[pv]['episode_returns'][-100:]) for pv in param_values]
    ddqn_std = [np.std(results_ddqn[pv]['episode_returns'][-100:]) for pv in param_values]
    ax.bar(x - width/2, dqn_std, width, label='DQN', alpha=0.8)
    ax.bar(x + width/2, ddqn_std, width, label='Double DQN', alpha=0.8)
    ax.set_xlabel('Epsilon Configuration')
    ax.set_ylabel('Std Dev (last 100 episodes)')
    ax.set_title('Training Stability')
    ax.set_xticks(x)
    ax.set_xticklabels([str(pv) for pv in param_values])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def create_summary_report(all_results):
    """Create a comprehensive summary report"""
    report = []
    report.append("="*80)
    report.append("DQN vs DOUBLE DQN HYPERPARAMETER ANALYSIS - SUMMARY REPORT")
    report.append("="*80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    for exp_name, exp_data in all_results.items():
        report.append(f"\n{'='*80}")
        report.append(f"EXPERIMENT: {exp_name}")
        report.append(f"{'='*80}\n")

        for param_val, results in exp_data.items():
            dqn_res = results['dqn']
            ddqn_res = results['ddqn']

            report.append(f"\nParameter Value: {param_val}")
            report.append("-" * 40)

            # DQN stats
            dqn_final = np.mean(dqn_res['episode_returns'][-100:])
            dqn_std = np.std(dqn_res['episode_returns'][-100:])
            dqn_min = np.min(dqn_res['episode_returns'][-100:])
            dqn_max = np.max(dqn_res['episode_returns'][-100:])

            report.append(f"DQN Final Performance:")
            report.append(f"  Mean:    {dqn_final:8.2f}")
            report.append(f"  Std:     {dqn_std:8.2f}")
            report.append(f"  Min:     {dqn_min:8.2f}")
            report.append(f"  Max:     {dqn_max:8.2f}")

            # Double DQN stats
            ddqn_final = np.mean(ddqn_res['episode_returns'][-100:])
            ddqn_std = np.std(ddqn_res['episode_returns'][-100:])
            ddqn_min = np.min(ddqn_res['episode_returns'][-100:])
            ddqn_max = np.max(ddqn_res['episode_returns'][-100:])

            report.append(f"\nDouble DQN Final Performance:")
            report.append(f"  Mean:    {ddqn_final:8.2f}")
            report.append(f"  Std:     {ddqn_std:8.2f}")
            report.append(f"  Min:     {ddqn_min:8.2f}")
            report.append(f"  Max:     {ddqn_max:8.2f}")

            report.append(f"\nComparison:")
            report.append(f"  Double DQN advantage: {ddqn_final - dqn_final:+.2f} points")
            report.append(f"  Stability improvement: {dqn_std - ddqn_std:+.2f} (lower std is better)")

    return "\n".join(report)


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("DQN vs DOUBLE DQN COMPREHENSIVE HYPERPARAMETER ANALYSIS")
    print("="*80 + "\n")

    # Base configuration
    base_config = {
        'hidden_size': 128,
        'learning_rate': 1e-3,
        'max_episodes': 500,  # Adjust for faster/slower analysis
        'batch_size': 64,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay_steps': 250000,
        'epsilon_decay_type': 'linear',
        'buffer_size': 100000,
        'min_buffer_size': 1000,
        'hidden_size': 128,
        'update_type': 'hard',
        'update_frequency': 1000,
        'tau': 0.005,
        'gradient_clipping': True,
    }

    all_results = {}

    print("\n" + "="*80)
    print("EXPERIMENT 1: BUFFER SIZE ANALYSIS")
    print("="*80)
    buffer_sizes = [10000, 50000, 100000, 200000]

    results_dqn_buffer = {}
    results_ddqn_buffer = {}

    for buf_size in buffer_sizes:
        config = base_config.copy()
        config['buffer_size'] = buf_size
        config['max_episodes'] = 300  # Reduced for speed

        env = gym.make('LunarLander-v3')
        env.action_space.seed(RANDOM_SEED)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        # DQN
        print(f"\nTraining DQN with buffer_size={buf_size}")
        agent_dqn = DQNAgent(state_size, action_size, config)
        result_dqn = train_agent(env, agent_dqn, config, use_double_dqn=False)
        results_dqn_buffer[buf_size] = result_dqn

        env = gym.make('LunarLander-v3')
        env.action_space.seed(RANDOM_SEED)

        # Double DQN
        print(f"Training Double DQN with buffer_size={buf_size}")
        agent_ddqn = DoubleDQNAgent(state_size, action_size, config)
        result_ddqn = train_agent(env, agent_ddqn, config, use_double_dqn=True)
        results_ddqn_buffer[buf_size] = result_ddqn

        all_results[f'buffer_size={buf_size}'] = {
            'dqn': result_dqn,
            'ddqn': result_ddqn
        }

    # Plot buffer analysis
    fig = plot_buffer_analysis(results_dqn_buffer, results_ddqn_buffer, buffer_sizes)
    plt.savefig(OUTPUT_DIR / '01_buffer_size_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: 01_buffer_size_analysis.png")
    plt.close()

    print("\n" + "="*80)
    print("EXPERIMENT 2: EPSILON DECAY ANALYSIS")
    print("="*80)
    epsilon_decay_steps = [50000, 150000, 250000, 400000]

    results_dqn_epsilon = {}
    results_ddqn_epsilon = {}

    for eps_steps in epsilon_decay_steps:
        config = base_config.copy()
        config['epsilon_decay_steps'] = eps_steps
        config['max_episodes'] = 300

        env = gym.make('LunarLander-v3')
        env.action_space.seed(RANDOM_SEED)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        print(f"\nTraining DQN with epsilon_decay_steps={eps_steps}")
        agent_dqn = DQNAgent(state_size, action_size, config)
        result_dqn = train_agent(env, agent_dqn, config, use_double_dqn=False)
        results_dqn_epsilon[eps_steps] = result_dqn

        env = gym.make('LunarLander-v3')
        env.action_space.seed(RANDOM_SEED)

        print(f"Training Double DQN with epsilon_decay_steps={eps_steps}")
        agent_ddqn = DoubleDQNAgent(state_size, action_size, config)
        result_ddqn = train_agent(env, agent_ddqn, config, use_double_dqn=True)
        results_ddqn_epsilon[eps_steps] = result_ddqn

        all_results[f'epsilon_decay_steps={eps_steps}'] = {
            'dqn': result_dqn,
            'ddqn': result_ddqn
        }

    # Plot epsilon analysis
    fig = plot_epsilon_analysis(results_dqn_epsilon, results_ddqn_epsilon, epsilon_decay_steps, 'epsilon_decay_steps')
    plt.savefig(OUTPUT_DIR / '02_epsilon_decay_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: 02_epsilon_decay_analysis.png")
    plt.close()

    print("\n" + "="*80)
    print("EXPERIMENT 3: TARGET NETWORK UPDATE STRATEGY")
    print("="*80)

    update_configs = [
        {'name': 'Hard-500', 'update_type': 'hard', 'update_frequency': 500},
        {'name': 'Hard-1000', 'update_type': 'hard', 'update_frequency': 1000},
        {'name': 'Hard-2000', 'update_type': 'hard', 'update_frequency': 2000},
        {'name': 'Soft-0.001', 'update_type': 'soft', 'tau': 0.001},
        {'name': 'Soft-0.005', 'update_type': 'soft', 'tau': 0.005},
    ]

    results_dqn_update = {}
    results_ddqn_update = {}

    for upd_cfg in update_configs:
        config = base_config.copy()
        config['update_type'] = upd_cfg['update_type']
        if upd_cfg['update_type'] == 'hard':
            config['update_frequency'] = upd_cfg['update_frequency']
        else:
            config['tau'] = upd_cfg['tau']
        config['max_episodes'] = 300

        env = gym.make('LunarLander-v3')
        env.action_space.seed(RANDOM_SEED)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        print(f"\nTraining DQN with {upd_cfg['name']}")
        agent_dqn = DQNAgent(state_size, action_size, config)
        result_dqn = train_agent(env, agent_dqn, config, use_double_dqn=False)
        results_dqn_update[upd_cfg['name']] = result_dqn

        env = gym.make('LunarLander-v3')
        env.action_space.seed(RANDOM_SEED)

        print(f"Training Double DQN with {upd_cfg['name']}")
        agent_ddqn = DoubleDQNAgent(state_size, action_size, config)
        result_ddqn = train_agent(env, agent_ddqn, config, use_double_dqn=True)
        results_ddqn_update[upd_cfg['name']] = result_ddqn

        all_results[f'update_{upd_cfg["name"]}'] = {
            'dqn': result_dqn,
            'ddqn': result_ddqn
        }

    # Plot update strategy analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    names = list(results_dqn_update.keys())

    ax = axes[0, 0]
    for name in names:
        ax.plot(results_dqn_update[name]['moving_avg_returns'], label=f'DQN: {name}', alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Moving Avg Return (100-ep)')
    ax.set_title('Update Strategy Impact - DQN')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for name in names:
        ax.plot(results_ddqn_update[name]['moving_avg_returns'], label=f'Double DQN: {name}', alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Moving Avg Return (100-ep)')
    ax.set_title('Update Strategy Impact - Double DQN')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    dqn_final = [np.mean(results_dqn_update[n]['episode_returns'][-100:]) for n in names]
    ddqn_final = [np.mean(results_ddqn_update[n]['episode_returns'][-100:]) for n in names]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, dqn_final, width, label='DQN', alpha=0.8)
    ax.bar(x + width/2, ddqn_final, width, label='Double DQN', alpha=0.8)
    ax.set_xlabel('Update Strategy')
    ax.set_ylabel('Final Avg Return')
    ax.set_title('Final Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 1]
    dqn_std = [np.std(results_dqn_update[n]['episode_returns'][-100:]) for n in names]
    ddqn_std = [np.std(results_ddqn_update[n]['episode_returns'][-100:]) for n in names]
    ax.bar(x - width/2, dqn_std, width, label='DQN', alpha=0.8)
    ax.bar(x + width/2, ddqn_std, width, label='Double DQN', alpha=0.8)
    ax.set_xlabel('Update Strategy')
    ax.set_ylabel('Std Dev (last 100 episodes)')
    ax.set_title('Training Stability')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_update_strategy_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: 03_update_strategy_analysis.png")
    plt.close()

    print("\n" + "="*80)
    print("EXPERIMENT 4: LEARNING RATE AND BATCH SIZE ANALYSIS")
    print("="*80)

    learning_rates = [5e-4, 1e-3, 2e-3]
    batch_sizes = [32, 64, 128]

    results_lr = {}
    results_bs = {}

    for lr in learning_rates:
        config = base_config.copy()
        config['learning_rate'] = lr
        config['max_episodes'] = 250

        env = gym.make('LunarLander-v3')
        env.action_space.seed(RANDOM_SEED)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        print(f"\nTraining DQN with learning_rate={lr}")
        agent_dqn = DQNAgent(state_size, action_size, config)
        result_dqn = train_agent(env, agent_dqn, config, use_double_dqn=False)

        env = gym.make('LunarLander-v3')
        env.action_space.seed(RANDOM_SEED)

        print(f"Training Double DQN with learning_rate={lr}")
        agent_ddqn = DoubleDQNAgent(state_size, action_size, config)
        result_ddqn = train_agent(env, agent_ddqn, config, use_double_dqn=True)

        results_lr[lr] = {'dqn': result_dqn, 'ddqn': result_ddqn}

    for bs in batch_sizes:
        config = base_config.copy()
        config['batch_size'] = bs
        config['max_episodes'] = 250

        env = gym.make('LunarLander-v3')
        env.action_space.seed(RANDOM_SEED)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        print(f"\nTraining DQN with batch_size={bs}")
        agent_dqn = DQNAgent(state_size, action_size, config)
        result_dqn = train_agent(env, agent_dqn, config, use_double_dqn=False)

        env = gym.make('LunarLander-v3')
        env.action_space.seed(RANDOM_SEED)

        print(f"Training Double DQN with batch_size={bs}")
        agent_ddqn = DoubleDQNAgent(state_size, action_size, config)
        result_ddqn = train_agent(env, agent_ddqn, config, use_double_dqn=True)

        results_bs[bs] = {'dqn': result_dqn, 'ddqn': result_ddqn}

    # Plot learning rate and batch size
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    ax = axes[0, 0]
    for lr in learning_rates:
        ax.plot(results_lr[lr]['dqn']['moving_avg_returns'], label=f'DQN: lr={lr}', alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Moving Avg Return (100-ep)')
    ax.set_title('Learning Rate Impact - DQN')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for bs in batch_sizes:
        ax.plot(results_bs[bs]['dqn']['moving_avg_returns'], label=f'DQN: bs={bs}', alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Moving Avg Return (100-ep)')
    ax.set_title('Batch Size Impact - DQN')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    dqn_final_lr = [np.mean(results_lr[lr]['dqn']['episode_returns'][-100:]) for lr in learning_rates]
    ddqn_final_lr = [np.mean(results_lr[lr]['ddqn']['episode_returns'][-100:]) for lr in learning_rates]
    x = np.arange(len(learning_rates))
    width = 0.35
    ax.bar(x - width/2, dqn_final_lr, width, label='DQN', alpha=0.8)
    ax.bar(x + width/2, ddqn_final_lr, width, label='Double DQN', alpha=0.8)
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Final Avg Return')
    ax.set_title('Learning Rate - Final Performance')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{lr:.1e}' for lr in learning_rates])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 1]
    dqn_final_bs = [np.mean(results_bs[bs]['dqn']['episode_returns'][-100:]) for bs in batch_sizes]
    ddqn_final_bs = [np.mean(results_bs[bs]['ddqn']['episode_returns'][-100:]) for bs in batch_sizes]
    x = np.arange(len(batch_sizes))
    ax.bar(x - width/2, dqn_final_bs, width, label='DQN', alpha=0.8)
    ax.bar(x + width/2, ddqn_final_bs, width, label='Double DQN', alpha=0.8)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Final Avg Return')
    ax.set_title('Batch Size - Final Performance')
    ax.set_xticks(x)
    ax.set_xticklabels([str(bs) for bs in batch_sizes])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_learning_rate_batch_size_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: 04_learning_rate_batch_size_analysis.png")
    plt.close()

    # Generate summary report
    report = create_summary_report(all_results)
    with open(OUTPUT_DIR / 'ANALYSIS_REPORT.txt', 'w') as f:
        f.write(report)
    print(f"\n✓ Saved: ANALYSIS_REPORT.txt")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - 01_buffer_size_analysis.png")
    print("  - 02_epsilon_decay_analysis.png")
    print("  - 03_update_strategy_analysis.png")
    print("  - 04_learning_rate_batch_size_analysis.png")
    print("  - ANALYSIS_REPORT.txt")


if __name__ == "__main__":
    main()

