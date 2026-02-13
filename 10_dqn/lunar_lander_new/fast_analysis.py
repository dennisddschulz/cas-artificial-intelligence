"""
Fast hyperparameter analysis for DQN vs Double DQN on Lunar Lander
Optimized version with fewer episodes for quicker results
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    def __init__(self, buffer_size, device=DEVICE):
        self.buffer_size = buffer_size
        self.device = device
        self.memory = deque(maxlen=buffer_size)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
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
    for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
        target_param.data.mul_(1.0 - tau)
        target_param.data.add_(tau * online_param.data)


def hard_update(target_net, online_net):
    target_net.load_state_dict(online_net.state_dict())


class DQNAgent:
    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        
        self.q_network = DQNNetwork(state_size, action_size, hidden_size=config['hidden_size']).to(DEVICE)
        self.target_network = DQNNetwork(state_size, action_size, hidden_size=config['hidden_size']).to(DEVICE)
        hard_update(self.target_network, self.q_network)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config['learning_rate'])
        self.memory = ReplayBuffer(config['buffer_size'], DEVICE)
        
        self.training_step = 0
        self.losses = []
        self.q_values = []
    
    def act(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.max(1)[1].item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self, batch_size, gamma, use_double_dqn=False):
        if not self.memory.is_ready(batch_size):
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            if use_double_dqn:
                next_actions = self.q_network(next_states).max(1)[1]
                next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q_values = self.target_network(next_states).max(1)[0]
            
            target_q_values = rewards + (1 - dones) * gamma * next_q_values
        
        loss = nn.functional.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.config.get('gradient_clipping', True):
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        self.training_step += 1
        if self.config['update_type'] == 'hard':
            if self.training_step % self.config['update_frequency'] == 0:
                hard_update(self.target_network, self.q_network)
        elif self.config['update_type'] == 'soft':
            soft_update(self.target_network, self.q_network, self.config['tau'])
        
        self.losses.append(loss.item())
        self.q_values.append(q_values.mean().item())
        
        return loss.item()


def train_agent(env, agent, config, use_double_dqn=False):
    max_episodes = config['max_episodes']
    batch_size = config['batch_size']
    gamma = config['gamma']
    epsilon_start = config['epsilon_start']
    epsilon_end = config['epsilon_end']
    epsilon_decay_steps = config['epsilon_decay_steps']
    min_buffer_size = config['min_buffer_size']
    epsilon_decay_type = config.get('epsilon_decay_type', 'linear')
    
    episode_returns = []
    episode_lengths = []
    moving_avg_returns = []
    
    for episode in tqdm(range(max_episodes), desc=f"{'Double DQN' if use_double_dqn else 'DQN'}"):
        state, _ = env.reset()
        episode_return = 0
        episode_length = 0
        
        if epsilon_decay_type == 'linear':
            epsilon = epsilon_start - (epsilon_start - epsilon_end) * (episode / epsilon_decay_steps)
        elif epsilon_decay_type == 'exponential':
            epsilon = max(epsilon_end, epsilon_start * np.exp(-3.0 * episode / epsilon_decay_steps))
        else:
            epsilon = epsilon_start
        
        epsilon = max(epsilon_end, epsilon)
        
        for step in range(500):
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


def main():
    print("\n" + "="*80)
    print("FAST DQN vs DOUBLE DQN HYPERPARAMETER ANALYSIS")
    print("="*80 + "\n")
    
    base_config = {
        'hidden_size': 128,
        'learning_rate': 1e-3,
        'max_episodes': 200,  # Reduced for speed
        'batch_size': 64,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay_steps': 250000,
        'epsilon_decay_type': 'linear',
        'buffer_size': 100000,
        'min_buffer_size': 1000,
        'update_type': 'hard',
        'update_frequency': 1000,
        'tau': 0.005,
        'gradient_clipping': True,
    }
    
    all_results = {}
    
    # ============================================================================
    # EXPERIMENT 1: Buffer Size
    # ============================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 1: BUFFER SIZE ANALYSIS")
    print("="*80)
    
    buffer_sizes = [10000, 50000, 100000, 200000]
    results_dqn_buffer = {}
    results_ddqn_buffer = {}
    
    for buf_size in buffer_sizes:
        config = base_config.copy()
        config['buffer_size'] = buf_size
        
        env = gym.make('LunarLander-v3')
        env.action_space.seed(RANDOM_SEED)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
        print(f"\nTraining DQN with buffer_size={buf_size}")
        agent_dqn = DQNAgent(state_size, action_size, config)
        result_dqn = train_agent(env, agent_dqn, config, use_double_dqn=False)
        results_dqn_buffer[buf_size] = result_dqn
        
        env = gym.make('LunarLander-v3')
        env.action_space.seed(RANDOM_SEED)
        
        print(f"Training Double DQN with buffer_size={buf_size}")
        agent_ddqn = DQNAgent(state_size, action_size, config)
        result_ddqn = train_agent(env, agent_ddqn, config, use_double_dqn=True)
        results_ddqn_buffer[buf_size] = result_ddqn
        
        all_results[f'buffer_{buf_size}'] = {'dqn': result_dqn, 'ddqn': result_ddqn}
    
    # Plot buffer analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Buffer Size Analysis', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    for buf_size in buffer_sizes:
        ax.plot(results_dqn_buffer[buf_size]['moving_avg_returns'], 
                label=f'{buf_size//1000}k', alpha=0.7, linewidth=2)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Moving Avg Return (100-ep)', fontsize=11)
    ax.set_title('DQN Episode Returns', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for buf_size in buffer_sizes:
        ax.plot(results_ddqn_buffer[buf_size]['moving_avg_returns'], 
                label=f'{buf_size//1000}k', alpha=0.7, linewidth=2)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Moving Avg Return (100-ep)', fontsize=11)
    ax.set_title('Double DQN Episode Returns', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    dqn_final = [np.mean(results_dqn_buffer[bz]['episode_returns'][-50:]) for bz in buffer_sizes]
    ddqn_final = [np.mean(results_ddqn_buffer[bz]['episode_returns'][-50:]) for bz in buffer_sizes]
    x = np.arange(len(buffer_sizes))
    width = 0.35
    ax.bar(x - width/2, dqn_final, width, label='DQN', alpha=0.8)
    ax.bar(x + width/2, ddqn_final, width, label='Double DQN', alpha=0.8)
    ax.set_xlabel('Buffer Size', fontsize=11)
    ax.set_ylabel('Final Avg Return (last 50 episodes)', fontsize=11)
    ax.set_title('Final Performance', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{bz//1000}k' for bz in buffer_sizes])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 1]
    dqn_std = [np.std(results_dqn_buffer[bz]['episode_returns'][-50:]) for bz in buffer_sizes]
    ddqn_std = [np.std(results_ddqn_buffer[bz]['episode_returns'][-50:]) for bz in buffer_sizes]
    ax.bar(x - width/2, dqn_std, width, label='DQN', alpha=0.8)
    ax.bar(x + width/2, ddqn_std, width, label='Double DQN', alpha=0.8)
    ax.set_xlabel('Buffer Size', fontsize=11)
    ax.set_ylabel('Std Dev (last 50 episodes)', fontsize=11)
    ax.set_title('Training Stability', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{bz//1000}k' for bz in buffer_sizes])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_buffer_size_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: 01_buffer_size_analysis.png")
    plt.close()
    
    # ============================================================================
    # EXPERIMENT 2: Epsilon Decay
    # ============================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 2: EPSILON DECAY ANALYSIS")
    print("="*80)
    
    epsilon_configs = [
        ('Fast (50k)', 50000),
        ('Medium (150k)', 150000),
        ('Slow (250k)', 250000),
        ('V. Slow (400k)', 400000),
    ]
    
    results_dqn_epsilon = {}
    results_ddqn_epsilon = {}
    
    for name, eps_steps in epsilon_configs:
        config = base_config.copy()
        config['epsilon_decay_steps'] = eps_steps
        
        env = gym.make('LunarLander-v3')
        env.action_space.seed(RANDOM_SEED)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
        print(f"\nTraining DQN with epsilon_decay={name}")
        agent_dqn = DQNAgent(state_size, action_size, config)
        result_dqn = train_agent(env, agent_dqn, config, use_double_dqn=False)
        results_dqn_epsilon[name] = result_dqn
        
        env = gym.make('LunarLander-v3')
        env.action_space.seed(RANDOM_SEED)
        
        print(f"Training Double DQN with epsilon_decay={name}")
        agent_ddqn = DQNAgent(state_size, action_size, config)
        result_ddqn = train_agent(env, agent_ddqn, config, use_double_dqn=True)
        results_ddqn_epsilon[name] = result_ddqn
        
        all_results[f'epsilon_{name}'] = {'dqn': result_dqn, 'ddqn': result_ddqn}
    
    # Plot epsilon analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Exploration Strategy (Epsilon Decay) Analysis', fontsize=16, fontweight='bold')
    
    names = [n for n, _ in epsilon_configs]
    
    ax = axes[0, 0]
    for name in names:
        ax.plot(results_dqn_epsilon[name]['moving_avg_returns'], 
                label=name, alpha=0.7, linewidth=2)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Moving Avg Return (100-ep)', fontsize=11)
    ax.set_title('DQN Episode Returns', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for name in names:
        ax.plot(results_ddqn_epsilon[name]['moving_avg_returns'], 
                label=name, alpha=0.7, linewidth=2)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Moving Avg Return (100-ep)', fontsize=11)
    ax.set_title('Double DQN Episode Returns', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    dqn_final = [np.mean(results_dqn_epsilon[n]['episode_returns'][-50:]) for n in names]
    ddqn_final = [np.mean(results_ddqn_epsilon[n]['episode_returns'][-50:]) for n in names]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, dqn_final, width, label='DQN', alpha=0.8)
    ax.bar(x + width/2, ddqn_final, width, label='Double DQN', alpha=0.8)
    ax.set_xlabel('Epsilon Decay Configuration', fontsize=11)
    ax.set_ylabel('Final Avg Return (last 50 episodes)', fontsize=11)
    ax.set_title('Final Performance', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 1]
    dqn_std = [np.std(results_dqn_epsilon[n]['episode_returns'][-50:]) for n in names]
    ddqn_std = [np.std(results_ddqn_epsilon[n]['episode_returns'][-50:]) for n in names]
    ax.bar(x - width/2, dqn_std, width, label='DQN', alpha=0.8)
    ax.bar(x + width/2, ddqn_std, width, label='Double DQN', alpha=0.8)
    ax.set_xlabel('Epsilon Decay Configuration', fontsize=11)
    ax.set_ylabel('Std Dev (last 50 episodes)', fontsize=11)
    ax.set_title('Training Stability', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_epsilon_decay_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: 02_epsilon_decay_analysis.png")
    plt.close()
    
    # ============================================================================
    # EXPERIMENT 3: Update Strategy
    # ============================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 3: TARGET NETWORK UPDATE STRATEGY")
    print("="*80)
    
    update_configs = [
        {'name': 'Hard-500', 'update_type': 'hard', 'update_frequency': 500},
        {'name': 'Hard-1000', 'update_type': 'hard', 'update_frequency': 1000},
        {'name': 'Hard-2000', 'update_type': 'hard', 'update_frequency': 2000},
        {'name': 'Soft-0.005', 'update_type': 'soft', 'tau': 0.005},
        {'name': 'Soft-0.01', 'update_type': 'soft', 'tau': 0.01},
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
        agent_ddqn = DQNAgent(state_size, action_size, config)
        result_ddqn = train_agent(env, agent_ddqn, config, use_double_dqn=True)
        results_ddqn_update[upd_cfg['name']] = result_ddqn
        
        all_results[f'update_{upd_cfg["name"]}'] = {'dqn': result_dqn, 'ddqn': result_ddqn}
    
    # Plot update strategy
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Target Network Update Strategy Analysis', fontsize=16, fontweight='bold')
    
    upd_names = [cfg['name'] for cfg in update_configs]
    
    ax = axes[0, 0]
    for name in upd_names:
        ax.plot(results_dqn_update[name]['moving_avg_returns'], 
                label=name, alpha=0.7, linewidth=2)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Moving Avg Return (100-ep)', fontsize=11)
    ax.set_title('DQN Episode Returns', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for name in upd_names:
        ax.plot(results_ddqn_update[name]['moving_avg_returns'], 
                label=name, alpha=0.7, linewidth=2)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Moving Avg Return (100-ep)', fontsize=11)
    ax.set_title('Double DQN Episode Returns', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    dqn_final = [np.mean(results_dqn_update[n]['episode_returns'][-50:]) for n in upd_names]
    ddqn_final = [np.mean(results_ddqn_update[n]['episode_returns'][-50:]) for n in upd_names]
    x = np.arange(len(upd_names))
    width = 0.35
    ax.bar(x - width/2, dqn_final, width, label='DQN', alpha=0.8)
    ax.bar(x + width/2, ddqn_final, width, label='Double DQN', alpha=0.8)
    ax.set_xlabel('Update Strategy', fontsize=11)
    ax.set_ylabel('Final Avg Return (last 50 episodes)', fontsize=11)
    ax.set_title('Final Performance', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(upd_names, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 1]
    dqn_std = [np.std(results_dqn_update[n]['episode_returns'][-50:]) for n in upd_names]
    ddqn_std = [np.std(results_ddqn_update[n]['episode_returns'][-50:]) for n in upd_names]
    ax.bar(x - width/2, dqn_std, width, label='DQN', alpha=0.8)
    ax.bar(x + width/2, ddqn_std, width, label='Double DQN', alpha=0.8)
    ax.set_xlabel('Update Strategy', fontsize=11)
    ax.set_ylabel('Std Dev (last 50 episodes)', fontsize=11)
    ax.set_title('Training Stability', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(upd_names, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_update_strategy_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: 03_update_strategy_analysis.png")
    plt.close()
    
    # ============================================================================
    # EXPERIMENT 4: Learning Parameters
    # ============================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 4: LEARNING RATE & BATCH SIZE ANALYSIS")
    print("="*80)
    
    learning_rates = [5e-4, 1e-3, 2e-3]
    batch_sizes = [32, 64, 128]
    
    results_lr = {}
    results_bs = {}
    
    for lr in learning_rates:
        config = base_config.copy()
        config['learning_rate'] = lr
        
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
        agent_ddqn = DQNAgent(state_size, action_size, config)
        result_ddqn = train_agent(env, agent_ddqn, config, use_double_dqn=True)
        
        results_lr[lr] = {'dqn': result_dqn, 'ddqn': result_ddqn}
    
    for bs in batch_sizes:
        config = base_config.copy()
        config['batch_size'] = bs
        
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
        agent_ddqn = DQNAgent(state_size, action_size, config)
        result_ddqn = train_agent(env, agent_ddqn, config, use_double_dqn=True)
        
        results_bs[bs] = {'dqn': result_dqn, 'ddqn': result_ddqn}
    
    # Plot learning parameters
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Learning Parameters (Learning Rate & Batch Size) Analysis', 
                 fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    for lr in learning_rates:
        ax.plot(results_lr[lr]['dqn']['moving_avg_returns'], 
                label=f'LR={lr:.1e}', alpha=0.7, linewidth=2)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Moving Avg Return (100-ep)', fontsize=11)
    ax.set_title('Learning Rate Impact - DQN', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for bs in batch_sizes:
        ax.plot(results_bs[bs]['dqn']['moving_avg_returns'], 
                label=f'BS={bs}', alpha=0.7, linewidth=2)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Moving Avg Return (100-ep)', fontsize=11)
    ax.set_title('Batch Size Impact - DQN', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    dqn_final_lr = [np.mean(results_lr[lr]['dqn']['episode_returns'][-50:]) for lr in learning_rates]
    ddqn_final_lr = [np.mean(results_lr[lr]['ddqn']['episode_returns'][-50:]) for lr in learning_rates]
    x = np.arange(len(learning_rates))
    width = 0.35
    ax.bar(x - width/2, dqn_final_lr, width, label='DQN', alpha=0.8)
    ax.bar(x + width/2, ddqn_final_lr, width, label='Double DQN', alpha=0.8)
    ax.set_xlabel('Learning Rate', fontsize=11)
    ax.set_ylabel('Final Avg Return (last 50 episodes)', fontsize=11)
    ax.set_title('Learning Rate - Final Performance', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{lr:.1e}' for lr in learning_rates])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 1]
    dqn_final_bs = [np.mean(results_bs[bs]['dqn']['episode_returns'][-50:]) for bs in batch_sizes]
    ddqn_final_bs = [np.mean(results_bs[bs]['ddqn']['episode_returns'][-50:]) for bs in batch_sizes]
    x = np.arange(len(batch_sizes))
    ax.bar(x - width/2, dqn_final_bs, width, label='DQN', alpha=0.8)
    ax.bar(x + width/2, ddqn_final_bs, width, label='Double DQN', alpha=0.8)
    ax.set_xlabel('Batch Size', fontsize=11)
    ax.set_ylabel('Final Avg Return (last 50 episodes)', fontsize=11)
    ax.set_title('Batch Size - Final Performance', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(bs) for bs in batch_sizes])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_learning_rate_batch_size_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: 04_learning_rate_batch_size_analysis.png")
    plt.close()
    
    # ============================================================================
    # Generate Summary Report
    # ============================================================================
    report = []
    report.append("="*80)
    report.append("DQN vs DOUBLE DQN HYPERPARAMETER ANALYSIS - SUMMARY REPORT")
    report.append("="*80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    for exp_name, exp_data in all_results.items():
        dqn_res = exp_data['dqn']
        ddqn_res = exp_data['ddqn']
        
        dqn_final = np.mean(dqn_res['episode_returns'][-50:])
        dqn_std = np.std(dqn_res['episode_returns'][-50:])
        ddqn_final = np.mean(ddqn_res['episode_returns'][-50:])
        ddqn_std = np.std(ddqn_res['episode_returns'][-50:])
        
        report.append(f"\n{exp_name}:")
        report.append(f"  DQN:        Mean={dqn_final:8.2f}, Std={dqn_std:8.2f}")
        report.append(f"  Double DQN: Mean={ddqn_final:8.2f}, Std={ddqn_std:8.2f}")
        report.append(f"  Improvement: {ddqn_final - dqn_final:+.2f} points")
    
    with open(OUTPUT_DIR / 'ANALYSIS_REPORT.txt', 'w') as f:
        f.write("\n".join(report))
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  ✓ 01_buffer_size_analysis.png")
    print("  ✓ 02_epsilon_decay_analysis.png")
    print("  ✓ 03_update_strategy_analysis.png")
    print("  ✓ 04_learning_rate_batch_size_analysis.png")
    print("  ✓ ANALYSIS_REPORT.txt")
    print("\nNow generating presentation slides...")


if __name__ == "__main__":
    main()

