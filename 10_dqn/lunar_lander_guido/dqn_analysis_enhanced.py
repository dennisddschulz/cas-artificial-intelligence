"""
Enhanced DQN vs Double DQN Analysis for LunarLander-v3
With wind variations, detailed statistics, and comprehensive visualizations
"""

import math, random, time
import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import imageio.v2 as imageio

import gymnasium as gym

# Set style for better looking plots
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('ggplot')
    except:
        pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


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


def make_env(seed=0, render_mode=None, wind_power=0.0, turbulence_power=0.0):
    """Create environment with optional wind settings"""
    env = gym.make(
        "LunarLander-v3",
        render_mode=render_mode,
        wind_power=wind_power,
        turbulence_power=turbulence_power,
        enable_wind=wind_power > 0 or turbulence_power > 0
    )
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


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


@torch.no_grad()
def select_action(q_net, obs, eps, n_actions):
    if random.random() < eps:
        return random.randrange(n_actions)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    q = q_net(obs_t)
    return int(torch.argmax(q, dim=1).item())


def train_step(
    q_online, q_target, optimizer, buffer,
    batch_size=64, gamma=0.99, double_dqn=False,
    grad_clip=10.0
):
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
        if not double_dqn:
            max_next = q_target(s2_t).max(dim=1)[0]
        else:
            a_star = torch.argmax(q_online(s2_t), dim=1)
            max_next = q_target(s2_t).gather(1, a_star.unsqueeze(1)).squeeze(1)

        target = r_t + gamma * (1.0 - d_t) * max_next

    loss = nn.functional.mse_loss(q_sa, target)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip is not None:
        nn.utils.clip_grad_norm_(q_online.parameters(), grad_clip)
    optimizer.step()

    return float(loss.item())


@torch.no_grad()
def bias_metrics(q_online, q_target, states_batch):
    s_t = torch.tensor(states_batch, dtype=torch.float32, device=DEVICE)
    q_on = q_online(s_t)
    q_tgt = q_target(s_t)

    mean_maxQ_online = float(q_on.max(dim=1)[0].mean().item())

    max_tgt = q_tgt.max(dim=1)[0]
    a_star = torch.argmax(q_on, dim=1)
    eval_tgt_on_astar = q_tgt.gather(1, a_star.unsqueeze(1)).squeeze(1)
    gap = float((max_tgt - eval_tgt_on_astar).mean().item())

    return mean_maxQ_online, gap


def train_agent(
    algo_name="DQN",
    seed=0,
    total_steps=500_000,
    start_learning=5_000,
    batch_size=64,
    buffer_size=100_000,
    gamma=0.99,
    lr=5e-4,
    hidden=256,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay_steps=200_000,
    target_update_every=1_000,
    train_every=4,
    double_dqn=False,
    wind_power=0.0,
    turbulence_power=0.0
):
    set_seed(seed)
    env = make_env(seed=seed, render_mode=None, wind_power=wind_power, turbulence_power=turbulence_power)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_online = QNet(obs_dim, n_actions, hidden=hidden).to(DEVICE)
    q_target = QNet(obs_dim, n_actions, hidden=hidden).to(DEVICE)
    q_target.load_state_dict(q_online.state_dict())
    q_target.eval()

    optimizer = optim.Adam(q_online.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=buffer_size)

    # Detailed logging
    episode_returns = []
    episode_lengths = []
    losses = []
    eps_log = []
    td_errors = []

    # Action distribution tracking
    action_counts = {i: 0 for i in range(n_actions)}

    # Bias logs
    mean_maxQ_log = []
    gap_log = []
    step_log = []

    # Q-value distribution tracking
    q_value_history = []

    obs, info = env.reset()
    ep_return = 0.0
    ep_len = 0

    for step in range(1, total_steps + 1):
        frac = min(1.0, step / max(1, eps_decay_steps))
        eps = eps_start + frac * (eps_end - eps_start)
        eps_log.append(eps)

        a = select_action(q_online, obs, eps, n_actions)
        action_counts[a] += 1

        obs2, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated

        buffer.push(obs, a, r, obs2, done)

        obs = obs2
        ep_return += r
        ep_len += 1

        if step >= start_learning and step % train_every == 0:
            loss = train_step(
                q_online, q_target, optimizer, buffer,
                batch_size=batch_size, gamma=gamma, double_dqn=double_dqn
            )
            if loss is not None:
                losses.append(loss)

        if step % target_update_every == 0:
            q_target.load_state_dict(q_online.state_dict())

        # Detailed bias probe
        if step % 5_000 == 0 and len(buffer) >= 5_000:
            s_batch, *_ = buffer.sample(512)
            mean_maxQ, gap = bias_metrics(q_online, q_target, s_batch)
            mean_maxQ_log.append(mean_maxQ)
            gap_log.append(gap)
            step_log.append(step)

            # Track Q-value distribution
            with torch.no_grad():
                s_t = torch.tensor(s_batch, dtype=torch.float32, device=DEVICE)
                q_vals = q_online(s_t).cpu().numpy()
                q_value_history.append({
                    'step': step,
                    'mean': q_vals.mean(),
                    'std': q_vals.std(),
                    'max': q_vals.max(),
                    'min': q_vals.min(),
                    'q25': np.percentile(q_vals, 25),
                    'q75': np.percentile(q_vals, 75)
                })

        if done:
            episode_returns.append(ep_return)
            episode_lengths.append(ep_len)

            if len(episode_returns) % 50 == 0:
                avg_return = np.mean(episode_returns[-50:])
                avg_length = np.mean(episode_lengths[-50:])
                print(f"[{algo_name}] Ep {len(episode_returns)}, Step {step}, "
                      f"Avg Ret: {avg_return:.1f}, Avg Len: {avg_length:.1f}, Eps: {eps:.3f}")

            obs, info = env.reset()
            ep_return = 0.0
            ep_len = 0

    env.close()

    return {
        "name": algo_name,
        "q_online": q_online,
        "q_target": q_target,
        "episode_returns": np.array(episode_returns, dtype=np.float32),
        "episode_lengths": np.array(episode_lengths, dtype=np.int32),
        "losses": np.array(losses, dtype=np.float32),
        "eps": np.array(eps_log, dtype=np.float32),
        "bias_steps": np.array(step_log, dtype=np.int32),
        "mean_maxQ": np.array(mean_maxQ_log, dtype=np.float32),
        "gap": np.array(gap_log, dtype=np.float32),
        "action_counts": action_counts,
        "q_value_history": q_value_history,
        "wind_power": wind_power,
        "turbulence_power": turbulence_power
    }


def evaluate_agent(q_net, n_episodes=100, seed=42, wind_power=0.0, turbulence_power=0.0):
    """Comprehensive evaluation of trained agent"""
    env = make_env(seed=seed, render_mode=None, wind_power=wind_power, turbulence_power=turbulence_power)

    returns = []
    lengths = []
    successes = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_return = 0.0
        ep_len = 0

        for _ in range(1000):
            a = select_action(q_net, obs, eps=0.0, n_actions=env.action_space.n)
            obs, r, terminated, truncated, _ = env.step(a)
            ep_return += r
            ep_len += 1

            if terminated or truncated:
                break

        returns.append(ep_return)
        lengths.append(ep_len)
        successes.append(1 if ep_return >= 200 else 0)

    env.close()

    return {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'min_return': np.min(returns),
        'max_return': np.max(returns),
        'median_return': np.median(returns),
        'mean_length': np.mean(lengths),
        'success_rate': np.mean(successes) * 100,
        'returns': returns,
        'lengths': lengths
    }


def record_video(q_net, filename, seed=42, max_steps=1000, wind_power=0.0, turbulence_power=0.0):
    """Record agent performance as video"""
    env = make_env(seed=seed, render_mode="rgb_array", wind_power=wind_power, turbulence_power=turbulence_power)
    obs, info = env.reset()
    frames = []
    total_reward = 0

    for step in range(max_steps):
        frame = env.render()
        frames.append(frame)

        a = select_action(q_net, obs, eps=0.0, n_actions=env.action_space.n)
        obs, r, terminated, truncated, info = env.step(a)
        total_reward += r

        if terminated or truncated:
            # Add final frames
            for _ in range(30):
                frames.append(env.render())
            break

    env.close()

    imageio.mimsave(filename, frames, fps=30)
    print(f"Saved {filename} (reward: {total_reward:.1f}, steps: {step+1})")

    return total_reward, step + 1


def create_comprehensive_plots(results_dict, output_dir="plots"):
    """Create comprehensive analysis plots"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    results = list(results_dict.values())

    # 1. Training Performance Overview
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Episode Returns
    ax = axes[0, 0]
    for res in results:
        window = 50
        smoothed = moving_average(res["episode_returns"], window)
        episodes = np.arange(window//2, window//2 + len(smoothed))
        ax.plot(episodes, smoothed, label=res["name"], linewidth=2, alpha=0.8)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Return (MA-50)", fontsize=11)
    ax.set_title("Episode Returns Over Training", fontsize=12, fontweight='bold')
    ax.axhline(y=200, color='green', linestyle='--', alpha=0.5, label='Solved (200)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Episode Lengths
    ax = axes[0, 1]
    for res in results:
        window = 50
        smoothed = moving_average(res["episode_lengths"], window)
        episodes = np.arange(window//2, window//2 + len(smoothed))
        ax.plot(episodes, smoothed, label=res["name"], linewidth=2, alpha=0.8)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Length (MA-50)", fontsize=11)
    ax.set_title("Episode Lengths Over Training", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Training Loss
    ax = axes[0, 2]
    for res in results:
        window = 1000
        smoothed = moving_average(res["losses"], window)
        steps = np.arange(window//2, window//2 + len(smoothed))
        ax.plot(steps, smoothed, label=res["name"], linewidth=2, alpha=0.8)
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Loss (MA-1000)", fontsize=11)
    ax.set_title("Training Loss", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Mean Max Q Values
    ax = axes[1, 0]
    for res in results:
        ax.plot(res["bias_steps"], res["mean_maxQ"],
                label=res["name"], linewidth=2, marker='o', markersize=4, alpha=0.8)
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Mean Max Q", fontsize=11)
    ax.set_title("Q-Value Estimates", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Overestimation Gap
    ax = axes[1, 1]
    for res in results:
        ax.plot(res["bias_steps"], res["gap"],
                label=res["name"], linewidth=2, marker='o', markersize=4, alpha=0.8)
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Overestimation Gap", fontsize=11)
    ax.set_title("Q-Value Overestimation", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)

    # Epsilon Decay
    ax = axes[1, 2]
    for res in results:
        steps = np.arange(len(res["eps"]))
        ax.plot(steps[::1000], res["eps"][::1000],
                label=res["name"], linewidth=2, alpha=0.8)
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Epsilon", fontsize=11)
    ax.set_title("Exploration Rate Decay", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_training_overview.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Return Distribution Analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Box plot
    ax = axes[0]
    data = [res["episode_returns"][-100:] for res in results]
    labels = [res["name"] for res in results]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
    for patch, color in zip(bp['boxes'], plt.cm.Set3(np.linspace(0, 1, len(results)))):
        patch.set_facecolor(color)
    ax.set_ylabel("Return", fontsize=11)
    ax.set_title("Return Distribution (Last 100 Episodes)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=200, color='green', linestyle='--', alpha=0.5)

    # Violin plot
    ax = axes[1]
    positions = np.arange(len(results))
    for i, res in enumerate(results):
        parts = ax.violinplot([res["episode_returns"][-100:]], positions=[i],
                              showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels([res["name"] for res in results])
    ax.set_ylabel("Return", fontsize=11)
    ax.set_title("Return Distribution (Violin Plot)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=200, color='green', linestyle='--', alpha=0.5)

    # Histogram
    ax = axes[2]
    for res in results:
        ax.hist(res["episode_returns"][-100:], bins=20, alpha=0.6, label=res["name"], density=True)
    ax.set_xlabel("Return", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Return Histogram (Last 100 Episodes)", fontsize=12, fontweight='bold')
    ax.axvline(x=200, color='green', linestyle='--', alpha=0.5, label='Solved')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_return_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Action Distribution
    fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 5))
    if len(results) == 1:
        axes = [axes]

    action_names = ['Nothing', 'Left', 'Main', 'Right']

    for ax, res in zip(axes, results):
        counts = res["action_counts"]
        total = sum(counts.values())
        percentages = [counts[i]/total*100 for i in range(len(counts))]

        colors = plt.cm.Set3(np.linspace(0, 1, len(percentages)))
        bars = ax.bar(action_names, percentages, color=colors, edgecolor='black', alpha=0.8)

        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

        ax.set_ylabel("Percentage (%)", fontsize=11)
        ax.set_title(f"Action Distribution: {res['name']}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_action_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Q-Value Evolution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for res in results:
        if not res["q_value_history"]:
            continue

        df = pd.DataFrame(res["q_value_history"])

        # Mean Q-values
        ax = axes[0, 0]
        ax.plot(df['step'], df['mean'], label=res["name"], linewidth=2, marker='o', markersize=4)

        # Q-value range
        ax = axes[0, 1]
        ax.plot(df['step'], df['max'], label=f"{res['name']} (max)", linewidth=2, alpha=0.7)
        ax.plot(df['step'], df['min'], label=f"{res['name']} (min)", linewidth=2, alpha=0.7)

        # Q-value std
        ax = axes[1, 0]
        ax.plot(df['step'], df['std'], label=res["name"], linewidth=2, marker='o', markersize=4)

        # Q-value IQR
        ax = axes[1, 1]
        ax.fill_between(df['step'], df['q25'], df['q75'], alpha=0.3, label=res["name"])
        ax.plot(df['step'], df['mean'], linewidth=2)

    axes[0, 0].set_xlabel("Step", fontsize=11)
    axes[0, 0].set_ylabel("Mean Q-Value", fontsize=11)
    axes[0, 0].set_title("Mean Q-Value Evolution", fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel("Step", fontsize=11)
    axes[0, 1].set_ylabel("Q-Value", fontsize=11)
    axes[0, 1].set_title("Q-Value Range (Min/Max)", fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("Step", fontsize=11)
    axes[1, 0].set_ylabel("Std Dev", fontsize=11)
    axes[1, 0].set_title("Q-Value Standard Deviation", fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel("Step", fontsize=11)
    axes[1, 1].set_ylabel("Q-Value", fontsize=11)
    axes[1, 1].set_title("Q-Value IQR (25th-75th Percentile)", fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/04_qvalue_evolution.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Plots saved to {output_dir}/")


def create_evaluation_plots(eval_results, output_dir="plots"):
    """Create evaluation comparison plots"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Performance metrics comparison
    ax = axes[0, 0]
    scenarios = list(eval_results.keys())
    means = [eval_results[s]['mean_return'] for s in scenarios]
    stds = [eval_results[s]['std_return'] for s in scenarios]

    x = np.arange(len(scenarios))
    colors = plt.cm.viridis(np.linspace(0, 1, len(scenarios)))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.set_ylabel("Mean Return", fontsize=11)
    ax.set_title("Performance Across Scenarios", fontsize=12, fontweight='bold')
    ax.axhline(y=200, color='green', linestyle='--', alpha=0.5, label='Solved')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    # Success rate
    ax = axes[0, 1]
    success_rates = [eval_results[s]['success_rate'] for s in scenarios]
    bars = ax.bar(x, success_rates, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.set_ylabel("Success Rate (%)", fontsize=11)
    ax.set_title("Success Rate (Return ≥ 200)", fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)

    # Episode length
    ax = axes[1, 0]
    lengths = [eval_results[s]['mean_length'] for s in scenarios]
    bars = ax.bar(x, lengths, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.set_ylabel("Mean Episode Length", fontsize=11)
    ax.set_title("Episode Length Across Scenarios", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Return distribution violin plot
    ax = axes[1, 1]
    data = [eval_results[s]['returns'] for s in scenarios]
    positions = np.arange(len(scenarios))
    parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)

    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.set_ylabel("Return Distribution", fontsize=11)
    ax.set_title("Return Distribution Across Scenarios", fontsize=12, fontweight='bold')
    ax.axhline(y=200, color='green', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/05_evaluation_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Evaluation plots saved to {output_dir}/")


def create_statistics_report(results_dict, eval_results, output_file="statistics_report.txt"):
    """Create detailed statistics report"""
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DQN vs Double DQN - Comprehensive Statistics Report\n")
        f.write("=" * 80 + "\n\n")

        # Training Statistics
        f.write("TRAINING STATISTICS\n")
        f.write("-" * 80 + "\n\n")

        for name, res in results_dict.items():
            f.write(f"{name}:\n")
            f.write(f"  Wind Power: {res['wind_power']}, Turbulence: {res['turbulence_power']}\n")
            f.write(f"  Total Episodes: {len(res['episode_returns'])}\n")

            # Last 100 episodes statistics
            last_100 = res['episode_returns'][-100:]
            f.write(f"\n  Last 100 Episodes:\n")
            f.write(f"    Mean Return: {np.mean(last_100):.2f} ± {np.std(last_100):.2f}\n")
            f.write(f"    Median Return: {np.median(last_100):.2f}\n")
            f.write(f"    Min Return: {np.min(last_100):.2f}\n")
            f.write(f"    Max Return: {np.max(last_100):.2f}\n")
            f.write(f"    25th Percentile: {np.percentile(last_100, 25):.2f}\n")
            f.write(f"    75th Percentile: {np.percentile(last_100, 75):.2f}\n")

            # Episode lengths
            last_100_len = res['episode_lengths'][-100:]
            f.write(f"\n  Episode Lengths (Last 100):\n")
            f.write(f"    Mean: {np.mean(last_100_len):.1f} ± {np.std(last_100_len):.1f}\n")
            f.write(f"    Median: {np.median(last_100_len):.1f}\n")

            # Loss statistics
            if len(res['losses']) > 0:
                f.write(f"\n  Training Loss:\n")
                f.write(f"    Mean: {np.mean(res['losses']):.4f}\n")
                f.write(f"    Final (last 1000): {np.mean(res['losses'][-1000:]):.4f}\n")

            # Q-value statistics
            if res['mean_maxQ'].size > 0:
                f.write(f"\n  Q-Values:\n")
                f.write(f"    Final Mean Max Q: {res['mean_maxQ'][-1]:.2f}\n")
                f.write(f"    Final Overestimation Gap: {res['gap'][-1]:.4f}\n")

            # Action distribution
            total_actions = sum(res['action_counts'].values())
            f.write(f"\n  Action Distribution:\n")
            action_names = ['Nothing', 'Left Engine', 'Main Engine', 'Right Engine']
            for i, name_a in enumerate(action_names):
                pct = res['action_counts'][i] / total_actions * 100
                f.write(f"    {name_a}: {pct:.2f}%\n")

            f.write("\n")

        # Evaluation Statistics
        f.write("\nEVALUATION STATISTICS (100 Episodes per Scenario)\n")
        f.write("-" * 80 + "\n\n")

        for scenario, stats in eval_results.items():
            f.write(f"{scenario}:\n")
            f.write(f"  Mean Return: {stats['mean_return']:.2f} ± {stats['std_return']:.2f}\n")
            f.write(f"  Median Return: {stats['median_return']:.2f}\n")
            f.write(f"  Min Return: {stats['min_return']:.2f}\n")
            f.write(f"  Max Return: {stats['max_return']:.2f}\n")
            f.write(f"  Mean Episode Length: {stats['mean_length']:.1f}\n")
            f.write(f"  Success Rate: {stats['success_rate']:.1f}%\n")
            f.write("\n")

        f.write("=" * 80 + "\n")

    print(f"✓ Statistics report saved to {output_file}")


def main():
    """Main execution function"""
    print("=" * 80)
    print("DQN vs Double DQN - Enhanced Analysis with Wind Variations")
    print("=" * 80)

    # Training configurations with different wind conditions
    configs = [
        # No wind (baseline)
        {
            'name': 'DQN_NoWind',
            'double_dqn': False,
            'seed': 0,
            'wind_power': 0.0,
            'turbulence_power': 0.0
        },
        {
            'name': 'DoubleDQN_NoWind',
            'double_dqn': True,
            'seed': 1,
            'wind_power': 0.0,
            'turbulence_power': 0.0
        },
        # Moderate wind
        {
            'name': 'DQN_ModerateWind',
            'double_dqn': False,
            'seed': 2,
            'wind_power': 10.0,
            'turbulence_power': 1.0
        },
        {
            'name': 'DoubleDQN_ModerateWind',
            'double_dqn': True,
            'seed': 3,
            'wind_power': 10.0,
            'turbulence_power': 1.0
        },
        # Strong wind
        {
            'name': 'DQN_StrongWind',
            'double_dqn': False,
            'seed': 4,
            'wind_power': 15.0,
            'turbulence_power': 1.5
        },
        {
            'name': 'DoubleDQN_StrongWind',
            'double_dqn': True,
            'seed': 5,
            'wind_power': 15.0,
            'turbulence_power': 1.5
        }
    ]

    # Train all configurations
    results = {}
    for config in configs:
        print(f"\n{'='*80}")
        print(f"Training: {config['name']}")
        print(f"{'='*80}")

        result = train_agent(
            algo_name=config['name'],
            seed=config['seed'],
            total_steps=500_000,
            double_dqn=config['double_dqn'],
            wind_power=config['wind_power'],
            turbulence_power=config['turbulence_power']
        )
        results[config['name']] = result

    # Create comprehensive training plots
    print("\n" + "="*80)
    print("Creating Training Analysis Plots...")
    print("="*80)
    create_comprehensive_plots(results, output_dir="plots")

    # Evaluate all trained agents
    print("\n" + "="*80)
    print("Evaluating Trained Agents...")
    print("="*80)

    eval_results = {}
    for name, res in results.items():
        print(f"\nEvaluating {name}...")
        eval_stats = evaluate_agent(
            res['q_online'],
            n_episodes=100,
            wind_power=res['wind_power'],
            turbulence_power=res['turbulence_power']
        )
        eval_results[name] = eval_stats
        print(f"  Mean Return: {eval_stats['mean_return']:.2f} ± {eval_stats['std_return']:.2f}")
        print(f"  Success Rate: {eval_stats['success_rate']:.1f}%")

    # Create evaluation plots
    create_evaluation_plots(eval_results, output_dir="plots")

    # Create statistics report
    create_statistics_report(results, eval_results, output_file="statistics_report.txt")

    # Record videos for each scenario
    print("\n" + "="*80)
    print("Recording Videos...")
    print("="*80)

    import os
    os.makedirs("videos", exist_ok=True)

    for name, res in results.items():
        video_file = f"videos/{name.lower().replace(' ', '_')}.mp4"
        print(f"\nRecording {name}...")
        reward, steps = record_video(
            res['q_online'],
            video_file,
            seed=42,
            wind_power=res['wind_power'],
            turbulence_power=res['turbulence_power']
        )

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print("\nGenerated files:")
    print("  Plots: plots/01-05_*.png")
    print("  Videos: videos/*.mp4")
    print("  Statistics: statistics_report.txt")
    print("\n")


if __name__ == "__main__":
    main()

