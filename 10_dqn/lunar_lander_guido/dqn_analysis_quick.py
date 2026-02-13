"""
Quick DQN vs Double DQN Analysis for LunarLander-v3
Reduced training for faster results with comprehensive visualizations
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio

import gymnasium as gym

# Set style
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
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
    total_steps=150_000,  # Reduced for faster execution
    start_learning=5_000,
    batch_size=64,
    buffer_size=50_000,
    gamma=0.99,
    lr=5e-4,
    hidden=256,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay_steps=100_000,
    target_update_every=1_000,
    train_every=4,
    double_dqn=False,
    wind_power=0.0,
    turbulence_power=0.0
):
    print(f"\nTraining {algo_name}...")
    print(f"  Steps: {total_steps}, Wind: {wind_power}, Turbulence: {turbulence_power}")

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

    episode_returns = []
    episode_lengths = []
    losses = []
    eps_log = []

    action_counts = {i: 0 for i in range(n_actions)}

    mean_maxQ_log = []
    gap_log = []
    step_log = []

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

        if step % 5_000 == 0 and len(buffer) >= 1_000:
            s_batch, *_ = buffer.sample(min(512, len(buffer)))
            mean_maxQ, gap = bias_metrics(q_online, q_target, s_batch)
            mean_maxQ_log.append(mean_maxQ)
            gap_log.append(gap)
            step_log.append(step)

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
                print(f"  Ep {len(episode_returns)}, Step {step}, "
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


def evaluate_agent(q_net, n_episodes=50, seed=42, wind_power=0.0, turbulence_power=0.0):
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
            for _ in range(30):
                frames.append(env.render())
            break

    env.close()

    imageio.mimsave(filename, frames, fps=30)
    print(f"  Saved {filename} (reward: {total_reward:.1f}, steps: {step+1})")

    return total_reward, step + 1


def create_comprehensive_plots(results_dict, output_dir="plots"):
    """Create comprehensive analysis plots"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    results = list(results_dict.values())

    # 1. Training Performance Overview
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Episode Returns
    ax = fig.add_subplot(gs[0, :2])
    for res in results:
        window = 50
        smoothed = moving_average(res["episode_returns"], window)
        episodes = np.arange(window//2, window//2 + len(smoothed))
        ax.plot(episodes, smoothed, label=res["name"], linewidth=2, alpha=0.8)
    ax.set_xlabel("Episode", fontsize=11, fontweight='bold')
    ax.set_ylabel("Return (MA-50)", fontsize=11, fontweight='bold')
    ax.set_title("Episode Returns Over Training", fontsize=13, fontweight='bold')
    ax.axhline(y=200, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Solved (200)')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    # Episode Lengths
    ax = fig.add_subplot(gs[0, 2:])
    for res in results:
        window = 50
        smoothed = moving_average(res["episode_lengths"], window)
        episodes = np.arange(window//2, window//2 + len(smoothed))
        ax.plot(episodes, smoothed, label=res["name"], linewidth=2, alpha=0.8)
    ax.set_xlabel("Episode", fontsize=11, fontweight='bold')
    ax.set_ylabel("Length (MA-50)", fontsize=11, fontweight='bold')
    ax.set_title("Episode Lengths Over Training", fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    # Training Loss
    ax = fig.add_subplot(gs[1, 0])
    for res in results:
        if len(res["losses"]) > 0:
            window = min(500, len(res["losses"])//5)
            smoothed = moving_average(res["losses"], window)
            steps = np.arange(window//2, window//2 + len(smoothed))
            ax.plot(steps, smoothed, label=res["name"], linewidth=2, alpha=0.8)
    ax.set_xlabel("Training Step", fontsize=10, fontweight='bold')
    ax.set_ylabel("Loss", fontsize=10, fontweight='bold')
    ax.set_title("Training Loss", fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # Mean Max Q Values
    ax = fig.add_subplot(gs[1, 1])
    for res in results:
        if len(res["bias_steps"]) > 0:
            ax.plot(res["bias_steps"], res["mean_maxQ"],
                    label=res["name"], linewidth=2, marker='o', markersize=5, alpha=0.8)
    ax.set_xlabel("Step", fontsize=10, fontweight='bold')
    ax.set_ylabel("Mean Max Q", fontsize=10, fontweight='bold')
    ax.set_title("Q-Value Estimates", fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # Overestimation Gap
    ax = fig.add_subplot(gs[1, 2])
    for res in results:
        if len(res["bias_steps"]) > 0:
            ax.plot(res["bias_steps"], res["gap"],
                    label=res["name"], linewidth=2, marker='o', markersize=5, alpha=0.8)
    ax.set_xlabel("Step", fontsize=10, fontweight='bold')
    ax.set_ylabel("Overestimation Gap", fontsize=10, fontweight='bold')
    ax.set_title("Q-Value Overestimation", fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

    # Epsilon Decay
    ax = fig.add_subplot(gs[1, 3])
    for res in results:
        steps = np.arange(len(res["eps"]))
        sample_idx = np.arange(0, len(steps), max(1, len(steps)//1000))
        ax.plot(steps[sample_idx], res["eps"][sample_idx],
                label=res["name"], linewidth=2, alpha=0.8)
    ax.set_xlabel("Step", fontsize=10, fontweight='bold')
    ax.set_ylabel("Epsilon", fontsize=10, fontweight='bold')
    ax.set_title("Exploration Rate Decay", fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # Return Distribution Box Plot
    ax = fig.add_subplot(gs[2, :2])
    data = [res["episode_returns"][-50:] for res in results]
    labels = [res["name"] for res in results]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True, notch=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(results)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Return", fontsize=11, fontweight='bold')
    ax.set_title("Return Distribution (Last 50 Episodes)", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=200, color='green', linestyle='--', alpha=0.5, linewidth=2)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

    # Action Distribution
    ax = fig.add_subplot(gs[2, 2:])
    action_names = ['Nothing', 'Left', 'Main', 'Right']
    x = np.arange(len(action_names))
    width = 0.8 / len(results)

    for i, res in enumerate(results):
        counts = res["action_counts"]
        total = sum(counts.values())
        percentages = [counts[j]/total*100 for j in range(len(counts))]
        offset = (i - len(results)/2 + 0.5) * width
        bars = ax.bar(x + offset, percentages, width, label=res["name"], alpha=0.8)

    ax.set_ylabel("Percentage (%)", fontsize=11, fontweight='bold')
    ax.set_title("Action Distribution", fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(action_names)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    plt.savefig(f"{output_dir}/01_comprehensive_training_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {output_dir}/01_comprehensive_training_analysis.png")

    # 2. Q-Value Evolution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for res in results:
        if not res["q_value_history"]:
            continue

        df = pd.DataFrame(res["q_value_history"])

        ax = axes[0, 0]
        ax.plot(df['step'], df['mean'], label=res["name"], linewidth=2, marker='o', markersize=4)
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel("Mean Q-Value", fontsize=11)
        ax.set_title("Mean Q-Value Evolution", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(df['step'], df['max'], label=f"{res['name']} (max)", linewidth=2, alpha=0.7)
        ax.plot(df['step'], df['min'], label=f"{res['name']} (min)", linewidth=2, alpha=0.7, linestyle='--')
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel("Q-Value", fontsize=11)
        ax.set_title("Q-Value Range (Min/Max)", fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(df['step'], df['std'], label=res["name"], linewidth=2, marker='o', markersize=4)
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel("Std Dev", fontsize=11)
        ax.set_title("Q-Value Standard Deviation", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.fill_between(df['step'], df['q25'], df['q75'], alpha=0.3, label=res["name"])
        ax.plot(df['step'], df['mean'], linewidth=2)
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel("Q-Value", fontsize=11)
        ax.set_title("Q-Value IQR (25th-75th Percentile)", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_qvalue_evolution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {output_dir}/02_qvalue_evolution.png")


def create_evaluation_plots(eval_results, output_dir="plots"):
    """Create evaluation comparison plots"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    scenarios = list(eval_results.keys())
    x = np.arange(len(scenarios))
    colors = plt.cm.viridis(np.linspace(0, 1, len(scenarios)))

    # Performance metrics
    ax = axes[0, 0]
    means = [eval_results[s]['mean_return'] for s in scenarios]
    stds = [eval_results[s]['std_return'] for s in scenarios]
    bars = ax.bar(x, means, yerr=stds, capsize=8, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("Mean Return", fontsize=12, fontweight='bold')
    ax.set_title("Performance Across Scenarios", fontsize=14, fontweight='bold')
    ax.axhline(y=200, color='green', linestyle='--', alpha=0.6, linewidth=2, label='Solved (200)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)

    # Success rate
    ax = axes[0, 1]
    success_rates = [eval_results[s]['success_rate'] for s in scenarios]
    bars = ax.bar(x, success_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("Success Rate (%)", fontsize=12, fontweight='bold')
    ax.set_title("Success Rate (Return ≥ 200)", fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{rate:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Episode length
    ax = axes[1, 0]
    lengths = [eval_results[s]['mean_length'] for s in scenarios]
    bars = ax.bar(x, lengths, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("Mean Episode Length", fontsize=12, fontweight='bold')
    ax.set_title("Episode Length Across Scenarios", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Return distribution violin
    ax = axes[1, 1]
    data = [eval_results[s]['returns'] for s in scenarios]
    positions = np.arange(len(scenarios))
    parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True, widths=0.7)

    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(scenarios, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("Return Distribution", fontsize=12, fontweight='bold')
    ax.set_title("Return Distribution Across Scenarios", fontsize=14, fontweight='bold')
    ax.axhline(y=200, color='green', linestyle='--', alpha=0.6, linewidth=2)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_evaluation_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {output_dir}/03_evaluation_comparison.png")


def create_wind_impact_analysis(eval_results, output_dir="plots"):
    """Analyze impact of wind conditions"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Separate DQN and Double DQN results by wind condition
    dqn_nowind = eval_results.get('DQN_NoWind', None)
    ddqn_nowind = eval_results.get('DoubleDQN_NoWind', None)
    dqn_moderate = eval_results.get('DQN_ModerateWind', None)
    ddqn_moderate = eval_results.get('DoubleDQN_ModerateWind', None)
    dqn_strong = eval_results.get('DQN_StrongWind', None)
    ddqn_strong = eval_results.get('DoubleDQN_StrongWind', None)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Wind conditions comparison
    ax = axes[0, 0]
    conditions = ['No Wind', 'Moderate Wind', 'Strong Wind']
    dqn_means = []
    ddqn_means = []
    dqn_stds = []
    ddqn_stds = []

    for res in [dqn_nowind, dqn_moderate, dqn_strong]:
        if res:
            dqn_means.append(res['mean_return'])
            dqn_stds.append(res['std_return'])
        else:
            dqn_means.append(0)
            dqn_stds.append(0)

    for res in [ddqn_nowind, ddqn_moderate, ddqn_strong]:
        if res:
            ddqn_means.append(res['mean_return'])
            ddqn_stds.append(res['std_return'])
        else:
            ddqn_means.append(0)
            ddqn_stds.append(0)

    x = np.arange(len(conditions))
    width = 0.35

    bars1 = ax.bar(x - width/2, dqn_means, width, yerr=dqn_stds, label='DQN',
                   capsize=5, color='skyblue', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, ddqn_means, width, yerr=ddqn_stds, label='Double DQN',
                   capsize=5, color='lightcoral', alpha=0.8, edgecolor='black')

    ax.set_ylabel('Mean Return', fontsize=12, fontweight='bold')
    ax.set_title('Performance Under Different Wind Conditions', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend(fontsize=11)
    ax.axhline(y=200, color='green', linestyle='--', alpha=0.5, linewidth=2)
    ax.grid(True, alpha=0.3, axis='y')

    # Success rate comparison
    ax = axes[0, 1]
    dqn_success = [res['success_rate'] if res else 0 for res in [dqn_nowind, dqn_moderate, dqn_strong]]
    ddqn_success = [res['success_rate'] if res else 0 for res in [ddqn_nowind, ddqn_moderate, ddqn_strong]]

    bars1 = ax.bar(x - width/2, dqn_success, width, label='DQN',
                   color='skyblue', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, ddqn_success, width, label='Double DQN',
                   color='lightcoral', alpha=0.8, edgecolor='black')

    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Success Rate Under Different Wind Conditions', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')

    # Robustness metric (lower is better)
    ax = axes[1, 0]
    dqn_std_returns = [res['std_return'] if res else 0 for res in [dqn_nowind, dqn_moderate, dqn_strong]]
    ddqn_std_returns = [res['std_return'] if res else 0 for res in [ddqn_nowind, ddqn_moderate, ddqn_strong]]

    bars1 = ax.bar(x - width/2, dqn_std_returns, width, label='DQN',
                   color='skyblue', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, ddqn_std_returns, width, label='Double DQN',
                   color='lightcoral', alpha=0.8, edgecolor='black')

    ax.set_ylabel('Return Std Dev', fontsize=12, fontweight='bold')
    ax.set_title('Return Variability (Robustness)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Return distribution heatmap
    ax = axes[1, 1]
    if all([dqn_nowind, ddqn_nowind, dqn_moderate, ddqn_moderate, dqn_strong, ddqn_strong]):
        data_matrix = []
        labels_y = []

        for name, res in [('DQN\nNo Wind', dqn_nowind), ('Double DQN\nNo Wind', ddqn_nowind),
                          ('DQN\nModerate', dqn_moderate), ('Double DQN\nModerate', ddqn_moderate),
                          ('DQN\nStrong', dqn_strong), ('Double DQN\nStrong', ddqn_strong)]:
            if res:
                data_matrix.append([
                    res['min_return'],
                    np.percentile(res['returns'], 25),
                    res['median_return'],
                    np.percentile(res['returns'], 75),
                    res['max_return']
                ])
                labels_y.append(name)

        data_matrix = np.array(data_matrix)
        im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=-300, vmax=300)

        ax.set_xticks(np.arange(5))
        ax.set_xticklabels(['Min', 'Q1', 'Median', 'Q3', 'Max'])
        ax.set_yticks(np.arange(len(labels_y)))
        ax.set_yticklabels(labels_y, fontsize=9)
        ax.set_title('Return Distribution Heatmap', fontsize=14, fontweight='bold')

        # Add values as text
        for i in range(len(labels_y)):
            for j in range(5):
                text = ax.text(j, i, f'{data_matrix[i, j]:.0f}',
                              ha="center", va="center", color="black", fontsize=9, fontweight='bold')

        plt.colorbar(im, ax=ax, label='Return Value')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/04_wind_impact_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {output_dir}/04_wind_impact_analysis.png")


def create_statistics_report(results_dict, eval_results, output_file="statistics_report.txt"):
    """Create detailed statistics report"""
    with open(output_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write(" " * 30 + "DQN vs Double DQN - Comprehensive Statistics Report\n")
        f.write("=" * 100 + "\n\n")

        f.write("TRAINING STATISTICS\n")
        f.write("-" * 100 + "\n\n")

        for name, res in results_dict.items():
            f.write(f"\n{name}:\n")
            f.write(f"  {'Wind Power:':<25} {res['wind_power']:.1f}\n")
            f.write(f"  {'Turbulence:':<25} {res['turbulence_power']:.1f}\n")
            f.write(f"  {'Total Episodes:':<25} {len(res['episode_returns'])}\n")

            last_n = min(50, len(res['episode_returns']))
            last_episodes = res['episode_returns'][-last_n:]
            f.write(f"\n  Last {last_n} Episodes:\n")
            f.write(f"    {'Mean Return:':<23} {np.mean(last_episodes):>8.2f} ± {np.std(last_episodes):>6.2f}\n")
            f.write(f"    {'Median Return:':<23} {np.median(last_episodes):>8.2f}\n")
            f.write(f"    {'Min Return:':<23} {np.min(last_episodes):>8.2f}\n")
            f.write(f"    {'Max Return:':<23} {np.max(last_episodes):>8.2f}\n")
            f.write(f"    {'25th Percentile:':<23} {np.percentile(last_episodes, 25):>8.2f}\n")
            f.write(f"    {'75th Percentile:':<23} {np.percentile(last_episodes, 75):>8.2f}\n")

            last_len = res['episode_lengths'][-last_n:]
            f.write(f"\n  Episode Lengths (Last {last_n}):\n")
            f.write(f"    {'Mean:':<23} {np.mean(last_len):>8.1f} ± {np.std(last_len):>6.1f}\n")
            f.write(f"    {'Median:':<23} {np.median(last_len):>8.1f}\n")

            if len(res['losses']) > 0:
                f.write(f"\n  Training Loss:\n")
                f.write(f"    {'Mean:':<23} {np.mean(res['losses']):>8.4f}\n")
                f.write(f"    {'Final (last 1000):':<23} {np.mean(res['losses'][-1000:]):>8.4f}\n")

            if res['mean_maxQ'].size > 0:
                f.write(f"\n  Q-Values:\n")
                f.write(f"    {'Final Mean Max Q:':<23} {res['mean_maxQ'][-1]:>8.2f}\n")
                f.write(f"    {'Final Overest. Gap:':<23} {res['gap'][-1]:>8.4f}\n")

            total_actions = sum(res['action_counts'].values())
            f.write(f"\n  Action Distribution:\n")
            action_names = ['Nothing', 'Left Engine', 'Main Engine', 'Right Engine']
            for i, act_name in enumerate(action_names):
                pct = res['action_counts'][i] / total_actions * 100
                count = res['action_counts'][i]
                f.write(f"    {act_name:<23} {pct:>6.2f}%  ({count:>8,} actions)\n")

        f.write("\n\n")
        f.write("EVALUATION STATISTICS\n")
        f.write("-" * 100 + "\n\n")

        for scenario, stats in eval_results.items():
            f.write(f"\n{scenario}:\n")
            f.write(f"  {'Mean Return:':<25} {stats['mean_return']:>8.2f} ± {stats['std_return']:>6.2f}\n")
            f.write(f"  {'Median Return:':<25} {stats['median_return']:>8.2f}\n")
            f.write(f"  {'Min Return:':<25} {stats['min_return']:>8.2f}\n")
            f.write(f"  {'Max Return:':<25} {stats['max_return']:>8.2f}\n")
            f.write(f"  {'Mean Episode Length:':<25} {stats['mean_length']:>8.1f}\n")
            f.write(f"  {'Success Rate:':<25} {stats['success_rate']:>7.1f}%\n")

        f.write("\n" + "=" * 100 + "\n")

    print(f"  ✓ Statistics report saved to {output_file}")


def main():
    """Main execution function"""
    print("\n" + "=" * 100)
    print(" " * 20 + "DQN vs Double DQN - Enhanced Analysis with Wind Variations")
    print("=" * 100 + "\n")

    configs = [
        {'name': 'DQN_NoWind', 'double_dqn': False, 'seed': 0, 'wind_power': 0.0, 'turbulence_power': 0.0},
        {'name': 'DoubleDQN_NoWind', 'double_dqn': True, 'seed': 1, 'wind_power': 0.0, 'turbulence_power': 0.0},
        {'name': 'DQN_ModerateWind', 'double_dqn': False, 'seed': 2, 'wind_power': 10.0, 'turbulence_power': 1.0},
        {'name': 'DoubleDQN_ModerateWind', 'double_dqn': True, 'seed': 3, 'wind_power': 10.0, 'turbulence_power': 1.0},
        {'name': 'DQN_StrongWind', 'double_dqn': False, 'seed': 4, 'wind_power': 15.0, 'turbulence_power': 1.5},
        {'name': 'DoubleDQN_StrongWind', 'double_dqn': True, 'seed': 5, 'wind_power': 15.0, 'turbulence_power': 1.5}
    ]

    # Train all configurations
    print("PHASE 1: Training Agents")
    print("-" * 100)
    results = {}
    for config in configs:
        result = train_agent(
            algo_name=config['name'],
            seed=config['seed'],
            total_steps=150_000,  # Reduced for faster execution
            double_dqn=config['double_dqn'],
            wind_power=config['wind_power'],
            turbulence_power=config['turbulence_power']
        )
        results[config['name']] = result

    # Create training plots
    print("\n" + "=" * 100)
    print("PHASE 2: Creating Training Analysis Plots")
    print("-" * 100)
    create_comprehensive_plots(results, output_dir="plots")

    # Evaluate agents
    print("\n" + "=" * 100)
    print("PHASE 3: Evaluating Trained Agents")
    print("-" * 100)

    eval_results = {}
    for name, res in results.items():
        print(f"\nEvaluating {name}...")
        eval_stats = evaluate_agent(
            res['q_online'],
            n_episodes=50,
            wind_power=res['wind_power'],
            turbulence_power=res['turbulence_power']
        )
        eval_results[name] = eval_stats
        print(f"  Mean Return: {eval_stats['mean_return']:.2f} ± {eval_stats['std_return']:.2f}")
        print(f"  Success Rate: {eval_stats['success_rate']:.1f}%")

    # Create evaluation plots
    create_evaluation_plots(eval_results, output_dir="plots")
    create_wind_impact_analysis(eval_results, output_dir="plots")

    # Create statistics report
    create_statistics_report(results, eval_results, output_file="statistics_report.txt")

    # Record videos
    print("\n" + "=" * 100)
    print("PHASE 4: Recording Videos")
    print("-" * 100)

    import os
    os.makedirs("videos", exist_ok=True)

    video_seeds = [42, 123, 456]  # Multiple seeds for variety
    for name, res in results.items():
        for vid_seed in video_seeds:
            video_file = f"videos/{name.lower().replace(' ', '_')}_seed{vid_seed}.mp4"
            print(f"\nRecording {name} (seed {vid_seed})...")
            reward, steps = record_video(
                res['q_online'],
                video_file,
                seed=vid_seed,
                wind_power=res['wind_power'],
                turbulence_power=res['turbulence_power']
            )

    print("\n" + "=" * 100)
    print(" " * 35 + "Analysis Complete!")
    print("=" * 100)
    print("\nGenerated files:")
    print("  📊 Plots:      plots/01-04_*.png")
    print("  🎥 Videos:     videos/*.mp4")
    print("  📄 Statistics: statistics_report.txt")
    print("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
    main()

