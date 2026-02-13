"""
Quick DQN vs Double DQN Visualization
Creates plots immediately with minimal training
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque
import random
import imageio.v2 as imageio
import os

# Set style
plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 150

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def moving_average(x, window=50):
    if len(x) < window:
        return x
    c = np.cumsum(np.insert(x, 0, 0))
    return (c[window:] - c[:-window]) / window

class ReplayBuffer:
    def __init__(self, capacity=50_000):
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

@torch.no_grad()
def select_action(q_net, obs, eps, n_actions):
    if random.random() < eps:
        return random.randrange(n_actions)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    q = q_net(obs_t)
    return int(torch.argmax(q, dim=1).item())

def train_step(q_online, q_target, optimizer, buffer, batch_size=64, gamma=0.99, double_dqn=False):
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

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_online.parameters(), 10.0)
    optimizer.step()

    return float(loss.item())

@torch.no_grad()
def compute_overestimation_gap(q_online, q_target, states_batch):
    s_t = torch.tensor(states_batch, dtype=torch.float32, device=DEVICE)
    q_on = q_online(s_t)
    q_tgt = q_target(s_t)

    mean_maxQ = float(q_on.max(dim=1)[0].mean().item())

    max_tgt = q_tgt.max(dim=1)[0]
    a_star = torch.argmax(q_on, dim=1)
    eval_tgt = q_tgt.gather(1, a_star.unsqueeze(1)).squeeze(1)
    gap = float((max_tgt - eval_tgt).mean().item())

    return mean_maxQ, gap

def train_agent(name, seed, double_dqn, steps=50_000, wind_power=0.0, turbulence_power=0.0):
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")

    set_seed(seed)
    env = gym.make("LunarLander-v3", wind_power=wind_power, turbulence_power=turbulence_power,
                   enable_wind=(wind_power > 0 or turbulence_power > 0))
    env.reset(seed=seed)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_online = QNet(obs_dim, n_actions, hidden=128).to(DEVICE)
    q_target = QNet(obs_dim, n_actions, hidden=128).to(DEVICE)
    q_target.load_state_dict(q_online.state_dict())
    q_target.eval()

    optimizer = optim.Adam(q_online.parameters(), lr=5e-4)
    buffer = ReplayBuffer(capacity=30_000)

    episode_returns = []
    episode_lengths = []
    losses = []
    mean_maxQ_log = []
    gap_log = []
    step_log = []

    obs, _ = env.reset()
    ep_return = 0.0
    ep_len = 0

    for step in range(1, steps + 1):
        eps = max(0.01, 1.0 - step / (steps * 0.8))

        a = select_action(q_online, obs, eps, n_actions)
        obs2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        buffer.push(obs, a, r, obs2, done)

        obs = obs2
        ep_return += r
        ep_len += 1

        if step >= 1000 and step % 4 == 0:
            loss = train_step(q_online, q_target, optimizer, buffer,
                            batch_size=64, gamma=0.99, double_dqn=double_dqn)
            if loss is not None:
                losses.append(loss)

        if step % 1000 == 0:
            q_target.load_state_dict(q_online.state_dict())

        if step % 2500 == 0 and len(buffer) >= 512:
            s_batch, *_ = buffer.sample(512)
            mean_q, gap = compute_overestimation_gap(q_online, q_target, s_batch)
            mean_maxQ_log.append(mean_q)
            gap_log.append(gap)
            step_log.append(step)

        if done:
            episode_returns.append(ep_return)
            episode_lengths.append(ep_len)

            if len(episode_returns) % 25 == 0:
                avg_ret = np.mean(episode_returns[-25:])
                print(f"  Ep {len(episode_returns):3d}, Step {step:6d}, Avg Return: {avg_ret:7.1f}")

            obs, _ = env.reset()
            ep_return = 0.0
            ep_len = 0

    env.close()

    return {
        'name': name,
        'q_online': q_online,
        'returns': np.array(episode_returns),
        'lengths': np.array(episode_lengths),
        'losses': np.array(losses),
        'mean_maxQ': np.array(mean_maxQ_log),
        'gap': np.array(gap_log),
        'step_log': np.array(step_log),
        'wind': wind_power,
        'turb': turbulence_power
    }

def create_comparison_plots(results, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)

    # Main comparison plot
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # 1. Episode Returns
    ax = fig.add_subplot(gs[0, :2])
    for res in results.values():
        window = 30
        smoothed = moving_average(res['returns'], window)
        episodes = np.arange(window//2, window//2 + len(smoothed))
        ax.plot(episodes, smoothed, label=res['name'], linewidth=2.5, alpha=0.85)
    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Return (MA-30)', fontsize=12, fontweight='bold')
    ax.set_title('Training Performance: Episode Returns', fontsize=14, fontweight='bold')
    ax.axhline(y=200, color='green', linestyle='--', linewidth=2, alpha=0.6, label='Solved (200)')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.4)

    # 2. Overestimation Gap (KEY PLOT!)
    ax = fig.add_subplot(gs[0, 2])
    for res in results.values():
        if len(res['gap']) > 0:
            ax.plot(res['step_log'], res['gap'], label=res['name'],
                   linewidth=2.5, marker='o', markersize=6, alpha=0.85)
    ax.set_xlabel('Training Step', fontsize=11, fontweight='bold')
    ax.set_ylabel('Overestimation Gap', fontsize=11, fontweight='bold')
    ax.set_title('Q-Value Overestimation\n(DQN > Double DQN)', fontsize=13, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.4)

    # 3. Mean Max Q
    ax = fig.add_subplot(gs[1, 0])
    for res in results.values():
        if len(res['mean_maxQ']) > 0:
            ax.plot(res['step_log'], res['mean_maxQ'], label=res['name'],
                   linewidth=2, marker='s', markersize=5, alpha=0.85)
    ax.set_xlabel('Training Step', fontsize=10, fontweight='bold')
    ax.set_ylabel('Mean Max Q', fontsize=10, fontweight='bold')
    ax.set_title('Q-Value Magnitude', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.4)

    # 4. Training Loss
    ax = fig.add_subplot(gs[1, 1])
    for res in results.values():
        if len(res['losses']) > 0:
            window = min(200, len(res['losses'])//5 + 1)
            smoothed = moving_average(res['losses'], window)
            steps = np.arange(window//2, window//2 + len(smoothed))
            ax.plot(steps, smoothed, label=res['name'], linewidth=2, alpha=0.85)
    ax.set_xlabel('Training Step', fontsize=10, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=10, fontweight='bold')
    ax.set_title('Training Loss (Smoothed)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.4)

    # 5. Episode Lengths
    ax = fig.add_subplot(gs[1, 2])
    for res in results.values():
        window = 30
        smoothed = moving_average(res['lengths'], window)
        episodes = np.arange(window//2, window//2 + len(smoothed))
        ax.plot(episodes, smoothed, label=res['name'], linewidth=2, alpha=0.85)
    ax.set_xlabel('Episode', fontsize=10, fontweight='bold')
    ax.set_ylabel('Length (MA-30)', fontsize=10, fontweight='bold')
    ax.set_title('Episode Lengths', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.4)

    # 6. Final Performance Bar Chart
    ax = fig.add_subplot(gs[2, 0])
    names = list(results.keys())
    final_returns = [np.mean(res['returns'][-30:]) if len(res['returns']) >= 30
                     else np.mean(res['returns']) for res in results.values()]
    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
    bars = ax.bar(range(len(names)), final_returns, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([str(i+1) for i in range(len(names))], fontsize=10)
    ax.set_ylabel('Mean Return (Last 30 Eps)', fontsize=10, fontweight='bold')
    ax.set_title('Final Performance', fontsize=12, fontweight='bold')
    ax.axhline(y=200, color='green', linestyle='--', linewidth=2, alpha=0.6)
    ax.grid(True, alpha=0.4, axis='y')

    for bar, val in zip(bars, final_returns):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 7. Return Distribution
    ax = fig.add_subplot(gs[2, 1])
    data = [res['returns'][-30:] if len(res['returns']) >= 30 else res['returns']
            for res in results.values()]
    bp = ax.boxplot(data, labels=[str(i+1) for i in range(len(names))],
                    patch_artist=True, showmeans=True, notch=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Return Distribution', fontsize=10, fontweight='bold')
    ax.set_title('Return Variability', fontsize=12, fontweight='bold')
    ax.axhline(y=200, color='green', linestyle='--', linewidth=2, alpha=0.6)
    ax.grid(True, alpha=0.4, axis='y')

    # 8. Overestimation Comparison Bar
    ax = fig.add_subplot(gs[2, 2])
    final_gaps = [res['gap'][-1] if len(res['gap']) > 0 else 0 for res in results.values()]
    bars = ax.bar(range(len(names)), final_gaps, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([str(i+1) for i in range(len(names))], fontsize=10)
    ax.set_ylabel('Final Overestimation Gap', fontsize=10, fontweight='bold')
    ax.set_title('Overestimation Comparison', fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    ax.grid(True, alpha=0.4, axis='y')

    for bar, val in zip(bars, final_gaps):
        height = bar.get_height()
        y_pos = height if height > 0 else height - 0.1
        va = 'bottom' if height > 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
               f'{val:.3f}', ha='center', va=va, fontsize=9, fontweight='bold')

    plt.savefig(f'{output_dir}/dqn_vs_double_dqn_comprehensive.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_dir}/dqn_vs_double_dqn_comprehensive.png")
    plt.close()

    # Legend
    print("\nLegend:")
    for i, name in enumerate(names, 1):
        print(f"  {i}: {name}")

def record_video(q_net, filename, seed=42, wind_power=0.0, turbulence_power=0.0):
    env = gym.make("LunarLander-v3", render_mode="rgb_array",
                   wind_power=wind_power, turbulence_power=turbulence_power,
                   enable_wind=(wind_power > 0 or turbulence_power > 0))
    env.reset(seed=seed)
    obs, _ = env.reset()
    frames = []
    total_reward = 0

    for step in range(1000):
        frame = env.render()
        frames.append(frame)

        a = select_action(q_net, obs, eps=0.0, n_actions=env.action_space.n)
        obs, r, terminated, truncated, _ = env.step(a)
        total_reward += r

        if terminated or truncated:
            for _ in range(30):
                frames.append(env.render())
            break

    env.close()

    imageio.mimsave(filename, frames, fps=30)
    print(f"  Saved video: {filename} (reward: {total_reward:.1f})")

    return total_reward

def print_statistics(results):
    print("\n" + "="*80)
    print(" "*25 + "TRAINING STATISTICS")
    print("="*80)

    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  Episodes: {len(res['returns'])}")
        print(f"  Final 30 Episodes:")
        final = res['returns'][-30:] if len(res['returns']) >= 30 else res['returns']
        print(f"    Mean Return:  {np.mean(final):8.2f} ± {np.std(final):6.2f}")
        print(f"    Median:       {np.median(final):8.2f}")
        print(f"    Min/Max:      {np.min(final):8.2f} / {np.max(final):8.2f}")

        if len(res['gap']) > 0:
            print(f"  Q-Value Analysis:")
            print(f"    Final Mean Max Q:      {res['mean_maxQ'][-1]:8.2f}")
            print(f"    Final Overest. Gap:    {res['gap'][-1]:8.4f}")

    print("\n" + "="*80)

def main():
    print("="*80)
    print(" "*20 + "DQN vs Double DQN - Quick Analysis")
    print("="*80)
    print("\nThis will train 4 agents with 50k steps each (~30-40 minutes total)")
    print("Adjust TRAINING_STEPS below for faster/slower training\n")

    TRAINING_STEPS = 50_000  # Adjust this: 25k=fast, 50k=good, 100k=excellent

    configs = [
        {'name': 'DQN_NoWind', 'seed': 0, 'double_dqn': False, 'wind': 0.0, 'turb': 0.0},
        {'name': 'DoubleDQN_NoWind', 'seed': 1, 'double_dqn': True, 'wind': 0.0, 'turb': 0.0},
        {'name': 'DQN_Wind', 'seed': 2, 'double_dqn': False, 'wind': 12.0, 'turb': 1.2},
        {'name': 'DoubleDQN_Wind', 'seed': 3, 'double_dqn': True, 'wind': 12.0, 'turb': 1.2},
    ]

    results = {}
    for cfg in configs:
        result = train_agent(
            name=cfg['name'],
            seed=cfg['seed'],
            double_dqn=cfg['double_dqn'],
            steps=TRAINING_STEPS,
            wind_power=cfg['wind'],
            turbulence_power=cfg['turb']
        )
        results[cfg['name']] = result

        # Save checkpoint
        torch.save(result['q_online'].state_dict(), f"{cfg['name']}_model.pt")

    print("\n" + "="*80)
    print("Creating Plots...")
    print("="*80)
    create_comparison_plots(results)

    print_statistics(results)

    # Generate videos
    print("\n" + "="*80)
    print("Generating Videos...")
    print("="*80)
    os.makedirs('videos', exist_ok=True)

    for name, res in results.items():
        video_file = f"videos/{name.lower()}.mp4"
        record_video(res['q_online'], video_file,
                    wind_power=res['wind'], turbulence_power=res['turb'])

    print("\n" + "="*80)
    print(" "*30 + "✓ COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  📊 Plot:  plots/dqn_vs_double_dqn_comprehensive.png")
    print("  🎥 Videos: videos/*.mp4")
    print("  💾 Models: *_model.pt")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()

