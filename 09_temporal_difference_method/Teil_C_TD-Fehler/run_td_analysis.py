#!/usr/bin/env python3
"""
Teil C: TD-Fehler (δ) Analyse - Standalone Script
SARSA (On-Policy) vs Q-Learning (Off-Policy)
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque

try:
    import gymnasium as gym
    _USE_GYMNASIUM = True
except ImportError:
    import gym
    _USE_GYMNASIUM = False


def make_env(seed=0):
    env = gym.make("Taxi-v3")
    if _USE_GYMNASIUM:
        env.reset(seed=seed)
    else:
        try:
            env.seed(seed)
        except Exception:
            pass
        env.reset()
    return env


def epsilon_greedy_action(Q, s, n_actions, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(0, n_actions))
    return int(np.argmax(Q[s]))


def moving_average(x, window=100):
    if len(x) == 0:
        return np.array([])
    w = min(window, len(x))
    out = np.empty(len(x))
    dq = deque()
    s = 0.0
    for i, v in enumerate(x):
        dq.append(v)
        s += v
        if len(dq) > w:
            s -= dq.popleft()
        out[i] = s / len(dq)
    return out


def train_sarsa_with_logging(
    num_episodes=5_000,
    alpha=0.1,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.1,
    eps_decay_episodes=3_500,
    seed=0
):
    """SARSA with TD error logging"""
    env = make_env(seed)
    n_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))
    rng = np.random.default_rng(seed)

    ep_returns = []
    ep_lengths = []
    td_errors = []

    for ep in range(num_episodes):
        frac = min(1.0, ep / max(1, eps_decay_episodes))
        eps = eps_start + frac * (eps_end - eps_start)

        reset_out = env.reset()
        s = reset_out[0] if _USE_GYMNASIUM else reset_out
        a = epsilon_greedy_action(Q, s, n_actions, eps, rng)

        done = False
        total_r = 0.0
        steps = 0

        while not done:
            step_out = env.step(a)
            if _USE_GYMNASIUM:
                s2, r, terminated, truncated, _info = step_out
                done = terminated or truncated
            else:
                s2, r, done, _info = step_out

            a2 = epsilon_greedy_action(Q, s2, n_actions, eps, rng) if not done else 0

            td_target = r + (0.0 if done else gamma * Q[s2][a2])
            td_error = float(td_target - Q[s][a])
            Q[s][a] += alpha * td_error

            td_errors.append(td_error)
            total_r += r
            steps += 1
            s, a = s2, a2

        ep_returns.append(total_r)
        ep_lengths.append(steps)

        if (ep + 1) % 1000 == 0:
            print(f"  SARSA Episode {ep+1}/{num_episodes}")

    env.close()
    return {
        "name": "SARSA (On-Policy)",
        "Q": Q,
        "ep_returns": np.array(ep_returns, dtype=np.float32),
        "ep_lengths": np.array(ep_lengths, dtype=np.int32),
        "td_errors": np.array(td_errors, dtype=np.float32),
    }


def train_q_learning_with_logging(
    num_episodes=5_000,
    alpha=0.1,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.1,
    eps_decay_episodes=3_500,
    seed=0
):
    """Q-Learning with TD error logging"""
    env = make_env(seed)
    n_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))
    rng = np.random.default_rng(seed)

    ep_returns = []
    ep_lengths = []
    td_errors = []

    for ep in range(num_episodes):
        frac = min(1.0, ep / max(1, eps_decay_episodes))
        eps = eps_start + frac * (eps_end - eps_start)

        reset_out = env.reset()
        s = reset_out[0] if _USE_GYMNASIUM else reset_out

        done = False
        total_r = 0.0
        steps = 0

        while not done:
            a = epsilon_greedy_action(Q, s, n_actions, eps, rng)

            step_out = env.step(a)
            if _USE_GYMNASIUM:
                s2, r, terminated, truncated, _info = step_out
                done = terminated or truncated
            else:
                s2, r, done, _info = step_out

            best_next = 0.0 if done else np.max(Q[s2])
            td_target = r + gamma * best_next
            td_error = float(td_target - Q[s][a])
            Q[s][a] += alpha * td_error

            td_errors.append(td_error)
            total_r += r
            steps += 1
            s = s2

        ep_returns.append(total_r)
        ep_lengths.append(steps)

        if (ep + 1) % 1000 == 0:
            print(f"  Q-Learning Episode {ep+1}/{num_episodes}")

    env.close()
    return {
        "name": "Q-Learning (Off-Policy)",
        "Q": Q,
        "ep_returns": np.array(ep_returns, dtype=np.float32),
        "ep_lengths": np.array(ep_lengths, dtype=np.int32),
        "td_errors": np.array(td_errors, dtype=np.float32),
    }


print("=" * 100)
print("TEIL C: TD-FEHLER (δ) ANALYSE")
print("=" * 100)
print()
print("Der TD-Fehler (Temporal Difference Error) δ = r + gamma * V(s') - V(s)")
print("ist zentral für das Verständnis, wie RL-Agenten lernen.")
print()

print("\nTraining SARSA und Q-Learning mit TD-Error Logging...")
print("Dies dauert ca. 2-3 Minuten...\n")

sarsa_log = train_sarsa_with_logging(num_episodes=5_000, alpha=0.1, seed=42)
qlearn_log = train_q_learning_with_logging(num_episodes=5_000, alpha=0.1, seed=42)

print("\n✓ Training completed!\n")

print("=" * 100)
print("C2: HISTOGRAMME - ANFANG vs ENDE")
print("=" * 100)
print()

chunk_size = 1000

sarsa_first = sarsa_log["td_errors"][:chunk_size]
sarsa_last = sarsa_log["td_errors"][-chunk_size:]

qlearn_first = qlearn_log["td_errors"][:chunk_size]
qlearn_last = qlearn_log["td_errors"][-chunk_size:]

print(f"Chunk size: {chunk_size} updates")
print()

print("SARSA Statistics:")
print("-" * 100)
print(f"  ANFANG (erste {chunk_size} updates):")
print(f"    Mean |δ|:    {np.mean(np.abs(sarsa_first)):.4f}")
print(f"    Median |δ|:  {np.median(np.abs(sarsa_first)):.4f}")
print(f"    Std |δ|:     {np.std(np.abs(sarsa_first)):.4f}")
print(f"    Max |δ|:     {np.max(np.abs(sarsa_first)):.4f}")
print()
print(f"  ENDE (letzte {chunk_size} updates):")
print(f"    Mean |δ|:    {np.mean(np.abs(sarsa_last)):.4f}")
print(f"    Median |δ|:  {np.median(np.abs(sarsa_last)):.4f}")
print(f"    Std |δ|:     {np.std(np.abs(sarsa_last)):.4f}")
print(f"    Max |δ|:     {np.max(np.abs(sarsa_last)):.4f}")
reduction = (np.mean(np.abs(sarsa_first)) - np.mean(np.abs(sarsa_last))) / np.mean(np.abs(sarsa_first)) * 100
print(f"  Reduktion Mean |δ|: {reduction:.1f}%")
print()

print("Q-Learning Statistics:")
print("-" * 100)
print(f"  ANFANG (erste {chunk_size} updates):")
print(f"    Mean |δ|:    {np.mean(np.abs(qlearn_first)):.4f}")
print(f"    Median |δ|:  {np.median(np.abs(qlearn_first)):.4f}")
print(f"    Std |δ|:     {np.std(np.abs(qlearn_first)):.4f}")
print(f"    Max |δ|:     {np.max(np.abs(qlearn_first)):.4f}")
print()
print(f"  ENDE (letzte {chunk_size} updates):")
print(f"    Mean |δ|:    {np.mean(np.abs(qlearn_last)):.4f}")
print(f"    Median |δ|:  {np.median(np.abs(qlearn_last)):.4f}")
print(f"    Std |δ|:     {np.std(np.abs(qlearn_last)):.4f}")
print(f"    Max |δ|:     {np.max(np.abs(qlearn_last)):.4f}")
reduction = (np.mean(np.abs(qlearn_first)) - np.mean(np.abs(qlearn_last))) / np.mean(np.abs(qlearn_first)) * 100
print(f"  Reduktion Mean |δ|: {reduction:.1f}%")
print()

# Create comprehensive TD error visualization
print("\nGenerating plot 1: Histogramme (Start vs Ende)...")
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# SARSA histograms
ax = fig.add_subplot(gs[0, 0])
ax.hist(sarsa_first, bins=60, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(np.mean(sarsa_first), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(sarsa_first):.3f}')
ax.set_xlabel('TD Error (δ)', fontsize=10)
ax.set_ylabel('Frequency', fontsize=10)
ax.set_title('SARSA: TD Error ANFANG\n(erste 1000 updates)', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[0, 1])
ax.hist(sarsa_last, bins=60, alpha=0.7, color='coral', edgecolor='black')
ax.axvline(np.mean(sarsa_last), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(sarsa_last):.3f}')
ax.set_xlabel('TD Error (δ)', fontsize=10)
ax.set_ylabel('Frequency', fontsize=10)
ax.set_title('SARSA: TD Error ENDE\n(letzte 1000 updates)', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[0, 2])
ax.hist(np.abs(sarsa_first), bins=60, alpha=0.5, label='Start', color='steelblue', edgecolor='black')
ax.hist(np.abs(sarsa_last), bins=60, alpha=0.5, label='End', color='coral', edgecolor='black')
ax.set_xlabel('|TD Error| (absolute)', fontsize=10)
ax.set_ylabel('Frequency', fontsize=10)
ax.set_title('SARSA: |δ| Vergleich\nStart vs. Ende', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Q-Learning histograms
ax = fig.add_subplot(gs[1, 0])
ax.hist(qlearn_first, bins=60, alpha=0.7, color='seagreen', edgecolor='black')
ax.axvline(np.mean(qlearn_first), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(qlearn_first):.3f}')
ax.set_xlabel('TD Error (δ)', fontsize=10)
ax.set_ylabel('Frequency', fontsize=10)
ax.set_title('Q-Learning: TD Error ANFANG\n(erste 1000 updates)', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 1])
ax.hist(qlearn_last, bins=60, alpha=0.7, color='lightsalmon', edgecolor='black')
ax.axvline(np.mean(qlearn_last), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(qlearn_last):.3f}')
ax.set_xlabel('TD Error (δ)', fontsize=10)
ax.set_ylabel('Frequency', fontsize=10)
ax.set_title('Q-Learning: TD Error ENDE\n(letzte 1000 updates)', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 2])
ax.hist(np.abs(qlearn_first), bins=60, alpha=0.5, label='Start', color='seagreen', edgecolor='black')
ax.hist(np.abs(qlearn_last), bins=60, alpha=0.5, label='End', color='lightsalmon', edgecolor='black')
ax.set_xlabel('|TD Error| (absolute)', fontsize=10)
ax.set_ylabel('Frequency', fontsize=10)
ax.set_title('Q-Learning: |δ| Vergleich\nStart vs. Ende', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Box plot
ax = fig.add_subplot(gs[2, 0])
data_to_plot = [np.abs(sarsa_first), np.abs(sarsa_last), np.abs(qlearn_first), np.abs(qlearn_last)]
labels = ['SARSA\nStart', 'SARSA\nEnd', 'Q-Learn\nStart', 'Q-Learn\nEnd']
bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
for patch, color in zip(bp['boxes'], ['steelblue', 'coral', 'seagreen', 'lightsalmon']):
    patch.set_facecolor(color)
ax.set_ylabel('|TD Error|', fontsize=10)
ax.set_title('Box Plot: |δ| Vergleich', fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# CDF
ax = fig.add_subplot(gs[2, 1])
sorted_sarsa = np.sort(np.abs(sarsa_first))
sorted_sarsa_last = np.sort(np.abs(sarsa_last))
sorted_qlearn = np.sort(np.abs(qlearn_first))
sorted_qlearn_last = np.sort(np.abs(qlearn_last))

ax.plot(sorted_sarsa, np.arange(len(sorted_sarsa)) / len(sorted_sarsa), label='SARSA Start', linewidth=2)
ax.plot(sorted_sarsa_last, np.arange(len(sorted_sarsa_last)) / len(sorted_sarsa_last), label='SARSA End', linewidth=2)
ax.plot(sorted_qlearn, np.arange(len(sorted_qlearn)) / len(sorted_qlearn), label='Q-Learn Start', linewidth=2)
ax.plot(sorted_qlearn_last, np.arange(len(sorted_qlearn_last)) / len(sorted_qlearn_last), label='Q-Learn End', linewidth=2)
ax.set_xlabel('|TD Error|', fontsize=10)
ax.set_ylabel('Cumulative Probability', fontsize=10)
ax.set_title('CDF: |δ| Verteilung', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Stats table
ax = fig.add_subplot(gs[2, 2])
ax.axis('off')
stats_text = "Statistics Summary\n" + "=" * 40 + "\n\n"
stats_text += "SARSA Start:  Mean={:.4f}\n".format(np.mean(np.abs(sarsa_first)))
stats_text += "SARSA End:    Mean={:.4f}\n".format(np.mean(np.abs(sarsa_last)))
red1 = (np.mean(np.abs(sarsa_first)) - np.mean(np.abs(sarsa_last))) / np.mean(np.abs(sarsa_first)) * 100
stats_text += f"SARSA Red: {red1:.1f}%\n\n"
stats_text += "Q-Learn Start: Mean={:.4f}\n".format(np.mean(np.abs(qlearn_first)))
stats_text += "Q-Learn End:   Mean={:.4f}\n".format(np.mean(np.abs(qlearn_last)))
red2 = (np.mean(np.abs(qlearn_first)) - np.mean(np.abs(qlearn_last))) / np.mean(np.abs(qlearn_first)) * 100
stats_text += f"Q-Learn Red: {red2:.1f}%\n"
ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('TD-Error (δ) Analyse: SARSA vs Q-Learning\nHistogramme und Statistiken',
             fontsize=14, fontweight='bold', y=0.995)
plt.savefig("C_03_TD_Error_Histograms_Start_End.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Plot saved: C_03_TD_Error_Histograms_Start_End.png\n")

# Time series analysis
print("Generating plot 2: Zeitreihe mit Moving Average...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

window = 500
sarsa_ma = moving_average(np.abs(sarsa_log["td_errors"]), window=window)
qlearn_ma = moving_average(np.abs(qlearn_log["td_errors"]), window=window)

# SARSA time series
ax = axes[0, 0]
ax.plot(sarsa_ma, linewidth=2, color='steelblue', label=f'Moving Avg (window={window})')
ax.fill_between(range(len(sarsa_ma)), 0, sarsa_ma, alpha=0.3, color='steelblue')
ax.set_xlabel('Update Step', fontsize=11)
ax.set_ylabel('|TD Error| (Moving Average)', fontsize=11)
ax.set_title('SARSA: |δ| Zeitreihe mit Moving Average', fontweight='bold', fontsize=12)
ax.grid(alpha=0.3)
ax.legend()

# SARSA early phase
ax = axes[0, 1]
early_steps = min(5000, len(sarsa_log["td_errors"]))
ax.scatter(range(early_steps), np.abs(sarsa_log["td_errors"][:early_steps]),
          alpha=0.3, s=10, color='steelblue', label='Raw |δ|')
sarsa_ma_early = moving_average(np.abs(sarsa_log["td_errors"][:early_steps]), window=100)
ax.plot(sarsa_ma_early, linewidth=2, color='red', label='Moving Avg (window=100)')
ax.set_xlabel('Update Step', fontsize=11)
ax.set_ylabel('|TD Error|', fontsize=11)
ax.set_title('SARSA: FRÜHE PHASE (erste 5000 updates)', fontweight='bold', fontsize=12)
ax.grid(alpha=0.3)
ax.legend()

# Q-Learning time series
ax = axes[1, 0]
ax.plot(qlearn_ma, linewidth=2, color='seagreen', label=f'Moving Avg (window={window})')
ax.fill_between(range(len(qlearn_ma)), 0, qlearn_ma, alpha=0.3, color='seagreen')
ax.set_xlabel('Update Step', fontsize=11)
ax.set_ylabel('|TD Error| (Moving Average)', fontsize=11)
ax.set_title('Q-Learning: |δ| Zeitreihe mit Moving Average', fontweight='bold', fontsize=12)
ax.grid(alpha=0.3)
ax.legend()

# Q-Learning early phase
ax = axes[1, 1]
early_steps = min(5000, len(qlearn_log["td_errors"]))
ax.scatter(range(early_steps), np.abs(qlearn_log["td_errors"][:early_steps]),
          alpha=0.3, s=10, color='seagreen', label='Raw |δ|')
qlearn_ma_early = moving_average(np.abs(qlearn_log["td_errors"][:early_steps]), window=100)
ax.plot(qlearn_ma_early, linewidth=2, color='darkred', label='Moving Avg (window=100)')
ax.set_xlabel('Update Step', fontsize=11)
ax.set_ylabel('|TD Error|', fontsize=11)
ax.set_title('Q-Learning: FRÜHE PHASE (erste 5000 updates)', fontweight='bold', fontsize=12)
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig("C_04_TD_Error_TimeSeries_MovingAverage.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Plot saved: C_04_TD_Error_TimeSeries_MovingAverage.png\n")

# Signed TD error analysis
print("Generating plot 3: Signed TD Error Analyse...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# SARSA signed
ax = axes[0, 0]
ax.hist(sarsa_log["td_errors"], bins=100, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='δ=0')
ax.axvline(np.mean(sarsa_log["td_errors"]), color='green', linestyle='--', linewidth=2,
          label=f'Mean δ = {np.mean(sarsa_log["td_errors"]):.3f}')
ax.set_xlabel('TD Error (δ)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title(f'SARSA: Signed TD Error Distribution\n(alle {len(sarsa_log["td_errors"]):.0f} updates)', fontweight='bold', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

# Q-Learning signed
ax = axes[0, 1]
ax.hist(qlearn_log["td_errors"], bins=100, alpha=0.7, color='seagreen', edgecolor='black')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='δ=0')
ax.axvline(np.mean(qlearn_log["td_errors"]), color='darkgreen', linestyle='--', linewidth=2,
          label=f'Mean δ = {np.mean(qlearn_log["td_errors"]):.3f}')
ax.set_xlabel('TD Error (δ)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title(f'Q-Learning: Signed TD Error Distribution\n(alle {len(qlearn_log["td_errors"]):.0f} updates)', fontweight='bold', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

# SARSA trend
ax = axes[1, 0]
sarsa_ma_signed = moving_average(sarsa_log["td_errors"], window=500)
ax.plot(sarsa_ma_signed, linewidth=2, color='steelblue', label='SARSA')
ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.fill_between(range(len(sarsa_ma_signed)), 0, sarsa_ma_signed, where=(sarsa_ma_signed >= 0),
                alpha=0.3, color='green', label='Underestimation (δ>0)')
ax.fill_between(range(len(sarsa_ma_signed)), 0, sarsa_ma_signed, where=(sarsa_ma_signed < 0),
                alpha=0.3, color='red', label='Overestimation (δ<0)')
ax.set_xlabel('Update Step', fontsize=11)
ax.set_ylabel('Mean TD Error (δ)', fontsize=11)
ax.set_title('SARSA: Signed TD Error Trend', fontweight='bold', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

# Q-Learning trend
ax = axes[1, 1]
qlearn_ma_signed = moving_average(qlearn_log["td_errors"], window=500)
ax.plot(qlearn_ma_signed, linewidth=2, color='seagreen', label='Q-Learning')
ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.fill_between(range(len(qlearn_ma_signed)), 0, qlearn_ma_signed, where=(qlearn_ma_signed >= 0),
                alpha=0.3, color='green', label='Underestimation (δ>0)')
ax.fill_between(range(len(qlearn_ma_signed)), 0, qlearn_ma_signed, where=(qlearn_ma_signed < 0),
                alpha=0.3, color='red', label='Overestimation (δ<0)')
ax.set_xlabel('Update Step', fontsize=11)
ax.set_ylabel('Mean TD Error (δ)', fontsize=11)
ax.set_title('Q-Learning: Signed TD Error Trend', fontweight='bold', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("C_05_TD_Error_Signed_Analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Plot saved: C_05_TD_Error_Signed_Analysis.png\n")

# Summary comparison
print("Generating plot 4: Summary Comparison...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Convergence
ax = axes[0, 0]
x = range(min(len(sarsa_ma), len(qlearn_ma)))
ax.semilogy(x, sarsa_ma[:len(x)], linewidth=2, label='SARSA', color='steelblue')
ax.semilogy(x, qlearn_ma[:len(x)], linewidth=2, label='Q-Learning', color='seagreen')
ax.set_xlabel('Update Step', fontsize=11)
ax.set_ylabel('|TD Error| - Moving Avg (log scale)', fontsize=11)
ax.set_title('Konvergenz: |δ| Abnahme über Zeit', fontweight='bold', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

# Reduction
ax = axes[0, 1]
methods = ['SARSA', 'Q-Learning']
reductions = [
    (np.mean(np.abs(sarsa_first)) - np.mean(np.abs(sarsa_last))) / np.mean(np.abs(sarsa_first)) * 100,
    (np.mean(np.abs(qlearn_first)) - np.mean(np.abs(qlearn_last))) / np.mean(np.abs(qlearn_first)) * 100
]
colors = ['steelblue', 'seagreen']
bars = ax.bar(methods, reductions, color=colors, edgecolor='black', linewidth=2)
ax.set_ylabel('Reduktion Mean |δ| (%)', fontsize=11)
ax.set_title('Reduktion: Start → Ende', fontweight='bold', fontsize=12)
ax.set_ylim(0, 100)
for bar, red in zip(bars, reductions):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{red:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# Bias comparison
ax = axes[1, 0]
sarsa_underest = np.sum(sarsa_log["td_errors"] > 0) / len(sarsa_log["td_errors"]) * 100
sarsa_overest = np.sum(sarsa_log["td_errors"] < 0) / len(sarsa_log["td_errors"]) * 100
qlearn_underest = np.sum(qlearn_log["td_errors"] > 0) / len(qlearn_log["td_errors"]) * 100
qlearn_overest = np.sum(qlearn_log["td_errors"] < 0) / len(qlearn_log["td_errors"]) * 100

underest = [sarsa_underest, qlearn_underest]
overest = [sarsa_overest, qlearn_overest]
x = np.arange(len(methods))
width = 0.35
ax.bar(x - width/2, underest, width, label='Unterschätzung (δ>0)', color='green', alpha=0.7)
ax.bar(x + width/2, overest, width, label='Überschätzung (δ<0)', color='red', alpha=0.7)
ax.set_ylabel('Prozentanteil (%)', fontsize=11)
ax.set_title('Bias: Unterschätzung vs Überschätzung', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
ax.set_ylim(0, 100)
ax.grid(alpha=0.3, axis='y')

# Magnitude comparison
ax = axes[1, 1]
means = [np.mean(np.abs(sarsa_log['td_errors'])), np.mean(np.abs(qlearn_log['td_errors']))]
stds = [np.std(np.abs(sarsa_log['td_errors'])), np.std(np.abs(qlearn_log['td_errors']))]
ax.bar(methods, means, yerr=stds, color=colors, edgecolor='black', linewidth=2, capsize=10)
ax.set_ylabel('Mean |TD Error|', fontsize=11)
ax.set_title('Magnitude: Durchschnittlicher |δ| mit Std Dev', fontweight='bold', fontsize=12)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("C_06_TD_Error_Summary_Comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Plot saved: C_06_TD_Error_Summary_Comparison.png\n")

# Print interpretation
print("=" * 100)
print("C5: INTERPRETATION DER ERGEBNISSE")
print("=" * 100)
print()

print("FRAGE 1: Wird δ im Verlauf kleiner? WARUM?")
print("-" * 100)
print()
print("ANTWORT: JA, |δ| nimmt im Verlauf deutlich ab.")
print()
print("Mathematische Erklärung:")
print("  δ_t = r_t + γ * V(s_{t+1}) - V(s_t)")
print()
print("  Zu Beginn:")
print("    • V(s) sind zufällige (alle ≈ 0)")
print("    • Große Unterschiede zwischen V(s_t) und V(s_{t+1})")
print("    • Große |δ|-Werte")
print()
print("  Im Verlauf:")
print("    • V(s) konvergieren zu realistischen Werten")
print("    • Bellman-Gleichung wird erfüllt: V(s) ≈ r + γ*V(s')")
print("    • Kleine |δ|-Werte (Agent hat alles gelernt)")
print()

print()
print("Empirische Beobachtung (aus den Daten):")
print()
print(f"SARSA:")
print(f"  • Unterschätzung (δ > 0): {sarsa_underest:.1f}%")
print(f"  • Überschätzung (δ < 0): {sarsa_overest:.1f}%")
print(f"  • Mean δ: {np.mean(sarsa_log['td_errors']):.4f}")
print()
print(f"Q-Learning:")
print(f"  • Unterschätzung (δ > 0): {qlearn_underest:.1f}%")
print(f"  • Überschätzung (δ < 0): {qlearn_overest:.1f}%")
print(f"  • Mean δ: {np.mean(qlearn_log['td_errors']):.4f}")
print()

print()
print("FRAGE 2: Was bedeutet großer POSITIVER δ?")
print("-" * 100)
print()
print("δ > 0  bedeutet:  r + γ*V(s') > V(s)  (Bellman-Gleichung verletzt)")
print()
print("Interpretation:")
print("  ✓ Agent UNTERSCHÄTZT den Zustand s")
print("  ✓ Wahrer Wert von s ist HÖHER als aktuell geschätzt")
print("  ✓ Q(s,a) wird erhöht um Schätzung zu korrigieren")
print()
print("Beispiel:")
print("  V(s) = 5.0, r = 10.0, V(s') = 20.0, γ = 0.99")
print("  δ = 10.0 + 0.99*20.0 - 5.0 = 24.8 > 0")
print("  → Deutliche Unterschätzung! Update: V(s) += α * 24.8")
print()

print()
print("FRAGE 3: Was bedeutet großer NEGATIVER δ?")
print("-" * 100)
print()
print("δ < 0  bedeutet:  r + γ*V(s') < V(s)  (Bellman-Gleichung verletzt)")
print()
print("Interpretation:")
print("  ✗ Agent ÜBERSCHÄTZT den Zustand s")
print("  ✗ Wahrer Wert von s ist NIEDRIGER als aktuell geschätzt")
print("  ✗ Q(s,a) wird reduziert um Schätzung zu korrigieren")
print()
print("Beispiel:")
print("  V(s) = 30.0, r = -5.0, V(s') = 10.0, γ = 0.99")
print("  δ = -5.0 + 0.99*10.0 - 30.0 = -25.1 < 0")
print("  → Deutliche Überschätzung! Update: V(s) += α * (-25.1)")
print()

print()
print("VERGLEICH SARSA vs Q-LEARNING:")
print("-" * 100)
if sarsa_underest > qlearn_underest:
    print(f"SARSA hat MEHR Unterschätzung ({sarsa_underest:.1f}% vs {qlearn_underest:.1f}%)")
    print("  → SARSA ist konservativer, vorsichtiger")
else:
    print(f"Q-Learning hat MEHR Unterschätzung ({qlearn_underest:.1f}% vs {sarsa_underest:.1f}%)")
    print("  → Q-Learning ist aggressiver")

print()
if abs(np.mean(sarsa_log['td_errors'])) > abs(np.mean(qlearn_log['td_errors'])):
    print(f"Q-Learning hat kleineren Mean δ ({np.mean(qlearn_log['td_errors']):.4f} vs {np.mean(sarsa_log['td_errors']):.4f})")
else:
    print(f"SARSA hat kleineren Mean δ ({np.mean(sarsa_log['td_errors']):.4f} vs {np.mean(qlearn_log['td_errors']):.4f})")
print()

print("\n" + "=" * 100)
print("TEIL C ABGESCHLOSSEN ✓")
print("=" * 100)
print()
print("Generierte Plots:")
print("  1. C_03_TD_Error_Histograms_Start_End.png")
print("  2. C_04_TD_Error_TimeSeries_MovingAverage.png")
print("  3. C_05_TD_Error_Signed_Analysis.png")
print("  4. C_06_TD_Error_Summary_Comparison.png")
print()
