#!/usr/bin/env python3
"""
TEIL A: Verbesserte Visualisierungen mit deutlicher Streuung
- Größere Fonts
- Deutlichere Min/Max Bereiche
- Bessere Farbgebung für Streuung
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import gymnasium as gym
import json
import os

# ============================================================================
# KONFIGURATION
# ============================================================================

SEEDS = [0, 1, 2, 3, 4]
NUM_EPISODES = 20_000
EVAL_EPISODES = 300
OUTPUT_DIR = "/home/isc-den/cas-artificial-intelligence/09_temporal_difference_method/Teil_A"

# ============================================================================
# HELPER FUNKTIONEN
# ============================================================================

def make_env(seed: int = 0):
    env = gym.make("Taxi-v3")
    env.reset(seed=seed)
    return env

def epsilon_greedy_action(Q, s, n_actions: int, eps: float, rng: np.random.Generator):
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

# ============================================================================
# TRAININGS-FUNKTIONEN (VERKÜRZT - nutzen gecachte Daten)
# ============================================================================

def load_or_train_all():
    """Lade trainierte Modelle oder trainiere sie"""
    import pickle

    results = {
        "MC": [],
        "SARSA": [],
        "Q-Learning": []
    }

    # Versuche, gespeicherte Daten zu laden
    cache_file = os.path.join(OUTPUT_DIR, "training_results_cache.pkl")

    if os.path.exists(cache_file):
        print("Lade gecachte Trainingsergebnisse...")
        try:
            with open(cache_file, "rb") as f:
                # Lade als dict statt pickle.load um defaultdict Problem zu vermeiden
                data = pickle.load(f)
                return data
        except:
            print("Cache konnte nicht geladen werden, trainiere neu...")

    # Falls Cache nicht existiert, trainiere
    print("Trainiere alle Algorithmen...")

    def train_mc_control_taxi(num_episodes=20_000, gamma=0.99, eps_start=1.0,
                             eps_end=0.05, eps_decay_episodes=15_000, seed=0):
        env = make_env(seed)
        n_actions = env.action_space.n
        Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))
        returns_sum = defaultdict(float)
        returns_count = defaultdict(int)
        rng = np.random.default_rng(seed)

        ep_returns = []
        ep_lengths = []

        for ep in range(num_episodes):
            frac = min(1.0, ep / max(1, eps_decay_episodes))
            eps = eps_start + frac * (eps_end - eps_start)

            reset_out = env.reset()
            s = reset_out[0]
            trajectory = []
            done = False
            total_r = 0.0
            steps = 0

            while not done:
                a = epsilon_greedy_action(Q, s, n_actions, eps, rng)
                s2, r, terminated, truncated, _info = env.step(a)
                done = terminated or truncated
                trajectory.append((s, a, r))
                total_r += r
                steps += 1
                s = s2

            G = 0.0
            for (s_t, a_t, r_t) in reversed(trajectory):
                G = r_t + gamma * G
                key = (s_t, a_t)
                returns_sum[key] += G
                returns_count[key] += 1
                Q[s_t][a_t] = returns_sum[key] / returns_count[key]

            ep_returns.append(total_r)
            ep_lengths.append(steps)

            if (ep + 1) % 5000 == 0:
                print(f"  MC Episode {ep+1}/{num_episodes}")

        env.close()
        return {
            "name": "MC",
            "Q": dict(Q),  # Konvertiere zu dict für Pickle
            "ep_returns": np.array(ep_returns, dtype=np.float32),
            "ep_lengths": np.array(ep_lengths, dtype=np.int32),
        }

    def train_sarsa_taxi(num_episodes=20_000, alpha=0.1, gamma=0.99, eps_start=1.0,
                         eps_end=0.05, eps_decay_episodes=15_000, seed=0):
        env = make_env(seed)
        n_actions = env.action_space.n
        Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))
        rng = np.random.default_rng(seed)

        ep_returns = []
        ep_lengths = []

        for ep in range(num_episodes):
            frac = min(1.0, ep / max(1, eps_decay_episodes))
            eps = eps_start + frac * (eps_end - eps_start)

            reset_out = env.reset()
            s = reset_out[0]
            a = epsilon_greedy_action(Q, s, n_actions, eps, rng)

            done = False
            total_r = 0.0
            steps = 0

            while not done:
                s2, r, terminated, truncated, _info = env.step(a)
                done = terminated or truncated

                a2 = epsilon_greedy_action(Q, s2, n_actions, eps, rng) if not done else 0
                td_target = r + (0.0 if done else gamma * Q[s2][a2])
                td_error = td_target - Q[s][a]
                Q[s][a] += alpha * td_error

                total_r += r
                steps += 1
                s, a = s2, a2

            ep_returns.append(total_r)
            ep_lengths.append(steps)

            if (ep + 1) % 5000 == 0:
                print(f"  SARSA Episode {ep+1}/{num_episodes}")

        env.close()
        return {
            "name": "SARSA",
            "Q": dict(Q),
            "ep_returns": np.array(ep_returns, dtype=np.float32),
            "ep_lengths": np.array(ep_lengths, dtype=np.int32),
        }

    def train_q_learning_taxi(num_episodes=20_000, alpha=0.1, gamma=0.99, eps_start=1.0,
                              eps_end=0.05, eps_decay_episodes=15_000, seed=0):
        env = make_env(seed)
        n_actions = env.action_space.n
        Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))
        rng = np.random.default_rng(seed)

        ep_returns = []
        ep_lengths = []

        for ep in range(num_episodes):
            frac = min(1.0, ep / max(1, eps_decay_episodes))
            eps = eps_start + frac * (eps_end - eps_start)

            reset_out = env.reset()
            s = reset_out[0]

            done = False
            total_r = 0.0
            steps = 0

            while not done:
                a = epsilon_greedy_action(Q, s, n_actions, eps, rng)
                s2, r, terminated, truncated, _info = env.step(a)
                done = terminated or truncated

                best_next = 0.0 if done else np.max(Q[s2])
                td_target = r + gamma * best_next
                td_error = td_target - Q[s][a]
                Q[s][a] += alpha * td_error

                total_r += r
                steps += 1
                s = s2

            ep_returns.append(total_r)
            ep_lengths.append(steps)

            if (ep + 1) % 5000 == 0:
                print(f"  Q-Learning Episode {ep+1}/{num_episodes}")

        env.close()
        return {
            "name": "Q-Learning",
            "Q": dict(Q),
            "ep_returns": np.array(ep_returns, dtype=np.float32),
            "ep_lengths": np.array(ep_lengths, dtype=np.int32),
        }

    print("\n=== TRAINIERE MIT 5 SEEDS ===\n")

    for seed in SEEDS:
        print(f"Seed {seed}:")
        results["MC"].append(train_mc_control_taxi(num_episodes=NUM_EPISODES, seed=seed))
        results["SARSA"].append(train_sarsa_taxi(num_episodes=NUM_EPISODES, seed=seed))
        results["Q-Learning"].append(train_q_learning_taxi(num_episodes=NUM_EPISODES, seed=seed))
        print()

    return results

# ============================================================================
# HAUPTPROGRAMM
# ============================================================================

print("=" * 80)
print("TEIL A: VERBESSERTE VISUALISIERUNGEN MIT DEUTLICHER STREUUNG")
print("=" * 80)

# Trainiere oder lade Daten
results = load_or_train_all()

greedy_results = {
    "MC": {"returns": [], "lengths": []},
    "SARSA": {"returns": [], "lengths": []},
    "Q-Learning": {"returns": [], "lengths": []}
}

print("\n=== GREEDY EVALUATION ===\n")

def evaluate_greedy(Q, num_episodes=300, seed=123):
    env = make_env(seed)
    n_actions = env.action_space.n

    returns = []
    lengths = []

    for ep in range(num_episodes):
        reset_out = env.reset()
        s = reset_out[0]

        done = False
        total_r = 0.0
        steps = 0

        while not done:
            a = int(np.argmax(Q[s])) if s in Q else 0
            s2, r, terminated, truncated, _info = env.step(a)
            done = terminated or truncated
            total_r += r
            steps += 1
            s = s2

        returns.append(total_r)
        lengths.append(steps)

    env.close()
    return float(np.mean(returns)), float(np.mean(lengths))

for alg_name in ["MC", "SARSA", "Q-Learning"]:
    print(f"{alg_name}:")
    for i, run in enumerate(results[alg_name]):
        mean_ret, mean_len = evaluate_greedy(run["Q"], num_episodes=EVAL_EPISODES, seed=999+i)
        greedy_results[alg_name]["returns"].append(mean_ret)
        greedy_results[alg_name]["lengths"].append(mean_len)
        print(f"  Seed {SEEDS[i]}: Return={mean_ret:7.2f}, Length={mean_len:7.2f}")

# ============================================================================
# PLOT 1: VERBESSERTE LEARNING CURVES MIT DEUTLICHER STREUUNG
# ============================================================================

print("\n=== GENERIERE VERBESSERTE VISUALISIERUNGEN ===\n")
print("Plot 1: Learning Curves (Return + Episode Length)...", end=" ", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Konfiguration für bessere Sichtbarkeit
colors_dict = {"MC": "#d62728", "SARSA": "#1f77b4", "Q-Learning": "#2ca02c"}
alphas_dict = {"MC": 0.25, "SARSA": 0.25, "Q-Learning": 0.25}

# ===== RETURNS PLOT =====
ax = axes[0]

for alg_name in ["MC", "SARSA", "Q-Learning"]:
    runs = results[alg_name]
    color = colors_dict[alg_name]

    # Berechne Moving Averages für alle Runs
    all_ma = np.array([moving_average(run["ep_returns"], window=200) for run in runs])

    # Zeichne alle 5 Seeds als dünne Linien
    for i, ma in enumerate(all_ma):
        ax.plot(ma, alpha=0.3, color=color, linewidth=0.9, label=f"{alg_name} Seed {i}" if i == 0 else "")

    # Berechne Mittelwert und Streuung
    mean_ma = all_ma.mean(axis=0)
    std_ma = all_ma.std(axis=0)
    min_ma = all_ma.min(axis=0)
    max_ma = all_ma.max(axis=0)

    # Zeichne dicke Mittelwertlinie
    ax.plot(mean_ma, label=f"{alg_name} Mean", color=color, linewidth=3.5, zorder=10)

    # Zeichne Streuung als breite schattierte Bereiche
    # ±1 Std Dev
    ax.fill_between(np.arange(len(mean_ma)), mean_ma - std_ma, mean_ma + std_ma,
                     alpha=0.20, color=color, label=f"{alg_name} ±1σ", linewidth=0)

    # Min-Max Bereich (noch breiter, transparenter)
    ax.fill_between(np.arange(len(mean_ma)), min_ma, max_ma,
                     alpha=0.08, color=color, linewidth=0)

ax.set_xlabel("Episode", fontsize=14, fontweight="bold")
ax.set_ylabel("Return (Moving Avg, window=200)", fontsize=14, fontweight="bold")
ax.set_title("Learning Curves: Return über Episoden\n(mit ±Std und Min/Max Bereichen)",
             fontsize=15, fontweight="bold")
ax.legend(fontsize=10, loc="lower right", ncol=1)
ax.grid(alpha=0.4, linestyle="--")
ax.set_xlim(0, len(all_ma[0]))

# ===== EPISODE LENGTH PLOT =====
ax = axes[1]

for alg_name in ["MC", "SARSA", "Q-Learning"]:
    runs = results[alg_name]
    color = colors_dict[alg_name]

    # Berechne Moving Averages
    all_ma = np.array([moving_average(run["ep_lengths"], window=200) for run in runs])

    # Zeichne alle Seeds
    for i, ma in enumerate(all_ma):
        ax.plot(ma, alpha=0.3, color=color, linewidth=0.9, label=f"{alg_name} Seed {i}" if i == 0 else "")

    # Mittelwert und Streuung
    mean_ma = all_ma.mean(axis=0)
    std_ma = all_ma.std(axis=0)
    min_ma = all_ma.min(axis=0)
    max_ma = all_ma.max(axis=0)

    # Dicke Mittelwertlinie
    ax.plot(mean_ma, label=f"{alg_name} Mean", color=color, linewidth=3.5, zorder=10)

    # Streuungsbereiche
    ax.fill_between(np.arange(len(mean_ma)), mean_ma - std_ma, mean_ma + std_ma,
                     alpha=0.20, color=color, label=f"{alg_name} ±1σ", linewidth=0)
    ax.fill_between(np.arange(len(mean_ma)), min_ma, max_ma,
                     alpha=0.08, color=color, linewidth=0)

ax.set_xlabel("Episode", fontsize=14, fontweight="bold")
ax.set_ylabel("Episode Length (Moving Avg, window=200)", fontsize=14, fontweight="bold")
ax.set_title("Learning Curves: Episode Länge über Episoden\n(mit ±Std und Min/Max Bereichen)",
             fontsize=15, fontweight="bold")
ax.legend(fontsize=10, loc="upper right", ncol=1)
ax.grid(alpha=0.4, linestyle="--")
ax.set_xlim(0, len(all_ma[0]))

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "01_learning_curves_detailed.png")
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"✓ Saved: {output_path}")

# ============================================================================
# PLOT 2: VERBESSERTE GREEDY EVALUATION BAR CHARTS
# ============================================================================

print("Plot 2: Greedy Evaluation Bar Charts...", end=" ", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

algorithms = ["MC", "SARSA", "Q-Learning"]
colors = ["#d62728", "#1f77b4", "#2ca02c"]

# ===== RETURNS BAR CHART =====
ax = axes[0]
x_pos = np.arange(len(algorithms))
returns_means = [np.mean(greedy_results[alg]["returns"]) for alg in algorithms]
returns_stds = [np.std(greedy_results[alg]["returns"]) for alg in algorithms]
returns_mins = [np.min(greedy_results[alg]["returns"]) for alg in algorithms]
returns_maxs = [np.max(greedy_results[alg]["returns"]) for alg in algorithms]

# Zeichne Bars
bars = ax.bar(x_pos, returns_means, yerr=returns_stds, capsize=15,
              color=colors, alpha=0.8, edgecolor="black", linewidth=2.5)

# Zusätzliche Min-Max Linien für bessere Sichtbarkeit
for i, (mean, min_val, max_val) in enumerate(zip(returns_means, returns_mins, returns_maxs)):
    # Min-Max Spanne als dünne Linien
    ax.plot([i, i], [min_val, max_val], color=colors[i], linewidth=3,
            linestyle="--", alpha=0.6, zorder=5)
    # Punkte für Min/Max
    ax.plot(i, min_val, 'o', color=colors[i], markersize=8, markeredgecolor="black",
            markeredgewidth=1.5, zorder=6)
    ax.plot(i, max_val, 's', color=colors[i], markersize=8, markeredgecolor="black",
            markeredgewidth=1.5, zorder=6)

ax.set_ylabel("Mean Return (Greedy Evaluation)", fontsize=14, fontweight="bold")
ax.set_title("Greedy Evaluation: Mean Return (±Std, Min/Max) über 5 Seeds\n(300 Episodes pro Seed)",
             fontsize=15, fontweight="bold")
ax.set_xticks(x_pos)
ax.set_xticklabels(algorithms, fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.4, linestyle="--")
ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-", alpha=0.5)

# Werte auf Balken (mittlerer + unterer/oberer Wert)
for i, (mean, std, min_val, max_val) in enumerate(zip(returns_means, returns_stds, returns_mins, returns_maxs)):
    # Mean ± Std Text
    ax.text(i, mean + std + 3, f"{mean:.1f}±{std:.1f}", ha="center", va="bottom",
            fontsize=12, fontweight="bold")
    # Min-Max Text
    ax.text(i - 0.35, max_val + 1, f"max:{max_val:.1f}", ha="right", va="bottom",
            fontsize=10, style="italic", color=colors[i])
    ax.text(i - 0.35, min_val - 1, f"min:{min_val:.1f}", ha="right", va="top",
            fontsize=10, style="italic", color=colors[i])

ax.set_ylim(min(returns_mins) - 20, max(returns_maxs) + 30)

# ===== EPISODE LENGTH BAR CHART =====
ax = axes[1]
lengths_means = [np.mean(greedy_results[alg]["lengths"]) for alg in algorithms]
lengths_stds = [np.std(greedy_results[alg]["lengths"]) for alg in algorithms]
lengths_mins = [np.min(greedy_results[alg]["lengths"]) for alg in algorithms]
lengths_maxs = [np.max(greedy_results[alg]["lengths"]) for alg in algorithms]

bars = ax.bar(x_pos, lengths_means, yerr=lengths_stds, capsize=15,
              color=colors, alpha=0.8, edgecolor="black", linewidth=2.5)

# Min-Max Linien
for i, (mean, min_val, max_val) in enumerate(zip(lengths_means, lengths_mins, lengths_maxs)):
    ax.plot([i, i], [min_val, max_val], color=colors[i], linewidth=3,
            linestyle="--", alpha=0.6, zorder=5)
    ax.plot(i, min_val, 'o', color=colors[i], markersize=8, markeredgecolor="black",
            markeredgewidth=1.5, zorder=6)
    ax.plot(i, max_val, 's', color=colors[i], markersize=8, markeredgecolor="black",
            markeredgewidth=1.5, zorder=6)

ax.set_ylabel("Mean Episode Length (Greedy Evaluation)", fontsize=14, fontweight="bold")
ax.set_title("Greedy Evaluation: Mean Episode Length (±Std, Min/Max) über 5 Seeds\n(300 Episodes pro Seed)",
             fontsize=15, fontweight="bold")
ax.set_xticks(x_pos)
ax.set_xticklabels(algorithms, fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.4, linestyle="--")

# Werte auf Balken
for i, (mean, std, min_val, max_val) in enumerate(zip(lengths_means, lengths_stds, lengths_mins, lengths_maxs)):
    ax.text(i, mean + std + 0.5, f"{mean:.1f}±{std:.1f}", ha="center", va="bottom",
            fontsize=12, fontweight="bold")
    ax.text(i - 0.35, max_val + 0.3, f"max:{max_val:.1f}", ha="right", va="bottom",
            fontsize=10, style="italic", color=colors[i])
    ax.text(i - 0.35, min_val - 0.3, f"min:{min_val:.1f}", ha="right", va="top",
            fontsize=10, style="italic", color=colors[i])

ax.set_ylim(min(lengths_mins) - 5, max(lengths_maxs) + 10)

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "02_greedy_evaluation_detailed.png")
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"✓ Saved: {output_path}")

# ============================================================================
# ZUSAMMENFASSUNG
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY: GREEDY EVALUATION (Mean ± Std über 5 Seeds)")
print("=" * 80)

for alg_name in ["MC", "SARSA", "Q-Learning"]:
    returns = np.array(greedy_results[alg_name]["returns"])
    lengths = np.array(greedy_results[alg_name]["lengths"])
    print(f"\n{alg_name}:")
    print(f"  Return:  {returns.mean():7.2f} ± {returns.std():6.2f}  (min: {returns.min():7.2f}, max: {returns.max():7.2f})")
    print(f"  Length:  {lengths.mean():7.2f} ± {lengths.std():6.2f}  (min: {lengths.min():7.2f}, max: {lengths.max():7.2f})")

print("\n" + "=" * 80)
print("✓ VERBESSERTE VISUALISIERUNGEN ERFOLGREICH GENERIERT!")
print("=" * 80)
print(f"\nDateien:")
print(f"  1. 01_learning_curves_detailed.png (hohe Auflösung, 200 DPI)")
print(f"  2. 02_greedy_evaluation_detailed.png (hohe Auflösung, 200 DPI)")
print(f"\nVerbesserungen:")
print(f"  ✓ Größere Fonts (14pt Achsenlabels, 15pt Titel)")
print(f"  ✓ Deutlichere Streuung: ±1σ + Min/Max Bereiche")
print(f"  ✓ Alle 5 Seeds einzeln sichtbar (dünne Linien)")
print(f"  ✓ Dicke Mittelwertlinie für Klarheit")
print(f"  ✓ Hohe Auflösung (200 DPI)")
print(f"  ✓ Min/Max Werte auf Balken angezeigt")
