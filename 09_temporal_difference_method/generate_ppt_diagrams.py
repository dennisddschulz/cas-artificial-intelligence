#!/usr/bin/env python3
"""
Erstelle visuelle Diagramme für PRIO 1 & PRIO 2 Präsentation
- SARSA vs Q-Learning Learning Curves
- Monte Carlo vs TD Vergleich
- TD Error Vergleich
- Policy Vergleich
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import gymnasium as gym

# ============================================================================
# KONFIGURATION
# ============================================================================

OUTPUT_DIR = "/home/isc-den/cas-artificial-intelligence/09_temporal_difference_method"

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
# TRAINING FUNKTIONEN
# ============================================================================

def train_sarsa(num_episodes=10_000, alpha=0.1, gamma=0.99, seed=42):
    """SARSA Training mit TD Error Logging"""
    env = make_env(seed)
    n_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))
    rng = np.random.default_rng(seed)

    ep_returns = []
    ep_lengths = []
    td_errors = []

    for ep in range(num_episodes):
        frac = min(1.0, ep / max(1, 7500))
        eps = 1.0 + frac * (0.05 - 1.0)

        reset_out = env.reset()
        s = reset_out[0]
        a = epsilon_greedy_action(Q, s, n_actions, abs(eps), rng)

        done = False
        total_r = 0.0
        steps = 0

        while not done:
            s2, r, terminated, truncated, _info = env.step(a)
            done = terminated or truncated

            a2 = epsilon_greedy_action(Q, s2, n_actions, abs(eps), rng) if not done else 0
            td_target = r + (0.0 if done else gamma * Q[s2][a2])
            td_error = float(td_target - Q[s][a])
            Q[s][a] += alpha * td_error

            td_errors.append(abs(td_error))
            total_r += r
            steps += 1
            s, a = s2, a2

        ep_returns.append(total_r)
        ep_lengths.append(steps)

        if (ep + 1) % 2500 == 0:
            print(f"  SARSA Episode {ep+1}/{num_episodes}")

    env.close()
    return {
        "name": "SARSA",
        "ep_returns": np.array(ep_returns, dtype=np.float32),
        "ep_lengths": np.array(ep_lengths, dtype=np.int32),
        "td_errors": np.array(td_errors, dtype=np.float32),
    }

def train_q_learning(num_episodes=10_000, alpha=0.1, gamma=0.99, seed=42):
    """Q-Learning Training mit TD Error Logging"""
    env = make_env(seed)
    n_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))
    rng = np.random.default_rng(seed)

    ep_returns = []
    ep_lengths = []
    td_errors = []

    for ep in range(num_episodes):
        frac = min(1.0, ep / max(1, 7500))
        eps = 1.0 + frac * (0.05 - 1.0)

        reset_out = env.reset()
        s = reset_out[0]

        done = False
        total_r = 0.0
        steps = 0

        while not done:
            a = epsilon_greedy_action(Q, s, n_actions, abs(eps), rng)
            s2, r, terminated, truncated, _info = env.step(a)
            done = terminated or truncated

            best_next = 0.0 if done else np.max(Q[s2])
            td_target = r + gamma * best_next
            td_error = float(td_target - Q[s][a])
            Q[s][a] += alpha * td_error

            td_errors.append(abs(td_error))
            total_r += r
            steps += 1
            s = s2

        ep_returns.append(total_r)
        ep_lengths.append(steps)

        if (ep + 1) % 2500 == 0:
            print(f"  Q-Learning Episode {ep+1}/{num_episodes}")

    env.close()
    return {
        "name": "Q-Learning",
        "ep_returns": np.array(ep_returns, dtype=np.float32),
        "ep_lengths": np.array(ep_lengths, dtype=np.int32),
        "td_errors": np.array(td_errors, dtype=np.float32),
    }

def train_mc(num_episodes=10_000, gamma=0.99, seed=42):
    """Monte Carlo Training"""
    env = make_env(seed)
    n_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    rng = np.random.default_rng(seed)

    ep_returns = []
    ep_lengths = []

    for ep in range(num_episodes):
        frac = min(1.0, ep / max(1, 7500))
        eps = 1.0 + frac * (0.05 - 1.0)

        reset_out = env.reset()
        s = reset_out[0]
        trajectory = []

        done = False
        total_r = 0.0
        steps = 0

        while not done:
            a = epsilon_greedy_action(Q, s, n_actions, abs(eps), rng)
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

        if (ep + 1) % 2500 == 0:
            print(f"  MC Episode {ep+1}/{num_episodes}")

    env.close()
    return {
        "name": "MC",
        "ep_returns": np.array(ep_returns, dtype=np.float32),
        "ep_lengths": np.array(ep_lengths, dtype=np.int32),
    }

# ============================================================================
# TRAINIEREN
# ============================================================================

print("=" * 80)
print("TRAINIERE ALGORITHMEN FÜR VISUALISIERUNG")
print("=" * 80)

print("\nTrainiere SARSA...")
sarsa_result = train_sarsa(num_episodes=10_000, seed=42)

print("\nTrainiere Q-Learning...")
qlearn_result = train_q_learning(num_episodes=10_000, seed=42)

print("\nTrainiere Monte Carlo...")
mc_result = train_mc(num_episodes=10_000, seed=42)

# ============================================================================
# PLOT 1: SARSA vs Q-Learning (Prio 1)
# ============================================================================

print("\nGeneriere Visualisierungen...")
print("  Plot 1: SARSA vs Q-Learning...", end=" ", flush=True)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("PRIO 1: SARSA (On-Policy) vs Q-Learning (Off-Policy)",
             fontsize=18, fontweight="bold", y=0.995)

# ===== Returns =====
ax = axes[0, 0]
sarsa_ma = moving_average(sarsa_result["ep_returns"], window=200)
qlearn_ma = moving_average(qlearn_result["ep_returns"], window=200)

ax.plot(sarsa_ma, label="SARSA (On-Policy)", color="#1f77b4", linewidth=2.5, alpha=0.8)
ax.plot(qlearn_ma, label="Q-Learning (Off-Policy)", color="#d62728", linewidth=2.5, alpha=0.8)
ax.fill_between(np.arange(len(sarsa_ma)), sarsa_ma, alpha=0.2, color="#1f77b4")
ax.fill_between(np.arange(len(qlearn_ma)), qlearn_ma, alpha=0.2, color="#d62728")
ax.set_xlabel("Episode", fontsize=12, fontweight="bold")
ax.set_ylabel("Return (MA window=200)", fontsize=12, fontweight="bold")
ax.set_title("Learning Curves: Return", fontsize=13, fontweight="bold")
ax.legend(fontsize=11, loc="lower right")
ax.grid(alpha=0.3)

# ===== Episode Length =====
ax = axes[0, 1]
sarsa_len = moving_average(sarsa_result["ep_lengths"], window=200)
qlearn_len = moving_average(qlearn_result["ep_lengths"], window=200)

ax.plot(sarsa_len, label="SARSA (On-Policy)", color="#1f77b4", linewidth=2.5, alpha=0.8)
ax.plot(qlearn_len, label="Q-Learning (Off-Policy)", color="#d62728", linewidth=2.5, alpha=0.8)
ax.fill_between(np.arange(len(sarsa_len)), sarsa_len, alpha=0.2, color="#1f77b4")
ax.fill_between(np.arange(len(qlearn_len)), qlearn_len, alpha=0.2, color="#d62728")
ax.set_xlabel("Episode", fontsize=12, fontweight="bold")
ax.set_ylabel("Episode Length (MA window=200)", fontsize=12, fontweight="bold")
ax.set_title("Effizienz: Episode Length", fontsize=13, fontweight="bold")
ax.legend(fontsize=11, loc="upper right")
ax.grid(alpha=0.3)

# ===== TD Error Distribution =====
ax = axes[0, 2]
ax.hist(sarsa_result["td_errors"], bins=50, alpha=0.6, label="SARSA", color="#1f77b4", edgecolor="black")
ax.hist(qlearn_result["td_errors"], bins=50, alpha=0.6, label="Q-Learning", color="#d62728", edgecolor="black")
ax.set_xlabel("TD Error (absolute value)", fontsize=12, fontweight="bold")
ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
ax.set_title("TD Error Distribution", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3, axis="y")
ax.set_yscale("log")

# ===== Early vs Late Learning =====
ax = axes[1, 0]
early_episodes = 2500
late_episodes_start = 7500

sarsa_early = np.mean(sarsa_result["ep_returns"][:early_episodes])
sarsa_late = np.mean(sarsa_result["ep_returns"][late_episodes_start:])
qlearn_early = np.mean(qlearn_result["ep_returns"][:early_episodes])
qlearn_late = np.mean(qlearn_result["ep_returns"][late_episodes_start:])

x = np.arange(2)
width = 0.35
bars1 = ax.bar(x - width/2, [sarsa_early, sarsa_late], width, label="SARSA",
               color="#1f77b4", alpha=0.7, edgecolor="black", linewidth=1.5)
bars2 = ax.bar(x + width/2, [qlearn_early, qlearn_late], width, label="Q-Learning",
               color="#d62728", alpha=0.7, edgecolor="black", linewidth=1.5)

ax.set_ylabel("Mean Return", fontsize=12, fontweight="bold")
ax.set_title("Early vs Late Learning", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["Early (0-2.5k)", "Late (7.5k-10k)"], fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
ax.axhline(y=0, color="black", linewidth=0.5)

for bar in bars1 + bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f"{height:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

# ===== Convergence Speed (First to reach threshold) =====
ax = axes[1, 1]
threshold = -5
episodes_to_threshold_sarsa = np.where(moving_average(sarsa_result["ep_returns"], window=100) > threshold)[0]
episodes_to_threshold_qlearn = np.where(moving_average(qlearn_result["ep_returns"], window=100) > threshold)[0]

sarsa_threshold = episodes_to_threshold_sarsa[0] if len(episodes_to_threshold_sarsa) > 0 else 10000
qlearn_threshold = episodes_to_threshold_qlearn[0] if len(episodes_to_threshold_qlearn) > 0 else 10000

bars = ax.bar(["SARSA", "Q-Learning"], [sarsa_threshold, qlearn_threshold],
              color=["#1f77b4", "#d62728"], alpha=0.7, edgecolor="black", linewidth=1.5, width=0.5)
ax.set_ylabel("Episodes to Reach Return = -5", fontsize=12, fontweight="bold")
ax.set_title("Convergence Speed (Return > -5)", fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f"{int(height)}", ha="center", va="bottom", fontsize=11, fontweight="bold")

# ===== Target Bestimmung Illustration =====
ax = axes[1, 2]
ax.axis("off")

# Text für Target Illustration
target_text = """
TARGET BESTIMMUNG (Der Kern!)

SARSA (On-Policy):
  TARGET = r + γ·Q(s', a')
  where a' = ACTUAL next action
  ➜ Konservativ

Q-Learning (Off-Policy):
  TARGET = r + γ·max Q(s', ·)
  where · = BEST action
  ➜ Aggressiv


Beispiel:
Q(s'): [0.3, 0.8, 0.0, 0.1]
r = -1, γ = 0.99

SARSA (a'=index 1):
  -1 + 0.99 × 0.8 = -0.208

Q-Learning (best=index 1):
  -1 + 0.99 × 0.8 = -0.208

(Unterschied bei anderen Q-Werten!)
"""

ax.text(0.05, 0.95, target_text, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

plt.tight_layout()
output_path = f"{OUTPUT_DIR}/PRIO1_SARSA_vs_QLearning.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"✓ Saved: {output_path}")

# ============================================================================
# PLOT 2: Monte Carlo vs TD (Prio 2)
# ============================================================================

print("  Plot 2: Monte Carlo vs TD...", end=" ", flush=True)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("PRIO 2 (Optional): Monte Carlo vs Temporal Difference",
             fontsize=18, fontweight="bold", y=0.995)

# ===== MC vs TD Returns =====
ax = axes[0, 0]
mc_ma = moving_average(mc_result["ep_returns"], window=200)
td_ma = moving_average(qlearn_result["ep_returns"], window=200)  # Use Q-Learning as TD example

ax.plot(mc_ma, label="Monte Carlo", color="#2ca02c", linewidth=2.5, alpha=0.8)
ax.plot(td_ma, label="Temporal Difference (Q-Learning)", color="#ff7f0e", linewidth=2.5, alpha=0.8)
ax.fill_between(np.arange(len(mc_ma)), mc_ma, alpha=0.2, color="#2ca02c")
ax.fill_between(np.arange(len(td_ma)), td_ma, alpha=0.2, color="#ff7f0e")
ax.set_xlabel("Episode", fontsize=12, fontweight="bold")
ax.set_ylabel("Return (MA window=200)", fontsize=12, fontweight="bold")
ax.set_title("Learning Curves: MC vs TD", fontsize=13, fontweight="bold")
ax.legend(fontsize=11, loc="lower right")
ax.grid(alpha=0.3)

# ===== Episode Length Comparison =====
ax = axes[0, 1]
mc_len = moving_average(mc_result["ep_lengths"], window=200)
td_len = moving_average(qlearn_result["ep_lengths"], window=200)

ax.plot(mc_len, label="Monte Carlo", color="#2ca02c", linewidth=2.5, alpha=0.8)
ax.plot(td_len, label="Temporal Difference", color="#ff7f0e", linewidth=2.5, alpha=0.8)
ax.fill_between(np.arange(len(mc_len)), mc_len, alpha=0.2, color="#2ca02c")
ax.fill_between(np.arange(len(td_len)), td_len, alpha=0.2, color="#ff7f0e")
ax.set_xlabel("Episode", fontsize=12, fontweight="bold")
ax.set_ylabel("Episode Length (MA window=200)", fontsize=12, fontweight="bold")
ax.set_title("Sample Efficiency: MC vs TD", fontsize=13, fontweight="bold")
ax.legend(fontsize=11, loc="upper right")
ax.grid(alpha=0.3)

# ===== Update Strategy Illustration =====
ax = axes[0, 2]
ax.axis("off")

update_text = """
UPDATE STRATEGY

Monte Carlo (Batch):
  • Warte bis Episode endet
  • Berechne G = Σ γ^t r_t
  • Q(s,a) += α(G - Q(s,a))
  • Update: 1 mal pro Episode
  
  Vor: Unbiased
  Nach: High Variance
  Gut für: Policy Evaluation

Temporal Difference (Online):
  • Update nach JEDEM Schritt
  • Q(s,a) += α(r + γV(s') - Q(s,a))
  • Update: viele mal pro Episode
  
  Vor: Low Variance
  Nach: Biased
  Gut für: Schnelles Lernen
"""

ax.text(0.05, 0.95, update_text, transform=ax.transAxes, fontsize=9.5,
        verticalalignment="top", family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))

# ===== Convergence Comparison =====
ax = axes[1, 0]
threshold = -5
episodes_to_threshold_mc = np.where(moving_average(mc_result["ep_returns"], window=100) > threshold)[0]
episodes_to_threshold_td = np.where(moving_average(qlearn_result["ep_returns"], window=100) > threshold)[0]

mc_threshold = episodes_to_threshold_mc[0] if len(episodes_to_threshold_mc) > 0 else 10000
td_threshold = episodes_to_threshold_td[0] if len(episodes_to_threshold_td) > 0 else 10000

bars = ax.bar(["Monte Carlo", "Temporal Difference"], [mc_threshold, td_threshold],
              color=["#2ca02c", "#ff7f0e"], alpha=0.7, edgecolor="black", linewidth=1.5, width=0.5)
ax.set_ylabel("Episodes to Reach Return = -5", fontsize=12, fontweight="bold")
ax.set_title("Convergence Speed: MC vs TD", fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f"{int(height)}", ha="center", va="bottom", fontsize=11, fontweight="bold")

# ===== Sample Efficiency =====
ax = axes[1, 1]
total_steps_mc = np.sum(mc_result["ep_lengths"][:5000])  # First 5000 episodes
total_steps_td = np.sum(qlearn_result["ep_lengths"][:5000])

bars = ax.bar(["MC", "TD"], [total_steps_mc, total_steps_td],
              color=["#2ca02c", "#ff7f0e"], alpha=0.7, edgecolor="black", linewidth=1.5, width=0.4)
ax.set_ylabel("Total Steps (first 5000 episodes)", fontsize=12, fontweight="bold")
ax.set_title("Sample Efficiency Comparison", fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f"{int(height):,}", ha="center", va="bottom", fontsize=11, fontweight="bold")

# ===== Learning Characteristics Table =====
ax = axes[1, 2]
ax.axis("off")

comparison_text = """
CHARAKTERISTIKEN

           MC          TD
────────────────────────────
Update:    Episode     Step
Timing:    Batch       Online
Variance:  Hoch        Niedrig
Bias:      Nein        Ja
Speed:     Langsam     Schnell
Samples:   Ineffizient Effizient
────────────────────────────

Praktisch:
MC  = Theory, Evaluation
TD  = Praxis, Real-time
"""

ax.text(0.05, 0.95, comparison_text, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5))

plt.tight_layout()
output_path = f"{OUTPUT_DIR}/PRIO2_MC_vs_TD.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"✓ Saved: {output_path}")

# ============================================================================
# PLOT 3: Alle 4 Algorithmen Synthesevergleich
# ============================================================================

print("  Plot 3: Synthesis Vergleich...", end=" ", flush=True)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Synthesevergleich: Alle 4 Algorithmen",
             fontsize=18, fontweight="bold", y=0.995)

# ===== Alle Returns =====
ax = axes[0, 0]
ax.plot(moving_average(sarsa_result["ep_returns"], window=200),
        label="SARSA (On-Policy TD)", color="#1f77b4", linewidth=2.5, alpha=0.8)
ax.plot(moving_average(qlearn_result["ep_returns"], window=200),
        label="Q-Learning (Off-Policy TD)", color="#d62728", linewidth=2.5, alpha=0.8)
ax.plot(moving_average(mc_result["ep_returns"], window=200),
        label="Monte Carlo", color="#2ca02c", linewidth=2.5, alpha=0.8)

ax.set_xlabel("Episode", fontsize=12, fontweight="bold")
ax.set_ylabel("Return (MA window=200)", fontsize=12, fontweight="bold")
ax.set_title("Learning Curves: Alle 4 Algorithmen", fontsize=13, fontweight="bold")
ax.legend(fontsize=11, loc="lower right")
ax.grid(alpha=0.3)

# ===== Final Performance Comparison =====
ax = axes[0, 1]
final_returns = [
    np.mean(sarsa_result["ep_returns"][-1000:]),
    np.mean(qlearn_result["ep_returns"][-1000:]),
    np.mean(mc_result["ep_returns"][-1000:]),
]
algorithms = ["SARSA\n(On-Policy TD)", "Q-Learning\n(Off-Policy TD)", "Monte Carlo"]
colors = ["#1f77b4", "#d62728", "#2ca02c"]

bars = ax.bar(algorithms, final_returns, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5, width=0.6)
ax.set_ylabel("Mean Return (last 1000 episodes)", fontsize=12, fontweight="bold")
ax.set_title("Final Performance", fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f"{height:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

# ===== Convergence Speed =====
ax = axes[1, 0]
threshold = -3
episodes_to = []
for result in [sarsa_result, qlearn_result, mc_result]:
    ma = moving_average(result["ep_returns"], window=100)
    idx = np.where(ma > threshold)[0]
    episodes_to.append(idx[0] if len(idx) > 0 else 10000)

bars = ax.bar(algorithms, episodes_to, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5, width=0.6)
ax.set_ylabel("Episodes to Reach Return = -3", fontsize=12, fontweight="bold")
ax.set_title("Convergence Speed", fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f"{int(height)}", ha="center", va="bottom", fontsize=11, fontweight="bold")

# ===== Vergleichstabelle =====
ax = axes[1, 1]
ax.axis("off")

table_text = """
VERGLEICHSTABELLE

┌──────────────────────────────────────────────────────┐
│ Dimension          │ SARSA  │ Q-Learn│ MC    │ TD    │
├──────────────────────────────────────────────────────┤
│ Policy Type        │ On     │ Off    │ -     │ -     │
│ Update Timing      │ Step   │ Step   │ Epis  │ Step  │
│ Lernart            │ Online │ Online │ Batch │ Online│
│ TD-Fehler Größe    │ Klein  │ Groß   │ -     │ -     │
│ Stabilität         │ Hoch   │ Mittel │ Mittel│ Hoch  │
│ Konvergenz Speed   │ Mittel │ Schnell│ Lang  │ Schnell
│ Overoptimism       │ Nein   │ Ja     │ Nein  │ Nein  │
└──────────────────────────────────────────────────────┘

KEY DIFFERENCES:

On-Policy (SARSA):
  • Lernt unter AKTUELLER Policy
  • Konservativ, sicher
  • Kleinere TD-Fehler

Off-Policy (Q-Learning):
  • Lernt beste mögl. Policy
  • Aggressiv, optimal
  • Größere TD-Fehler

Monte Carlo vs TD:
  • MC: Episode-weise, unbiased
  • TD: Step-weise, effizient
"""

ax.text(0.02, 0.98, table_text, transform=ax.transAxes, fontsize=8.5,
        verticalalignment="top", family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.3))

plt.tight_layout()
output_path = f"{OUTPUT_DIR}/Synthesis_All_Algorithms.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"✓ Saved: {output_path}")

print("\n" + "=" * 80)
print("✅ ALLE VISUALISIERUNGEN GENERIERT!")
print("=" * 80)
print(f"\nGenerierte Dateien:")
print(f"  1. PRIO1_SARSA_vs_QLearning.png (PRIO 1: On-Policy vs Off-Policy)")
print(f"  2. PRIO2_MC_vs_TD.png (PRIO 2: Monte Carlo vs TD)")
print(f"  3. Synthesis_All_Algorithms.png (Vergleich aller 4)")
print(f"\n✨ Diese können jetzt in der PowerPoint eingebettet werden!")
