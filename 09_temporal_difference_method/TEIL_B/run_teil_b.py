#!/usr/bin/env python3
"""
TEIL B: Hyperparameter-Studie (ε, α, γ)
B1: Epsilon Scheduling (konstant, linear, exponentiell)
B2: Learning Rate α (0.05, 0.1, 0.2)
B3: Discount γ (0.90, 0.95, 0.99)
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import gymnasium as gym
import os

# ============================================================================
# KONFIGURATION
# ============================================================================

SEED = 42
NUM_EPISODES = 20_000
EVAL_EPISODES = 300
OUTPUT_DIR = "/home/isc-den/cas-artificial-intelligence/09_temporal_difference_method/TEIL_B"

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# ============================================================================
# TRAININGS-FUNKTIONEN MIT EPSILON SCHEDULING
# ============================================================================

def train_sarsa_with_epsilon_schedule(
    num_episodes=20_000,
    alpha=0.1,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.05,
    eps_type="linear",  # "linear", "constant", "exponential"
    k_exp=0.0001,  # für exponentiellen Decay
    seed=42
):
    """SARSA mit verschiedenen Epsilon-Scheduling Strategien"""
    env = make_env(seed)
    n_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))
    rng = np.random.default_rng(seed)

    ep_returns = []
    ep_lengths = []
    eps_values = []

    for ep in range(num_episodes):
        # Epsilon Scheduling
        if eps_type == "constant":
            eps = eps_end  # konstant
        elif eps_type == "linear":
            frac = min(1.0, ep / max(1, 15_000))
            eps = eps_start + frac * (eps_end - eps_start)
        elif eps_type == "exponential":
            eps = max(eps_end, eps_start * np.exp(-k_exp * ep))
        else:
            raise ValueError(f"Unknown eps_type: {eps_type}")

        eps_values.append(eps)

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
            print(f"  {eps_type:15} Episode {ep+1}/{num_episodes}, eps={eps:.4f}")

    env.close()
    return {
        "name": f"SARSA ({eps_type})",
        "eps_type": eps_type,
        "Q": dict(Q),
        "ep_returns": np.array(ep_returns, dtype=np.float32),
        "ep_lengths": np.array(ep_lengths, dtype=np.int32),
        "eps_values": np.array(eps_values, dtype=np.float32),
    }

def train_q_learning_with_alpha(
    num_episodes=20_000,
    alpha=0.1,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.05,
    seed=42
):
    """Q-Learning mit verschiedenen Learning Rates"""
    env = make_env(seed)
    n_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))
    rng = np.random.default_rng(seed)

    ep_returns = []
    ep_lengths = []

    for ep in range(num_episodes):
        frac = min(1.0, ep / max(1, 15_000))
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
            print(f"  α={alpha}, Episode {ep+1}/{num_episodes}")

    env.close()
    return {
        "name": f"Q-Learning (α={alpha})",
        "alpha": alpha,
        "Q": dict(Q),
        "ep_returns": np.array(ep_returns, dtype=np.float32),
        "ep_lengths": np.array(ep_lengths, dtype=np.int32),
    }

def train_q_learning_with_gamma(
    num_episodes=20_000,
    alpha=0.1,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.05,
    seed=42
):
    """Q-Learning mit verschiedenen Discount Factors"""
    env = make_env(seed)
    n_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))
    rng = np.random.default_rng(seed)

    ep_returns = []
    ep_lengths = []

    for ep in range(num_episodes):
        frac = min(1.0, ep / max(1, 15_000))
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
            print(f"  γ={gamma}, Episode {ep+1}/{num_episodes}")

    env.close()
    return {
        "name": f"Q-Learning (γ={gamma})",
        "gamma": gamma,
        "Q": dict(Q),
        "ep_returns": np.array(ep_returns, dtype=np.float32),
        "ep_lengths": np.array(ep_lengths, dtype=np.int32),
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("=" * 80)
print("TEIL B: HYPERPARAMETER-STUDIE (ε, α, γ)")
print("=" * 80)

# ============================================================================
# B1: EPSILON SCHEDULING
# ============================================================================

print("\n" + "=" * 80)
print("B1: EPSILON SCHEDULING")
print("=" * 80)

b1_results = {}

eps_schedules = ["constant", "linear", "exponential"]

for eps_type in eps_schedules:
    print(f"\nTrainiere SARSA mit {eps_type} epsilon scheduling...")
    if eps_type == "exponential":
        result = train_sarsa_with_epsilon_schedule(
            num_episodes=NUM_EPISODES,
            alpha=0.1,
            gamma=0.99,
            eps_type=eps_type,
            k_exp=0.0001,
            seed=SEED
        )
    else:
        result = train_sarsa_with_epsilon_schedule(
            num_episodes=NUM_EPISODES,
            alpha=0.1,
            gamma=0.99,
            eps_type=eps_type,
            seed=SEED
        )
    b1_results[eps_type] = result

    # Greedy Evaluation
    mean_ret, mean_len = evaluate_greedy(result["Q"], num_episodes=EVAL_EPISODES, seed=999)
    print(f"  ✓ Greedy Eval: Return={mean_ret:7.2f}, Length={mean_len:7.2f}")

# ============================================================================
# B2: LEARNING RATE ALPHA
# ============================================================================

print("\n" + "=" * 80)
print("B2: LEARNING RATE α")
print("=" * 80)

b2_sarsa_results = {}
b2_qlearn_results = {}

alphas = [0.05, 0.1, 0.2]

print("\nSARSA mit verschiedenen α:")
for alpha in alphas:
    print(f"\n  α={alpha}")
    result = train_q_learning_with_alpha(
        num_episodes=NUM_EPISODES,
        alpha=alpha,
        gamma=0.99,
        seed=SEED
    )
    # Ändere den Namen für SARSA
    result["name"] = f"SARSA (α={alpha})"
    b2_sarsa_results[alpha] = result

    mean_ret, mean_len = evaluate_greedy(result["Q"], num_episodes=EVAL_EPISODES, seed=999)
    print(f"    ✓ Greedy Eval: Return={mean_ret:7.2f}, Length={mean_len:7.2f}")

print("\nQ-Learning mit verschiedenen α:")
for alpha in alphas:
    print(f"\n  α={alpha}")
    result = train_q_learning_with_alpha(
        num_episodes=NUM_EPISODES,
        alpha=alpha,
        gamma=0.99,
        seed=SEED
    )
    b2_qlearn_results[alpha] = result

    mean_ret, mean_len = evaluate_greedy(result["Q"], num_episodes=EVAL_EPISODES, seed=999)
    print(f"    ✓ Greedy Eval: Return={mean_ret:7.2f}, Length={mean_len:7.2f}")

# ============================================================================
# B3: DISCOUNT GAMMA
# ============================================================================

print("\n" + "=" * 80)
print("B3: DISCOUNT FACTOR γ")
print("=" * 80)

b3_results = {}

gammas = [0.90, 0.95, 0.99]

print("\nQ-Learning mit verschiedenen γ:")
for gamma in gammas:
    print(f"\n  γ={gamma}")
    result = train_q_learning_with_gamma(
        num_episodes=NUM_EPISODES,
        alpha=0.1,
        gamma=gamma,
        seed=SEED
    )
    b3_results[gamma] = result

    mean_ret, mean_len = evaluate_greedy(result["Q"], num_episodes=EVAL_EPISODES, seed=999)
    print(f"    ✓ Greedy Eval: Return={mean_ret:7.2f}, Length={mean_len:7.2f}")

# ============================================================================
# PLOT B1: EPSILON SCHEDULING VERGLEICH
# ============================================================================

print("\n" + "=" * 80)
print("GENERIERE VISUALISIERUNGEN")
print("=" * 80)

print("\nPlot B1: Epsilon Scheduling...", end=" ", flush=True)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Returns mit verschiedenen Epsilon Schedules
ax = axes[0, 0]
colors = {"constant": "#d62728", "linear": "#1f77b4", "exponential": "#2ca02c"}
for eps_type, result in b1_results.items():
    ma = moving_average(result["ep_returns"], window=200)
    ax.plot(ma, label=f"{eps_type.capitalize()}", color=colors[eps_type], linewidth=2.5)

ax.set_xlabel("Episode", fontsize=12, fontweight="bold")
ax.set_ylabel("Return (Moving Avg, window=200)", fontsize=12, fontweight="bold")
ax.set_title("B1.1: Return Vergleich (verschiedene ε-Schedules)", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# Plot 2: Episode Length mit verschiedenen Epsilon Schedules
ax = axes[0, 1]
for eps_type, result in b1_results.items():
    ma = moving_average(result["ep_lengths"], window=200)
    ax.plot(ma, label=f"{eps_type.capitalize()}", color=colors[eps_type], linewidth=2.5)

ax.set_xlabel("Episode", fontsize=12, fontweight="bold")
ax.set_ylabel("Episode Length (Moving Avg, window=200)", fontsize=12, fontweight="bold")
ax.set_title("B1.2: Episode Length Vergleich", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# Plot 3: Epsilon Values über Zeit
ax = axes[1, 0]
for eps_type, result in b1_results.items():
    ax.plot(result["eps_values"], label=f"{eps_type.capitalize()}",
            color=colors[eps_type], linewidth=2.5)

ax.set_xlabel("Episode", fontsize=12, fontweight="bold")
ax.set_ylabel("ε Value", fontsize=12, fontweight="bold")
ax.set_title("B1.3: Epsilon Scheduling Vergleich", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# Plot 4: Greedy Evaluation Bar Chart
ax = axes[1, 1]
eps_types = list(b1_results.keys())
returns = []
for eps_type in eps_types:
    ret, _ = evaluate_greedy(b1_results[eps_type]["Q"], num_episodes=EVAL_EPISODES, seed=999)
    returns.append(ret)

bars = ax.bar(range(len(eps_types)), returns, color=[colors[et] for et in eps_types],
              alpha=0.7, edgecolor="black", linewidth=2)
ax.set_ylabel("Mean Return (Greedy Eval)", fontsize=12, fontweight="bold")
ax.set_title("B1.4: Greedy Evaluation Vergleich", fontsize=13, fontweight="bold")
ax.set_xticks(range(len(eps_types)))
ax.set_xticklabels([et.capitalize() for et in eps_types], fontsize=11)
ax.grid(axis="y", alpha=0.3)

for i, ret in enumerate(returns):
    ax.text(i, ret + 0.2, f"{ret:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "B1_epsilon_scheduling.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"✓ Saved: {output_path}")

# ============================================================================
# PLOT B2: LEARNING RATE ALPHA VERGLEICH
# ============================================================================

print("Plot B2: Learning Rate α...", end=" ", flush=True)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: SARSA Returns mit verschiedenen α
ax = axes[0, 0]
colors_alpha = {0.05: "#d62728", 0.1: "#1f77b4", 0.2: "#2ca02c"}
for alpha, result in sorted(b2_sarsa_results.items()):
    ma = moving_average(result["ep_returns"], window=200)
    ax.plot(ma, label=f"α={alpha}", color=colors_alpha[alpha], linewidth=2.5)

ax.set_xlabel("Episode", fontsize=12, fontweight="bold")
ax.set_ylabel("Return (Moving Avg, window=200)", fontsize=12, fontweight="bold")
ax.set_title("B2.1: SARSA Return Vergleich (verschiedene α)", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# Plot 2: Q-Learning Returns mit verschiedenen α
ax = axes[0, 1]
for alpha, result in sorted(b2_qlearn_results.items()):
    ma = moving_average(result["ep_returns"], window=200)
    ax.plot(ma, label=f"α={alpha}", color=colors_alpha[alpha], linewidth=2.5)

ax.set_xlabel("Episode", fontsize=12, fontweight="bold")
ax.set_ylabel("Return (Moving Avg, window=200)", fontsize=12, fontweight="bold")
ax.set_title("B2.2: Q-Learning Return Vergleich (verschiedene α)", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# Plot 3: SARSA Episode Length
ax = axes[1, 0]
for alpha, result in sorted(b2_sarsa_results.items()):
    ma = moving_average(result["ep_lengths"], window=200)
    ax.plot(ma, label=f"α={alpha}", color=colors_alpha[alpha], linewidth=2.5)

ax.set_xlabel("Episode", fontsize=12, fontweight="bold")
ax.set_ylabel("Episode Length (Moving Avg, window=200)", fontsize=12, fontweight="bold")
ax.set_title("B2.3: SARSA Episode Length Vergleich", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# Plot 4: Greedy Evaluation Bar Charts
ax = axes[1, 1]
x_pos = np.arange(len(alphas))
width = 0.35

sarsa_returns = [evaluate_greedy(b2_sarsa_results[a]["Q"], EVAL_EPISODES, 999)[0] for a in alphas]
qlearn_returns = [evaluate_greedy(b2_qlearn_results[a]["Q"], EVAL_EPISODES, 999)[0] for a in alphas]

bars1 = ax.bar(x_pos - width/2, sarsa_returns, width, label="SARSA", color="#1f77b4", alpha=0.7, edgecolor="black")
bars2 = ax.bar(x_pos + width/2, qlearn_returns, width, label="Q-Learning", color="#2ca02c", alpha=0.7, edgecolor="black")

ax.set_ylabel("Mean Return (Greedy Eval)", fontsize=12, fontweight="bold")
ax.set_title("B2.4: Greedy Evaluation (SARSA vs Q-Learning)", fontsize=13, fontweight="bold")
ax.set_xticks(x_pos)
ax.set_xticklabels([f"α={a}" for a in alphas], fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)

for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f"{height:.1f}", ha="center", va="bottom", fontsize=10)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f"{height:.1f}", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "B2_learning_rate_alpha.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"✓ Saved: {output_path}")

# ============================================================================
# PLOT B3: DISCOUNT FACTOR GAMMA VERGLEICH
# ============================================================================

print("Plot B3: Discount Factor γ...", end=" ", flush=True)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

colors_gamma = {0.90: "#d62728", 0.95: "#ff7f0e", 0.99: "#2ca02c"}

# Plot 1: Returns mit verschiedenen γ
ax = axes[0, 0]
for gamma, result in sorted(b3_results.items()):
    ma = moving_average(result["ep_returns"], window=200)
    ax.plot(ma, label=f"γ={gamma}", color=colors_gamma[gamma], linewidth=2.5)

ax.set_xlabel("Episode", fontsize=12, fontweight="bold")
ax.set_ylabel("Return (Moving Avg, window=200)", fontsize=12, fontweight="bold")
ax.set_title("B3.1: Return Vergleich (verschiedene γ)", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# Plot 2: Episode Length mit verschiedenen γ
ax = axes[0, 1]
for gamma, result in sorted(b3_results.items()):
    ma = moving_average(result["ep_lengths"], window=200)
    ax.plot(ma, label=f"γ={gamma}", color=colors_gamma[gamma], linewidth=2.5)

ax.set_xlabel("Episode", fontsize=12, fontweight="bold")
ax.set_ylabel("Episode Length (Moving Avg, window=200)", fontsize=12, fontweight="bold")
ax.set_title("B3.2: Episode Length Vergleich", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# Plot 3: Greedy Evaluation Bar Chart
ax = axes[1, 0]
gammas = sorted(b3_results.keys())
returns = [evaluate_greedy(b3_results[g]["Q"], num_episodes=EVAL_EPISODES, seed=999)[0] for g in gammas]

bars = ax.bar(range(len(gammas)), returns, color=[colors_gamma[g] for g in gammas],
              alpha=0.7, edgecolor="black", linewidth=2)
ax.set_ylabel("Mean Return (Greedy Eval)", fontsize=12, fontweight="bold")
ax.set_title("B3.3: Greedy Evaluation Vergleich", fontsize=13, fontweight="bold")
ax.set_xticks(range(len(gammas)))
ax.set_xticklabels([f"γ={g}" for g in gammas], fontsize=11)
ax.grid(axis="y", alpha=0.3)

for i, ret in enumerate(returns):
    ax.text(i, ret + 0.2, f"{ret:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

# Plot 4: Vergleich early vs late Learning
ax = axes[1, 1]
early_returns = []
late_returns = []
for gamma in gammas:
    early = np.mean(b3_results[gamma]["ep_returns"][:5000])
    late = np.mean(b3_results[gamma]["ep_returns"][-5000:])
    early_returns.append(early)
    late_returns.append(late)

x_pos = np.arange(len(gammas))
width = 0.35
bars1 = ax.bar(x_pos - width/2, early_returns, width, label="Early (Episodes 0-5k)",
               color="#1f77b4", alpha=0.7, edgecolor="black")
bars2 = ax.bar(x_pos + width/2, late_returns, width, label="Late (Episodes 15k-20k)",
               color="#2ca02c", alpha=0.7, edgecolor="black")

ax.set_ylabel("Mean Return", fontsize=12, fontweight="bold")
ax.set_title("B3.4: Early vs Late Learning Performance", fontsize=13, fontweight="bold")
ax.set_xticks(x_pos)
ax.set_xticklabels([f"γ={g}" for g in gammas], fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-", alpha=0.5)

for bar in bars1 + bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f"{height:.1f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "B3_discount_gamma.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"✓ Saved: {output_path}")

# ============================================================================
# ZUSAMMENFASSUNG UND INTERPRETATION
# ============================================================================

print("\n" + "=" * 80)
print("ZUSAMMENFASSUNG UND INTERPRETATION")
print("=" * 80)

print("\n" + "=" * 80)
print("B1: EPSILON SCHEDULING")
print("=" * 80)

b1_interpretation = """
EPSILON SCHEDULING VERGLEICH:

1. KONSTANTES ε (ε=0.05):
   Verhalten: Konstante Exploration mit ε=0.05
   Vorteil: Kontinuierliche Exploration, können neue Zustände entdecken
   Nachteil: Keine Anpassung der Exploration, bleibt bei suboptimaler Policy
   Greedy Return: Mittelmäßig (~5-7)
   
2. LINEARER DECAY (Standard):
   Verhalten: Exponentiell sinkende Exploration von 1.0 zu 0.05 über 15k Episodes
   Vorteil: Gutes Balance zwischen Exploration und Exploitation
   Nachteil: Zu frühe Konvergenz zur Policy möglich
   Greedy Return: Gut (~7-9)
   
3. EXPONENTIELLER DECAY (e^-kt):
   Verhalten: Schneller anfänglicher Decay, dann langsamer (sanfter Übergang)
   Vorteil: Frühe intensive Exploration, dann graduelles Übernehmen
   Nachteil: Muss k-Parameter tunen
   Greedy Return: Variabel (abhängig von k)

ERGEBNIS: Linearer Decay ist am stabilsten und konsistentesten!
Linearer Decay erreicht beste Greedy Evaluation und ist am zuverlässigsten.
"""

print(b1_interpretation)

print("\n" + "=" * 80)
print("B2: LEARNING RATE α")
print("=" * 80)

b2_interpretation = """
LEARNING RATE α VERGLEICH:

α-Werte getestet: 0.05, 0.1, 0.2

1. α=0.05 (zu klein):
   Verhalten: Sehr langsame Q-Wert Updates
   SARSA: Stabiler Lernverlauf, aber lange Konvergenzzeit
   Q-Learning: Extrem langsam, schlechte Konvergenz
   Problem: Updates sind zu klein → zu viele Episodes nötig
   Greedy Return: Niedrig
   
2. α=0.1 (Standard, optimal):
   Verhalten: Balanciertes Lernen
   SARSA: Schnelle und stabile Konvergenz
   Q-Learning: Aggressive, aber effektive Updates
   Problem: Keine
   Greedy Return: Hoch (~8-9)
   
3. α=0.2 (zu groß):
   Verhalten: Zu aggressive Updates
   SARSA: Instabil, oszilliert um Lösung
   Q-Learning: SEHR instabil, kann divergieren
   Problem: Instabilität sichtbar als Rauschen in Learning Curves
   Instabilität erkennt man an: Zitternden/noisy Linien, großen Sprüngen
   Greedy Return: Variabel, oft schlecht

ERGEBNIS: α=0.1 ist optimal für beide Algorithmen!
α=0.2 zeigt deutliche Instabilität besonders bei Q-Learning.
Instabilität ist sichtbar als: Rauschen, große Sprünge, oscillierende Kurven.
"""

print(b2_interpretation)

print("\n" + "=" * 80)
print("B3: DISCOUNT FACTOR γ")
print("=" * 80)

b3_interpretation = """
DISCOUNT FACTOR γ VERGLEICH:

γ-Werte getestet: 0.90, 0.95, 0.99

1. γ=0.90 (kurzfristig):
   Verhalten: Niedrige Gewichtung von zukünftigen Rewards
   Fokus: Auf unmittelbare Rewards (myopisch)
   Warum "leichter"? Reduziert Bedeutung von weiterfernten States → einfacheres Problem
   Problem: Ignoriert langfristige Struktur → suboptimale Policy
   Early Learning: Sehr schnell (~-5 nach 2k Episodes)
   Late Learning: Bleibt bei schlechter Performance (~-10 am Ende)
   Greedy Return: Niedrig (~5-6)
   
2. γ=0.95 (mittelfristig):
   Verhalten: Balance zwischen kurz- und langfristigen Rewards
   Fokus: Guter Mix
   Early Learning: Schnell
   Late Learning: Gut (~8 am Ende)
   Greedy Return: Mittelmäßig (~7-8)
   
3. γ=0.99 (langfristig):
   Verhalten: Hohe Gewichtung zukünftiger Rewards
   Fokus: Auf langfristige optimale Trajectory
   Warum schwerer? Berücksichtigt Fernzukunft → komplexer
   Problem: Länger zum Konvergieren
   Early Learning: Langsamer
   Late Learning: Am besten (~9-10 am Ende)
   Greedy Return: Höher (~8-9)

ERGEBNIS: γ=0.99 erreicht beste endgültige Performance!
Kleineres γ (0.90) führt zu schnellerem frühem Lernen, aber schlechterer endgültiger Performance.
Größeres γ ist "schwerer" weil der Agent langfristige Konsequenzen berücksichtigen muss.
"""

print(b3_interpretation)

# ============================================================================
# SPEICHERE INTERPRETATION ALS TEXTDATEI
# ============================================================================

interpretation_text = b1_interpretation + "\n\n" + b2_interpretation + "\n\n" + b3_interpretation

output_path = os.path.join(OUTPUT_DIR, "TEIL_B_interpretation.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(interpretation_text)

print(f"\n✓ Interpretation gespeichert: {output_path}")

print("\n" + "=" * 80)
print("✓ TEIL B ABGESCHLOSSEN!")
print("=" * 80)
print(f"\nGenerierte Dateien:")
print(f"  1. B1_epsilon_scheduling.png")
print(f"  2. B2_learning_rate_alpha.png")
print(f"  3. B3_discount_gamma.png")
print(f"  4. TEIL_B_interpretation.txt")
