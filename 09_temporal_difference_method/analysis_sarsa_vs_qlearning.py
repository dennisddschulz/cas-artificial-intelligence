#!/usr/bin/env python3
"""
PRIO 1: ON-POLICY (SARSA) vs OFF-POLICY (Q-Learning) ANALYSE
Vergleich der Unterschiede beim Lernen mit Fokus auf Target Bestimmung und Verhalten
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

try:
    import gymnasium as gym
    _USE_GYMNASIUM = True
except ImportError:
    import gym
    _USE_GYMNASIUM = False


def make_env(seed: int = 0):
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
        dq.append(v); s += v
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
    """SARSA with detailed logging for analysis"""
    env = make_env(seed)
    n_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))
    rng = np.random.default_rng(seed)

    ep_returns = []
    ep_lengths = []
    td_errors_per_step = []
    q_value_ranges = []

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

            # SARSA: use actual next action from policy
            a2 = epsilon_greedy_action(Q, s2, n_actions, eps, rng) if not done else 0

            # TARGET uses a' (ACTUAL next action)
            td_target = r + (0.0 if done else gamma * Q[s2][a2])
            td_error = float(td_target - Q[s][a])
            Q[s][a] += alpha * td_error

            td_errors_per_step.append(abs(td_error))
            total_r += r
            steps += 1
            s, a = s2, a2

        ep_returns.append(total_r)
        ep_lengths.append(steps)

        # Log Q-value range every 500 episodes
        if (ep + 1) % 500 == 0:
            if Q:
                all_q = np.concatenate([Q[s] for s in Q.keys()])
                q_value_ranges.append({
                    'ep': ep + 1,
                    'min': float(np.min(all_q)),
                    'max': float(np.max(all_q)),
                    'mean': float(np.mean(all_q))
                })

        if (ep + 1) % 1000 == 0:
            print(f"  SARSA: Episode {ep + 1}/{num_episodes}")

    env.close()
    return {
        "name": "SARSA (On-Policy)",
        "Q": Q,
        "ep_returns": np.array(ep_returns, dtype=np.float32),
        "ep_lengths": np.array(ep_lengths, dtype=np.int32),
        "td_errors": np.array(td_errors_per_step, dtype=np.float32),
        "q_ranges": q_value_ranges,
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
    """Q-Learning with detailed logging for analysis"""
    env = make_env(seed)
    n_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))
    rng = np.random.default_rng(seed)

    ep_returns = []
    ep_lengths = []
    td_errors_per_step = []
    q_value_ranges = []

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

            # Q-LEARNING: use BEST next action (max)
            best_next = 0.0 if done else np.max(Q[s2])

            # TARGET uses a* = argmax (BEST action)
            td_target = r + gamma * best_next
            td_error = float(td_target - Q[s][a])
            Q[s][a] += alpha * td_error

            td_errors_per_step.append(abs(td_error))
            total_r += r
            steps += 1
            s = s2

        ep_returns.append(total_r)
        ep_lengths.append(steps)

        # Log Q-value range
        if (ep + 1) % 500 == 0:
            if Q:
                all_q = np.concatenate([Q[s] for s in Q.keys()])
                q_value_ranges.append({
                    'ep': ep + 1,
                    'min': float(np.min(all_q)),
                    'max': float(np.max(all_q)),
                    'mean': float(np.mean(all_q))
                })

        if (ep + 1) % 1000 == 0:
            print(f"  Q-Learning: Episode {ep + 1}/{num_episodes}")

    env.close()
    return {
        "name": "Q-Learning (Off-Policy)",
        "Q": Q,
        "ep_returns": np.array(ep_returns, dtype=np.float32),
        "ep_lengths": np.array(ep_lengths, dtype=np.int32),
        "td_errors": np.array(td_errors_per_step, dtype=np.float32),
        "q_ranges": q_value_ranges,
    }


def evaluate_greedy(Q, num_episodes=200, seed=123):
    env = make_env(seed)
    n_actions = env.action_space.n

    returns = []
    lengths = []

    for ep in range(num_episodes):
        reset_out = env.reset()
        s = reset_out[0] if _USE_GYMNASIUM else reset_out

        done = False
        total_r = 0.0
        steps = 0

        while not done:
            a = int(np.argmax(Q[s])) if s in Q else 0

            step_out = env.step(a)
            if _USE_GYMNASIUM:
                s2, r, terminated, truncated, _info = step_out
                done = terminated or truncated
            else:
                s2, r, done, _info = step_out

            total_r += r
            steps += 1
            s = s2

        returns.append(total_r)
        lengths.append(steps)

    env.close()
    return float(np.mean(returns)), float(np.mean(lengths))


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("\n" + "=" * 90)
print("PRIO 1: ON-POLICY (SARSA) vs OFF-POLICY (Q-Learning) - DETAILLIERTE ANALYSE")
print("=" * 90)
print()

print("THEORETISCHER HINTERGRUND:")
print("-" * 90)
print()
print("SARSA - State-Action-Reward-State-Action (On-Policy):")
print("  Formel: Q(s,a) += alpha * [r + gamma * Q(s',a') - Q(s,a)]")
print("  where a' = epsilon_greedy_policy(s')  ← ACTUAL next action from policy")
print()
print("  Intuition:")
print("    ✓ Lernt den Wert einer Action unter der AKTUELLEN explorierenden Policy")
print("    ✓ Konservativ: Berücksichtigt, dass nächste Aktion zufällig sein könnte")
print("    ✓ Weniger Overoptimism, aber auch langsameres Lernen")
print()
print()
print("Q-Learning - Off-Policy Temporal Difference:")
print("  Formel: Q(s,a) += alpha * [r + gamma * max Q(s',·) - Q(s,a)]")
print("  where a' = argmax Q(s',·)  ← BEST possible action, unabhängig von Policy")
print()
print("  Intuition:")
print("    ✓ Lernt den Wert einer Action unter der OPTIMALEN möglichen Policy")
print("    ✓ Aggressiv: Ignoriert Explorations-Risiken")
print("    ✓ Schneller Lernen, aber kann Werte überschätzen")
print()
print()

print("=" * 90)
print("TRAINING... (Dies dauert ca. 2-3 Minuten)")
print("=" * 90)
print()

sarsa_log = train_sarsa_with_logging(num_episodes=5_000, alpha=0.1, seed=42)
qlearn_log = train_q_learning_with_logging(num_episodes=5_000, alpha=0.1, seed=42)

runs_prio1 = [sarsa_log, qlearn_log]
print("✓ Training abgeschlossen!\n")

# ============================================================================
# VISUALISIERUNG 1: Vergleich der Lernkurven
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("PRIO 1: On-Policy (SARSA) vs Off-Policy (Q-Learning) Vergleich",
             fontsize=16, fontweight='bold')

# Return curves
ax = axes[0, 0]
for r in runs_prio1:
    y = moving_average(r["ep_returns"], window=100)
    ax.plot(y, label=r["name"], linewidth=2.5, alpha=0.8)
ax.set_xlabel("Episode", fontsize=11)
ax.set_ylabel("Return (moving avg, window=100)", fontsize=11)
ax.set_title("Lernkurven: Return pro Episode\n(Wie schnell verbessert sich der Agent?)", fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# Episode length curves
ax = axes[0, 1]
for r in runs_prio1:
    y = moving_average(r["ep_lengths"], window=100)
    ax.plot(y, label=r["name"], linewidth=2.5, alpha=0.8)
ax.set_xlabel("Episode", fontsize=11)
ax.set_ylabel("Episode Länge (moving avg, window=100)", fontsize=11)
ax.set_title("Episode Länge: Wie schnell erreicht der Agent das Ziel?\n(Weniger Schritte = effizienter)", fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# TD error magnitude distribution
ax = axes[1, 0]
ax.hist(np.abs(sarsa_log["td_errors"]), bins=100, alpha=0.6, label="SARSA", density=True, color='blue')
ax.hist(np.abs(qlearn_log["td_errors"]), bins=100, alpha=0.6, label="Q-Learning", density=True, color='orange')
ax.set_xlabel("|TD Error| (Absolute Wert)", fontsize=11)
ax.set_ylabel("Häufigkeit (normalisiert)", fontsize=11)
ax.set_title("TD Error Verteilung\n(Kleinere Fehler = stabileres Lernen)", fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xlim(0, 5)
ax.grid(alpha=0.3)

# Q-value ranges over training
ax = axes[1, 1]
for r in runs_prio1:
    qs = r["q_ranges"]
    if qs:
        eps = [q['ep'] for q in qs]
        maxes = [q['max'] for q in qs]
        ax.plot(eps, maxes, marker='o', label=f"{r['name']}", linewidth=2.5, markersize=6)
ax.set_xlabel("Episode", fontsize=11)
ax.set_ylabel("Max Q-Wert", fontsize=11)
ax.set_title("Q-Wert Evolution über Zeit\n(Optimismus des Algorithmus)", fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("01_SARSA_vs_QLearning_Comparison.png", dpi=150, bbox_inches='tight')
print("✓ Plot saved: 01_SARSA_vs_QLearning_Comparison.png\n")
plt.close()

# ============================================================================
# EVALUIERUNG: Greedy Policy Performance
# ============================================================================

print("=" * 90)
print("EVALUIERUNG: GREEDY POLICY PERFORMANCE (OHNE EXPLORATION)")
print("=" * 90)
print()

for r in runs_prio1:
    mean_ret, mean_len = evaluate_greedy(r["Q"], num_episodes=300, seed=999)
    print(f"{r['name']:>20} | Return: {mean_ret:8.2f} | Avg Steps: {mean_len:7.2f}")

print()
print("Interpretation:")
print("  • Höherer Return = besser gelernte Policy")
print("  • Weniger Schritte = effizienter (schneller zum Ziel)")
print()

# ============================================================================
# DETAILLIERTE ANALYSE: STEP-BY-STEP BEISPIEL
# ============================================================================

print("=" * 90)
print("DETAILLIERTES BEISPIEL: ZIEL DER TARGET BESTIMMUNG")
print("=" * 90)
print()

print("Szenario:")
print("-" * 90)
print("Aktueller Zustand:     s = 10")
print("Q-Werte s=10:          Q(10,UP)=0.2, Q(10,RIGHT)=-0.1, Q(10,DOWN)=0.5, Q(10,LEFT)=0.0")
print("Gewählte Action:       a = RIGHT (indx 1) [via epsilon-greedy mit eps=0.3]")
print("Transition:            (s=10, a=RIGHT) → (s'=15, r=-1)")
print("Q-Werte s'=15:         Q(15,UP)=0.3, Q(15,RIGHT)=0.8, Q(15,DOWN)=0.0, Q(15,LEFT)=0.1")
print()

s, a, r, s2, gamma = 10, 1, -1, 15, 0.99
Q_s = np.array([0.2, -0.1, 0.5, 0.0])
Q_s2 = np.array([0.3, 0.8, 0.0, 0.1])

print("\nSARSA (On-Policy):")
print("-" * 90)
print(f"  Target = r + gamma * Q(s', a')")
print(f"  Der Agent wählt a' via ε-greedy:")
print(f"    • Mit Prob 0.7: Greedy → best_action = {np.argmax(Q_s2)} (Q-value={Q_s2[np.argmax(Q_s2)]:.1f})")
print(f"    • Mit Prob 0.3: Random → one of [0,1,2,3]")
print(f"  Annahme: Greedy action wird gewählt → a' = {np.argmax(Q_s2)}")
print()
print(f"  Target = {r} + {gamma} * Q({s2}, {np.argmax(Q_s2)})")
print(f"         = {r} + {gamma} * {Q_s2[np.argmax(Q_s2)]:.1f}")
print(f"         = {r + gamma * Q_s2[np.argmax(Q_s2)]:.4f}")
print()
print(f"  TD Update: Q({s}, {a}) += 0.1 * ({r + gamma * Q_s2[np.argmax(Q_s2)]:.4f} - {Q_s[a]:.2f})")
print(f"           = {Q_s[a]:.2f} + 0.1 * ({r + gamma * Q_s2[np.argmax(Q_s2)] - Q_s[a]:.4f})")
print(f"           = {Q_s[a] + 0.1 * (r + gamma * Q_s2[np.argmax(Q_s2)] - Q_s[a]):.4f}")
print()

print("\nQ-Learning (Off-Policy):")
print("-" * 90)
print(f"  Target = r + gamma * max Q(s', a')")
print(f"  Q-Learning ignoriert, welche Aktion tatsächlich gewählt wird!")
print(f"  Es verwendet IMMER die beste bekannte Aktion:")
print()
print(f"  Best_action = argmax Q(s',:) = {np.argmax(Q_s2)}")
print(f"  Q-values for s'={s2}: {' '.join([f'Q({i})={q:.1f}' for i, q in enumerate(Q_s2)])}")
print()
print(f"  Target = {r} + {gamma} * max(Q({s2}, ·))")
print(f"         = {r} + {gamma} * {np.max(Q_s2):.1f}")
print(f"         = {r + gamma * np.max(Q_s2):.4f}")
print()
print(f"  TD Update: Q({s}, {a}) += 0.1 * ({r + gamma * np.max(Q_s2):.4f} - {Q_s[a]:.2f})")
print(f"           = {Q_s[a]:.2f} + 0.1 * ({r + gamma * np.max(Q_s2) - Q_s[a]:.4f})")
print(f"           = {Q_s[a] + 0.1 * (r + gamma * np.max(Q_s2) - Q_s[a]):.4f}")
print()

print("\n" + "-" * 90)
print("UNTERSCHIED (TARGET VALUES):")
print("-" * 90)
sarsa_target = r + gamma * Q_s2[np.argmax(Q_s2)]
q_target = r + gamma * np.max(Q_s2)
print(f"SARSA target:     {sarsa_target:.4f}")
print(f"Q-Learning target: {q_target:.4f}")
print(f"Differenz:        {q_target - sarsa_target:.4f}")
print()
print("Implikationen:")
print("  → Q-Learning ist OPTIMISTISCHER (höheres target value)")
print("  → Q-Learning geht davon aus, dass nächste Action die beste sein wird")
print("  → SARSA ist vorsichtiger (conservative) mit Explorations-Risiken")
print()

# ============================================================================
# STATISTIK-ZUSAMMENFASSUNG
# ============================================================================

print("\n" + "=" * 90)
print("STATISTIK-ZUSAMMENFASSUNG")
print("=" * 90)
print()

for r in runs_prio1:
    avg_return_last_500 = float(np.mean(r["ep_returns"][-500:]))
    avg_length_last_500 = float(np.mean(r["ep_lengths"][-500:]))
    max_td_error = float(np.max(np.abs(r["td_errors"])))
    mean_td_error = float(np.mean(np.abs(r["td_errors"])))

    print(f"{r['name']}:")
    print(f"  Average Return (last 500 episodes): {avg_return_last_500:8.2f}")
    print(f"  Average Episode Length (last 500):  {avg_length_last_500:8.2f} steps")
    print(f"  Max |TD Error|:                     {max_td_error:8.4f}")
    print(f"  Mean |TD Error|:                    {mean_td_error:8.4f}")
    print()

# ============================================================================
# ZUSAMMENFASSUNG
# ============================================================================

print("\n" + "=" * 90)
print("ZUSAMMENFASSUNG: WANN SARSA vs Q-LEARNING?")
print("=" * 90)
print()
print("┌─────────────────────────────────────────────────────────────────────────────────┐")
print("│ SARSA (On-Policy):                                                              │")
print("├─────────────────────────────────────────────────────────────────────────────────┤")
print("│ • Lernt unter der EXPLORIERENDEN Policy (ε-greedy)                             │")
print("│ • Konservativ: berücksichtigt Explorations-Risiken                             │")
print("│ • Kleinere TD-Fehler, stabiler Lernen                                          │")
print("│ • Langsameres Konvergieren                                                      │")
print("│                                                                                  │")
print("│ WANN NUTZEN:")
print("│ ✓ Robotik / Sicherheit-kritische Anwendungen (Exploration ist teuer)           │")
print("│ ✓ Wenn man eher konservativ sein möchte                                         │")
print("│ ✓ Wenn Exploration Schaden anrichten kann                                       │")
print("└─────────────────────────────────────────────────────────────────────────────────┘")
print()
print("┌─────────────────────────────────────────────────────────────────────────────────┐")
print("│ Q-Learning (Off-Policy):                                                        │")
print("├─────────────────────────────────────────────────────────────────────────────────┤")
print("│ • Lernt unter der OPTIMALEN Policy (unabhängig von Exploration)                │")
print("│ • Aggressiv: ignoriert Explorations-Risiken                                    │")
print("│ • Größere TD-Fehler, schneller Lernen                                          │")
print("│ • Schnelleres Konvergieren zur optimalen Policy                                │")
print("│                                                                                  │")
print("│ WANN NUTZEN:")
print("│ ✓ Wenn optimale Policy das Hauptziel ist                                       │")
print("│ ✓ Simulation / sichere Umgebungen (Exploration ist günstig)                    │")
print("│ ✓ Wenn schneller Lernen wichtig ist                                            │")
print("└─────────────────────────────────────────────────────────────────────────────────┘")
print()

print("=" * 90)
print("Analyse abgeschlossen! Plots gespeichert in:")
print("  - 01_SARSA_vs_QLearning_Comparison.png")
print("=" * 90)
