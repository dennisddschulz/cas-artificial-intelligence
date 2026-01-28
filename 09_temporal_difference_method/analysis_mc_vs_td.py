#!/usr/bin/env python3
"""
Zusätzliche Visualisierungen für PRIO 2: MC vs TD Vergleich
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict, deque

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


print("\n" + "=" * 90)
print("PRIO 2: Monte Carlo vs Temporal Difference - VISUELLE ERKLÄRUNG")
print("=" * 90)
print()

# Create a detailed comparison visualization
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

fig.suptitle("PRIO 2: Monte Carlo vs Temporal Difference - Theoretischer Vergleich",
             fontsize=16, fontweight='bold')

# ============================================================================
# Panel 1: Update Timing
# ============================================================================
ax = fig.add_subplot(gs[0, 0])
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.set_title("Monte Carlo: Update Timing", fontsize=12, fontweight='bold')

# MC timeline
ax.plot([0, 9], [2.5, 2.5], 'b-', linewidth=3, label='Episode')
for i in range(0, 10, 1):
    ax.plot([i, i], [2.45, 2.55], 'b-', linewidth=2)
    if i < 9:
        ax.text(i + 0.5, 2.8, f"Step {i}", ha='center', fontsize=8)

ax.plot([9], [2.5], 'ro', markersize=15, label='UPDATE HERE')
ax.text(9, 2.0, 'Nach ganzer Episode\n(Ende erreicht)', ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

ax.set_xticks([])
ax.set_yticks([])
ax.legend(loc='upper left', fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# ============================================================================
# Panel 2: TD Update Timing
# ============================================================================
ax = fig.add_subplot(gs[0, 1])
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.set_title("TD (SARSA/Q-Learning): Update Timing", fontsize=12, fontweight='bold')

# TD timeline
ax.plot([0, 9], [2.5, 2.5], 'orange', linewidth=3, label='Episode')
for i in range(0, 10, 1):
    ax.plot([i, i], [2.45, 2.55], 'orange', linewidth=2)
    if i < 9:
        ax.text(i + 0.5, 2.8, f"Step {i}", ha='center', fontsize=8)

# Red dots after every step
for i in range(0, 9):
    ax.plot([i], [2.5], 'ro', markersize=10)

ax.text(4.5, 1.8, 'UPDATE NACH JEDEM SCHRITT!', ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

ax.set_xticks([])
ax.set_yticks([])
ax.legend(loc='upper left', fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# ============================================================================
# Panel 3: MC Update Calculation
# ============================================================================
ax = fig.add_subplot(gs[0, 2])
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

text = """Monte Carlo Update:

G(t) = R(t) + γR(t+1) + γ²R(t+2) + ...
       (gesamte Rückgabe ab Schritt t)

Q(s,a) += α[G(t) - Q(s,a)]

Eigenschaften:
• Unbiased (wahre Rückgabe)
• High Variance (große Schwankungen)
• Speicherintensiv
"""

ax.text(0.5, 9.5, text, fontsize=10, family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
        verticalalignment='top')

# ============================================================================
# Panel 4: SARSA Update Calculation
# ============================================================================
ax = fig.add_subplot(gs[1, 0])
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

text = """SARSA Update (On-Policy TD):

target = r + γQ(s', a')

Q(s,a) += α[target - Q(s,a)]

where a' ~ ε-greedy(π)

Eigenschaften:
• Biased (von Q-Schätzung abhängig)
• Low Variance (stabile Updates)
• On-Policy
"""

ax.text(0.5, 9.5, text, fontsize=10, family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
        verticalalignment='top')

# ============================================================================
# Panel 5: Q-Learning Update
# ============================================================================
ax = fig.add_subplot(gs[1, 1])
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

text = """Q-Learning Update (Off-Policy TD):

target = r + γ max Q(s', ·)

Q(s,a) += α[target - Q(s,a)]

Eigenschaften:
• Biased (von Q-Schätzung abhängig)
• Low Variance (stabile Updates)
• Off-Policy
• Kann alte Daten replays nutzen
"""

ax.text(0.5, 9.5, text, fontsize=10, family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7),
        verticalalignment='top')

# ============================================================================
# Panel 6: Bias-Varianz Spektrum
# ============================================================================
ax = fig.add_subplot(gs[1, 2])
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

algorithms = ['MC', 'TD(λ)', 'TD(0)', 'Q-Learn']
bias = [1, 3, 5, 7]
variance = [9, 7, 5, 3]
colors = ['blue', 'green', 'orange', 'red']

for alg, b, v, c in zip(algorithms, bias, variance, colors):
    ax.scatter(b, v, s=800, alpha=0.6, color=c, label=alg)
    ax.text(b, v, alg, ha='center', va='center', fontsize=9, fontweight='bold')

ax.set_xlabel("Bias ↑", fontsize=11, fontweight='bold')
ax.set_ylabel("Variance ↑", fontsize=11, fontweight='bold')
ax.set_title("Bias-Varianz Tradeoff", fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add annotation
ax.text(5, 0.5, "Bessere Sample Efficiency ←", ha='center', fontsize=9, style='italic')

# ============================================================================
# Panel 7: MC Beispiel (Trajectory)
# ============================================================================
ax = fig.add_subplot(gs[2, 0])
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

text = """MC Beispiel:

Episode: s₀→s₁→s₂→s₃ (DONE)
Rewards: -1, -1, +10

Returns (backward):
G(s₃) = 10
G(s₂) = -1 + γ(10) ≈ 8.9
G(s₁) = -1 + γ(8.9) ≈ 7.8
G(s₀) = -1 + γ(7.8) ≈ 6.7

All Q-values updated mit
diesen G-Werten!

Pro: Unbiased
Kontra: Muss ganze Episode
        speichern & warten
"""

ax.text(0.3, 9.7, text, fontsize=9, family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
        verticalalignment='top')

# ============================================================================
# Panel 8: TD Beispiel
# ============================================================================
ax = fig.add_subplot(gs[2, 1])
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

text = """TD Beispiel (SARSA):

Step 0: s₀→a→s₁, r=-1
  target = -1 + γQ(s₁,a₁)
  Update Q(s₀,a) SOFORT

Step 1: s₁→a'→s₂, r=-1
  target = -1 + γQ(s₂,a₂)
  Update Q(s₁,a') SOFORT

Pro: Online learning
     Schnelle Updates
     Speicherarm

Kontra: Abhängig von
        Q-Schätzung
"""

ax.text(0.3, 9.7, text, fontsize=9, family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
        verticalalignment='top')

# ============================================================================
# Panel 9: Summary Table
# ============================================================================
ax = fig.add_subplot(gs[2, 2])
ax.axis('off')

summary_text = """
Vergleich (Zusammenfassung)

MC:
✓ Unbiased (wahre Return)
✗ High Variance
✗ Langsam (wartet auf Episode)

TD:
✓ Low Variance  
✓ Fast Updates (online)
✗ Biased (bootstrapping)

→ TD ist praktisch besser!
  (schneller, sample-effizient)
"""

ax.text(0.5, 9.5, summary_text, fontsize=10, family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        verticalalignment='top', horizontalalignment='left')

plt.savefig("02_MC_vs_TD_Detailed_Comparison.png", dpi=150, bbox_inches='tight')
print("✓ Detailliertes MC vs TD Vergleich-Plot gespeichert: 02_MC_vs_TD_Detailed_Comparison.png\n")
plt.close()

# ============================================================================
# Create a comprehensive comparison table visualization
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('tight')
ax.axis('off')

# Create table data
table_data = [
    ['Merkmal', 'Monte Carlo', 'Temporal Difference', 'SARSA (On-Policy)', 'Q-Learning (Off-Policy)'],
    ['Update Timing', 'Nach Episode', 'Nach jedem Schritt', 'Nach jedem Schritt', 'Nach jedem Schritt'],
    ['Target Berechnung', 'G = ΣγᵗRₜ\n(full trajectory)', 'r + γV(s\')', 'r + γQ(s\',a\')\n(actual next)', 'r + γmax Q(s\',·)\n(best next)'],
    ['Bias', 'Unbiased', 'Leicht biased', 'Leicht biased\n(auf Policy)', 'Biased\n(overoptimistic)'],
    ['Varianz', 'HIGH', 'LOW', 'MEDIUM-LOW', 'LOW'],
    ['Online Learning', '✗ Offline', '✓ Online', '✓ Online', '✓ Online'],
    ['Off-Policy Support', '✗ Nein', '✓ Ja (TD)', '✗ Nein', '✓ Ja'],
    ['Sample Efficiency', '✗ Schlecht', '✓ Gut', '✓ Gut', '✓✓ Sehr gut'],
    ['Konvergenz Speed', 'Langsam', 'Schnell', 'Mittel-Schnell', 'Schnell'],
    ['Speicher', 'Hoch\n(ganze Episode)', 'Niedrig\n(ein Schritt)', 'Niedrig', 'Niedrig'],
    ['Beste Anwendung', 'Episodische\nProbleme\n(kurz)', 'Episodisch &\nKontinuierlich', 'Sichere\nUmgebungen\nRisikobewusst', 'Optimale Policy\nSimulationen'],
]

# Create table
table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.13, 0.22, 0.22, 0.22, 0.22])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 3)

# Color header
for i in range(5):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color rows alternately
for i in range(1, len(table_data)):
    for j in range(5):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')
        else:
            table[(i, j)].set_facecolor('#ffffff')

        # Highlight TD column
        if j > 1:
            table[(i, j)].set_facecolor('#ffffcc')

plt.title('Umfassender Vergleich: MC vs TD vs SARSA vs Q-Learning',
          fontsize=14, fontweight='bold', pad=20)

plt.savefig("03_MC_vs_TD_Comparison_Table.png", dpi=150, bbox_inches='tight')
print("✓ Vergleichstabelle gespeichert: 03_MC_vs_TD_Comparison_Table.png\n")
plt.close()

print("=" * 90)
print("PRIO 2: Alle Visualisierungen erstellt!")
print("=" * 90)
print("\nErstellt Dateien:")
print("  1. 02_MC_vs_TD_Detailed_Comparison.png - Visuelle Erklärung")
print("  2. 03_MC_vs_TD_Comparison_Table.png - Vergleichstabelle")
print()
print("=" * 90)
