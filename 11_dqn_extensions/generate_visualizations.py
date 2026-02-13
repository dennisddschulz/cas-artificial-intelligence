#!/usr/bin/env python3
"""
Generiere detaillierte Visualisierungen für PER-Analyse
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Setup
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 9)
plt.rcParams['font.size'] = 12

output_dir = Path('/home/isc-den/cas-artificial-intelligence/11_dqn_extensions')

# Versuche Daten zu laden
results_file = output_dir / 'results_summary.json'

if results_file.exists():
    with open(results_file, 'r') as f:
        results = json.load(f)
    print("✓ Trainingsergebnisse geladen")
else:
    print("❌ results_summary.json nicht gefunden!")
    print("Stelle sicher, dass das Training abgeschlossen ist.")
    exit(1)

# ============================================
# VISUALISIERUNG 1: Evaluation Return Comparison
# ============================================

fig, ax = plt.subplots(figsize=(16, 8))

uniform_means = np.array(results['uniform']['eval_means'])
uniform_stds = np.array(results['uniform']['eval_stds'])
uniform_steps = np.array(results['uniform']['eval_steps']) / 1000

per_means = np.array(results['per']['eval_means'])
per_stds = np.array(results['per']['eval_stds'])
per_steps = np.array(results['per']['eval_steps']) / 1000

# Plot
ax.plot(uniform_steps, uniform_means, 'o-', linewidth=3.5, markersize=10,
        label='Uniform Replay (Baseline)', color='#1f77b4', alpha=0.9, zorder=3)
ax.fill_between(uniform_steps, uniform_means - uniform_stds, uniform_means + uniform_stds,
                alpha=0.2, color='#1f77b4', zorder=1)

ax.plot(per_steps, per_means, 's-', linewidth=3.5, markersize=10,
        label='Prioritized Experience Replay (PER)', color='#ff7f0e', alpha=0.9, zorder=3)
ax.fill_between(per_steps, per_means - per_stds, per_means + per_stds,
                alpha=0.2, color='#ff7f0e', zorder=1)

ax.axhline(y=200, color='green', linestyle='--', linewidth=2.5, alpha=0.5, label='Target Return (200)', zorder=2)

ax.set_xlabel('Training Steps (thousands)', fontsize=14, fontweight='bold')
ax.set_ylabel('Evaluation Return', fontsize=14, fontweight='bold')
ax.set_title('Performance Comparison: Uniform Replay vs. Prioritized Experience Replay\n(LunarLander-v3, 300k steps, 5 evaluation episodes)',
            fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='lower right', framealpha=0.95, edgecolor='black', fancybox=True)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.set_xlim([0, 300])

plt.tight_layout()
plt.savefig(output_dir / 'VIZ_01_eval_return.png', dpi=300, bbox_inches='tight')
print("✓ Saved: VIZ_01_eval_return.png")
plt.close()

# ============================================
# VISUALISIERUNG 2: Detailed Metrics Comparison
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Subplot 1: Final Performance
ax = axes[0, 0]
methods = ['Uniform Replay', 'PER']
final_returns = [
    results['uniform']['eval_means'][-1],
    results['per']['eval_means'][-1]
]
colors = ['#1f77b4', '#ff7f0e']
bars = ax.bar(methods, final_returns, color=colors, alpha=0.8, edgecolor='black', linewidth=2.5, width=0.6)
for bar, val in zip(bars, final_returns):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{val:.1f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.set_ylabel('Final Return', fontsize=12, fontweight='bold')
ax.set_title('Final Performance after 300k steps', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.set_ylim([0, max(final_returns) * 1.15])

# Subplot 2: Maximum Performance
ax = axes[0, 1]
max_returns = [
    max(results['uniform']['eval_means']),
    max(results['per']['eval_means'])
]
bars = ax.bar(methods, max_returns, color=colors, alpha=0.8, edgecolor='black', linewidth=2.5, width=0.6)
for bar, val in zip(bars, max_returns):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{val:.1f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.set_ylabel('Maximum Return', fontsize=12, fontweight='bold')
ax.set_title('Peak Performance During Training', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.set_ylim([0, max(max_returns) * 1.15])

# Subplot 3: Convergence Speed
ax = axes[1, 0]
threshold = 200
convergence_steps = []
for res_name in ['uniform', 'per']:
    means = np.array(results[res_name]['eval_means'])
    steps = np.array(results[res_name]['eval_steps'])
    converged_idx = np.where(means >= threshold)[0]
    if len(converged_idx) > 0:
        convergence_steps.append(steps[converged_idx[0]] / 1000)
    else:
        convergence_steps.append(300)

bars = ax.bar(methods, convergence_steps, color=colors, alpha=0.8, edgecolor='black', linewidth=2.5, width=0.6)
for bar, val in zip(bars, convergence_steps):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{val:.1f}k', ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.set_ylabel('Training Steps (thousands)', fontsize=12, fontweight='bold')
ax.set_title(f'Convergence Speed (Target Return: {threshold})', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.axhline(y=300, color='red', linestyle='--', linewidth=2, alpha=0.3)
ax.set_ylim([0, 320])

# Subplot 4: Improvement Metrics Table
ax = axes[1, 1]
ax.axis('off')

improvement = results['per']['eval_means'][-1] - results['uniform']['eval_means'][-1]
improvement_pct = (improvement / abs(results['uniform']['eval_means'][-1])) * 100

metrics_text = f"""
VERGLEICH: UNIFORM VS. PER

Final Return Difference:
  Uniform:      {results['uniform']['eval_means'][-1]:>8.2f}
  PER:          {results['per']['eval_means'][-1]:>8.2f}
  Differenz:    {improvement:>8.2f} ({improvement_pct:>+6.1f}%)

Maximum Return:
  Uniform:      {max(results['uniform']['eval_means']):>8.2f}
  PER:          {max(results['per']['eval_means']):>8.2f}

Convergence (steps to {threshold}):
  Uniform:      {convergence_steps[0]:>8.1f}k
  PER:          {convergence_steps[1]:>8.1f}k
  Speedup:      {convergence_steps[0]/convergence_steps[1]:>8.2f}x

Statistical Significance:
  p-value:      {results.get('p_value', 'N/A'):>8}
"""

ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
       fontsize=11, verticalalignment='top', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='#ffffcc', alpha=0.9, pad=1.5, linewidth=2))

plt.suptitle('Detailed Performance Metrics: Uniform Replay vs. PER',
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / 'VIZ_02_detailed_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: VIZ_02_detailed_comparison.png")
plt.close()

# ============================================
# VISUALISIERUNG 3: Learning Curves Over Time
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Smoothing function
def smooth(arr, window=3):
    return np.convolve(arr, np.ones(window)/window, mode='valid')

# Plot 1: Uniform Replay Learning Curve
ax = axes[0]
ax.plot(uniform_steps, uniform_means, 'o-', linewidth=3, markersize=8,
        color='#1f77b4', alpha=0.9, label='Mean Return')
ax.fill_between(uniform_steps, uniform_means - uniform_stds, uniform_means + uniform_stds,
                alpha=0.2, color='#1f77b4', label='±1 Std Dev')
ax.axhline(y=200, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target')
ax.set_xlabel('Training Steps (thousands)', fontsize=12, fontweight='bold')
ax.set_ylabel('Evaluation Return', fontsize=12, fontweight='bold')
ax.set_title('Uniform Replay - Learning Progression', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3, linestyle='--')

# Plot 2: PER Learning Curve
ax = axes[1]
ax.plot(per_steps, per_means, 's-', linewidth=3, markersize=8,
        color='#ff7f0e', alpha=0.9, label='Mean Return')
ax.fill_between(per_steps, per_means - per_stds, per_means + per_stds,
                alpha=0.2, color='#ff7f0e', label='±1 Std Dev')
ax.axhline(y=200, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target')
ax.set_xlabel('Training Steps (thousands)', fontsize=12, fontweight='bold')
ax.set_ylabel('Evaluation Return', fontsize=12, fontweight='bold')
ax.set_title('PER - Learning Progression', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3, linestyle='--')

plt.suptitle('Individual Learning Curves with Confidence Intervals',
            fontsize=15, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(output_dir / 'VIZ_03_learning_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: VIZ_03_learning_curves.png")
plt.close()

# ============================================
# VISUALISIERUNG 4: Performance Improvement Analysis
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Plot 1: Performance Difference Over Time
ax = axes[0, 0]
min_steps = min(len(uniform_means), len(per_means))
perf_diff = per_means[:min_steps] - uniform_means[:min_steps]
steps_aligned = uniform_steps[:min_steps]

colors_diff = ['green' if x > 0 else 'red' for x in perf_diff]
ax.bar(steps_aligned, perf_diff, color=colors_diff, alpha=0.7, edgecolor='black', linewidth=1.5, width=10)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Training Steps (thousands)', fontsize=12, fontweight='bold')
ax.set_ylabel('Return Difference (PER - Uniform)', fontsize=12, fontweight='bold')
ax.set_title('Performance Advantage of PER Over Time', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

# Plot 2: Stability Analysis
ax = axes[0, 1]
uniform_diffs = np.diff(uniform_means)
per_diffs = np.diff(per_means)

window = 5
uniform_volatility = np.convolve(np.abs(uniform_diffs), np.ones(window)/window, mode='valid')
per_volatility = np.convolve(np.abs(per_diffs), np.ones(window)/window, mode='valid')

ax.plot(uniform_steps[1:][window-1:], uniform_volatility, 'o-', linewidth=2.5, markersize=7,
        color='#1f77b4', alpha=0.8, label='Uniform Replay')
ax.plot(per_steps[1:][window-1:], per_volatility, 's-', linewidth=2.5, markersize=7,
        color='#ff7f0e', alpha=0.8, label='PER')
ax.set_xlabel('Training Steps (thousands)', fontsize=12, fontweight='bold')
ax.set_ylabel('Performance Volatility (smoothed)', fontsize=12, fontweight='bold')
ax.set_title('Training Stability Analysis', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')

# Plot 3: Convergence Profile
ax = axes[1, 0]
# Calculate how "far" each evaluation is from final return
uniform_distance = np.abs(uniform_means - uniform_means[-1])
per_distance = np.abs(per_means - per_means[-1])

ax.semilogy(uniform_steps, uniform_distance, 'o-', linewidth=2.5, markersize=7,
            color='#1f77b4', alpha=0.8, label='Uniform Replay')
ax.semilogy(per_steps, per_distance, 's-', linewidth=2.5, markersize=7,
            color='#ff7f0e', alpha=0.8, label='PER')
ax.set_xlabel('Training Steps (thousands)', fontsize=12, fontweight='bold')
ax.set_ylabel('Distance to Final Return (log scale)', fontsize=12, fontweight='bold')
ax.set_title('Convergence Speed (Log Scale)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--', which='both')

# Plot 4: Performance Statistics
ax = axes[1, 1]
stats_data = {
    'Uniform': {
        'mean': np.mean(uniform_means),
        'median': np.median(uniform_means),
        'std': np.std(uniform_means),
        'min': np.min(uniform_means),
        'max': np.max(uniform_means),
    },
    'PER': {
        'mean': np.mean(per_means),
        'median': np.median(per_means),
        'std': np.std(per_means),
        'min': np.min(per_means),
        'max': np.max(per_means),
    }
}

x_pos = np.arange(len(stats_data))
means = [stats_data['Uniform']['mean'], stats_data['PER']['mean']]
stds = [stats_data['Uniform']['std'], stats_data['PER']['std']]
mins = [stats_data['Uniform']['min'], stats_data['PER']['min']]
maxs = [stats_data['Uniform']['max'], stats_data['PER']['max']]

ax.bar(x_pos, means, color=['#1f77b4', '#ff7f0e'], alpha=0.8, edgecolor='black', linewidth=2, width=0.6)
ax.errorbar(x_pos, means, yerr=stds, fmt='none', ecolor='black', capsize=10, capthick=2, linewidth=2)

for i, (mean, std, mn, mx) in enumerate(zip(means, stds, mins, maxs)):
    ax.text(i, mean + std + 5, f'μ={mean:.1f}\nσ={std:.1f}', ha='center', fontsize=10, fontweight='bold')
    ax.text(i, mn - 15, f'min={mn:.1f}', ha='center', fontsize=9)
    ax.text(i, mx + 5, f'max={mx:.1f}', ha='center', fontsize=9)

ax.set_xticks(x_pos)
ax.set_xticklabels(['Uniform Replay', 'PER'])
ax.set_ylabel('Return', fontsize=12, fontweight='bold')
ax.set_title('Performance Statistics Summary', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

plt.suptitle('Comprehensive Performance Analysis',
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / 'VIZ_04_performance_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: VIZ_04_performance_analysis.png")
plt.close()

# ============================================
# VISUALISIERUNG 5: Side-by-Side Comparison Table
# ============================================

fig, ax = plt.subplots(figsize=(16, 8))
ax.axis('tight')
ax.axis('off')

# Create comparison table
table_data = [
    ['Metric', 'Uniform Replay', 'PER', 'Difference'],
    ['Final Return', f"{results['uniform']['eval_means'][-1]:.2f}",
     f"{results['per']['eval_means'][-1]:.2f}",
     f"{results['per']['eval_means'][-1] - results['uniform']['eval_means'][-1]:+.2f}"],
    ['Max Return', f"{max(results['uniform']['eval_means']):.2f}",
     f"{max(results['per']['eval_means']):.2f}",
     f"{max(results['per']['eval_means']) - max(results['uniform']['eval_means']):+.2f}"],
    ['Mean Return', f"{np.mean(results['uniform']['eval_means']):.2f}",
     f"{np.mean(results['per']['eval_means']):.2f}",
     f"{np.mean(results['per']['eval_means']) - np.mean(results['uniform']['eval_means']):+.2f}"],
    ['Std Dev', f"{np.std(results['uniform']['eval_means']):.2f}",
     f"{np.std(results['per']['eval_means']):.2f}",
     f"{np.std(results['per']['eval_means']) - np.std(results['uniform']['eval_means']):+.2f}"],
    ['Min Return', f"{np.min(results['uniform']['eval_means']):.2f}",
     f"{np.min(results['per']['eval_means']):.2f}",
     f"{np.min(results['per']['eval_means']) - np.min(results['uniform']['eval_means']):+.2f}"],
]

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.25, 0.25, 0.25, 0.25])

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2.8)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white', size=13)

# Style data rows
for i in range(1, len(table_data)):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E7E6E6')
        else:
            table[(i, j)].set_facecolor('#F2F2F2')
        table[(i, j)].set_text_props(weight='bold' if j == 0 else 'normal')

        # Highlight positive differences
        if j == 3 and i > 0:
            cell_text = table[(i, j)].get_text().get_text()
            if cell_text.startswith('+'):
                table[(i, j)].set_facecolor('#90EE90')
            elif cell_text.startswith('-'):
                table[(i, j)].set_facecolor('#FFB6C6')

plt.title('Detailed Metrics Comparison Table\nUniform Replay vs. Prioritized Experience Replay',
         fontsize=15, fontweight='bold', pad=20)
plt.savefig(output_dir / 'VIZ_05_metrics_table.png', dpi=300, bbox_inches='tight')
print("✓ Saved: VIZ_05_metrics_table.png")
plt.close()

# ============================================
# VISUALISIERUNG 6: Summary Infographic
# ============================================

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

# Title
fig.suptitle('PER vs. Uniform Replay - Comprehensive Summary',
            fontsize=18, fontweight='bold', y=0.98)

# Plot 1: Main comparison
ax1 = fig.add_subplot(gs[0, :])
methods = ['Uniform\nReplay', 'PER']
final_returns = [results['uniform']['eval_means'][-1], results['per']['eval_means'][-1]]
colors = ['#1f77b4', '#ff7f0e']
bars = ax1.bar(methods, final_returns, color=colors, alpha=0.8, edgecolor='black', linewidth=3, width=0.5)
for bar, val in zip(bars, final_returns):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}', ha='center', va='bottom', fontsize=16, fontweight='bold')
ax1.set_ylabel('Final Return', fontsize=13, fontweight='bold')
ax1.set_title('Final Performance Comparison',fontsize=14, fontweight='bold')
ax1.set_ylim([0, max(final_returns) * 1.2])
ax1.grid(True, alpha=0.3, axis='y', linestyle='--')

# Plot 2: Advantages of PER
ax2 = fig.add_subplot(gs[1, 0])
ax2.axis('off')
advantages_text = """
✅ VORTEILE VON PER

• Schnellere Konvergenz
  Focus auf schwierige Samples

• Bessere Sample-Effizienz
  Mehr Lernen pro Trainings-Step

• Robustere Policy
  Schwierige Szenarien trainiert

• Höhere Peak-Performance
  Bessere finale Scores erreicht
"""
ax2.text(0.05, 0.95, advantages_text, transform=ax2.transAxes,
        fontsize=11, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#90EE90', alpha=0.8, pad=1, linewidth=2))

# Plot 3: Nachteile von PER
ax3 = fig.add_subplot(gs[1, 1])
ax3.axis('off')
disadvantages_text = """
⚠️  NACHTEILE VON PER

• Höhere Komplexität
  SumTree Implementation

• Hyperparameter-sensitiv
  Alpha, Beta, Epsilon tuning

• Potenzielle Instabilität
  Bei falscher Parametrisierung

• Recency Bias möglich
  Neue Samples überrepräsentiert
"""
ax3.text(0.05, 0.95, disadvantages_text, transform=ax3.transAxes,
        fontsize=11, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#FFB6C6', alpha=0.8, pad=1, linewidth=2))

# Plot 4: Wenn PER besonders hilft
ax4 = fig.add_subplot(gs[2, 0])
ax4.axis('off')
when_per_text = """
🎯 PER HILFT BESONDERS BEI:

• Sparse-Reward Environments
• Sample-Effizienz kritisch
• Komplexe Szenarien
• Begrenzte Compute-Ressourcen
"""
ax4.text(0.05, 0.95, when_per_text, transform=ax4.transAxes,
        fontsize=11, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#FFFFCC', alpha=0.8, pad=1, linewidth=2))

# Plot 5: Wenn Uniform besser
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')
when_uniform_text = """
✓ UNIFORM IST BESSER BEI:

• Einfachen Umgebungen
• Genug Compute vorhanden
• Debugging wichtig
• Reproduzierbarkeit kritisch
"""
ax5.text(0.05, 0.95, when_uniform_text, transform=ax5.transAxes,
        fontsize=11, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#ADD8E6', alpha=0.8, pad=1, linewidth=2))

plt.savefig(output_dir / 'VIZ_06_summary_infographic.png', dpi=300, bbox_inches='tight')
print("✓ Saved: VIZ_06_summary_infographic.png")
plt.close()

print("\n" + "="*80)
print("✅ ALLE VISUALISIERUNGEN ERFOLGREICH ERSTELLT!")
print("="*80)
print("\n📊 Generierte Visualisierungen:")
print("   1. VIZ_01_eval_return.png - Haupt-Vergleich")
print("   2. VIZ_02_detailed_comparison.png - Detaillierte Metriken")
print("   3. VIZ_03_learning_curves.png - Lernkurven einzeln")
print("   4. VIZ_04_performance_analysis.png - Umfassende Analyse")
print("   5. VIZ_05_metrics_table.png - Metriken-Tabelle")
print("   6. VIZ_06_summary_infographic.png - Zusammenfassung")
print("="*80)

