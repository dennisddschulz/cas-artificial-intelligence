#!/usr/bin/env python3
"""
Comprehensive Visualization Script for PER Analysis
Erstellt detaillierte Plots für Uniform Replay vs. PER
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Configure matplotlib
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 7

output_dir = Path("/home/isc-den/cas-artificial-intelligence/11_dqn_extensions")

# Load results
with open(output_dir / 'results_summary_CORRECTED.json', 'r') as f:
    results = json.load(f)

uniform_data = results['uniform']
per_data = results['per']

# Extract data
uniform_steps = np.array(uniform_data['eval_steps'])
uniform_means = np.array(uniform_data['eval_means'])
uniform_stds = np.array(uniform_data['eval_stds'])

per_steps = np.array(per_data['eval_steps'])
per_means = np.array(per_data['eval_means'])
per_stds = np.array(per_data['eval_stds'])

# Colors
color_uniform = '#1f77b4'  # Blue
color_per = '#ff7f0e'      # Orange

print("="*80)
print("GENERATING COMPREHENSIVE VISUALIZATIONS FOR PER ANALYSIS")
print("="*80)
print(f"\nUniform Replay Stats:")
print(f"  Final Return: {uniform_means[-1]:.2f} ± {uniform_stds[-1]:.2f}")
print(f"  Max Return: {np.max(uniform_means):.2f}")
print(f"  Mean Return: {np.mean(uniform_means):.2f}")
print(f"  Std Dev: {np.std(uniform_means):.2f}")

print(f"\nPER Stats:")
print(f"  Final Return: {per_means[-1]:.2f} ± {per_stds[-1]:.2f}")
print(f"  Max Return: {np.max(per_means):.2f}")
print(f"  Mean Return: {np.mean(per_means):.2f}")
print(f"  Std Dev: {np.std(per_means):.2f}")

# Improvement calculation
improvement = ((per_means[-1] - uniform_means[-1]) / abs(uniform_means[-1])) * 100
print(f"\nPER Improvement: {improvement:.1f}%")

t_stat, p_value = stats.ttest_ind(per_means, uniform_means)
print(f"T-Test p-value: {p_value:.6f}")

# ============================================
# FIGURE 1: Main Comparison with Confidence Bands
# ============================================
fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(uniform_steps, uniform_means, 'o-', color=color_uniform,
        label='Uniform Replay (Baseline)', linewidth=3, markersize=8, alpha=0.9)
ax.fill_between(uniform_steps,
                uniform_means - uniform_stds,
                uniform_means + uniform_stds,
                color=color_uniform, alpha=0.2, linewidth=0)

ax.plot(per_steps, per_means, 's-', color=color_per,
        label='Prioritized Experience Replay (PER)', linewidth=3, markersize=8, alpha=0.9)
ax.fill_between(per_steps,
                per_means - per_stds,
                per_means + per_stds,
                color=color_per, alpha=0.2, linewidth=0)

ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
ax.set_ylabel('Evaluation Return', fontsize=12, fontweight='bold')
ax.set_title('Performance Comparison: Uniform Replay vs. Prioritized Experience Replay\n(LunarLander-v3)',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=12, framealpha=0.95)

# Add statistics box
textstr = f'Final Performance:\nUniform: {uniform_means[-1]:.1f} ± {uniform_stds[-1]:.1f}\nPER: {per_means[-1]:.1f} ± {per_stds[-1]:.1f}\n\nImprovement: {improvement:.1f}%\np-value: {p_value:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props, family='monospace')

plt.tight_layout()
plt.savefig(output_dir / 'viz_01_main_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: viz_01_main_comparison.png")
plt.close()

# ============================================
# FIGURE 2: Smoothed Curves (Moving Average)
# ============================================
fig, ax = plt.subplots(figsize=(14, 8))

# 3-point moving average for smoothing
def moving_average(data, window=3):
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    smoothed = np.convolve(data, kernel, mode='valid')
    # Pad to match original length
    pad = (len(data) - len(smoothed)) // 2
    return np.pad(smoothed, (pad, len(data) - len(smoothed) - pad), mode='edge')

uniform_smooth = moving_average(uniform_means, window=3)
per_smooth = moving_average(per_means, window=3)

ax.plot(uniform_steps, uniform_smooth, 'o-', color=color_uniform,
        label='Uniform Replay (Smoothed)', linewidth=3, markersize=8, alpha=0.9)
ax.plot(per_steps, per_smooth, 's-', color=color_per,
        label='PER (Smoothed)', linewidth=3, markersize=8, alpha=0.9)

# Add raw points with transparency
ax.plot(uniform_steps, uniform_means, 'o', color=color_uniform, alpha=0.2, markersize=5)
ax.plot(per_steps, per_means, 's', color=color_per, alpha=0.2, markersize=5)

ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
ax.set_ylabel('Evaluation Return', fontsize=12, fontweight='bold')
ax.set_title('Smoothed Performance Curves with Raw Data (faint)\n(3-point Moving Average)',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=12, framealpha=0.95)

plt.tight_layout()
plt.savefig(output_dir / 'viz_02_smoothed_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: viz_02_smoothed_curves.png")
plt.close()

# ============================================
# FIGURE 3: Performance Difference Over Time
# ============================================
fig, ax = plt.subplots(figsize=(14, 8))

# Interpolate to same steps for comparison
common_steps = np.minimum(uniform_steps, per_steps)
uniform_interp = np.interp(common_steps, uniform_steps, uniform_means)
per_interp = np.interp(common_steps, per_steps, per_means)

difference = per_interp - uniform_interp

colors_diff = ['#2ecc71' if d > 0 else '#e74c3c' for d in difference]

bars = ax.bar(common_steps, difference, width=10000, color=colors_diff, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.axhline(y=0, color='black', linestyle='-', linewidth=2)

ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
ax.set_ylabel('PER Return - Uniform Return', fontsize=12, fontweight='bold')
ax.set_title('Performance Difference: PER vs. Uniform Replay\n(Positive = PER Better, Negative = Uniform Better)',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (step, diff) in enumerate(zip(common_steps, difference)):
    label = f'{diff:.0f}'
    y_pos = diff + (10 if diff > 0 else -15)
    ax.text(step, y_pos, label, ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'viz_03_difference.png', dpi=300, bbox_inches='tight')
print("✓ Saved: viz_03_difference.png")
plt.close()

# ============================================
# FIGURE 4: Variance Comparison (Stability)
# ============================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Subplots for variance
ax1.bar(range(len(uniform_steps)), uniform_stds, color=color_uniform, alpha=0.7, label='Uniform Replay', edgecolor='black')
ax1.set_xlabel('Evaluation Step Index', fontsize=11, fontweight='bold')
ax1.set_ylabel('Std Dev (Error Bars)', fontsize=11, fontweight='bold')
ax1.set_title('Uniform Replay: Variance Across Evaluations', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

ax2.bar(range(len(per_steps)), per_stds, color=color_per, alpha=0.7, label='PER', edgecolor='black')
ax2.set_xlabel('Evaluation Step Index', fontsize=11, fontweight='bold')
ax2.set_ylabel('Std Dev (Error Bars)', fontsize=11, fontweight='bold')
ax2.set_title('PER: Variance Across Evaluations', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

fig.suptitle('Variance Comparison: Which Method is More Stable?', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(output_dir / 'viz_04_variance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: viz_04_variance.png")
plt.close()

# ============================================
# FIGURE 5: Learning Progress Quartiles
# ============================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Quartile analysis
n_uniform = len(uniform_means)
q_size = n_uniform // 4

uniform_q1 = uniform_means[:q_size]
uniform_q2 = uniform_means[q_size:2*q_size]
uniform_q3 = uniform_means[2*q_size:3*q_size]
uniform_q4 = uniform_means[3*q_size:]

n_per = len(per_means)
q_size_per = n_per // 4
per_q1 = per_means[:q_size_per]
per_q2 = per_means[q_size_per:2*q_size_per]
per_q3 = per_means[2*q_size_per:3*q_size_per]
per_q4 = per_means[3*q_size_per:]

# Plot for Uniform
quartiles_uniform = [
    ('Early\n(0-25%)', np.mean(uniform_q1), np.std(uniform_q1)),
    ('Mid-Early\n(25-50%)', np.mean(uniform_q2), np.std(uniform_q2)),
    ('Mid-Late\n(50-75%)', np.mean(uniform_q3), np.std(uniform_q3)),
    ('Late\n(75-100%)', np.mean(uniform_q4), np.std(uniform_q4)),
]

x_pos = np.arange(len(quartiles_uniform))
means_u = [q[1] for q in quartiles_uniform]
stds_u = [q[2] for q in quartiles_uniform]

ax1.bar(x_pos, means_u, yerr=stds_u, color=color_uniform, alpha=0.7, capsize=10,
        edgecolor='black', linewidth=1.5, error_kw={'linewidth': 2})
ax1.set_xticks(x_pos)
ax1.set_xticklabels([q[0] for q in quartiles_uniform])
ax1.set_ylabel('Average Return', fontsize=11, fontweight='bold')
ax1.set_title('Uniform Replay: Learning Progress by Phase', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Plot for PER
quartiles_per = [
    ('Early\n(0-25%)', np.mean(per_q1), np.std(per_q1)),
    ('Mid-Early\n(25-50%)', np.mean(per_q2), np.std(per_q2)),
    ('Mid-Late\n(50-75%)', np.mean(per_q3), np.std(per_q3)),
    ('Late\n(75-100%)', np.mean(per_q4), np.std(per_q4)),
]

means_p = [q[1] for q in quartiles_per]
stds_p = [q[2] for q in quartiles_per]

ax2.bar(x_pos, means_p, yerr=stds_p, color=color_per, alpha=0.7, capsize=10,
        edgecolor='black', linewidth=1.5, error_kw={'linewidth': 2})
ax2.set_xticks(x_pos)
ax2.set_xticklabels([q[0] for q in quartiles_per])
ax2.set_ylabel('Average Return', fontsize=11, fontweight='bold')
ax2.set_title('PER: Learning Progress by Phase', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

fig.suptitle('Learning Progression: Early vs. Late Training', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(output_dir / 'viz_05_quartiles.png', dpi=300, bbox_inches='tight')
print("✓ Saved: viz_05_quartiles.png")
plt.close()

# ============================================
# FIGURE 6: Box Plots for Distribution Comparison
# ============================================
fig, ax = plt.subplots(figsize=(12, 8))

bp = ax.boxplot([uniform_means, per_means],
                 labels=['Uniform Replay', 'Prioritized Experience Replay'],
                 patch_artist=True,
                 widths=0.6,
                 showmeans=True,
                 meanprops=dict(marker='D', markerfacecolor='red', markersize=10,
                               label='Mean'))

for patch, color in zip(bp['boxes'], [color_uniform, color_per]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Styling
for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color='black', linewidth=2)

ax.set_ylabel('Evaluation Return', fontsize=12, fontweight='bold')
ax.set_title('Return Distribution Comparison\n(Box Plot with Mean)',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='y')

# Add statistics
stats_text = f"""Statistics:
Uniform:  μ={np.mean(uniform_means):.1f}, σ={np.std(uniform_means):.1f}, median={np.median(uniform_means):.1f}
PER:      μ={np.mean(per_means):.1f}, σ={np.std(per_means):.1f}, median={np.median(per_means):.1f}
"""
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='right', bbox=props, family='monospace')

plt.tight_layout()
plt.savefig(output_dir / 'viz_06_boxplot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: viz_06_boxplot.png")
plt.close()

# ============================================
# FIGURE 7: Cumulative Performance
# ============================================
fig, ax = plt.subplots(figsize=(14, 8))

uniform_cumsum = np.cumsum(uniform_means)
per_cumsum = np.cumsum(per_means)

ax.plot(uniform_steps, uniform_cumsum, 'o-', color=color_uniform,
        label='Uniform Replay', linewidth=3, markersize=8, alpha=0.9)
ax.plot(per_steps, per_cumsum, 's-', color=color_per,
        label='Prioritized Experience Replay', linewidth=3, markersize=8, alpha=0.9)

ax.fill_between(uniform_steps, 0, uniform_cumsum, color=color_uniform, alpha=0.1)
ax.fill_between(per_steps, 0, per_cumsum, color=color_per, alpha=0.1)

ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
ax.set_ylabel('Cumulative Return', fontsize=12, fontweight='bold')
ax.set_title('Cumulative Performance Over Training\n(Integral of Evaluation Returns)',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=12, framealpha=0.95)

plt.tight_layout()
plt.savefig(output_dir / 'viz_07_cumulative.png', dpi=300, bbox_inches='tight')
print("✓ Saved: viz_07_cumulative.png")
plt.close()

# ============================================
# FIGURE 8: Convergence Speed Analysis
# ============================================
fig, ax = plt.subplots(figsize=(14, 8))

# Find when each method reaches 50% of max performance
uniform_max = np.max(uniform_means)
per_max = np.max(per_means)

# Define threshold (e.g., 75% of max)
threshold_percent = 0.75
uniform_threshold = uniform_max * threshold_percent
per_threshold = per_max * threshold_percent

# Find first index above threshold
try:
    uniform_convergence_idx = np.where(uniform_means >= uniform_threshold)[0][0]
    uniform_convergence_step = uniform_steps[uniform_convergence_idx]
except:
    uniform_convergence_step = uniform_steps[-1]

try:
    per_convergence_idx = np.where(per_means >= per_threshold)[0][0]
    per_convergence_step = per_steps[per_convergence_idx]
except:
    per_convergence_step = per_steps[-1]

ax.plot(uniform_steps, uniform_means, 'o-', color=color_uniform,
        label='Uniform Replay', linewidth=3, markersize=8, alpha=0.9)
ax.plot(per_steps, per_means, 's-', color=color_per,
        label='Prioritized Experience Replay', linewidth=3, markersize=8, alpha=0.9)

ax.axhline(y=uniform_threshold, color=color_uniform, linestyle='--', alpha=0.5, linewidth=2, label=f'Uniform 75% Threshold')
ax.axvline(x=uniform_convergence_step, color=color_uniform, linestyle=':', alpha=0.5, linewidth=2)

ax.axhline(y=per_threshold, color=color_per, linestyle='--', alpha=0.5, linewidth=2, label=f'PER 75% Threshold')
ax.axvline(x=per_convergence_step, color=color_per, linestyle=':', alpha=0.5, linewidth=2)

ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
ax.set_ylabel('Evaluation Return', fontsize=12, fontweight='bold')
ax.set_title('Convergence Speed: When Do Methods Reach 75% of Peak Performance?',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=11, framealpha=0.95)

convergence_text = f'Convergence to 75%:\nUniform: Step {uniform_convergence_step:,}\nPER: Step {per_convergence_step:,}\n\nDifference: {abs(per_convergence_step - uniform_convergence_step):,} steps'
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
ax.text(0.02, 0.97, convergence_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props, family='monospace')

plt.tight_layout()
plt.savefig(output_dir / 'viz_08_convergence.png', dpi=300, bbox_inches='tight')
print("✓ Saved: viz_08_convergence.png")
plt.close()

# ============================================
# FIGURE 9: Statistical Summary Dashboard
# ============================================
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Plot 1: Mean comparison
ax1 = fig.add_subplot(gs[0, 0])
methods = ['Uniform', 'PER']
means = [np.mean(uniform_means), np.mean(per_means)]
colors_methods = [color_uniform, color_per]
bars1 = ax1.bar(methods, means, color=colors_methods, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('Mean Return', fontweight='bold')
ax1.set_title('Average Performance', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
for bar, mean in zip(bars1, means):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{mean:.1f}', ha='center', va='bottom', fontweight='bold')

# Plot 2: Std dev comparison
ax2 = fig.add_subplot(gs[0, 1])
stds = [np.std(uniform_means), np.std(per_means)]
bars2 = ax2.bar(methods, stds, color=colors_methods, alpha=0.7, edgecolor='black', linewidth=2)
ax2.set_ylabel('Std Deviation', fontweight='bold')
ax2.set_title('Variability (Stability)', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
for bar, std in zip(bars2, stds):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{std:.1f}', ha='center', va='bottom', fontweight='bold')

# Plot 3: Max performance
ax3 = fig.add_subplot(gs[0, 2])
maxes = [np.max(uniform_means), np.max(per_means)]
bars3 = ax3.bar(methods, maxes, color=colors_methods, alpha=0.7, edgecolor='black', linewidth=2)
ax3.set_ylabel('Max Return', fontweight='bold')
ax3.set_title('Peak Performance', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
for bar, mx in zip(bars3, maxes):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{mx:.1f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Median comparison
ax4 = fig.add_subplot(gs[1, 0])
medians = [np.median(uniform_means), np.median(per_means)]
bars4 = ax4.bar(methods, medians, color=colors_methods, alpha=0.7, edgecolor='black', linewidth=2)
ax4.set_ylabel('Median Return', fontweight='bold')
ax4.set_title('Median Performance', fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
for bar, med in zip(bars4, medians):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{med:.1f}', ha='center', va='bottom', fontweight='bold')

# Plot 5: Min performance
ax5 = fig.add_subplot(gs[1, 1])
mins = [np.min(uniform_means), np.min(per_means)]
bars5 = ax5.bar(methods, mins, color=colors_methods, alpha=0.7, edgecolor='black', linewidth=2)
ax5.set_ylabel('Min Return', fontweight='bold')
ax5.set_title('Worst Performance', fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')
for bar, mn in zip(bars5, mins):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{mn:.1f}', ha='center', va='bottom', fontweight='bold')

# Plot 6: Coefficient of Variation
ax6 = fig.add_subplot(gs[1, 2])
cv_uniform = (np.std(uniform_means) / abs(np.mean(uniform_means))) * 100
cv_per = (np.std(per_means) / abs(np.mean(per_means))) * 100
cvs = [cv_uniform, cv_per]
bars6 = ax6.bar(methods, cvs, color=colors_methods, alpha=0.7, edgecolor='black', linewidth=2)
ax6.set_ylabel('CV (%)', fontweight='bold')
ax6.set_title('Relative Variability', fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')
for bar, cv in zip(bars6, cvs):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
            f'{cv:.1f}%', ha='center', va='bottom', fontweight='bold')

# Plot 7: Final vs Initial
ax7 = fig.add_subplot(gs[2, :2])
phases = ['Initial (First 5)', 'Middle (Mid 10)', 'Final (Last 5)']
uniform_phases = [
    np.mean(uniform_means[:5]),
    np.mean(uniform_means[5:15]),
    np.mean(uniform_means[-5:])
]
per_phases = [
    np.mean(per_means[:5]),
    np.mean(per_means[5:15]),
    np.mean(per_means[-5:])
]

x = np.arange(len(phases))
width = 0.35

bars_u = ax7.bar(x - width/2, uniform_phases, width, label='Uniform',
                 color=color_uniform, alpha=0.7, edgecolor='black', linewidth=1.5)
bars_p = ax7.bar(x + width/2, per_phases, width, label='PER',
                 color=color_per, alpha=0.7, edgecolor='black', linewidth=1.5)

ax7.set_ylabel('Average Return', fontweight='bold')
ax7.set_title('Training Progress: Early → Middle → Late', fontweight='bold')
ax7.set_xticks(x)
ax7.set_xticklabels(phases)
ax7.legend(fontsize=10)
ax7.grid(True, alpha=0.3, axis='y')

# Plot 8: Summary text box
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis('off')

summary_text = f"""KEY FINDINGS

Final Performance:
  Uniform: {uniform_means[-1]:.1f}±{uniform_stds[-1]:.1f}
  PER: {per_means[-1]:.1f}±{per_stds[-1]:.1f}

Improvement: {improvement:+.1f}%

Statistical Test:
  t-stat: {t_stat:.3f}
  p-value: {p_value:.4f}
  
Stability (Std Dev):
  Uniform: {np.std(uniform_means):.1f}
  PER: {np.std(per_means):.1f}
"""

ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

fig.suptitle('Statistical Summary: Uniform Replay vs. Prioritized Experience Replay',
             fontsize=15, fontweight='bold', y=0.995)
plt.savefig(output_dir / 'viz_09_summary_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Saved: viz_09_summary_dashboard.png")
plt.close()

# ============================================
# FIGURE 10: Combined Analysis with Annotations
# ============================================
fig, ax = plt.subplots(figsize=(16, 10))

# Main plot
ax.plot(uniform_steps, uniform_means, 'o-', color=color_uniform,
        label='Uniform Replay (Baseline)', linewidth=3.5, markersize=9, alpha=0.95, zorder=3)
ax.fill_between(uniform_steps,
                uniform_means - uniform_stds,
                uniform_means + uniform_stds,
                color=color_uniform, alpha=0.15, linewidth=0)

ax.plot(per_steps, per_means, 's-', color=color_per,
        label='Prioritized Experience Replay (PER)', linewidth=3.5, markersize=9, alpha=0.95, zorder=3)
ax.fill_between(per_steps,
                per_means - per_stds,
                per_means + per_stds,
                color=color_per, alpha=0.15, linewidth=0)

# Add reference lines
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.4, linewidth=1.5, zorder=1)
ax.axhline(y=np.mean(uniform_means), color=color_uniform, linestyle=':', alpha=0.4, linewidth=2, zorder=1)
ax.axhline(y=np.mean(per_means), color=color_per, linestyle=':', alpha=0.4, linewidth=2, zorder=1)

# Styling
ax.set_xlabel('Training Steps', fontsize=13, fontweight='bold')
ax.set_ylabel('Evaluation Return', fontsize=13, fontweight='bold')
ax.set_title('Comprehensive Comparison: Uniform Replay vs. Prioritized Experience Replay\nLunarLander-v3 Training Results with Confidence Bands (±1 std)',
             fontsize=15, fontweight='bold', pad=25)
ax.grid(True, alpha=0.25, linestyle='--')
ax.legend(loc='best', fontsize=13, framealpha=0.95, edgecolor='black', fancybox=True)

# Enhanced statistics box
textstr = f'''FINAL EVALUATION METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    Uniform Replay    PER
Final Return:         {uniform_means[-1]:8.1f}    {per_means[-1]:8.1f}
Max Return:           {np.max(uniform_means):8.1f}    {np.max(per_means):8.1f}
Mean Return:          {np.mean(uniform_means):8.1f}    {np.mean(per_means):8.1f}
Median Return:        {np.median(uniform_means):8.1f}    {np.median(per_means):8.1f}
Std Deviation:        {np.std(uniform_means):8.1f}    {np.std(per_means):8.1f}

STATISTICAL TEST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Improvement:          {improvement:+7.1f}%
p-value:              {p_value:8.6f}
'''

props = dict(boxstyle='round,pad=0.8', facecolor='#f0f0f0', alpha=0.92, edgecolor='black', linewidth=2)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props, family='monospace', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'viz_10_comprehensive.png', dpi=300, bbox_inches='tight')
print("✓ Saved: viz_10_comprehensive.png")
plt.close()

print("\n" + "="*80)
print("✅ ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
print("="*80)
print("\nGenerated files:")
print("  1. viz_01_main_comparison.png - Main performance comparison")
print("  2. viz_02_smoothed_curves.png - Smoothed curves with raw data")
print("  3. viz_03_difference.png - PER vs Uniform difference")
print("  4. viz_04_variance.png - Stability comparison")
print("  5. viz_05_quartiles.png - Learning progress by phase")
print("  6. viz_06_boxplot.png - Distribution comparison")
print("  7. viz_07_cumulative.png - Cumulative performance")
print("  8. viz_08_convergence.png - Convergence speed analysis")
print("  9. viz_09_summary_dashboard.png - Statistical summary")
print("  10. viz_10_comprehensive.png - Comprehensive comparison")

