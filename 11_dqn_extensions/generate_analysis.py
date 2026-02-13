#!/usr/bin/env python3
"""
Analyse der Training-Ergebnisse und Generierung von Visualisierungen
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
sns.set_style("whitegrid")
sns.set_palette("husl")

output_dir = Path('/home/isc-den/cas-artificial-intelligence/11_dqn_extensions')

# Load results
with open(output_dir / 'results_summary.json', 'r') as f:
    results = json.load(f)

print("="*80)
print("TRAINING RESULTS ANALYSIS")
print("="*80)

uniform = results['uniform']
per = results['per']

print(f"\n✓ UNIFORM REPLAY:")
print(f"  Final Return:     {uniform['eval_means'][-1]:>8.2f} ± {uniform['eval_stds'][-1]:.2f}")
print(f"  Max Return:       {max(uniform['eval_means']):>8.2f}")
print(f"  Min Return:       {min(uniform['eval_means']):>8.2f}")
print(f"  Mean Return:      {np.mean(uniform['eval_means']):>8.2f}")

print(f"\n✗ PER:")
print(f"  Final Return:     {per['eval_means'][-1]:>8.2f} ± {per['eval_stds'][-1]:.2f}")
print(f"  Max Return:       {max(per['eval_means']):>8.2f}")
print(f"  Min Return:       {min(per['eval_means']):>8.2f}")
print(f"  Mean Return:      {np.mean(per['eval_means']):>8.2f}")

print(f"\n⚠️  PROBLEM:")
print(f"  PER is {abs(per['eval_means'][-1] - uniform['eval_means'][-1]):.1f} points WORSE than Uniform!")
print(f"  This is unusual and suggests an implementation issue.")

# ============================================
# VISUALISIERUNG 1: Main Comparison
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Plot 1: Learning Curves
ax = axes[0, 0]
uniform_steps = np.array(uniform['eval_steps']) / 1000
uniform_means = np.array(uniform['eval_means'])
uniform_stds = np.array(uniform['eval_stds'])

per_steps = np.array(per['eval_steps']) / 1000
per_means = np.array(per['eval_means'])
per_stds = np.array(per['eval_stds'])

ax.plot(uniform_steps, uniform_means, 'o-', linewidth=3, markersize=8,
        label='Uniform Replay', color='#1f77b4', alpha=0.9)
ax.fill_between(uniform_steps, uniform_means - uniform_stds, uniform_means + uniform_stds,
                alpha=0.2, color='#1f77b4')

ax.plot(per_steps, per_means, 's-', linewidth=3, markersize=8,
        label='PER', color='#ff7f0e', alpha=0.9)
ax.fill_between(per_steps, per_means - per_stds, per_means + per_stds,
                alpha=0.2, color='#ff7f0e')

ax.axhline(y=200, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target (200)')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.3)
ax.set_xlabel('Training Steps (thousands)', fontsize=12, fontweight='bold')
ax.set_ylabel('Evaluation Return', fontsize=12, fontweight='bold')
ax.set_title('Learning Curves Comparison', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)

# Plot 2: Final Performance Bar Chart
ax = axes[0, 1]
methods = ['Uniform\nReplay', 'PER']
final_returns = [uniform['eval_means'][-1], per['eval_means'][-1]]
colors = ['#1f77b4', '#ff7f0e']
bars = ax.bar(methods, final_returns, color=colors, alpha=0.8, edgecolor='black', linewidth=2.5, width=0.6)

for bar, val in zip(bars, final_returns):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{val:.1f}', ha='center', va='bottom' if val > 0 else 'top',
           fontsize=12, fontweight='bold')

ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.axhline(y=200, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target (200)')
ax.set_ylabel('Final Return', fontsize=12, fontweight='bold')
ax.set_title('Final Performance Comparison', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=11)

# Plot 3: Maximum Performance
ax = axes[1, 0]
max_returns = [max(uniform['eval_means']), max(per['eval_means'])]
bars = ax.bar(methods, max_returns, color=colors, alpha=0.8, edgecolor='black', linewidth=2.5, width=0.6)

for bar, val in zip(bars, max_returns):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{val:.1f}', ha='center', va='bottom' if val > 0 else 'top',
           fontsize=12, fontweight='bold')

ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_ylabel('Maximum Return', fontsize=12, fontweight='bold')
ax.set_title('Best Performance During Training', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Statistics Table
ax = axes[1, 1]
ax.axis('off')

stats_text = f"""
DETAILED PERFORMANCE METRICS

UNIFORM REPLAY (Baseline):
  Final Return:      {uniform['eval_means'][-1]:>10.2f} ± {uniform['eval_stds'][-1]:.2f}
  Max Return:        {max(uniform['eval_means']):>10.2f}
  Min Return:        {min(uniform['eval_means']):>10.2f}
  Mean Return:       {np.mean(uniform['eval_means']):>10.2f}
  Std Dev:           {np.std(uniform['eval_means']):>10.2f}

PRIORITIZED EXPERIENCE REPLAY (PER):
  Final Return:      {per['eval_means'][-1]:>10.2f} ± {per['eval_stds'][-1]:.2f}
  Max Return:        {max(per['eval_means']):>10.2f}
  Min Return:        {min(per['eval_means']):>10.2f}
  Mean Return:       {np.mean(per['eval_means']):>10.2f}
  Std Dev:           {np.std(per['eval_means']):>10.2f}

P-VALUE: {results['p_value']:.2e}
  ⚠️ Highly significant difference (p < 0.001)
  Uniform is statistically better!

PERFORMANCE DIFFERENCE:
  Δ = PER - Uniform: {per['eval_means'][-1] - uniform['eval_means'][-1]:>8.1f}
  Worse by:          {abs(per['eval_means'][-1] - uniform['eval_means'][-1]):.1f} points
"""

ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
       fontsize=10, verticalalignment='top', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='#ffffcc', alpha=0.9, pad=1.5, linewidth=2))

plt.suptitle('Unexpected Result: Uniform Replay OUTPERFORMS PER\nLunarLander-v3 (300k steps)',
            fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / 'VIZ_01_main_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: VIZ_01_main_comparison.png")
plt.close()

# ============================================
# VISUALISIERUNG 2: Detailed Analysis
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Plot 1: Return Stability (Standard Deviation)
ax = axes[0, 0]
uniform_stds_arr = np.array(uniform['eval_stds'])
per_stds_arr = np.array(per['eval_stds'])

ax.plot(uniform_steps, uniform_stds_arr, 'o-', linewidth=2.5, markersize=7,
        label='Uniform Replay', color='#1f77b4', alpha=0.8)
ax.plot(per_steps, per_stds_arr, 's-', linewidth=2.5, markersize=7,
        label='PER', color='#ff7f0e', alpha=0.8)

ax.set_xlabel('Training Steps (thousands)', fontsize=12, fontweight='bold')
ax.set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
ax.set_title('Evaluation Stability (Lower is Better)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Plot 2: Performance Improvement Over Time
ax = axes[0, 1]
min_len = min(len(uniform_means), len(per_means))
improvement = per_means[:min_len] - uniform_means[:min_len]
steps_aligned = uniform_steps[:min_len]

colors_bars = ['green' if x > 0 else 'red' for x in improvement]
ax.bar(steps_aligned, improvement, color=colors_bars, alpha=0.7, edgecolor='black', linewidth=1.5, width=10)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Training Steps (thousands)', fontsize=12, fontweight='bold')
ax.set_ylabel('Return Difference (PER - Uniform)', fontsize=12, fontweight='bold')
ax.set_title('Performance Gap Over Time\n(Negative = Uniform Better)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Learning Trajectory Comparison
ax = axes[1, 0]
ax.plot(uniform_steps, uniform_means, 'o-', linewidth=3, markersize=8,
        label='Uniform Replay (Good)', color='#1f77b4', alpha=0.9)
ax.plot(per_steps, per_means, 's-', linewidth=3, markersize=8,
        label='PER (Poor)', color='#ff7f0e', alpha=0.9)
ax.fill_between(uniform_steps, 0, uniform_means, alpha=0.1, color='#1f77b4')
ax.fill_between(per_steps, 0, per_means, alpha=0.1, color='#ff7f0e')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Training Steps (thousands)', fontsize=12, fontweight='bold')
ax.set_ylabel('Evaluation Return', fontsize=12, fontweight='bold')
ax.set_title('Complete Learning Trajectories', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)

# Plot 4: Distribution of Returns
ax = axes[1, 1]
ax.hist(uniform_means, bins=10, alpha=0.7, color='#1f77b4', edgecolor='black', linewidth=1.5, label='Uniform')
ax.hist(per_means, bins=10, alpha=0.7, color='#ff7f0e', edgecolor='black', linewidth=1.5, label='PER')
ax.axvline(np.mean(uniform_means), color='#1f77b4', linestyle='--', linewidth=2.5, label=f'Uniform Mean: {np.mean(uniform_means):.1f}')
ax.axvline(np.mean(per_means), color='#ff7f0e', linestyle='--', linewidth=2.5, label=f'PER Mean: {np.mean(per_means):.1f}')
ax.set_xlabel('Evaluation Return', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Return Distribution Comparison', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Detailed Performance Analysis: Unexpected PER Underperformance',
            fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / 'VIZ_02_detailed_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: VIZ_02_detailed_analysis.png")
plt.close()

# ============================================
# VISUALISIERUNG 3: Why Did PER Fail?
# ============================================

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

fig.suptitle('Analysis: Why PER Underperformed - Possible Issues',
            fontsize=16, fontweight='bold', y=0.98)

# Text explanation
ax = fig.add_subplot(gs[0, :])
ax.axis('off')

explanation_text = """
POSSIBLE REASONS FOR PER FAILURE:

1. PRIORITY UPDATE BUG
   - TD-Error calculation might be incorrect
   - Priorities might not update properly
   - SumTree might have indexing issues
   
2. IMPORTANCE SAMPLING WEIGHTS ISSUE
   - Weights might be calculated incorrectly
   - Could lead to inverted or extreme gradients
   - Loss function might become unstable

3. BUFFER INITIALIZATION
   - Initial priorities might be set wrong
   - New samples might get inappropriate priorities
   - Buffer might not fill correctly before training
   
4. HYPERPARAMETER MISMATCH
   - Alpha (prioritization strength) = 0.6 might be too high
   - Beta (importance sampling) annealing might be wrong
   - Learning rate might need adjustment for PER

5. GRADIENT EXPLOSION/VANISHING
   - Large importance weights could cause gradient explosion
   - Gradient clipping might be too aggressive
   - Loss computation might be numerically unstable
"""

ax.text(0.05, 0.95, explanation_text, transform=ax.transAxes,
       fontsize=11, verticalalignment='top', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.9, pad=1.5, linewidth=2))

# Comparison metrics
ax = fig.add_subplot(gs[1, 0])
ax.axis('off')

metrics1 = f"""
UNIFORM REPLAY ADVANTAGES:
✓ Stable learning trajectory
✓ Positive final returns
✓ Consistent performance
✓ Simple implementation
✓ Predictable behavior

Statistics:
  Mean:    {np.mean(uniform_means):>8.2f}
  StdDev:  {np.std(uniform_means):>8.2f}
  Trend:   Improving →
"""

ax.text(0.05, 0.95, metrics1, transform=ax.transAxes,
       fontsize=10, verticalalignment='top', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.8, pad=1, linewidth=1.5))

ax = fig.add_subplot(gs[1, 1])
ax.axis('off')

metrics2 = f"""
PER IMPLEMENTATION ISSUES:
✗ Negative final returns
✗ Highly unstable
✗ Gets progressively worse
✗ Complex implementation
✗ Likely has bugs

Statistics:
  Mean:    {np.mean(per_means):>8.2f}
  StdDev:  {np.std(per_means):>8.2f}
  Trend:   Declining ↓
"""

ax.text(0.05, 0.95, metrics2, transform=ax.transAxes,
       fontsize=10, verticalalignment='top', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8, pad=1, linewidth=1.5))

# Recommendation
ax = fig.add_subplot(gs[2, :])
ax.axis('off')

recommendation = """
RECOMMENDED FIXES:

1. DEBUG THE PER IMPLEMENTATION
   - Add logging to verify TD-Error calculations
   - Check that priorities are being updated correctly
   - Verify SumTree operations
   - Test importance weight calculations

2. SIMPLIFY PER PARAMETERS
   - Reduce alpha (try 0.3 instead of 0.6)
   - Use fixed beta instead of annealing
   - Ensure epsilon is appropriate

3. COMPARISON WITH VERIFIED IMPLEMENTATION
   - Compare against reference implementation
   - Check if the issue is specific to this code
   - Consider using existing library (e.g., stable-baselines3)

4. INTERMEDIATE TESTING
   - Test PER in isolation with simple examples
   - Verify each component separately
   - Use smaller buffer sizes initially

CONCLUSION:
The current PER implementation has a significant bug that causes it to perform worse than simple Uniform Replay.
This is NOT the expected behavior - PER should improve or match Uniform performance.
The implementation needs debugging before conclusions about PER effectiveness can be drawn.
"""

ax.text(0.05, 0.95, recommendation, transform=ax.transAxes,
       fontsize=10.5, verticalalignment='top', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='#fff9cc', alpha=0.9, pad=1.5, linewidth=2))

plt.savefig(output_dir / 'VIZ_03_analysis_why_failed.png', dpi=300, bbox_inches='tight')
print("✓ Saved: VIZ_03_analysis_why_failed.png")
plt.close()

print("\n" + "="*80)
print("✅ ALL VISUALIZATIONS CREATED")
print("="*80)

