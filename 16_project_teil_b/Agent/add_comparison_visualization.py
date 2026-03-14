#!/usr/bin/env python3
"""
Add a comprehensive comparison visualization showing all three strategies
"""
import json
import sys

notebook_path = '/home/isc-den/cas-artificial-intelligence/16_project_teil_b/Agent/Project_Part_2_Final_Architecture.ipynb'

viz_comparison_code = '''# ============================================================
# VISUALIZATION: Three-Strategy Comparison
# ============================================================

print("\\nGenerating comprehensive strategy comparison visualizations...")

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

dates = np.arange(len(forecast_only_equity))

# Panel 1: Equity Curves Comparison
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(dates, forecast_only_equity, label='Forecast-Only Strategy', 
         linewidth=2.5, color='#C73E1D', alpha=0.8)
ax1.plot(dates, equity_baseline, label='PPO (Without Forecast)', 
         linewidth=2.5, color='#2CA02C', alpha=0.8)
ax1.plot(dates, equity, label='PPO (With Forecast)', 
         linewidth=2.5, color='#1f77b4', alpha=0.8)
ax1.axhline(y=INITIAL_EQUITY, color='black', linestyle='--', alpha=0.3, label='Initial Equity')
ax1.set_title('Equity Curves: Three Strategy Comparison', fontsize=14, fontweight='bold')
ax1.set_xlabel('Trading Days')
ax1.set_ylabel('Equity Value ($)')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

# Panel 2: Cumulative Returns Comparison
ax2 = fig.add_subplot(gs[1, 0])
forecast_cumret = (forecast_only_equity / INITIAL_EQUITY - 1) * 100
baseline_cumret = (equity_baseline / INITIAL_EQUITY - 1) * 100
rl_cumret = (equity / INITIAL_EQUITY - 1) * 100

ax2.plot(dates, forecast_cumret, label='Forecast-Only', linewidth=2, color='#C73E1D')
ax2.plot(dates, baseline_cumret, label='PPO (No Forecast)', linewidth=2, color='#2CA02C')
ax2.plot(dates, rl_cumret, label='PPO (With Forecast)', linewidth=2, color='#1f77b4')
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.2)
ax2.set_title('Cumulative Returns (%)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Trading Days')
ax2.set_ylabel('Return (%)')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Panel 3: Daily Returns Distribution
ax3 = fig.add_subplot(gs[1, 1])
ax3.hist(forecast_only_returns * 100, bins=30, alpha=0.5, label='Forecast-Only', color='#C73E1D')
ax3.hist(baseline_returns * 100, bins=30, alpha=0.5, label='PPO (No Forecast)', color='#2CA02C')
ax3.hist(rl_returns * 100, bins=30, alpha=0.5, label='PPO (With Forecast)', color='#1f77b4')
ax3.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
ax3.set_xlabel('Daily Return (%)')
ax3.set_ylabel('Frequency')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

# Panel 4: Metrics Comparison (Bar Chart)
ax4 = fig.add_subplot(gs[2, 0])
strategies_list = ['Forecast-Only', 'PPO (No Forecast)', 'PPO (With Forecast)']
returns_list = [
    strategies['Forecast-Only']['total_return'] * 100,
    strategies['PPO (No Forecast)']['total_return'] * 100,
    strategies['PPO (With Forecast)']['total_return'] * 100
]
colors_list = ['#C73E1D', '#2CA02C', '#1f77b4']

bars = ax4.bar(strategies_list, returns_list, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.5)
ax4.set_title('Total Return Comparison', fontsize=12, fontweight='bold')
ax4.set_ylabel('Total Return (%)')
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Panel 5: Sharpe Ratio Comparison
ax5 = fig.add_subplot(gs[2, 1])
sharpe_list = [
    strategies['Forecast-Only']['sharpe'],
    strategies['PPO (No Forecast)']['sharpe'],
    strategies['PPO (With Forecast)']['sharpe']
]

bars = ax5.bar(strategies_list, sharpe_list, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.5)
ax5.set_title('Sharpe Ratio Comparison', fontsize=12, fontweight='bold')
ax5.set_ylabel('Sharpe Ratio')
ax5.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Overall title
fig.suptitle('Comprehensive Trading Strategy Comparison: Does Forecast Improve RL Performance?', 
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
plt.show()

print("✓ Comparison visualization complete!")
'''

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Find the COMPREHENSIVE STRATEGY COMPARISON cell and add visualization after it
    comparison_idx = None
    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'COMPREHENSIVE STRATEGY COMPARISON' in source and 'THREE-STRATEGY' in source:
                comparison_idx = idx
                break
    
    if comparison_idx is not None:
        new_cell = {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [line if line.endswith('\n') else line + '\n' for line in viz_comparison_code.split('\n')[:-1]]
        }
        
        nb['cells'].insert(comparison_idx + 1, new_cell)
        print(f"Inserted comparison visualization cell after index {comparison_idx}", file=sys.stderr)
    else:
        print("WARNING: Could not find comparison cell. Appending visualization.", file=sys.stderr)
        new_cell = {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [line if line.endswith('\n') else line + '\n' for line in viz_comparison_code.split('\n')[:-1]]
        }
        nb['cells'].append(new_cell)
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print("\n✓ Successfully added comparison visualization!", file=sys.stderr)
    
except Exception as e:
    print(f"✗ Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

