#!/usr/bin/env python3
"""
Add comprehensive comparison analysis and results table to the notebook
"""
import json
import sys

notebook_path = '/home/isc-den/cas-artificial-intelligence/16_project_teil_b/Agent/Project_Part_2_Final_Architecture.ipynb'

comparison_code = '''# ============================================================
# COMPREHENSIVE STRATEGY COMPARISON & ANALYSIS
# ============================================================

print("\\n" + "="*80)
print("STRATEGY COMPARISON ANALYSIS - Does Forecast Improve RL Performance?")
print("="*80)

# Ensure equity is a numpy array
if not isinstance(equity, np.ndarray):
    equity = np.array(equity)

# Calculate metrics for RL Agent Strategy
rl_return = (equity[-1] - INITIAL_EQUITY) / INITIAL_EQUITY
rl_returns = np.diff(equity) / equity[:-1]
rl_sharpe = (np.mean(rl_returns) / np.std(rl_returns) * np.sqrt(252)) if np.std(rl_returns) > 0 else 0.0
rl_vol = np.std(rl_returns) * np.sqrt(252)
rl_maxdd = 1.0 - np.min(equity) / np.max(equity[:np.argmax(np.maximum.accumulate(equity) - equity)])

# Calculate metrics for Forecast-Only Strategy (already computed)
forecast_vol = np.std(forecast_only_returns) * np.sqrt(252) if len(forecast_only_returns) > 0 else 0.0

# Calculate turnover (position changes)
rl_turnover = np.sum(np.abs(np.diff(pos_hist))) / len(pos_hist) if len(pos_hist) > 0 else 0.0
forecast_turnover = np.sum(np.abs(np.diff(forecast_only_positions))) / len(forecast_only_positions)

# Create comparison dataframe
comparison_data = {
    'Strategy': ['Forecast-Only', 'RL Agent (with Forecast)'],
    'Final Equity': [f"${forecast_only_equity[-1]:,.0f}", f"${equity[-1]:,.0f}"],
    'Total Return': [f"{forecast_only_return*100:.2f}%", f"{rl_return*100:.2f}%"],
    'Sharpe Ratio': [f"{forecast_only_sharpe:.4f}", f"{rl_sharpe:.4f}"],
    'Volatility (Annual)': [f"{forecast_vol*100:.2f}%", f"{rl_vol*100:.2f}%"],
    'Max Drawdown': [f"{forecast_only_maxdd*100:.2f}%", f"{rl_maxdd*100:.2f}%"],
    'Avg Turnover': [f"{forecast_turnover:.4f}", f"{rl_turnover:.4f}"],
}

df_comparison = pd.DataFrame(comparison_data)

print("\\n📊 PERFORMANCE METRICS COMPARISON\\n")
print(df_comparison.to_string(index=False))

# Calculate improvement metrics
print("\\n" + "="*80)
print("FORECAST IMPACT ANALYSIS")
print("="*80)

return_diff = rl_return - forecast_only_return
return_improvement = (return_diff / abs(forecast_only_return)) * 100 if forecast_only_return != 0 else 0

sharpe_diff = rl_sharpe - forecast_only_sharpe
sharpe_improvement = (sharpe_diff / abs(forecast_only_sharpe)) * 100 if forecast_only_sharpe != 0 else 0

print(f"\\n📈 Return Comparison:")
print(f"  Forecast-Only Return:     {forecast_only_return*100:>8.2f}%")
print(f"  RL Agent Return:          {rl_return*100:>8.2f}%")
print(f"  Difference (RL - Forecast): {return_diff*100:>8.2f}%")
if return_diff > 0:
    print(f"  ✓ RL Agent OUTPERFORMED by {abs(return_improvement):.1f}%")
else:
    print(f"  ✗ Forecast-Only OUTPERFORMED by {abs(return_improvement):.1f}%")

print(f"\\n📊 Risk-Adjusted Return (Sharpe Ratio):")
print(f"  Forecast-Only Sharpe:     {forecast_only_sharpe:>8.4f}")
print(f"  RL Agent Sharpe:          {rl_sharpe:>8.4f}")
print(f"  Difference (RL - Forecast): {sharpe_diff:>8.4f}")
if sharpe_diff > 0:
    print(f"  ✓ RL Agent has BETTER risk-adjusted returns")
else:
    print(f"  ✗ Forecast-Only has BETTER risk-adjusted returns")

print(f"\\n🎯 Volatility & Risk:")
print(f"  Forecast-Only Volatility: {forecast_vol*100:>8.2f}%")
print(f"  RL Agent Volatility:      {rl_vol*100:>8.2f}%")
print(f"  Forecast-Only Max DD:     {forecast_only_maxdd*100:>8.2f}%")
print(f"  RL Agent Max DD:          {rl_maxdd*100:>8.2f}%")

print(f"\\n🔄 Trading Activity:")
print(f"  Forecast-Only Turnover:   {forecast_turnover:>8.4f}")
print(f"  RL Agent Turnover:        {rl_turnover:>8.4f}")

# Statistical significance test (Sharpe ratio improvement)
print("\\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

if rl_return > forecast_only_return:
    print(f"\\n✓ YES - Forecast IMPROVES RL performance!")
    print(f"  The RL agent trained WITH forecast achieves {return_improvement:.1f}% higher returns")
else:
    print(f"\\n✗ NO - Forecast does NOT improve RL performance")
    print(f"  The simple forecast-only strategy outperforms the RL agent by {abs(return_improvement):.1f}%")

if rl_sharpe > forecast_only_sharpe:
    print(f"\\n✓ RL agent has better risk-adjusted returns (Sharpe: {rl_sharpe:.4f} vs {forecast_only_sharpe:.4f})")
else:
    print(f"\\n✗ Forecast-only strategy has better risk-adjusted returns (Sharpe: {forecast_only_sharpe:.4f} vs {rl_sharpe:.4f})")

if rl_maxdd < forecast_only_maxdd:
    print(f"\\n✓ RL agent manages downside risk better (Max DD: {rl_maxdd*100:.2f}% vs {forecast_only_maxdd*100:.2f}%)")
else:
    print(f"\\n✗ Forecast-only strategy has smaller max drawdown (Max DD: {forecast_only_maxdd*100:.2f}% vs {rl_maxdd*100:.2f}%)")

print("\\n" + "="*80)

# Export results for report
results_summary = {
    'forecast_only_return': float(forecast_only_return),
    'forecast_only_sharpe': float(forecast_only_sharpe),
    'forecast_only_vol': float(forecast_vol),
    'forecast_only_maxdd': float(forecast_only_maxdd),
    'forecast_only_turnover': float(forecast_turnover),
    'rl_return': float(rl_return),
    'rl_sharpe': float(rl_sharpe),
    'rl_vol': float(rl_vol),
    'rl_maxdd': float(rl_maxdd),
    'rl_turnover': float(rl_turnover),
    'return_improvement_pct': float(return_improvement),
    'sharpe_improvement': float(sharpe_diff),
}

print("\\n📋 Results saved for technical report generation")
'''

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Find the last visualization cell and add the comparison after it
    last_viz_idx = None
    for idx in range(len(nb['cells']) - 1, -1, -1):
        cell = nb['cells'][idx]
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'plt.tight_layout()' in source or 'plt.show()' in source:
                last_viz_idx = idx
                break
    
    if last_viz_idx is not None:
        print(f"Found last visualization cell at index {last_viz_idx}", file=sys.stderr)
        
        # Create new cell with comparison code
        new_cell = {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [line if line.endswith('\n') else line + '\n' for line in comparison_code.split('\n')[:-1]]
        }
        
        # Insert after the last visualization
        nb['cells'].insert(last_viz_idx + 1, new_cell)
        print(f"Inserted comparison analysis cell after index {last_viz_idx}", file=sys.stderr)
    else:
        print("WARNING: Could not find visualization cell. Appending to end.", file=sys.stderr)
        new_cell = {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [line if line.endswith('\n') else line + '\n' for line in comparison_code.split('\n')[:-1]]
        }
        nb['cells'].append(new_cell)
    
    # Save the notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print("\n✓ Successfully added comprehensive comparison analysis!", file=sys.stderr)
    
except Exception as e:
    print(f"✗ Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

