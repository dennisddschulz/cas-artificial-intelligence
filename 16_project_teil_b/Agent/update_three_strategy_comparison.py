#!/usr/bin/env python3
"""
Update the comparison analysis to include PPO baseline without forecast
"""
import json
import sys

notebook_path = '/home/isc-den/cas-artificial-intelligence/16_project_teil_b/Agent/Project_Part_2_Final_Architecture.ipynb'

enhanced_comparison_code = '''# ============================================================
# COMPREHENSIVE THREE-STRATEGY COMPARISON
# ============================================================
# Compare: 1) Forecast-Only, 2) PPO without Forecast, 3) PPO with Forecast

print("\\n" + "="*90)
print("THREE-STRATEGY COMPARISON: Does Forecast Improve RL Performance?")
print("="*90)

# Ensure all equity arrays are numpy arrays
if not isinstance(equity, np.ndarray):
    equity = np.array(equity)
if not isinstance(equity_baseline, np.ndarray):
    equity_baseline = np.array(equity_baseline)
if not isinstance(forecast_only_equity, np.ndarray):
    forecast_only_equity = np.array(forecast_only_equity)

# Calculate metrics for all three strategies
strategies = {}

# 1. Forecast-Only Strategy
strategies['Forecast-Only'] = {
    'equity': forecast_only_equity,
    'returns': forecast_only_returns,
    'positions': forecast_only_positions,
    'final_equity': forecast_only_equity[-1],
    'total_return': forecast_only_return,
    'sharpe': forecast_only_sharpe,
    'volatility': np.std(forecast_only_returns) * np.sqrt(252) if len(forecast_only_returns) > 0 else 0.0,
    'maxdd': forecast_only_maxdd,
    'turnover': np.sum(np.abs(np.diff(forecast_only_positions))) / len(forecast_only_positions),
    'color': '#C73E1D'  # Orange
}

# 2. PPO WITHOUT Forecast (Baseline)
baseline_returns = np.diff(equity_baseline) / equity_baseline[:-1]
strategies['PPO (No Forecast)'] = {
    'equity': equity_baseline,
    'returns': baseline_returns,
    'positions': pos_baseline,
    'final_equity': equity_baseline[-1],
    'total_return': (equity_baseline[-1] - INITIAL_EQUITY) / INITIAL_EQUITY,
    'sharpe': (np.mean(baseline_returns) / np.std(baseline_returns) * np.sqrt(252)) if np.std(baseline_returns) > 0 else 0.0,
    'volatility': np.std(baseline_returns) * np.sqrt(252),
    'maxdd': 1.0 - np.min(equity_baseline) / np.max(equity_baseline[:np.argmax(np.maximum.accumulate(equity_baseline) - equity_baseline)]),
    'turnover': np.sum(np.abs(np.diff(pos_baseline))) / len(pos_baseline),
    'color': '#2CA02C'  # Green
}

# 3. PPO WITH Forecast
rl_returns = np.diff(equity) / equity[:-1]
strategies['PPO (With Forecast)'] = {
    'equity': equity,
    'returns': rl_returns,
    'positions': pos_hist,
    'final_equity': equity[-1],
    'total_return': (equity[-1] - INITIAL_EQUITY) / INITIAL_EQUITY,
    'sharpe': (np.mean(rl_returns) / np.std(rl_returns) * np.sqrt(252)) if np.std(rl_returns) > 0 else 0.0,
    'volatility': np.std(rl_returns) * np.sqrt(252),
    'maxdd': 1.0 - np.min(equity) / np.max(equity[:np.argmax(np.maximum.accumulate(equity) - equity)]),
    'turnover': np.sum(np.abs(np.diff(pos_hist))) / len(pos_hist),
    'color': '#1f77b4'  # Blue
}

# Create comparison dataframe
comparison_rows = []
for strategy_name, metrics in strategies.items():
    comparison_rows.append({
        'Strategy': strategy_name,
        'Final Equity': f"${metrics['final_equity']:,.0f}",
        'Total Return': f"{metrics['total_return']*100:.2f}%",
        'Sharpe Ratio': f"{metrics['sharpe']:.4f}",
        'Volatility': f"{metrics['volatility']*100:.2f}%",
        'Max Drawdown': f"{metrics['maxdd']*100:.2f}%",
        'Turnover': f"{metrics['turnover']:.4f}",
    })

df_comparison = pd.DataFrame(comparison_rows)

print("\\n📊 STRATEGY PERFORMANCE COMPARISON\\n")
print(df_comparison.to_string(index=False))

# Detailed analysis
print("\\n" + "="*90)
print("DETAILED ANALYSIS - Impact of Forecast on RL Performance")
print("="*90)

# Compare PPO with vs without forecast
print("\\n🎯 FORECAST IMPACT (PPO with Forecast vs PPO without Forecast)\\n")

for_return = strategies['PPO (With Forecast)']['total_return']
no_for_return = strategies['PPO (No Forecast)']['total_return']
return_diff = for_return - no_for_return
return_improvement = (return_diff / abs(no_for_return)) * 100 if no_for_return != 0 else 0

for_sharpe = strategies['PPO (With Forecast)']['sharpe']
no_for_sharpe = strategies['PPO (No Forecast)']['sharpe']

print(f"Return Improvement from Forecast:")
print(f"  PPO with Forecast:     {for_return*100:>8.2f}%")
print(f"  PPO without Forecast:  {no_for_return*100:>8.2f}%")
print(f"  Difference:            {return_diff*100:>8.2f}%")
if return_diff > 0:
    print(f"  ✓ Forecast IMPROVES return by {abs(return_improvement):.1f}%")
else:
    print(f"  ✗ Forecast HURTS return by {abs(return_improvement):.1f}%")

print(f"\\nRisk-Adjusted Performance (Sharpe):")
print(f"  PPO with Forecast:     {for_sharpe:>8.4f}")
print(f"  PPO without Forecast:  {no_for_sharpe:>8.4f}")
if for_sharpe > no_for_sharpe:
    print(f"  ✓ Forecast IMPROVES Sharpe ratio by {for_sharpe - no_for_sharpe:.4f}")
else:
    print(f"  ✗ Forecast HURTS Sharpe ratio by {no_for_sharpe - for_sharpe:.4f}")

# Compare best PPO vs Forecast-Only
print("\\n🏆 PPO vs FORECAST-ONLY STRATEGY\\n")

best_rl = strategies['PPO (With Forecast)']
best_rl_return = best_rl['total_return']
forecast_return = strategies['Forecast-Only']['total_return']
rl_vs_forecast = (best_rl_return - forecast_return) / abs(forecast_return) * 100

print(f"Best PPO (with Forecast): {best_rl_return*100:>8.2f}%")
print(f"Forecast-Only Strategy:   {forecast_return*100:>8.2f}%")
if best_rl_return > forecast_return:
    print(f"✓ PPO OUTPERFORMS forecast-only by {rl_vs_forecast:.1f}%")
else:
    print(f"✗ Forecast-only OUTPERFORMS PPO by {abs(rl_vs_forecast):.1f}%")

# Ranking
print("\\n📈 STRATEGY RANKING (by Total Return)\\n")
ranked = sorted(strategies.items(), key=lambda x: x[1]['total_return'], reverse=True)
for rank, (name, metrics) in enumerate(ranked, 1):
    print(f"  {rank}. {name:25s} → {metrics['total_return']*100:>8.2f}% return")

print("\\n📊 STRATEGY RANKING (by Sharpe Ratio)\\n")
ranked_sharpe = sorted(strategies.items(), key=lambda x: x[1]['sharpe'], reverse=True)
for rank, (name, metrics) in enumerate(ranked_sharpe, 1):
    print(f"  {rank}. {name:25s} → {metrics['sharpe']:>8.4f} Sharpe")

# Key findings
print("\\n" + "="*90)
print("KEY FINDINGS")
print("="*90)

if for_return > no_for_return:
    print(f"\\n✓ YES - Forecast IMPROVES RL Agent Performance!")
    print(f"  → Adding LSTM forecast signal to state increases returns")
    print(f"  → The agent learns to effectively utilize price movement predictions")
else:
    print(f"\\n✗ NO - Forecast does NOT improve RL Agent Performance")
    print(f"  → The agent may be overfitting to noisy forecast signals")
    print(f"  → Or the forecast signal may not contain useful information for trading")

if for_sharpe > no_for_sharpe:
    print(f"\\n✓ Forecast improves RISK-ADJUSTED returns (Sharpe)")
    print(f"  → The agent produces more stable profits with forecast information")
else:
    print(f"\\n✗ Forecast increases volatility (worse Sharpe ratio)")
    print(f"  → The forecast may encourage overtrading")

# Comparison with simple baseline
if best_rl_return > forecast_return:
    print(f"\\n✓ PPO (with Forecast) beats the simple Forecast-Only strategy")
    print(f"  → Deep RL successfully exploits additional market dynamics")
else:
    print(f"\\n✗ Simple Forecast-Only strategy outperforms RL")
    print(f"  → The market may not have exploitable patterns beyond forecast signal")
    print(f"  → RL might need: more data, better rewards, or longer training")

print("\\n" + "="*90)
print("SUMMARY FOR TECHNICAL REPORT")
print("="*90)
print(f"""
Experiment: Do LSTM Forecast Signals Improve RL Trading Performance?

Setup:
  - Asset: {df_test['symbol'].iloc[0] if 'symbol' in df_test.columns else 'Unknown'}
  - Period: {len(df_test)} trading days
  - Initial Equity: ${INITIAL_EQUITY:,.0f}

Strategies Tested:
  1. Forecast-Only: Simple rule (long if forecast > 0.5, else short)
  2. PPO Baseline: RL agent without forecast signal
  3. PPO with Forecast: RL agent WITH LSTM forecast in observation

Results:
  Forecast-Only → Return: {strategies['Forecast-Only']['total_return']*100:.2f}%, Sharpe: {strategies['Forecast-Only']['sharpe']:.4f}
  PPO (No Forecast) → Return: {strategies['PPO (No Forecast)']['total_return']*100:.2f}%, Sharpe: {strategies['PPO (No Forecast)']['sharpe']:.4f}
  PPO (With Forecast) → Return: {strategies['PPO (With Forecast)']['total_return']*100:.2f}%, Sharpe: {strategies['PPO (With Forecast)']['sharpe']:.4f}

Conclusion:
  Forecast Improves RL by {return_improvement:.1f}% (return difference)
  Best Strategy: {ranked[0][0]} with {ranked[0][1]['total_return']*100:.2f}% return
""")

print("="*90)
'''

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb['cells']
    
    # Find and replace the existing COMPREHENSIVE STRATEGY COMPARISON cell
    found_idx = None
    for idx, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'COMPREHENSIVE STRATEGY COMPARISON & ANALYSIS' in source:
                found_idx = idx
                break
    
    if found_idx is not None:
        # Replace the cell
        new_cell = {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [line if line.endswith('\n') else line + '\n' for line in enhanced_comparison_code.split('\n')[:-1]]
        }
        nb['cells'][found_idx] = new_cell
        print(f"Updated comparison analysis cell at index {found_idx}", file=sys.stderr)
    else:
        print("WARNING: Could not find existing comparison cell. Appending new one.", file=sys.stderr)
        new_cell = {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [line if line.endswith('\n') else line + '\n' for line in enhanced_comparison_code.split('\n')[:-1]]
        }
        nb['cells'].append(new_cell)
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print("\n✓ Successfully updated comparison analysis with three-strategy comparison!", file=sys.stderr)
    
except Exception as e:
    print(f"✗ Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

