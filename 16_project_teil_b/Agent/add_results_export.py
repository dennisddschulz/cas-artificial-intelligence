#!/usr/bin/env python3
"""
Add a results export cell that saves all metrics for the technical report
"""
import json
import sys

notebook_path = '/home/isc-den/cas-artificial-intelligence/16_project_teil_b/Agent/Project_Part_2_Final_Architecture.ipynb'

export_code = '''# ============================================================
# EXPORT RESULTS FOR TECHNICAL REPORT
# ============================================================

print("\\n" + "="*80)
print("EXPORTING RESULTS FOR TECHNICAL REPORT")
print("="*80)

# Create comprehensive results dictionary
results_export = {
    'experiment': 'Forecast-Augmented RL Trading: Impact of LSTM Predictions on PPO',
    'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'data': {
        'asset': df_test['symbol'].iloc[0] if 'symbol' in df_test.columns else 'Unknown',
        'period_length': len(df_test),
        'initial_equity': INITIAL_EQUITY,
        'test_start_date': str(df_test.index[0]) if hasattr(df_test.index[0], '__str__') else 'N/A',
        'test_end_date': str(df_test.index[-1]) if hasattr(df_test.index[-1], '__str__') else 'N/A',
    },
    'hyperparameters': {
        'trading_fee': FEE,
        'kappa_penalty': KAPPA,
        'max_leverage': LEVERAGE_MAX,
        'lstm_lookback': LOOKBACK,
        'lstm_forecast_horizon': FORECAST_HORIZON,
    },
    'strategies': {
        'Forecast-Only': {
            'description': 'Simple rule: position = 1.0 if forecast_prob > 0.5 else -1.0',
            'final_equity': float(strategies['Forecast-Only']['final_equity']),
            'total_return': float(strategies['Forecast-Only']['total_return']),
            'sharpe_ratio': float(strategies['Forecast-Only']['sharpe']),
            'volatility': float(strategies['Forecast-Only']['volatility']),
            'max_drawdown': float(strategies['Forecast-Only']['maxdd']),
            'average_turnover': float(strategies['Forecast-Only']['turnover']),
            'num_days': len(strategies['Forecast-Only']['equity']),
        },
        'PPO-Baseline': {
            'description': 'PPO agent trained WITHOUT forecast signal (state: position, cash ratio, momentum, volatility)',
            'final_equity': float(strategies['PPO (No Forecast)']['final_equity']),
            'total_return': float(strategies['PPO (No Forecast)']['total_return']),
            'sharpe_ratio': float(strategies['PPO (No Forecast)']['sharpe']),
            'volatility': float(strategies['PPO (No Forecast)']['volatility']),
            'max_drawdown': float(strategies['PPO (No Forecast)']['maxdd']),
            'average_turnover': float(strategies['PPO (No Forecast)']['turnover']),
            'num_days': len(strategies['PPO (No Forecast)']['equity']),
        },
        'PPO-With-Forecast': {
            'description': 'PPO agent trained WITH LSTM forecast signal included in state',
            'final_equity': float(strategies['PPO (With Forecast)']['final_equity']),
            'total_return': float(strategies['PPO (With Forecast)']['total_return']),
            'sharpe_ratio': float(strategies['PPO (With Forecast)']['sharpe']),
            'volatility': float(strategies['PPO (With Forecast)']['volatility']),
            'max_drawdown': float(strategies['PPO (With Forecast)']['maxdd']),
            'average_turnover': float(strategies['PPO (With Forecast)']['turnover']),
            'num_days': len(strategies['PPO (With Forecast)']['equity']),
        },
    },
    'key_findings': {
        'forecast_improves_return': bool(
            strategies['PPO (With Forecast)']['total_return'] > strategies['PPO (No Forecast)']['total_return']
        ),
        'return_improvement_pct': float(
            (strategies['PPO (With Forecast)']['total_return'] - strategies['PPO (No Forecast)']['total_return']) / 
            abs(strategies['PPO (No Forecast)']['total_return']) * 100
            if strategies['PPO (No Forecast)']['total_return'] != 0 else 0
        ),
        'forecast_improves_sharpe': bool(
            strategies['PPO (With Forecast)']['sharpe'] > strategies['PPO (No Forecast)']['sharpe']
        ),
        'sharpe_improvement': float(
            strategies['PPO (With Forecast)']['sharpe'] - strategies['PPO (No Forecast)']['sharpe']
        ),
        'best_strategy': max(strategies.items(), key=lambda x: x[1]['total_return'])[0],
        'best_strategy_return': float(max(strategies.items(), key=lambda x: x[1]['total_return'])[1]['total_return']),
    },
}

# Save to JSON
import json
json_path = 'experiment_results.json'
with open(json_path, 'w') as f:
    json.dump(results_export, f, indent=2)
print(f"✓ Results saved to {json_path}")

# Create a markdown summary for the report
markdown_summary = f'''# Trading Strategy Comparison: Forecast-Augmented RL

## Executive Summary

This experiment investigates whether LSTM price movement forecasts improve Deep Reinforcement Learning (PPO) trading performance.

### Key Question
**Does forecast improve RL performance?**

### Answer
{'✓ YES' if results_export['key_findings']['forecast_improves_return'] else '✗ NO'} - Forecast {'IMPROVES' if results_export['key_findings']['forecast_improves_return'] else 'HURTS'} RL performance by {abs(results_export['key_findings']['return_improvement_pct']):.1f}%

## Experiment Setup

| Parameter | Value |
|-----------|-------|
| Asset | {results_export['data']['asset']} |
| Test Period | {results_export['data']['period_length']} days |
| Initial Equity | ${results_export['data']['initial_equity']:,.0f} |
| Trading Fee | {results_export['hyperparameters']['trading_fee']*100:.02f}% |
| Kappa Penalty | {results_export['hyperparameters']['kappa_penalty']} |
| Max Leverage | {results_export['hyperparameters']['max_leverage']}x |

## Results

### Performance Metrics Comparison

| Strategy | Return | Sharpe Ratio | Volatility | Max Drawdown | Final Equity |
|----------|--------|--------------|------------|--------------|--------------|
| Forecast-Only | {results_export['strategies']['Forecast-Only']['total_return']*100:.2f}% | {results_export['strategies']['Forecast-Only']['sharpe_ratio']:.4f} | {results_export['strategies']['Forecast-Only']['volatility']*100:.2f}% | {results_export['strategies']['Forecast-Only']['max_drawdown']*100:.2f}% | ${results_export['strategies']['Forecast-Only']['final_equity']:,.0f} |
| PPO (No Forecast) | {results_export['strategies']['PPO-Baseline']['total_return']*100:.2f}% | {results_export['strategies']['PPO-Baseline']['sharpe_ratio']:.4f} | {results_export['strategies']['PPO-Baseline']['volatility']*100:.2f}% | {results_export['strategies']['PPO-Baseline']['max_drawdown']*100:.2f}% | ${results_export['strategies']['PPO-Baseline']['final_equity']:,.0f} |
| PPO (With Forecast) | {results_export['strategies']['PPO-With-Forecast']['total_return']*100:.2f}% | {results_export['strategies']['PPO-With-Forecast']['sharpe_ratio']:.4f} | {results_export['strategies']['PPO-With-Forecast']['volatility']*100:.2f}% | {results_export['strategies']['PPO-With-Forecast']['max_drawdown']*100:.2f}% | ${results_export['strategies']['PPO-With-Forecast']['final_equity']:,.0f} |

### Impact of Forecast Information

- **Return Improvement**: {results_export['key_findings']['return_improvement_pct']:.1f}% (PPO with Forecast vs PPO without Forecast)
- **Sharpe Improvement**: {results_export['key_findings']['sharpe_improvement']:.4f} points
- **Best Strategy**: {results_export['key_findings']['best_strategy']} ({results_export['key_findings']['best_strategy_return']*100:.2f}% return)

## Conclusions

{'✓ FORECAST HELPS' if results_export['key_findings']['forecast_improves_return'] else '✗ FORECAST HURTS'} RL performance.

The LSTM price movement forecast {'provides valuable information that the PPO agent can leverage for improved returns' if results_export['key_findings']['forecast_improves_return'] else 'may introduce noise or overconfidence that hurts trading performance'}.

## Recommendations

1. {'✓ Continue using forecast signal in RL state' if results_export['key_findings']['forecast_improves_return'] else '✗ Remove forecast signal from RL state'}
2. {'✓ Forecast complements RL learning' if results_export['key_findings']['forecast_improves_sharpe'] else '✗ Forecast increases risk without commensurate return'}
3. Explore ensemble methods combining all three strategies

---
Generated: {results_export['date']}
'''

md_path = 'experiment_summary.md'
with open(md_path, 'w') as f:
    f.write(markdown_summary)
print(f"✓ Summary report saved to {md_path}")

# Create CSV for easy inclusion in technical report
import pandas as pd

comparison_df = pd.DataFrame([
    {
        'Strategy': 'Forecast-Only',
        'Final Equity': f"${results_export['strategies']['Forecast-Only']['final_equity']:,.0f}",
        'Total Return': f"{results_export['strategies']['Forecast-Only']['total_return']*100:.2f}%",
        'Sharpe Ratio': f"{results_export['strategies']['Forecast-Only']['sharpe_ratio']:.4f}",
        'Volatility': f"{results_export['strategies']['Forecast-Only']['volatility']*100:.2f}%",
        'Max Drawdown': f"{results_export['strategies']['Forecast-Only']['max_drawdown']*100:.2f}%",
        'Turnover': f"{results_export['strategies']['Forecast-Only']['average_turnover']:.4f}",
    },
    {
        'Strategy': 'PPO (Without Forecast)',
        'Final Equity': f"${results_export['strategies']['PPO-Baseline']['final_equity']:,.0f}",
        'Total Return': f"{results_export['strategies']['PPO-Baseline']['total_return']*100:.2f}%",
        'Sharpe Ratio': f"{results_export['strategies']['PPO-Baseline']['sharpe_ratio']:.4f}",
        'Volatility': f"{results_export['strategies']['PPO-Baseline']['volatility']*100:.2f}%",
        'Max Drawdown': f"{results_export['strategies']['PPO-Baseline']['max_drawdown']*100:.2f}%",
        'Turnover': f"{results_export['strategies']['PPO-Baseline']['average_turnover']:.4f}",
    },
    {
        'Strategy': 'PPO (With Forecast)',
        'Final Equity': f"${results_export['strategies']['PPO-With-Forecast']['final_equity']:,.0f}",
        'Total Return': f"{results_export['strategies']['PPO-With-Forecast']['total_return']*100:.2f}%",
        'Sharpe Ratio': f"{results_export['strategies']['PPO-With-Forecast']['sharpe_ratio']:.4f}",
        'Volatility': f"{results_export['strategies']['PPO-With-Forecast']['volatility']*100:.2f}%",
        'Max Drawdown': f"{results_export['strategies']['PPO-With-Forecast']['max_drawdown']*100:.2f}%",
        'Turnover': f"{results_export['strategies']['PPO-With-Forecast']['average_turnover']:.4f}",
    },
])

csv_path = 'results_comparison_table.csv'
comparison_df.to_csv(csv_path, index=False)
print(f"✓ Results table saved to {csv_path}")

print("\\n" + "="*80)
print("EXPORT COMPLETE")
print("="*80)
print(f"""
Files generated:
  1. {json_path} - Complete results in JSON format
  2. {md_path} - Markdown summary for technical report
  3. {csv_path} - Comparison table for report inclusion
  
Use these files as the basis for your technical report (10-15 pages).
""")
'''

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Add export cell at the very end
    new_cell = {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [line if line.endswith('\n') else line + '\n' for line in export_code.split('\n')[:-1]]
    }
    
    nb['cells'].append(new_cell)
    print("Appended results export cell to end of notebook", file=sys.stderr)
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print("\n✓ Successfully added results export cell!", file=sys.stderr)
    sys.exit(0)
    
except Exception as e:
    print(f"✗ Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

