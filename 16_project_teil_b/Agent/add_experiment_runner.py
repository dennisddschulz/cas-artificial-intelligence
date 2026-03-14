#!/usr/bin/env python3
"""
Create a comprehensive experiment runner cell for the notebook
"""
import json

notebook_path = '/home/isc-den/cas-artificial-intelligence/16_project_teil_b/Agent/Project_Part_2_Final_Architecture.ipynb'

experiment_cell_code = '''# ============================================================
# COMPREHENSIVE EXPERIMENT RUNNER WITH W&B TRACKING
# ============================================================
# This cell orchestrates all experiments with proper W&B logging

import sys
sys.path.insert(0, '/home/isc-den/cas-artificial-intelligence/16_project_teil_b/Agent')

from experiment_framework import (
    ExperimentConfig, ExperimentTracker, MetricsCalculator, 
    TradingEnv_2, ExperimentRunner
)

print("\\n" + "="*80)
print("LAUNCHING COMPREHENSIVE EXPERIMENT SUITE")
print("="*80)

# Initialize experiment runner
config = ExperimentConfig()
runner = ExperimentRunner(config=config)

# Prepare data
print("\\nPreparing data...")
print(f"  Train/Test split: {config.TEST_SPLIT*100:.0f}% / {(1-config.TEST_SPLIT)*100:.0f}%")
print(f"  Trading period: {len(df_test)} days")
print(f"  Initial equity: ${config.INITIAL_EQUITY:,.0f}")

# Run all experiments
results = runner.run_all_experiments(df_test, forecast_probs_aligned)

print("\\n" + "="*80)
print("EXPERIMENT RESULTS SUMMARY")
print("="*80)

# Create summary table
summary_data = []
for strategy, metrics in sorted(results.items()):
    summary_data.append({
        'Strategy': strategy,
        'Final Equity': f"${metrics['final_equity']:,.0f}",
        'Return': f"{metrics['total_return']*100:.2f}%",
        'Sharpe': f"{metrics['sharpe_ratio']:.4f}",
        'Volatility': f"{metrics['volatility']*100:.2f}%",
        'Max DD': f"{metrics['max_drawdown']*100:.2f}%",
        'Turnover': f"{metrics['turnover']:.4f}",
    })

summary_df = pd.DataFrame(summary_data)
print("\\n" + summary_df.to_string(index=False))

# Analyze results
print("\\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

returns_dict = {k: v['total_return'] for k, v in results.items()}
best_strategy = max(returns_dict, key=returns_dict.get)
best_return = returns_dict[best_strategy]

print(f"\\n✓ Best performing strategy: {best_strategy}")
print(f"✓ Best return: {best_return*100:.2f}%")

# Forecast impact analysis
if 'ppo-with-forecast' in results and 'ppo-without-forecast' in results:
    with_forecast_return = results['ppo-with-forecast']['total_return']
    without_forecast_return = results['ppo-without-forecast']['total_return']
    improvement = (with_forecast_return - without_forecast_return) / abs(without_forecast_return) * 100
    
    print(f"\\n📊 FORECAST IMPACT ANALYSIS:")
    print(f"  PPO with forecast:    {with_forecast_return*100:.2f}%")
    print(f"  PPO without forecast: {without_forecast_return*100:.2f}%")
    print(f"  Improvement:          {improvement:+.1f}%")
    
    if improvement > 0:
        print(f"  ✓ YES - Forecast IMPROVES RL performance!")
    else:
        print(f"  ✗ NO - Forecast does NOT improve RL performance")

# Reward definition impact
print(f"\\n💰 REWARD DEFINITION COMPARISON:")
reward_results = {k: v for k, v in results.items() if 'reward' in k.lower()}
if reward_results:
    for strategy, metrics in sorted(reward_results.items(), 
                                    key=lambda x: x[1]['total_return'], 
                                    reverse=True):
        print(f"  {strategy:30s}: {metrics['total_return']*100:>8.2f}%")

print("\\n" + "="*80)
print("✓ EXPERIMENT SUITE COMPLETE")
print("="*80)
print("\\nResults logged to W&B (offline mode)")
print("Check ./wandb/ directory for experiment logs")
'''

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Add the experiment runner cell at a good position
    # We'll add it right after the main analysis sections
    new_cell = {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [line if line.endswith('\n') else line + '\n' for line in experiment_cell_code.split('\n')[:-1]]
    }
    
    nb['cells'].append(new_cell)
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print("✓ Added comprehensive experiment runner cell to notebook")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

