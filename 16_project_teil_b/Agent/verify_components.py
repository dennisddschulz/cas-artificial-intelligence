#!/usr/bin/env python3
"""
VERIFICATION SCRIPT
Checks if all components from the original notebook are included in the Python scripts
"""

import os
from pathlib import Path

def check_file_content(filepath, keywords):
    """Check if file contains all keywords"""
    if not Path(filepath).exists():
        return False, []
    
    with open(filepath, 'r') as f:
        content = f.read().lower()
    
    found = []
    missing = []
    for keyword in keywords:
        if keyword.lower() in content:
            found.append(keyword)
        else:
            missing.append(keyword)
    
    return found, missing

def main():
    print("\n" + "="*80)
    print("VERIFICATION: All Components from Original Notebook")
    print("="*80)
    
    # Components that should be in main.py
    lstm_components = [
        'LSTMForecaster',
        'forecast',
        'lstm',
        'binary classification',
        'sigmoid',
        'torch.nn',
    ]
    
    trading_env_components = [
        'TradingEnv',
        'gym.Env',
        'observation_space',
        'action_space',
        'step',
        'reset',
        'fee',
        'kappa',
    ]
    
    ppo_components = [
        'PPO',
        'stable_baselines3',
        'policy',
        'value',
        'learn',
    ]
    
    comparison_components = [
        'Forecast-Only',
        'PPO-Without-Forecast',
        'PPO-With-Forecast',
        'metrics',
        'sharpe',
        'return',
        'drawdown',
        'volatility',
    ]
    
    wandb_components = [
        'wandb',
        'W&B',
        'offline',
        'metrics.pkl',
    ]
    
    visualization_components = [
        'matplotlib',
        'seaborn',
        'plot',
        'figure',
        'subplot',
    ]
    
    # Check main.py
    print("\n✓ Checking main.py...")
    found_lstm, missing_lstm = check_file_content('main.py', lstm_components)
    found_env, missing_env = check_file_content('main.py', trading_env_components)
    found_ppo, missing_ppo = check_file_content('main.py', ppo_components)
    found_comp, missing_comp = check_file_content('main.py', comparison_components)
    found_wandb, missing_wandb = check_file_content('main.py', wandb_components)
    
    print(f"\n  1. LSTM Forecaster: {len(found_lstm)}/{len(lstm_components)} ✓")
    if missing_lstm:
        print(f"     Missing: {missing_lstm}")
    
    print(f"  2. Trading Environment: {len(found_env)}/{len(trading_env_components)} ✓")
    if missing_env:
        print(f"     Missing: {missing_env}")
    
    print(f"  3. PPO Agent: {len(found_ppo)}/{len(ppo_components)} ✓")
    if missing_ppo:
        print(f"     Missing: {missing_ppo}")
    
    print(f"  4. Experiment Comparison: {len(found_comp)}/{len(comparison_components)} ✓")
    if missing_comp:
        print(f"     Missing: {missing_comp}")
    
    print(f"  5. W&B Logging: {len(found_wandb)}/{len(wandb_components)} ✓")
    if missing_wandb:
        print(f"     Missing: {missing_wandb}")
    
    # Check analyze_results.py
    print("\n✓ Checking analyze_results.py...")
    found_viz, missing_viz = check_file_content('analyze_results.py', visualization_components)
    found_pkl, missing_pkl = check_file_content('analyze_results.py', ['metrics.pkl', 'pickle'])
    
    print(f"  1. Visualizations: {len(found_viz)}/{len(visualization_components)} ✓")
    if missing_viz:
        print(f"     Missing: {missing_viz}")
    
    print(f"  2. Pickle Loading: {len(found_pkl)}/2 ✓")
    if missing_pkl:
        print(f"     Missing: {missing_pkl}")
    
    # Check configuration
    print("\n✓ Checking Configuration...")
    found_btc, _ = check_file_content('main.py', ['BTC-USD', 'Bitcoin'])
    print(f"  Bitcoin (BTC-USD): {'✓ YES' if found_btc else '✗ NO'}")
    
    found_features, _ = check_file_content('main.py', ['rsi', 'macd', 'momentum', 'sma', 'volatility'])
    print(f"  Technical Features: {'✓ YES (found)' if len(found_features) >= 3 else '✗ NO'}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    total_found = (len(found_lstm) + len(found_env) + len(found_ppo) + 
                  len(found_comp) + len(found_wandb) + len(found_viz) + len(found_pkl))
    total_expected = (len(lstm_components) + len(trading_env_components) + 
                     len(ppo_components) + len(comparison_components) + 
                     len(wandb_components) + len(visualization_components) + 2)
    
    completion = (total_found / total_expected) * 100
    
    print(f"\n✓ Components Found: {total_found}/{total_expected}")
    print(f"✓ Completion: {completion:.1f}%")
    
    if completion >= 95:
        print(f"\n✓✓✓ ALL MAJOR COMPONENTS INCLUDED ✓✓✓")
        print("\nReady to run: python3 complete_workflow.py")
    elif completion >= 80:
        print(f"\n✓✓ MOSTLY COMPLETE - Minor items missing")
    else:
        print(f"\n⚠️  INCOMPLETE - Major components missing")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()

