#!/usr/bin/env python3
"""Quick test to see if experiment 1 completes without hanging"""

import sys
import os

# Prevent hanging on imports
os.environ['WANDB_MODE'] = 'offline'
os.environ['MPLBACKEND'] = 'Agg'

print("Testing trading_framework imports...")
try:
    from trading_config import get_ppo_without_forecast_config
    from trading_framework import ExperimentRunner
    print("✓ Imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

print("\nCreating experiment config...")
config = get_ppo_without_forecast_config(
    name="PPO-Test",
    experiment_type="test",
    variant="v1"
)

print(f"  - Forecast: {config.forecast_mode.value}")
print(f"  - Reward: {config.reward_type.value}")

print("\nCreating experiment runner...")
runner = ExperimentRunner(config)

print(f"✓ Runner created successfully")
print(f"\nNow running experiment 1 (this will take ~15-25 minutes)...")
print("If this hangs, check the hanging point in logs.\n")

try:
    results = runner.run()
    print(f"\n✓ Experiment completed successfully!")
    if results:
        print(f"  - Final Equity: ${results['equity'][-1]:,.2f}")
        print(f"  - Total Return: {results['metrics'].get('total_return', 0)*100:.2f}%")
        print(f"  - Sharpe Ratio: {results['metrics'].get('sharpe_ratio', 0):.4f}")
except KeyboardInterrupt:
    print("\n⚠ Experiment interrupted by user")
except Exception as e:
    print(f"\n✗ Experiment failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("TEST COMPLETED - Script did not hang!")
print("="*70)

