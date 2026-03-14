#!/usr/bin/env python3
"""
Quick Test Script - Run Only Experiment 2 (PPO With Forecast)
For fast validation without waiting for all 6 experiments
"""

import os
import sys

# Configure environment - COMPREHENSIVE SSL/NETWORK FIXES
os.environ['MPLBACKEND'] = 'Agg'
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'

# Disable SSL verification for requests library
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure SSL
import ssl
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

os.environ['VERIFY_SSL'] = 'false'

# Try to disable WandB network issues
os.environ['WANDB_OFFLINE'] = 'false'  # We want online, but with fallback
os.environ['WANDB_SILENT'] = 'false'
os.environ['WANDB_CONSOLE'] = 'off'

print("\n" + "="*80)
print("QUICK TEST: EXPERIMENT 2 ONLY (PPO WITH FORECAST)")
print("="*80)
print("\nThis is a fast test with reduced training updates")
print("Total PPO Updates: 100 (instead of 3000)")
print("LSTM Epochs: 20 (instead of 100)")
print("Expected duration: 10-15 minutes")
print("\nWandB Mode: Online (with offline fallback if network issues occur)\n")

from trading_config import get_all_experiments
from trading_framework import ExperimentRunner

# Get all experiments
experiments = get_all_experiments()

# Select Experiment 2: PPO With Forecast
config = experiments['PPO_With_Forecast']

# REDUCE TRAINING FOR QUICK TEST
print("Reducing configuration for quick test...")
config.ppo.total_updates = 100          # Was 3000
config.ppo.num_envs = 4                 # Was 8 (save memory)
config.ppo.n_steps = 64                 # Was 256 (shorter rollouts)
config.forecasting.epochs = 20          # Was 100 (fewer LSTM epochs)
config.use_wandb = True                 # Still log to WandB
config.wandb_mode = "online"            # Try online first

print(f"✓ PPO Updates: {config.ppo.total_updates}")
print(f"✓ Parallel Envs: {config.ppo.num_envs}")
print(f"✓ Rollout Steps: {config.ppo.n_steps}")
print(f"✓ LSTM Epochs: {config.forecasting.epochs}")
print(f"✓ WandB Mode: {config.wandb_mode}")

# If you experience SSL errors, you can switch to offline mode:
# Uncomment the line below to use offline mode:
# config.wandb_mode = "offline"
# print(f"\n⚠ Using OFFLINE mode due to network issues")

print(f"\n{'='*80}")
print("RUNNING EXPERIMENT 2: PPO WITH FORECAST")
print(f"{'='*80}\n")

try:
    runner = ExperimentRunner(config)
    results = runner.run()
    
    print(f"\n{'='*80}")
    print("✓ EXPERIMENT 2 COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}\n")
    
    # Show key results
    if 'metrics' in results:
        metrics = results['metrics']
        print("KEY METRICS:")
        print(f"  Total Return:    {metrics.get('total_return', 0)*100:>8.2f}%")
        print(f"  Sharpe Ratio:    {metrics.get('sharpe_ratio', 0):>8.4f}")
        print(f"  Max Drawdown:    {metrics.get('max_drawdown', 0)*100:>8.2f}%")
        print(f"  Volatility:      {metrics.get('volatility', 0)*100:>8.2f}%")
        print(f"  Win Rate:        {metrics.get('win_rate', 0)*100:>8.2f}%")
        print()
    
    # Show results location
    print(f"Results saved to: ./results/")
    print(f"Plots saved to: ./plots/")
    print(f"WandB: {config.wandb_project} (group: {config.wandb_group})")
    
except Exception as e:
    print(f"\n✗ EXPERIMENT FAILED: {e}")
    import traceback
    traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("⚠ NETWORK/SSL ERROR - SOLUTION OPTIONS:")
    print(f"{'='*80}")
    print("\nOption 1: Try again with offline mode")
    print("  - Edit this file")
    print("  - Uncomment: config.wandb_mode = 'offline'")
    print("  - Then run: python test_exp2_only.py")
    print("\nOption 2: Sync offline data later to cloud")
    print("  - Once training completes with offline mode")
    print("  - Run: wandb sync ./wandb/offline-run-*/")
    print("\nOption 3: Check network connection")
    print("  - Verify internet is working")
    print("  - Try disabling VPN if active")
    print("  - Check firewall settings")
    print("\nOption 4: Use global offline mode")
    print("  - Run: export WANDB_MODE=offline")
    print("  - Then: python test_exp2_only.py")
    print(f"{'='*80}\n")
    
    sys.exit(1)

print(f"\n{'='*80}")
print("NEXT STEPS:")
print(f"{'='*80}")
print("1. Check results: ls ./results/")
print("2. View metrics: python load_and_inspect_metrics.py")
print("3. Run full suite: python run_all_experiments.py")
print(f"{'='*80}\n")

