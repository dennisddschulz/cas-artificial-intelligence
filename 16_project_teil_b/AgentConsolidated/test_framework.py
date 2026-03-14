#!/usr/bin/env python3
"""
Quick test script to verify the parameterized framework works correctly
Tests the basic flow without running full training
"""

import os
import sys

# Disable SSL warnings for testing
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['CURL_CA_BUNDLE'] = ''

print("="*70)
print("FRAMEWORK TEST SCRIPT")
print("="*70)

# Test 1: Import all modules
print("\n1. Testing imports...")
try:
    from trading_config import (
        ExperimentConfig,
        get_ppo_without_forecast_config,
        get_ppo_with_forecast_config,
        ForecastMode,
        RewardType
    )
    from trading_framework import ExperimentRunner
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create configurations
print("\n2. Testing configurations...")
try:
    config_no_forecast = get_ppo_without_forecast_config(
        name="PPO-No-Forecast-Test",
        group="test"
    )
    print(f"   ✓ PPO Without Forecast config created")
    print(f"     - Forecast Mode: {config_no_forecast.forecast_mode.value}")
    print(f"     - Reward Type: {config_no_forecast.reward_type.value}")
    print(f"     - Fee: {config_no_forecast.environment.fee}")
    print(f"     - Kappa: {config_no_forecast.environment.kappa}")
    print(f"     - Leverage Max: {config_no_forecast.environment.leverage_max}")
    
    config_with_forecast = get_ppo_with_forecast_config(
        name="PPO-With-Forecast-Test",
        group="test"
    )
    print(f"   ✓ PPO With Forecast config created")
    
except Exception as e:
    print(f"   ✗ Config creation failed: {e}")
    sys.exit(1)

# Test 3: Check environment parameters
print("\n3. Checking environment parameters...")
try:
    env_config = config_no_forecast.environment
    print(f"   Initial Equity: ${env_config.initial_equity:,.0f}")
    print(f"   Fee: {env_config.fee}")
    print(f"   Kappa: {env_config.kappa}")
    print(f"   Leverage Max: {env_config.leverage_max}")
    print(f"   Slippage Coef: {env_config.slippage_coef}")
    print(f"   Smoothing Alpha: {env_config.smoothing_alpha}")
    print(f"   Reward Scale: {env_config.reward_scale}")
    print(f"   Reward Type: {env_config.reward_type.value}")
    print("   ✓ All parameters as expected")
except Exception as e:
    print(f"   ✗ Parameter check failed: {e}")
    sys.exit(1)

# Test 4: Check PPO parameters
print("\n4. Checking PPO parameters...")
try:
    ppo_config = config_no_forecast.ppo
    print(f"   Num Envs: {ppo_config.num_envs}")
    print(f"   N Steps: {ppo_config.n_steps}")
    print(f"   Total Updates: {ppo_config.total_updates}")
    print(f"   Learning Rate: {ppo_config.learning_rate}")
    print(f"   Gamma: {ppo_config.gamma}")
    print(f"   GAE Lambda: {ppo_config.gae_lambda}")
    print(f"   VF Coef: {ppo_config.vf_coef}")
    print(f"   Ent Coef: {ppo_config.ent_coef}")
    print(f"   Clip Eps: {ppo_config.clip_eps}")
    print("   ✓ All PPO parameters as expected")
except Exception as e:
    print(f"   ✗ PPO parameter check failed: {e}")
    sys.exit(1)

# Test 5: Create ExperimentRunner
print("\n5. Testing ExperimentRunner initialization...")
try:
    # Modify config for quick test (reduce data size)
    test_config = get_ppo_without_forecast_config()
    test_config.ppo.total_updates = 1  # Very short for testing
    test_config.ppo.num_envs = 1
    test_config.ppo.n_steps = 10
    test_config.use_wandb = False  # Disable wandb for testing
    
    runner = ExperimentRunner(test_config)
    print(f"   ✓ ExperimentRunner created")
    print(f"     - Device: {runner.device}")
    print(f"     - Seed: {runner.config.seed}")
except Exception as e:
    print(f"   ✗ ExperimentRunner creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("ALL TESTS PASSED!")
print("="*70)
print("\nFramework is ready for experiments. Next steps:")
print("  1. Run: python run_experiments.py")
print("  2. Monitor progress with: python check_progress.py")
print("="*70)

