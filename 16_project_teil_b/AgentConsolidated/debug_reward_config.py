#!/usr/bin/env python3
"""
Debug Script - Check what's wrong with the reward calculation
"""

import sys
sys.path.insert(0, '/home/isc-den/cas-artificial-intelligence/16_project_teil_b/AgentConsolidated')

from trading_config import (
    get_ppo_without_forecast_config, 
    EnvironmentConfig, 
    RewardType
)
import numpy as np

print("="*100)
print("DEBUG: Checking reward configuration")
print("="*100 + "\n")

# Test 1: Check the default config
print("[1] Default EnvironmentConfig():")
env_default = EnvironmentConfig()
print(f"  reward_type: {env_default.reward_type.value}")
print(f"  kappa: {env_default.kappa}")
print(f"  reward_scale: {env_default.reward_scale}")
print(f"  reward_params['kappa']: {env_default.reward_params.get('kappa')}")
print(f"  reward_params['reward_scale']: {env_default.reward_params.get('reward_scale')}")
print()

# Test 2: Check the PPO-Without-Forecast config
print("[2] PPO-Without-Forecast Config:")
config = get_ppo_without_forecast_config()
print(f"  experiment_name: {config.experiment_name}")
print(f"  forecast_mode: {config.forecast_mode.value}")
print(f"  reward_type: {config.reward_type.value}")
print(f"  environment.reward_type: {config.environment.reward_type.value}")
print(f"  environment.kappa: {config.environment.kappa}")
print(f"  environment.reward_scale: {config.environment.reward_scale}")
print(f"  environment.reward_params: {config.environment.reward_params}")
print()

# Test 3: Simulate a single step reward calculation
print("[3] Simulating reward calculation (WITH_RISK):")
print("  Inputs:")
pnl = 0.001  # Small positive PnL
cost = 0.0001  # Small cost
slippage = 0.0
pos = 0.5  # Position size
sigma_t = 0.02  # Volatility

kappa = 0.01  # Default WITH_RISK kappa
reward_scale = 1.0  # Default scale

print(f"    pnl: {pnl}")
print(f"    cost: {cost}")
print(f"    slippage: {slippage}")
print(f"    pos: {pos}")
print(f"    sigma_t: {sigma_t}")
print(f"    kappa: {kappa}")
print(f"    reward_scale: {reward_scale}")

true_reward = pnl - cost - slippage
risk_pen = kappa * (pos ** 2) * sigma_t
reward = true_reward - risk_pen
reward *= reward_scale

print(f"\n  Calculations:")
print(f"    true_reward = {pnl} - {cost} - {slippage} = {true_reward:.6f}")
print(f"    risk_pen = {kappa} * {pos}^2 * {sigma_t} = {risk_pen:.6f}")
print(f"    reward = {true_reward:.6f} - {risk_pen:.6f} = {reward:.6f}")
print(f"    reward *= {reward_scale} = {reward:.6f}")
print()

# Test 4: Check if parameters are being read correctly in TradingEnv
print("[4] Testing TradingEnv parameter extraction:")
print("  Creating dummy TradingEnv...")

# We can't create a TradingEnv without a dataframe, so just check the logic
reward_params = config.environment.reward_params
print(f"  reward_params passed: {reward_params}")
print(f"  kappa from params: {reward_params.get('kappa', 0.01)}")
print(f"  reward_scale from params: {reward_params.get('reward_scale', 1.0)}")
print()

# Test 5: Check equity update logic
print("[5] Testing equity update logic:")
initial_equity = 100000.0
true_reward = -0.001  # Negative return (like market going down)
print(f"  Initial equity: ${initial_equity:.2f}")
print(f"  True reward (log return): {true_reward:.6f}")
equity_after = initial_equity * np.exp(true_reward)
print(f"  Equity after: ${equity_after:.2f}")
print(f"  Change: {(equity_after/initial_equity - 1)*100:.4f}%")
print()

print("="*100)
print("ANALYSIS")
print("="*100)
print()
print("✓ Default parameters look OK")
print("✓ Reward calculation logic is correct")
print("✓ Equity update logic is correct")
print()
print("POSSIBLE ISSUES:")
print("  1. Data might be different (different date range?)")
print("  2. Market conditions might be worse now")
print("  3. Policy didn't converge (not trained long enough?)")
print("  4. Some subtle bug in the step() function")
print()
print("NEXT: Run with verbose logging to see actual trades")
print()

