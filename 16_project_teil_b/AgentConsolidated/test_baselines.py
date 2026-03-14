#!/usr/bin/env python3
"""
CRITICAL: Test baseline strategies to validate if RL makes sense
Before spending time on complex experiments, we need to know:
1. What does Buy-and-Hold return?
2. What does Random Agent return?
3. What does Simple "Always Long" return?
4. Can PPO beat these baselines?
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

print("\n" + "="*100)
print("BASELINE STRATEGY TESTS")
print("="*100 + "\n")

# Download Bitcoin data
print("Loading Bitcoin data (2018-01-01 to 2026-03-14)...")
df = yf.download("BTC-USD", start="2018-01-01", end="2026-03-14", progress=False)
df = df[['Close', 'Volume']]
df.columns = ['close', 'volume']
df['r'] = np.log(df['close'] / df['close'].shift(1))  # log returns
df = df.dropna()

print(f"✓ Data loaded: {len(df)} days\n")

# Split: train=60%, val=20%, test=20%
n = len(df)
train_idx = int(0.6 * n)
val_idx = train_idx + int(0.2 * n)

df_train = df.iloc[:train_idx]
df_val = df.iloc[train_idx:val_idx]
df_test = df.iloc[val_idx:]

initial_equity = 100000.0
fee = 0.0001  # Transaction cost

print(f"Train: {len(df_train)} days ({df_train.index[0].date()} to {df_train.index[-1].date()})")
print(f"Val:   {len(df_val)} days ({df_val.index[0].date()} to {df_val.index[-1].date()})")
print(f"Test:  {len(df_test)} days ({df_test.index[0].date()} to {df_test.index[-1].date()})")
print(f"\nInitial Equity: ${initial_equity:,.0f}\n")

# ============================================================================
# BASELINE 1: BUY AND HOLD
# ============================================================================
print("\n" + "="*100)
print("BASELINE 1: BUY AND HOLD (Buy on day 1, Sell on last day)")
print("="*100)

btc_bought = initial_equity / df_test.iloc[0]['close']
final_value = btc_bought * df_test.iloc[-1]['close']
buy_hold_return = (final_value - initial_equity) / initial_equity

print(f"Entry Price:      ${df_test.iloc[0]['close']:,.2f}")
print(f"Exit Price:       ${df_test.iloc[-1]['close']:,.2f}")
print(f"BTC Bought:       {btc_bought:.6f}")
print(f"Initial Equity:   ${initial_equity:,.2f}")
print(f"Final Equity:     ${final_value:,.2f}")
print(f"Total Return:     {buy_hold_return*100:+.2f}%")
print(f"Status:           {'✓ PROFIT' if buy_hold_return > 0 else '✗ LOSS'}\n")

# ============================================================================
# BASELINE 2: SIMPLE ALWAYS LONG (1.0 leverage constantly)
# ============================================================================
print("\n" + "="*100)
print("BASELINE 2: SIMPLE 'ALWAYS LONG' (Hold 1.0 leverage at all times)")
print("="*100)

equity = initial_equity
position = 1.0  # Full long
prev_pos = 1.0

for idx in range(len(df_test)):
    r_t = df_test.iloc[idx]['r']
    
    # No position changes = no costs
    turnover = 0
    cost = fee * turnover
    
    # PnL from position
    pnl = position * r_t
    
    # Update equity
    equity = equity * (1.0 + pnl)

simple_long_return = (equity - initial_equity) / initial_equity

print(f"Position:         Always 1.0 (full long)")
print(f"Initial Equity:   ${initial_equity:,.2f}")
print(f"Final Equity:     ${equity:,.2f}")
print(f"Total Return:     {simple_long_return*100:+.2f}%")
print(f"Status:           {'✓ PROFIT' if simple_long_return > 0 else '✗ LOSS'}\n")

# ============================================================================
# BASELINE 3: RANDOM AGENT (Random position each day)
# ============================================================================
print("\n" + "="*100)
print("BASELINE 3: RANDOM AGENT (Random position [-1, 1] each day)")
print("="*100)

np.random.seed(42)
equity = initial_equity
position = 0.0

costs_total = 0
for idx in range(len(df_test)):
    r_t = df_test.iloc[idx]['r']
    
    # Random new position
    new_pos = np.random.uniform(-1.0, 1.0)
    
    # Turnover and cost
    turnover = abs(new_pos - position)
    cost = fee * turnover
    costs_total += cost
    
    # PnL
    pnl = position * r_t
    
    # Update
    equity = equity * (1.0 + (pnl - cost))
    position = new_pos

random_return = (equity - initial_equity) / initial_equity

print(f"Position:         Random [-1.0, 1.0] daily")
print(f"Initial Equity:   ${initial_equity:,.2f}")
print(f"Final Equity:     ${equity:,.2f}")
print(f"Total Transaction Costs: ${costs_total:,.2f}")
print(f"Total Return:     {random_return*100:+.2f}%")
print(f"Status:           {'✓ PROFIT' if random_return > 0 else '✗ LOSS'}\n")

# ============================================================================
# BASELINE 4: ALWAYS CASH (Do nothing - hold USD)
# ============================================================================
print("\n" + "="*100)
print("BASELINE 4: ALWAYS CASH (Do nothing - hold USD at 0% interest)")
print("="*100)

cash_return = 0.0  # No return

print(f"Position:         Always 0.0 (cash)")
print(f"Initial Equity:   ${initial_equity:,.2f}")
print(f"Final Equity:     ${initial_equity:,.2f}")
print(f"Total Return:     {cash_return*100:+.2f}%")
print(f"Status:           ✓ Break even\n")

# ============================================================================
# COMPARISON TABLE
# ============================================================================
print("\n" + "="*100)
print("COMPARISON: WHICH BASELINE WINS?")
print("="*100 + "\n")

baselines = [
    ("Buy-and-Hold", buy_hold_return),
    ("Always Long", simple_long_return),
    ("Random Agent", random_return),
    ("Always Cash", cash_return),
]

# Sort by return
baselines_sorted = sorted(baselines, key=lambda x: x[1], reverse=True)

for rank, (name, ret) in enumerate(baselines_sorted, 1):
    status = "✓ BEST" if rank == 1 else "✗" if ret < 0 else "○"
    print(f"{rank}. {name:20s}: {ret*100:+7.2f}%  {status}")

print("\n" + "="*100)
print("CRITICAL FINDINGS")
print("="*100 + "\n")

best_baseline, best_return = baselines_sorted[0]

print(f"Best Baseline Return: {best_baseline} with {best_return*100:+.2f}%")
print(f"\nFor PPO to make sense, it must beat: {best_baseline} ({best_return*100:+.2f}%)")
print(f"\nIf PPO < {best_baseline}, then:")
print(f"  ✗ The RL formulation is broken")
print(f"  ✗ The experiments don't make sense")
print(f"  ✗ We're wasting time on complex rewards/ablations")

if simple_long_return > random_return:
    print(f"\nKey Insight: BTC is in BULLISH TREND!")
    print(f"  → 'Always Long' ({simple_long_return*100:+.2f}%) > Random ({random_return*100:+.2f}%)")
    print(f"  → This means: Market timing is HARD, just holding is better")
    print(f"  → PPO needs to beat 'Always Long' to be useful")
else:
    print(f"\nKey Insight: BTC is CHOPPY (not trending)")
    print(f"  → Random ({random_return*100:+.2f}%) ≈ Always Long ({simple_long_return*100:+.2f}%)")
    print(f"  → This means: Trading skill doesn't help much")

print("\n" + "="*100 + "\n")

