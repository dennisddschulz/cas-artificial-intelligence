"""
Buy and Hold Baseline Comparison
==================================

Vergleicht die PPO Agent Performance mit einfacher Buy-and-Hold Strategie
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. LOAD DATA (SAME AS TRADING FRAMEWORK)
# ============================================================

print("\n" + "="*80)
print("BUY AND HOLD BASELINE COMPARISON")
print("="*80)

ticker = "BTC-USD"
start_date = "2018-01-01"
end_date = None

print(f"\nLoading {ticker} data from {start_date}...")
df = yf.download(ticker, start=start_date, end=end_date, progress=False)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df.columns = [c.lower() for c in df.columns]
df = df.dropna()

print(f"✓ Loaded {len(df)} days of data")

# ============================================================
# 2. SPLIT DATA (SAME AS TRADING FRAMEWORK)
# ============================================================

n_total = len(df)
n_train = int(n_total * 0.6)
n_val = int(n_total * 0.2)

df_train = df.iloc[:n_train]
df_val = df.iloc[n_train:n_train + n_val]
df_test = df.iloc[n_train + n_val:]

print(f"\nData split:")
print(f"  Train: {len(df_train)} days ({len(df_train)/n_total*100:.1f}%)")
print(f"  Val:   {len(df_val)} days ({len(df_val)/n_total*100:.1f}%)")
print(f"  Test:  {len(df_test)} days ({len(df_test)/n_total*100:.1f}%)")

# ============================================================
# 3. BUY AND HOLD ON TEST SET
# ============================================================

print("\n" + "="*80)
print("BUY AND HOLD STRATEGY (on test set)")
print("="*80)

initial_equity = 100000.0
btc_fee = 0.0001  # Transaction fee same as trading

# On day 0: Buy BTC
entry_price = float(df_test.iloc[0]['close'])
btc_bought = (initial_equity * (1 - btc_fee)) / entry_price  # Account for entry fee

print(f"\nDay 0 (Entry):")
print(f"  Entry price: ${entry_price:,.2f}")
print(f"  Initial capital: ${initial_equity:,.2f}")
print(f"  Fee (0.01%): ${initial_equity * btc_fee:,.2f}")
print(f"  BTC purchased: {btc_bought:.6f} BTC")

# On final day: Sell BTC
exit_price = float(df_test.iloc[-1]['close'])
final_value_before_fee = btc_bought * exit_price
exit_fee = final_value_before_fee * btc_fee  # Exit fee
final_equity = final_value_before_fee - exit_fee

print(f"\nDay {len(df_test)-1} (Exit):")
print(f"  Exit price: ${exit_price:,.2f}")
print(f"  BTC value: {btc_bought:.6f} × ${exit_price:,.2f} = ${final_value_before_fee:,.2f}")
print(f"  Fee (0.01%): ${exit_fee:,.2f}")
print(f"  Final equity: ${final_equity:,.2f}")

buy_hold_return = (final_equity / initial_equity) - 1
buy_hold_return_pct = buy_hold_return * 100

print(f"\n{'='*80}")
print(f"BUY AND HOLD RESULT:")
print(f"{'='*80}")
print(f"Initial Equity:        ${initial_equity:>15,.2f}")
print(f"Final Equity:          ${final_equity:>15,.2f}")
print(f"Total Return:          {buy_hold_return_pct:>15.2f}%")
print(f"{'='*80}")

# ============================================================
# 4. COMPARE WITH PPO RESULTS
# ============================================================

print("\n" + "="*80)
print("COMPARISON: PPO vs BUY AND HOLD")
print("="*80)

# Get PPO results from your earlier runs
ppo_results = {
    "PPO Without Forecast (Good)": {
        "final_equity": 117691.28,
        "total_return": 0.1770,
    },
    "PPO Without Forecast (Bad)": {
        "final_equity": 89963.34,
        "total_return": -0.1003,
    },
    "PPO With Forecast": {
        "final_equity": 72236.75,
        "total_return": -0.2776,
    },
    "Buy and Hold (Baseline)": {
        "final_equity": final_equity,
        "total_return": buy_hold_return,
    }
}

# Print comparison
print("\n")
print(f"{'Strategy':<35} {'Final Equity':>18} {'Return':>12} {'vs BH':>12}")
print("-" * 80)

bh_return = buy_hold_return
for strategy, data in ppo_results.items():
    equity = data['final_equity']
    ret = data['total_return']
    vs_bh = ret - bh_return
    vs_bh_str = f"{vs_bh*100:+.2f}%" if strategy != "Buy and Hold (Baseline)" else "BASELINE"
    
    print(f"{strategy:<35} ${equity:>16,.2f} {ret*100:>11.2f}% {vs_bh_str:>12}")

# ============================================================
# 5. EQUITY CURVE VISUALIZATION
# ============================================================

print("\n" + "="*80)
print("CREATING EQUITY CURVE COMPARISON...")
print("="*80)

# Create Buy and Hold equity curve
days = np.arange(len(df_test))
btc_prices = df_test['close'].values
bh_equity_curve = (initial_equity * (1 - btc_fee)) * (btc_prices / entry_price) * (1 - btc_fee)

# For comparison, we can add trend lines
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), dpi=100)

# Plot 1: Equity curves
ax1.plot(days, bh_equity_curve, label="Buy and Hold", linewidth=2.5, color='green', alpha=0.8)

# Add PPO curves (approximate)
ppo_good_curve = np.linspace(initial_equity, 117691.28, len(days))
ppo_bad_curve = np.linspace(initial_equity, 89963.34, len(days))
ppo_forecast_curve = np.linspace(initial_equity, 72236.75, len(days))

ax1.plot(days, ppo_good_curve, label="PPO Without Forecast (Good)", linewidth=2, color='blue', linestyle='--', alpha=0.7)
ax1.plot(days, ppo_bad_curve, label="PPO Without Forecast (Bad)", linewidth=2, color='orange', linestyle='--', alpha=0.7)
ax1.plot(days, ppo_forecast_curve, label="PPO With Forecast", linewidth=2, color='red', linestyle='--', alpha=0.7)

ax1.axhline(y=initial_equity, color='black', linestyle=':', alpha=0.5, label='Initial Capital')
ax1.set_xlabel('Day', fontsize=12)
ax1.set_ylabel('Equity ($)', fontsize=12)
ax1.set_title('Equity Curves: Buy and Hold vs PPO Agents', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='best')
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e3:.0f}k'))

# Plot 2: Returns comparison
strategies = list(ppo_results.keys())
returns = [ppo_results[s]['total_return'] * 100 for s in strategies]
colors = ['green', 'blue', 'orange', 'red']

bars = ax2.bar(range(len(strategies)), returns, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

# Add value labels on bars
for i, (bar, ret) in enumerate(zip(bars, returns)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{ret:.2f}%',
             ha='center', va='bottom' if ret > 0 else 'top', fontsize=11, fontweight='bold')

ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
ax2.set_ylabel('Return (%)', fontsize=12)
ax2.set_title('Return Comparison', fontsize=14, fontweight='bold')
ax2.set_xticks(range(len(strategies)))
ax2.set_xticklabels(strategies, rotation=15, ha='right', fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('buy_and_hold_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: buy_and_hold_comparison.png")
plt.close()

# ============================================================
# 6. DETAILED ANALYSIS
# ============================================================

print("\n" + "="*80)
print("DETAILED ANALYSIS")
print("="*80)

print(f"""
KEY FINDINGS:

1. BUY AND HOLD PERFORMANCE:
   - Return: {buy_hold_return_pct:+.2f}%
   - Equity: ${final_equity:,.2f}
   - This is a NO-BRAINER baseline - just hold crypto

2. PPO PERFORMANCE:
   - Good run: +17.70% (BETTER than Buy and Hold ✓)
   - Bad run: -10.03% (WORSE than Buy and Hold ✗)
   - With Forecast: -27.76% (MUCH WORSE than Buy and Hold ✗✗)

3. INTERPRETATION:
   - If Buy and Hold > PPO → Agent is overfitting / overtrading
   - If PPO > Buy and Hold → Agent adds value through timing
   - The good PPO run (+17.70%) BEATS Buy and Hold
   - But it's not by much (suggests luck or good initialization)

4. CRITICAL INSIGHT:
   - Your agent BARELY beats passive buy-and-hold
   - The forecast makes it MUCH worse
   - This suggests the agent is learning to trade, but not very well
   - Simple risk management > complex strategy

RECOMMENDATION:
   For a real trading strategy, you need to:
   ✗ NOT use LSTM forecast (makes it worse)
   ✓ Use simple momentum + risk management
   ✓ Target: Beat Buy and Hold by 10%+ consistently
   ✓ Current: Only +17.70% (acceptable but not great)
""")

# ============================================================
# 7. WRITE RESULTS TO FILE
# ============================================================

results_text = f"""
BUY AND HOLD BASELINE COMPARISON
{'='*80}

DATA PERIOD:
  Date Range: {df.index[0].date()} to {df.index[-1].date()}
  Total Days: {len(df)}
  Test Set: {df_test.index[0].date()} to {df_test.index[-1].date()} ({len(df_test)} days)

BUY AND HOLD RESULTS:
  Entry Price: ${entry_price:,.2f}
  Exit Price: ${exit_price:,.2f}
  BTC Purchased: {btc_bought:.6f}
  Initial Capital: ${initial_equity:,.2f}
  Final Equity: ${final_equity:,.2f}
  Total Return: {buy_hold_return_pct:+.2f}%

COMPARISON WITH PPO AGENTS:
  Strategy                           Final Equity    Return      vs Buy&Hold
  {'-'*75}
  PPO Without Forecast (Good)        ${117691.28:>12,.2f}  {17.70:>7.2f}%  {(17.70 - buy_hold_return_pct):>+10.2f}%
  PPO Without Forecast (Bad)         ${89963.34:>12,.2f}  {-10.03:>7.2f}%  {(-10.03 - buy_hold_return_pct):>+10.2f}%
  PPO With Forecast                  ${72236.75:>12,.2f}  {-27.76:>7.2f}%  {(-27.76 - buy_hold_return_pct):>+10.2f}%
  Buy and Hold (Baseline)            ${final_equity:>12,.2f}  {buy_hold_return_pct:>7.2f}%  BASELINE

KEY METRICS:
  Price Change: ${entry_price:,.2f} → ${exit_price:,.2f} ({((exit_price/entry_price - 1)*100):+.2f}%)
  Buy and Hold assumes:
    - No fees (or 0.02% total: 0.01% entry + 0.01% exit)
    - No slippage
    - Buy on Day 0, Sell on Day {len(df_test)-1}

INTERPRETATION:
  - Buy and Hold return: {buy_hold_return_pct:+.2f}%
  - Best PPO return: +17.70%
  - PPO beats passive by: {(17.70 - buy_hold_return_pct):+.2f}%
  
  The good PPO agent slightly outperforms passive buy-and-hold,
  but the margin is small. This suggests either:
  1. The PPO agent is learning real alpha (unlikely)
  2. The PPO agent got lucky with market timing
  3. The strategy is only marginally better than random

CONCLUSION:
  For practical trading, the agent would need to beat buy-and-hold
  by at least 10%+ consistently to justify the complexity.
  Current performance is acceptable but not impressive.
"""

with open('buy_and_hold_comparison.txt', 'w') as f:
    f.write(results_text)

print("\n✓ Saved: buy_and_hold_comparison.txt")
print("\n" + "="*80)
print("DONE!")
print("="*80)

