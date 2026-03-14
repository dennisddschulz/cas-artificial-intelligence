# Enhanced WandB Metrics Logging Guide

## Overview

The trading framework now logs **comprehensive metrics to WandB** in two ways:

1. **Timeseries Metrics** - Tracked at every 10 steps during evaluation
2. **Summary Metrics** - Final metrics grouped by category

This enables rich visualizations and trend analysis in WandB.

## Timeseries Metrics (Updated Every 10 Steps)

These metrics are logged continuously during evaluation for trend analysis:

```
timeseries/equity              - Current portfolio equity value
timeseries/total_return        - Cumulative return from start
timeseries/total_costs         - Cumulative transaction costs
timeseries/avg_daily_pnl       - Average daily profit/loss (20-step window)
timeseries/position_size       - Current position size
timeseries/drawdown            - Current drawdown from peak
timeseries/cumulative_return   - Cumulative return percentage
timeseries/step                - Evaluation step number
```

### What This Means

- **Equity curve**: Watch how portfolio value changes over test period
- **Returns progression**: See if returns are accumulating or declining
- **Cost accumulation**: Track if transaction costs are building up
- **Position dynamics**: Monitor position sizing decisions
- **Drawdown evolution**: See when maximum drawdowns occur

### Visualization in WandB

In your WandB dashboard:
1. Go to Charts section
2. Create line charts for each timeseries metric
3. Compare multiple runs side-by-side
4. Analyze when specific events occur (e.g., max drawdown)

## Summary Metrics (Logged at End of Evaluation)

### Budget & Liquidity Summary
```
budget/initial_equity       - Starting capital ($100,000)
budget/final_equity         - Ending capital ($143,611.60)
budget/total_return         - Return as decimal (0.4361)
budget/total_return_pct     - Return as percentage (43.61%)
budget/total_pnl            - Total profit/loss in dollars
budget/total_costs          - Total transaction costs
budget/avg_daily_pnl        - Average daily PnL
budget/avg_position_size    - Average position size (0.9123)
budget/num_steps            - Number of trading steps
```

### Risk Metrics
```
risk/max_drawdown           - Maximum drawdown (-41.85%)
risk/max_drawdown_pct       - Drawdown as percentage
risk/volatility             - Daily volatility
risk/volatility_pct         - Daily volatility percentage
risk/annualized_volatility  - Annualized volatility (37.88%)
risk/annualized_volatility_pct - As percentage
```

### Performance Metrics
```
performance/sharpe_ratio    - Risk-adjusted return (0.5480)
performance/annualized_return - Annual return (22.76%)
performance/annualized_return_pct - As percentage
performance/calmar_ratio    - Return / max drawdown (0.5438)
performance/sortino_ratio   - Downside risk adjusted (0.9971)
performance/win_rate        - Winning trades % (49.31%)
performance/profit_factor   - Gross profit / gross loss (1.119)
```

### Trading Metrics
```
trading/turnover            - Average position change (435.38)
trading/cost_ratio          - Costs / total return (0.00)
```

### Distribution Metrics
```
distribution/kurtosis       - Tail heaviness (6.79)
distribution/skewness       - Asymmetry (0.96)
```

## How to View in WandB

### Method 1: Summary Page
After run completes, all metrics appear on the run summary page:
- Organized by category (budget, risk, performance, etc.)
- Single values visible at a glance

### Method 2: Charts
1. Click "Charts" tab
2. Create custom charts:
   - **Performance dashboard**: Compare sharpe_ratio, calmar_ratio, sortino_ratio
   - **Equity growth**: Plot budget/final_equity and budget/total_return_pct
   - **Risk analysis**: Show risk/max_drawdown and risk/annualized_volatility
   - **Trend analysis**: Use timeseries/equity to see progression

### Method 3: Compare Runs
1. Select multiple runs
2. Compare metrics side-by-side
3. Identify best performers
4. Understand trade-offs

### Method 4: Reports
1. Create a report
2. Add charts and insights
3. Share with team

## Interpreting the Metrics

### Your Current Results (Seed 42)

```
Initial Equity:        $     100,000.00
Final Equity:          $     143,611.60  ✓ Good (43.61% gain)
Total Return:                    43.61%  ✓ Strong return
Total Costs:           $           0.04  ✓ Minimal costs
Total PnL:             $      43,611.60  ✓ Profitable
Avg Daily PnL:         $           0.00  ⚠ Very small per-day
Max Drawdown:                    -41.85%  ⚠ Significant risk
Avg Position Size:              0.9123   ✓ Good diversification
Number of Steps:                   583   ✓ Good sample size

Sharpe Ratio:          0.5480            ⚠ Moderate (>1 is good)
Annualized Return:     22.76%            ✓ Good annual return
Annualized Volatility: 37.88%            ⚠ High volatility
Calmar Ratio:          0.5438            ⚠ Moderate (return/drawdown)
Sortino Ratio:         0.9971            ⚠ Moderate (downside focus)
Win Rate:              49.31%             ⚠ Close to 50% (luck?)
Profit Factor:         1.1190            ⚠ Marginal (only 11.9% more wins)
```

### Interpretation

**Strengths:**
- ✓ Positive 43.61% return
- ✓ Very low transaction costs ($0.04)
- ✓ Adequate sample size (583 steps)
- ✓ Good diversification (0.91 avg position)

**Weaknesses:**
- ⚠ High volatility (37.88% annualized)
- ⚠ Significant max drawdown (-41.85%)
- ⚠ Sharpe ratio below 1.0
- ⚠ Win rate close to 50% (suggests luck vs skill)
- ⚠ Profit factor only 1.12 (barely profitable)

**Verdict:** Strategy is profitable but risky. Returns come with high volatility.

## Multi-Seed Comparison Using WandB

To compare different seeds:

1. **Run experiments with different seeds:**
   ```bash
   # Seed 10
   python seed_utilities.py --set-seed 10
   python run_reward_ablation.py --max-exp 1

   # Seed 20
   python seed_utilities.py --set-seed 20
   python run_reward_ablation.py --max-exp 1

   # Seed 30
   python seed_utilities.py --set-seed 30
   python run_reward_ablation.py --max-exp 1
   ```

2. **In WandB Dashboard:**
   - Go to Runs table
   - Select all 3 runs (seed 10, 20, 30)
   - Compare metrics across seeds
   - Look for stability (low variance = stable strategy)

3. **Expected to see:**
   - Seed 10: ~45% return
   - Seed 20: ~42% return
   - Seed 30: ~43% return
   - **If similar**: Strategy is robust ✓
   - **If very different**: Strategy depends on luck ✗

## Creating WandB Charts

### Example 1: Equity Growth Over Time
```
Metric: timeseries/equity
Type: Line chart
Title: "Portfolio Equity Growth During Evaluation"
X-axis: Step
Y-axis: Equity Value ($)
```

### Example 2: Return vs Risk
```
Metrics: performance/annualized_return vs risk/annualized_volatility
Type: Scatter plot
Title: "Risk-Return Tradeoff"
Show trend line
```

### Example 3: Performance Comparison
```
Metrics: Sharpe Ratio, Calmar Ratio, Sortino Ratio
Type: Bar chart
Title: "Risk-Adjusted Performance Metrics"
Group by: Reward Type or Seed
```

### Example 4: Stability Across Seeds
```
Metrics: budget/total_return_pct by seed (10, 20, 30)
Type: Bar chart with error bars
Title: "Return Stability Across Seeds"
Show: Mean ± Std Dev
```

## Common WandB Visualizations

### Dashboard 1: Quick Summary
- Final equity (gauge or stat)
- Total return (stat)
- Sharpe ratio (stat)
- Max drawdown (stat with color coding)

### Dashboard 2: Risk Analysis
- Max drawdown over time (timeseries)
- Annualized volatility (gauge)
- Kurtosis (stat)
- Skewness (stat)

### Dashboard 3: Equity Growth
- Equity curve (timeseries)
- Position size (timeseries)
- Cumulative PnL (timeseries)
- Drawdown (timeseries)

### Dashboard 4: Reward Comparison
- Total return by reward type (bar)
- Sharpe ratio by reward type (bar)
- Max drawdown by reward type (bar)

## Troubleshooting

### Metrics not appearing in WandB?
1. Check WandB mode: `WANDB_MODE=online` required for online mode
2. Verify: `use_wandb: True` in config
3. Check credentials are valid
4. Offline mode saves to `./wandb/offline-run-*`

### Timeseries not showing?
1. Timeseries logged every 10 steps
2. If evaluation < 10 steps, no timeseries logged
3. Check `timeseries/*` metrics in run
4. Create line chart explicitly

### Can't compare runs?
1. Ensure runs have same metric names
2. Use "Runs" table, not "Projects"
3. Select multiple runs with checkboxes
4. Click "Compare" button

## Metric Definitions

**Sharpe Ratio**: Return / Volatility
- >1.0: Good
- >2.0: Excellent
- Your result (0.548): Below average

**Calmar Ratio**: Annual Return / Max Drawdown
- >0.5: Good
- >1.0: Excellent
- Your result (0.544): Acceptable

**Sortino Ratio**: Return / Downside Volatility (losses only)
- >1.0: Good
- >2.0: Excellent
- Your result (0.997): Below 1.0

**Win Rate**: % of profitable trades
- >50%: Wins more often
- <50%: Loses more often
- Your result (49.31%): Nearly even (concerning)

**Profit Factor**: Gross Profit / Gross Loss
- >1.5: Good (50% more wins)
- >2.0: Excellent (100% more wins)
- Your result (1.119): Marginal (only 11.9% more wins)

## Next Steps

1. **Run with different seeds:**
   ```bash
   python multi_seed_testing.py --seeds 10 20 30 --mode rewards
   ```

2. **Create WandB dashboard:**
   - Open your WandB project
   - Create custom dashboard
   - Add charts for key metrics
   - Share with team

3. **Analyze results:**
   - Look at timeseries for patterns
   - Compare performance across reward types
   - Identify best strategy
   - Document findings

4. **Generate report:**
   - Create WandB report
   - Include charts and insights
   - Document assumptions
   - Share findings

## Summary

| Metric | Your Value | Interpretation |
|--------|-----------|-----------------|
| Total Return | 43.61% | ✓ Good |
| Sharpe Ratio | 0.548 | ⚠ Below average |
| Max Drawdown | -41.85% | ⚠ High risk |
| Volatility | 37.88% | ⚠ High |
| Win Rate | 49.31% | ⚠ Nearly random |
| Profit Factor | 1.119 | ⚠ Marginal |
| **Overall** | **Mixed** | **Profitable but risky** |


