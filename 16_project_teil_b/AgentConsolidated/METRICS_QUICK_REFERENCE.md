# Quick Reference: All Logged Metrics

## 📋 Complete Metric Reference Table

| Category | Metric Name | W&B Key | Logged At | Type | Example Value |
|----------|-------------|---------|-----------|------|---|
| **TRAINING** | Update Number | `training/update` | Every 100 PPO updates | Counter | 100, 200, 300... |
| | Episode Return (100 mean) | `training/episode_return_mean_100` | Every 100 PPO updates | Float | 0.045 |
| | Episode Return (100 std) | `training/episode_return_std_100` | Every 100 PPO updates | Float | 0.028 |
| | Log Std (Exploration) | `training/log_std` | Every 100 PPO updates | Float | -1.234 |
| | Total Episodes | `training/total_episodes` | Every 100 PPO updates | Counter | 800 |
| | Policy Loss | `training/policy_loss` | Every 100 PPO updates | Float | 0.0234 |
| | Value Loss | `training/value_loss` | Every 100 PPO updates | Float | 0.0156 |
| | Entropy Loss | `training/entropy_loss` | Every 100 PPO updates | Float | 0.0089 |
| | Total Loss | `training/total_loss` | Every 100 PPO updates | Float | 0.0479 |
| | Approx KL Divergence | `training/approx_kl` | Every 100 PPO updates | Float | 0.0045 |
| **EQUITY CURVE (TIMESERIES)** | Equity Value | `timeseries/equity` | Every step (1700+ points) | Float | 100000.00, 100150.25, ... |
| | Total Return | `timeseries/total_return` | Every step | Float | 0.0015, 0.0042, ... |
| | Cumulative PnL | `timeseries/cumulative_pnl` | Every step | Float | 1500.00, 4200.00, ... |
| **COSTS** | Daily Cost | `timeseries/daily_cost` | Every step | Float | 5.25, 2.50, ... |
| | Cumulative Costs | `timeseries/cumulative_costs` | Every step | Float | 5.25, 7.75, ... |
| | Daily PnL | `timeseries/daily_pnl` | Every step | Float | 150.25, -50.10, ... |
| | Avg Daily PnL (20-day) | `timeseries/avg_daily_pnl` | Every step | Float | 125.50 |
| **POSITIONS** | Position Size | `timeseries/position_size` | Every step | Float | 0.50, -0.25, 1.0, ... |
| | Position Absolute | `timeseries/position_absolute` | Every step | Float | 0.50, 0.25, 1.0, ... |
| | Avg Position | `timeseries/avg_position` | Every step | Float | 0.45 |
| **DRAWDOWN** | Current Drawdown | `timeseries/drawdown` | Every step | Float | -0.05, -0.08, ... |
| | Drawdown from Peak | `timeseries/drawdown_from_peak` | Every step | Float | -0.05, -0.15, ... |
| | Max Drawdown So Far | `timeseries/max_drawdown_so_far` | Every step | Float | -0.05, -0.08, -0.12, ... |
| **RETURNS** | Cumulative Return | `timeseries/cumulative_return` | Every step | Float | 0.015, 0.042, ... |
| | Daily Return | `timeseries/daily_return` | Every step | Float | 0.0015, -0.0008, ... |
| **VOLATILITY** | Rolling Vol (20-day) | `timeseries/rolling_volatility_20` | Every step | Float | 0.020, 0.018, ... |
| **STEP COUNTER** | Step Number | `timeseries/step` | Every step | Counter | 1, 2, 3, ..., 1700 |
| **BUDGET SUMMARY** | Initial Equity | `budget/initial_equity` | End of evaluation | Float | 100000.00 |
| | Final Equity | `budget/final_equity` | End of evaluation | Float | 125000.00 |
| | Total Return | `budget/total_return` | End of evaluation | Float | 0.25 |
| | Total Return % | `budget/total_return_pct` | End of evaluation | Float | 25.00 |
| | Total PnL | `budget/total_pnl` | End of evaluation | Float | 25000.00 |
| | Total Costs | `budget/total_costs` | End of evaluation | Float | 5000.00 |
| | Avg Daily PnL | `budget/avg_daily_pnl` | End of evaluation | Float | 14.71 |
| | Avg Position Size | `budget/avg_position_size` | End of evaluation | Float | 0.45 |
| | Num Steps | `budget/num_steps` | End of evaluation | Counter | 1700 |
| **RISK METRICS** | Max Drawdown | `risk/max_drawdown` | End of evaluation | Float | -0.15 |
| | Max Drawdown % | `risk/max_drawdown_pct` | End of evaluation | Float | -15.00 |
| | Daily Volatility | `risk/volatility` | End of evaluation | Float | 0.020 |
| | Daily Volatility % | `risk/volatility_pct` | End of evaluation | Float | 2.00 |
| | Annualized Volatility | `risk/annualized_volatility` | End of evaluation | Float | 0.315 |
| | Annualized Volatility % | `risk/annualized_volatility_pct` | End of evaluation | Float | 31.50 |
| **PERFORMANCE** | Sharpe Ratio | `performance/sharpe_ratio` | End of evaluation | Float | 1.25 |
| | Annualized Return | `performance/annualized_return` | End of evaluation | Float | 0.60 |
| | Annualized Return % | `performance/annualized_return_pct` | End of evaluation | Float | 60.00 |
| | Calmar Ratio | `performance/calmar_ratio` | End of evaluation | Float | 1.67 |
| | Sortino Ratio | `performance/sortino_ratio` | End of evaluation | Float | 1.50 |
| | Win Rate | `performance/win_rate` | End of evaluation | Float | 0.55 |
| | Win Rate % | `performance/win_rate_pct` | End of evaluation | Float | 55.00 |
| | Profit Factor | `performance/profit_factor` | End of evaluation | Float | 1.80 |
| **TRADING** | Turnover | `trading/turnover` | End of evaluation | Float | 0.42 |
| | Cost Ratio | `trading/cost_ratio` | End of evaluation | Float | 0.08 |
| **COSTS BREAKDOWN** | Total Costs | `costs/total` | End of evaluation | Float | 5000.00 |
| | Avg Cost | `costs/average` | End of evaluation | Float | 2.94 |
| | Max Cost | `costs/max` | End of evaluation | Float | 125.00 |
| **POSITIONS STATS** | Avg Absolute | `positions/avg_absolute` | End of evaluation | Float | 0.45 |
| | Max Absolute | `positions/max_absolute` | End of evaluation | Float | 1.00 |
| | Min | `positions/min` | End of evaluation | Float | -1.00 |
| **DRAWDOWN STATS** | Max | `drawdown/max` | End of evaluation | Float | -0.15 |
| | Average | `drawdown/average` | End of evaluation | Float | -0.05 |
| **DISTRIBUTION** | Kurtosis | `distribution/kurtosis` | End of evaluation | Float | 2.50 |
| | Skewness | `distribution/skewness` | End of evaluation | Float | -0.30 |
| **DAILY RETURNS** | Mean | `daily_returns/mean` | End of evaluation | Float | 0.0015 |
| | Std | `daily_returns/std` | End of evaluation | Float | 0.020 |
| | Min | `daily_returns/min` | End of evaluation | Float | -0.082 |
| | Max | `daily_returns/max` | End of evaluation | Float | 0.065 |
| | Positive Count | `daily_returns/positive_count` | End of evaluation | Counter | 935 |
| | Negative Count | `daily_returns/negative_count` | End of evaluation | Counter | 765 |
| | Zero Count | `daily_returns/zero_count` | End of evaluation | Counter | 0 |

---

## 📊 Metrics by Category

### Training (10 metrics)
- `training/update`, `training/episode_return_mean_100`, `training/episode_return_std_100`
- `training/log_std`, `training/total_episodes`, `training/policy_loss`, `training/value_loss`
- `training/entropy_loss`, `training/total_loss`, `training/approx_kl`

### Timeseries (15 metrics × 1700 steps = 25,500+ points)
- Equity: `equity`, `total_return`, `cumulative_pnl`
- Costs: `daily_cost`, `cumulative_costs`, `daily_pnl`, `avg_daily_pnl`
- Positions: `position_size`, `position_absolute`, `avg_position`
- Drawdown: `drawdown`, `drawdown_from_peak`, `max_drawdown_so_far`
- Returns: `cumulative_return`, `daily_return`
- Risk: `rolling_volatility_20`
- Meta: `step`

### Budget Summary (9 metrics)
- `budget/initial_equity`, `budget/final_equity`, `budget/total_return`, `budget/total_return_pct`
- `budget/total_pnl`, `budget/total_costs`, `budget/avg_daily_pnl`, `budget/avg_position_size`, `budget/num_steps`

### Risk (6 metrics)
- `risk/max_drawdown`, `risk/max_drawdown_pct`, `risk/volatility`, `risk/volatility_pct`
- `risk/annualized_volatility`, `risk/annualized_volatility_pct`

### Performance (8 metrics)
- `performance/sharpe_ratio`, `performance/annualized_return`, `performance/annualized_return_pct`
- `performance/calmar_ratio`, `performance/sortino_ratio`, `performance/win_rate`, `performance/win_rate_pct`
- `performance/profit_factor`

### Trading (2 metrics)
- `trading/turnover`, `trading/cost_ratio`

### Cost Details (3 metrics)
- `costs/total`, `costs/average`, `costs/max`

### Position Stats (3 metrics)
- `positions/avg_absolute`, `positions/max_absolute`, `positions/min`

### Drawdown Stats (2 metrics)
- `drawdown/max`, `drawdown/average`

### Distribution (2 metrics)
- `distribution/kurtosis`, `distribution/skewness`

### Daily Returns (7 metrics)
- `daily_returns/mean`, `daily_returns/std`, `daily_returns/min`, `daily_returns/max`
- `daily_returns/positive_count`, `daily_returns/negative_count`, `daily_returns/zero_count`

---

## 🎯 Total Metrics Summary

| Category | Count | Logged When |
|----------|-------|------------|
| Training | 10 | Every 100 updates (30 checkpoints) |
| Timeseries | 15 × 1700 | Every step (25,500+ points) |
| Summary | 60+ | Once at end |
| Visualizations | 2 | Once at end |
| **TOTAL** | **~25,500+ data points** | Throughout experiment |

---

## ✅ What This Means

- ✅ **Equity curve:** Complete with 1700+ daily values
- ✅ **Drawdown tracking:** Exact underwater periods
- ✅ **Cost breakdown:** Every transaction tracked
- ✅ **Position history:** All positions recorded
- ✅ **Risk metrics:** Full risk assessment
- ✅ **Performance ratios:** All key ratios calculated
- ✅ **Distribution:** Returns distribution analyzed
- ✅ **Visualizations:** Charts auto-generated
- ✅ **Multi-seed comparison:** Easy seed-to-seed comparison

Everything is logged automatically! 🎉

