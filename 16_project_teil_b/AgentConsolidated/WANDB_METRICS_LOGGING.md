# W&B Metrics Logging - Complete Overview

## ✅ YES - All Metrics ARE Being Logged!

The additional metrics you mentioned (liquidity, budget, equity curve, drawdown, etc.) **ARE being logged to W&B**, but they're logged **at different stages** of the experiment lifecycle.

---

## 📊 Metrics Logged During Training

### Location: `train_ppo()` method (Line ~760-790)

**Training Progress Metrics (logged every 100 PPO updates):**
```
training/update                    - Current update number
training/episode_return_mean_100   - Mean of last 100 episode returns
training/episode_return_std_100    - Std of last 100 episode returns
training/log_std                   - Log standard deviation (exploration)
training/total_episodes            - Total episodes trained so far
training/policy_loss               - PPO policy loss
training/value_loss                - Value function loss
training/entropy_loss              - Entropy regularization loss
training/total_loss                - Total combined loss
training/approx_kl                 - Approximate KL divergence
```

**Logged:** Every 100 PPO updates during training phase

---

## 🎯 Timeseries Metrics Logged During Evaluation

### Location: `BudgetTracker._log_step_to_wandb()` (Line ~60-140 in budget_tracker.py)

**Real-time tracking during evaluation (logged EVERY STEP):**

#### Equity Metrics:
```
timeseries/equity                  - Current portfolio equity
timeseries/total_return            - Total return since start
timeseries/cumulative_pnl          - Cumulative profit/loss
```

#### Cost & Transaction Metrics:
```
timeseries/total_costs             - Sum of all transaction costs
timeseries/daily_cost              - Transaction cost for this step
timeseries/cumulative_costs        - Running total of costs
```

#### PnL Metrics:
```
timeseries/daily_pnl               - PnL for current step
timeseries/avg_daily_pnl           - 20-step rolling average PnL
```

#### Position Metrics:
```
timeseries/position_size           - Current position (-1 to +1)
timeseries/position_absolute       - Absolute value of position
timeseries/avg_position            - Average absolute position over time
```

#### Drawdown Metrics:
```
timeseries/drawdown                - Current drawdown value
timeseries/drawdown_from_peak      - Drawdown from equity peak
timeseries/max_drawdown_so_far     - Maximum drawdown observed
```

#### Return Metrics:
```
timeseries/cumulative_return       - Cumulative return over time
timeseries/daily_return            - Daily return (step-to-step)
```

#### Risk Metrics:
```
timeseries/rolling_volatility_20   - 20-step rolling volatility
```

#### Step Counter:
```
timeseries/step                    - Current evaluation step number
```

**Logged:** EVERY STEP during test evaluation phase

---

## 📈 Summary Metrics Logged After Evaluation

### Location: `evaluate()` method (Line ~1020-1030)

**Comprehensive Summary Metrics (logged once at end):**

#### Budget & Liquidity Summary:
```
budget/initial_equity              - Starting capital ($100,000)
budget/final_equity                - Ending portfolio value
budget/total_return                - Total return (decimal: 0.25 = 25%)
budget/total_return_pct            - Total return (percentage: 25%)
budget/total_pnl                   - Total profit/loss in dollars
budget/total_costs                 - Total transaction costs paid
budget/avg_daily_pnl               - Average daily PnL
budget/avg_position_size           - Average position size held
budget/num_steps                   - Number of trading steps
```

#### Risk Metrics:
```
risk/max_drawdown                  - Maximum drawdown (decimal)
risk/max_drawdown_pct              - Maximum drawdown (percentage)
risk/volatility                    - Daily volatility (decimal)
risk/volatility_pct                - Daily volatility (percentage)
risk/annualized_volatility         - Annualized volatility (decimal)
risk/annualized_volatility_pct     - Annualized volatility (percentage)
```

#### Performance Metrics:
```
performance/sharpe_ratio           - Risk-adjusted return metric
performance/annualized_return      - Annualized return (decimal)
performance/annualized_return_pct  - Annualized return (percentage)
performance/calmar_ratio           - Return / Max Drawdown ratio
performance/sortino_ratio          - Downside risk adjusted return
performance/win_rate               - Percentage of positive days
performance/win_rate_pct           - Win rate as percentage
performance/profit_factor          - Gross profit / Gross loss ratio
```

#### Trading Metrics:
```
trading/turnover                   - Position turnover ratio
trading/cost_ratio                 - Cost as % of returns
```

#### Distribution Metrics:
```
distribution/kurtosis              - Distribution kurtosis (tail risk)
distribution/skewness              - Distribution skewness
```

#### Daily Returns Statistics:
```
daily_returns/mean                 - Mean of daily returns
daily_returns/std                  - Std of daily returns
daily_returns/min                  - Minimum daily return
daily_returns/max                  - Maximum daily return
daily_returns/positive_count       - Number of positive days
daily_returns/negative_count       - Number of negative days
daily_returns/zero_count           - Number of flat days
```

#### Transaction Costs:
```
costs/total                        - Total costs paid
costs/average                      - Average cost per transaction
costs/max                          - Maximum single transaction cost
```

#### Position Statistics:
```
positions/avg_absolute             - Average absolute position
positions/max_absolute             - Maximum position held
positions/min                      - Minimum position value
```

#### Drawdown Statistics:
```
drawdown/max                       - Maximum observed drawdown
drawdown/average                   - Average drawdown value
```

**Logged:** ONCE after entire evaluation completes

---

## 📊 Visualizations Logged to W&B

### Location: `evaluate()` method (Line ~1050-1085)

1. **Equity Curve Plot**
   - Portfolio value over time
   - Logged as: `evaluation/equity_curve`

2. **Returns Distribution Plot**
   - Histogram of daily returns
   - Logged as: `evaluation/returns_distribution`

Both are logged as `wandb.Image()` objects

---

## 🔄 Timeline of Metric Logging

```
TRAINING PHASE:
├─ Every 100 PPO updates → training/* metrics logged
└─ (PPO model training for 3000 updates = 30 log points)

EVALUATION PHASE:
├─ Every step → timeseries/* metrics logged
│  (thousands of log entries - one per trading day)
│
└─ After evaluation complete:
   ├─ Summary budget/*, risk/*, performance/*, etc. logged
   ├─ Daily returns statistics logged
   ├─ Position statistics logged
   ├─ Drawdown statistics logged
   ├─ Equity curve visualization logged
   └─ Returns distribution visualization logged
```

---

## 📍 How to Find These in W&B UI

### Real-time Metrics During Runs:
1. Open your W&B run
2. Look at the **Charts** tab
3. Filter metrics by prefix:
   - `training/*` → Training progress
   - `timeseries/*` → Step-by-step evaluation (thousands of points)
   - `budget/*` → Financial summary
   - `risk/*` → Risk metrics
   - `performance/*` → Performance ratios
   - `costs/*` → Transaction costs

### Grouped by Seed:
- All metrics are **automatically grouped by seed** in W&B
- Navigate to: `2026-03-13/ppo_baseline/v1/seed_10/` (for example)
- Each seed group contains all its metrics

### Compare Across Seeds:
In W&B UI:
1. Select multiple runs from different seeds
2. The same metric will be compared across all selected seeds
3. Example: `timeseries/equity` across `seed_10`, `seed_20`, `seed_30`

---

## 🎯 Summary: What's Logged Where

| Metric Category | When Logged | Details |
|---|---|---|
| **Training** | Every 100 PPO updates | Loss, KL, episode returns |
| **Equity Curve** | Every evaluation step | Real-time portfolio value |
| **Liquidity/Cash** | Every evaluation step | Position sizes, turnover |
| **Budget Summary** | End of evaluation | Initial/final equity, returns |
| **Drawdown** | Every step + summary | Current DD, max DD, recovery |
| **Transaction Costs** | Every step + summary | Daily costs, totals, averages |
| **Risk Metrics** | End of evaluation | Volatility, Sharpe, Calmar, Sortino |
| **Win Rate** | End of evaluation | Percentage of positive days |
| **Visualizations** | End of evaluation | Equity curve plot, returns histogram |

---

## ✨ Key Points

✅ **Equity curves** - Logged at EVERY STEP (timeseries/equity)
✅ **Drawdown** - Logged at EVERY STEP (timeseries/drawdown*)
✅ **Budget/Liquidity** - Logged at end (budget/*, timeseries/*)
✅ **Transaction costs** - Logged at EVERY STEP (timeseries/costs, costs/*)
✅ **Position tracking** - Logged at EVERY STEP (timeseries/position*, positions/*)
✅ **Risk metrics** - Logged at end (risk/*, performance/*)
✅ **All grouped by seed** - Automatic W&B group hierarchy

**NO manual intervention needed** - everything is automatically logged!

---

## 🔗 Code References

- **Budget Tracker Logging:** `budget_tracker.py` lines 60-140
- **Evaluation Summary Metrics:** `trading_framework.py` lines 960-1030
- **Training Progress Metrics:** `trading_framework.py` lines 760-790
- **Integration in evaluate():** `trading_framework.py` lines 807-1130

All metrics are **automatically captured and logged to W&B** during the normal experiment flow!

