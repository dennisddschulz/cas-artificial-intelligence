# What You'll See in W&B After Running Experiments

## 📊 Project Structure in W&B UI

```
PPO_Bitcoin_Trading_Experiments_Dennis (Project)
│
├─── 2026-03-13 (Date Group - shows TODAY's experiments)
│    │
│    ├─── ppo_baseline (Experiment Type)
│    │    │
│    │    └─── v1 (Version)
│    │         │
│    │         ├─── seed_10 (Group for seed 10)
│    │         │    ├─ Run 1: 2026-03-13_PPO-Baseline_with_risk_10
│    │         │    ├─ Run 2: 2026-03-13_PPO-Baseline_lstm_10
│    │         │    └─ Run 3: 2026-03-13_PPO-Baseline_basic_10
│    │         │
│    │         ├─── seed_20 (Group for seed 20)
│    │         │    ├─ Run 1: 2026-03-13_PPO-Baseline_with_risk_20
│    │         │    ├─ Run 2: 2026-03-13_PPO-Baseline_lstm_20
│    │         │    └─ Run 3: 2026-03-13_PPO-Baseline_basic_20
│    │         │
│    │         └─── seed_30 (Group for seed 30)
│    │              ├─ Run 1: 2026-03-13_PPO-Baseline_with_risk_30
│    │              ├─ Run 2: 2026-03-13_PPO-Baseline_lstm_30
│    │              └─ Run 3: 2026-03-13_PPO-Baseline_basic_30
│    │
│    └─── reward_ablation (Reward Ablation Experiments)
│         │
│         ├─── v1
│         │    ├─── seed_10
│         │    ├─── seed_20
│         │    └─── seed_30
│         │
│         └─── (multiple rewards: with_risk, basic, with_sharpe, etc.)
```

---

## 📈 Metrics You'll See in Each Run

### 1. In the "Charts" Tab

**Training Phase Metrics** (visible as you're training):
```
Charts
├─ training/update                     → Line chart (0 to 3000)
├─ training/episode_return_mean_100    → Line chart (average returns)
├─ training/episode_return_std_100     → Line chart (volatility)
├─ training/policy_loss                → Line chart (decreasing)
├─ training/value_loss                 → Line chart (decreasing)
├─ training/total_loss                 → Line chart (decreasing)
├─ training/approx_kl                  → Line chart (KL divergence)
└─ training/log_std                    → Line chart (exploration)
```

**Real-Time Evaluation Metrics** (step-by-step over 1700 days):
```
Charts
├─ timeseries/equity                    → Line chart (complete equity curve!)
│                                          1700+ points showing portfolio value
│
├─ timeseries/drawdown_from_peak        → Line chart (drawdown trajectory)
│                                          Shows how deep underwater you go
│
├─ timeseries/position_size             → Line chart (position over time)
│                                          Shows how much leveraged each day
│
├─ timeseries/daily_cost                → Bar chart (costs per day)
│
├─ timeseries/daily_pnl                 → Bar chart (P&L per day)
│
├─ timeseries/cumulative_return         → Line chart (total return curve)
│
├─ timeseries/rolling_volatility_20     → Line chart (20-day rolling vol)
│
└─ timeseries/step                      → Counter (1 to 1700)
```

**Summary Performance Metrics** (final values):
```
Charts
├─── Budget & Liquidity
│    ├─ budget/initial_equity           → 100000.00
│    ├─ budget/final_equity             → 125000.00 (or your result)
│    ├─ budget/total_return             → 0.25 (25%)
│    ├─ budget/total_return_pct         → 25.00
│    ├─ budget/total_pnl                → 25000.00
│    ├─ budget/total_costs              → 5000.00 (transaction fees)
│    ├─ budget/avg_daily_pnl            → 14.71
│    └─ budget/avg_position_size        → 0.45
│
├─── Risk Metrics
│    ├─ risk/max_drawdown               → -0.15 (15% loss)
│    ├─ risk/volatility                 → 0.020 (2.0% daily vol)
│    ├─ risk/annualized_volatility      → 0.315 (31.5% annual)
│    └─ risk/annualized_volatility_pct  → 31.50
│
├─── Performance Ratios
│    ├─ performance/sharpe_ratio        → 1.25
│    ├─ performance/calmar_ratio        → 1.67
│    ├─ performance/sortino_ratio       → 1.50
│    ├─ performance/annualized_return   → 0.60 (60% per year)
│    ├─ performance/win_rate            → 0.55 (55% of days)
│    └─ performance/profit_factor       → 1.8
│
├─── Cost Breakdown
│    ├─ costs/total                     → 5000.00
│    ├─ costs/average                   → 2.94
│    └─ costs/max                       → 125.00
│
├─── Position Statistics
│    ├─ positions/avg_absolute          → 0.45
│    ├─ positions/max_absolute          → 1.0
│    └─ positions/min                   → -1.0
│
├─── Drawdown Statistics
│    ├─ drawdown/max                    → -0.15
│    └─ drawdown/average                → -0.05
│
└─── Distribution
     ├─ distribution/kurtosis           → 2.5
     └─ distribution/skewness           → -0.3
```

---

### 2. In the "Media" Tab

**Visualizations:**
```
Media
├─ evaluation/equity_curve
│  └─ [IMAGE] Line plot showing:
│     - Starting equity: $100,000 (baseline)
│     - Final equity: $125,000 (your result)
│     - Equity progression over 1700 days
│     - Peak equity: $128,000
│     - Lowest equity: $85,000
│
└─ evaluation/returns_distribution
   └─ [IMAGE] Histogram showing:
      - Distribution of daily returns
      - Mean return: +0.15%
      - Std deviation: 2.0%
      - Number of positive/negative days
```

---

### 3. Summary Section

```
Summary
├─ Training Configuration
│  ├─ forecast_mode: lstm (or none)
│  ├─ reward_type: with_risk
│  ├─ initial_equity: 100000.0
│  ├─ ppo_updates: 3000
│  ├─ seed: 10
│  └─ wandb_mode: offline
│
└─ Results
   ├─ budget/final_equity: 125000.00
   ├─ performance/sharpe_ratio: 1.25
   ├─ risk/max_drawdown: -0.15
   └─ ... (all key metrics)
```

---

## 🔄 Example: What Each Metric Tells You

### Equity Curve Example
```
If you open: timeseries/equity

You'll see a line chart with 1700 points:
├─ Day 1:   $100,000 (starting)
├─ Day 100: $105,500 (5.5% gain)
├─ Day 500: $120,000 (peak equity)
├─ Day 600: $115,000 (drawdown from peak)
└─ Day 1700: $125,000 (final result)

This shows the COMPLETE trading journey step-by-step!
```

### Drawdown Curve Example
```
If you open: timeseries/drawdown_from_peak

You'll see when portfolio is underwater:
├─ Day 100: 0% (no drawdown yet)
├─ Day 500: 0% (at peak)
├─ Day 600: -4.2% (4.2% below peak)
├─ Day 700: -8.5% (8.5% below peak - worst drawdown)
└─ Day 1700: -2% (recovered most, still 2% below peak)

The MINIMUM value here = your max_drawdown (-8.5%)
```

---

## 🎯 How to Compare Multiple Seeds

### In W&B UI:

1. **View All Runs:**
   - Go to: `2026-03-13 > ppo_baseline > v1 > seed_10`
   - Click: "Seed 10" group name
   - See all runs with seed_10

2. **Compare Across Seeds:**
   - Select "seed_10" group
   - Shift-Click "seed_20" group
   - Ctrl-Click "seed_30" group
   - Click "Compare" button

3. **View Side-by-Side:**
   - In Charts tab, same metric appears for all selected seeds
   - Example: `timeseries/equity` shows 3 curves overlaid
   - Easy to see which seed performs best

4. **Create Summary Table:**
   - Click: "Create Chart" → "Grouped Bar Chart"
   - Select: `performance/sharpe_ratio`
   - Group by: "seeds"
   - See all Sharpe ratios in one table:
     ```
     seed_10:  1.25
     seed_20:  1.18
     seed_30:  1.32
     ```

---

## 📊 Example Run After Completion

```
Run Name: 2026-03-13_PPO-Baseline_with_risk_10
Status: Finished ✓
Duration: 45 minutes

Tags:
  ├─ 2026-03-13
  ├─ ppo_baseline
  ├─ forecast_lstm
  ├─ reward_with_risk
  ├─ version_v1
  └─ seed_10

Key Metrics:
  ├─ Initial Equity:        $100,000
  ├─ Final Equity:          $125,000
  ├─ Total Return:          +25%
  ├─ Sharpe Ratio:          1.25
  ├─ Max Drawdown:          -15%
  ├─ Annualized Volatility: 31.5%
  ├─ Win Rate:              55%
  └─ Total Costs:           $5,000

Charts (in this run):
  ├─ 30 training metrics
  ├─ 1,700 timeseries equity points
  ├─ 1,700 timeseries drawdown points
  ├─ 60+ summary scalar metrics
  └─ 2 visualizations (equity curve + returns histogram)

Total Data Points: ~25,500+
Storage: ~500KB (very efficient)
```

---

## ✨ What Makes This Powerful

1. **Complete Equity Curve:** 1700 points showing exact portfolio value each day
2. **Drawdown Tracking:** See exact underwater periods
3. **Real-time Logging:** Step-by-step tracking (not just final results)
4. **Budget Transparency:** All costs tracked and visible
5. **Multi-Seed Comparison:** Easy to compare seeds [10, 20, 30]
6. **Automatic Organization:** Date-based hierarchical grouping
7. **Rich Visualizations:** Charts + images automatically generated

---

## 🚀 Next: Running Your Experiments

Once you run experiments, you'll get:
- Complete equity curves in W&B UI
- All drawdown metrics
- All budget/liquidity metrics
- All risk metrics
- Automatically organized by seed
- Ready for comparison across seeds

Everything is logged automatically - no manual work needed! 🎉

