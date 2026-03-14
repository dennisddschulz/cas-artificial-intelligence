# WandB Logging Structure - Exact Data Sent

## Overview

This document shows the EXACT data structure being logged to WandB at each stage.

---

## Stage 1: PPO Training (Every 100 Updates)

**When:** During `train_ppo()` at lines 792-806
**Frequency:** Every 100 updates
**Destination:** WandB (logged individually)

```python
# EXACT LOG:
wandb.log({
    "training/update": 100,                          # int
    "training/episode_return_mean_100": 1234.56,     # float
    "training/episode_return_std_100": 89.12,        # float
    "training/log_std": -0.45,                       # float
    "training/total_episodes": 500,                  # int
    "training/policy_loss": 0.0234,                  # float
    "training/value_loss": 0.0456,                   # float
    "training/entropy_loss": -0.0123,                # float
    "training/total_loss": 0.0567,                   # float
    "training/approx_kl": 0.0089,                    # float
})
```

**Data in WandB:** ✅ YES - 10 individual metrics per 100 updates

---

## Stage 2: Forecast Training (After LSTM, Lines 465-495)

**When:** After `train_forecast_model()` completes
**Frequency:** Once (if using LSTM)
**Destination:** WandB (logged individually)

```python
# EXACT LOG:
wandb.log({
    "forecast/train_accuracy": 0.6234,               # float
    "forecast/train_smape": 45.67,                   # float
    "forecast/train_mape": 52.34,                    # float
    "forecast/val_accuracy": 0.5892,                 # float
    "forecast/val_smape": 48.12,                     # float
    "forecast/val_mape": 55.67,                      # float
    "forecast/test_accuracy": 0.5745,                # float
    "forecast/test_smape": 49.34,                    # float
    "forecast/test_mape": 57.89,                     # float
    "forecast/best_val_loss": 0.4523,                # float
})
```

**Data in WandB:** ✅ YES - 10 individual metrics

---

## Stage 3: Evaluation - Timeseries Logging

**When:** During `evaluate()` at lines 844-882 (EVERY step)
**Frequency:** Every evaluation step
**Destination:** WandB (logged individually at each step)

```python
# EXACT LOG AT EACH STEP (Example: Step 42 of 500):
wandb.log({
    # Equity metrics
    "timeseries/equity": 102500.50,                  # float - current portfolio value
    "timeseries/total_return": 0.025,                # float - 2.5% cumulative return
    "timeseries/cumulative_pnl": 2500.50,            # float - $2500.50 P&L

    # Cost metrics
    "timeseries/total_costs": 0.05,                  # float - total fees
    "timeseries/daily_cost": 0.001,                  # float - fee this step
    "timeseries/cumulative_costs": 0.05,             # float - total fees

    # PnL metrics
    "timeseries/daily_pnl": 150.25,                  # float - profit this step
    "timeseries/avg_daily_pnl": 125.50,              # float - 20-step rolling avg

    # Position metrics
    "timeseries/position_size": 1.50,                # float - 150% long
    "timeseries/position_absolute": 1.50,            # float - absolute value
    "timeseries/avg_position": 1.25,                 # float - average position

    # Drawdown metrics
    "timeseries/drawdown": -0.0850,                  # float - 8.5% below peak
    "timeseries/drawdown_from_peak": 0.0850,         # float - distance from peak
    "timeseries/max_drawdown_so_far": -0.1250,       # float - worst drawdown yet

    # Return metrics
    "timeseries/cumulative_return": 0.025,           # float - 2.5% total return
    "timeseries/daily_return": 0.0015,               # float - 0.15% daily return

    # Risk metrics
    "timeseries/rolling_volatility_20": 0.0125,      # float - 1.25% vol

    # Step counter
    "timeseries/step": 42,                           # int - step number
})

# Then IMMEDIATELY after evaluation continues to step 43
wandb.log({
    "timeseries/equity": 102650.75,
    "timeseries/total_return": 0.02651,
    # ... etc ...
    "timeseries/step": 43,
})

# ... continues for all 500 steps ...
```

**Data in WandB:** ✅ YES - 14 metrics × 500 steps = **7,000 data points**

---

## Stage 4: Evaluation Summary Metrics

**When:** After evaluation completes, at lines 1048-1141
**Frequency:** Once (at end)
**Destination:** WandB (logged individually)

```python
# EXACT LOG:
wandb.log({
    # Budget & Liquidity Summary (9 metrics)
    "budget/initial_equity": 100000.0,               # float
    "budget/final_equity": 102500.50,                # float
    "budget/total_return": 0.025,                    # float - decimal
    "budget/total_return_pct": 2.5,                  # float - percentage
    "budget/total_pnl": 2500.50,                     # float
    "budget/total_costs": 0.05,                      # float
    "budget/avg_daily_pnl": 4.29,                    # float
    "budget/avg_position_size": 1.25,                # float
    "budget/num_steps": 500,                         # int

    # Risk Metrics (6 metrics)
    "risk/max_drawdown": -0.1250,                    # float - decimal
    "risk/max_drawdown_pct": -12.50,                 # float - percentage
    "risk/volatility": 0.0150,                       # float - daily vol
    "risk/volatility_pct": 1.50,                     # float
    "risk/annualized_volatility": 0.2381,            # float
    "risk/annualized_volatility_pct": 23.81,         # float

    # Performance Metrics (8 metrics)
    "performance/sharpe_ratio": 1.2345,              # float
    "performance/annualized_return": 0.1234,         # float
    "performance/annualized_return_pct": 12.34,      # float
    "performance/calmar_ratio": 0.9876,              # float
    "performance/sortino_ratio": 1.5432,             # float
    "performance/win_rate": 0.5234,                  # float - 52.34%
    "performance/win_rate_pct": 52.34,               # float
    "performance/profit_factor": 1.2500,             # float

    # Trading Metrics (2 metrics)
    "trading/turnover": 0.1234,                      # float
    "trading/cost_ratio": 0.0002,                    # float

    # Distribution Metrics (2 metrics)
    "distribution/kurtosis": 3.4567,                 # float
    "distribution/skewness": -0.1234,                # float

    # Daily Returns Statistics (7 metrics)
    "daily_returns/mean": 0.00043,                   # float - 0.043%
    "daily_returns/std": 0.00150,                    # float - 0.150%
    "daily_returns/min": -0.0234,                    # float
    "daily_returns/max": 0.0278,                     # float
    "daily_returns/positive_count": 261,             # int - days with gains
    "daily_returns/negative_count": 237,             # int - days with losses
    "daily_returns/zero_count": 2,                   # int - flat days

    # Transaction Cost Metrics (3 metrics)
    "costs/total": 0.05,                             # float
    "costs/average": 0.0001,                         # float
    "costs/max": 0.0015,                             # float

    # Position Metrics (3 metrics)
    "positions/avg_absolute": 1.25,                  # float
    "positions/max_absolute": 2.00,                  # float
    "positions/min": -1.50,                          # float

    # Drawdown Statistics (2 metrics)
    "drawdown/max": -0.1250,                         # float
    "drawdown/average": -0.0345,                     # float
})
```

**Data in WandB:** ✅ YES - 42 individual summary metrics

---

## Stage 5: Visualizations (Lines 997-1042)

**When:** During evaluation summary
**Frequency:** Once
**Destination:** WandB (as images)

```python
# EXACT LOG:

# Equity curve visualization
wandb.log({
    "evaluation/equity_curve": wandb.Image(fig),  # Matplotlib figure as image
})

# Returns distribution visualization
wandb.log({
    "evaluation/returns_distribution": wandb.Image(fig),  # Histogram as image
})
```

**Data in WandB:** ✅ YES - 2 professional matplotlib visualizations

---

## Stage 6: Local Files (NOT Sent to WandB)

**When:** After evaluation completes
**Frequency:** Once
**Destination:** Local filesystem ONLY (NOT WandB)

```python
# SAVED LOCALLY (files.pkl):
metrics_to_save = {
    'experiment_name': 'PPO_LSTM_SHARPE_seed42',     # str
    'forecast_mode': 'lstm',                         # str
    'reward_type': 'with_risk',                      # str
    'timestamp': '2024-03-12T15:30:45.123456',       # str ISO format
    'config': {                                      # dict
        'initial_equity': 100000.0,
        'fee': 0.0001,
        'kappa': 0.01,
        'leverage_max': 2.0,
        'total_updates': 1000,
    },
    'metrics': {                                     # dict - all 42 metrics
        'total_return': 0.025,
        'sharpe_ratio': 1.2345,
        # ... all metrics dict ...
    },
    'equity_curve': [100000.0, 100123.45, ..., 102500.50],  # list
    'daily_returns': [0.00123, -0.00089, ..., 0.00234],     # list
    'positions': [1.0, 1.05, ..., 1.50],                     # list
    'costs': [0.0001, 0.00012, ..., 0.00015],               # list
    'tracker_summary': {...},                               # dict
}

# Files created:
# 1. results/metrics.pkl (binary) - Contains above dict
# 2. results/metrics_summary.csv (text) - Single row with all metrics
```

**Data in WandB:** ❌ NO - Files NOT sent as artifacts

---

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     PPO TRAINING                             │
│                                                              │
│  Every 100 updates:                                          │
│  wandb.log({training/*: 10 metrics})                         │
│                                                              │
│  Total: 1000 updates / 100 = 10 logs = 100 data points       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              FORECAST MODEL TRAINING (If LSTM)               │
│                                                              │
│  After training completes:                                   │
│  wandb.log({forecast/*: 10 metrics})                         │
│                                                              │
│  Total: 1 log = 10 data points                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                       EVALUATION                             │
│                                                              │
│  Every step (500 steps):                                     │
│  wandb.log({timeseries/*: 14 metrics})                       │
│                                                              │
│  Total: 500 steps × 14 metrics = 7,000 data points           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  SUMMARY METRICS                             │
│                                                              │
│  At end of evaluation:                                       │
│  wandb.log({budget/*, risk/*, performance/*, ...})           │
│                                                              │
│  Total: 42 summary metrics = 42 data points                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  VISUALIZATIONS                              │
│                                                              │
│  At end of evaluation:                                       │
│  wandb.log({evaluation/equity_curve: Image})                 │
│  wandb.log({evaluation/returns_distribution: Image})         │
│                                                              │
│  Total: 2 images                                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│            LOCAL FILES (BACKUP ONLY)                         │
│                                                              │
│  Save locally but NOT send to WandB:                         │
│  results/metrics.pkl                                         │
│  results/metrics_summary.csv                                 │
│                                                              │
│  NOT logged as artifacts to WandB                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              WANDB DASHBOARD CONTAINS:                       │
│                                                              │
│  ✅ 100+ individual metrics                                   │
│  ✅ 7,000+ timeseries data points                             │
│  ✅ 2 professional visualizations                             │
│  ✅ Complete training curve                                   │
│  ✅ Complete evaluation trajectory                            │
│                                                              │
│  ❌ NO file artifacts                                         │
│  ❌ NO pickle downloads                                       │
│  ❌ NO CSV artifacts                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Total Data Logged to WandB

```
Training Metrics:        10 metrics × (1000/100) updates    =    100 points
Forecast Metrics:        10 metrics × 1 (if LSTM)           =     10 points
Timeseries Metrics:      14 metrics × 500 steps             =  7,000 points
Summary Metrics:         42 metrics × 1                     =     42 points
Visualizations:          2 images                           =      2 images
                                                            ─────────────
                         TOTAL in WandB:                   7,152 data points
                                                            + 2 visualizations
```

**All logged as INDIVIDUAL METRICS, NOT as file artifacts.**

---

## What's NOT in WandB

```
metrics.pkl file     → NOT uploaded as artifact
CSV file             → NOT uploaded as artifact

Both saved locally only for backup.
```

---

## Summary

✅ **Everything is in WandB as individual metrics**
✅ **Complete timeseries data available**
✅ **All summary metrics logged**
✅ **Professional visualizations included**
❌ **No file artifacts**
❌ **No pickle or CSV uploads**

Perfect setup for analysis and monitoring!


