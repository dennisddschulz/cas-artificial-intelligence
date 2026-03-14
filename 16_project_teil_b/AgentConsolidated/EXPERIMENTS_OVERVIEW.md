
# 📊 6 Experiments Overview Table

## Complete Experiment Configuration

| # | Type | Forecast | Reward Type | Purpose | WandB Group | Config Name |
|---|------|----------|-------------|---------|-------------|-------------|
| 1 | PPO | ❌ None | WITH_RISK | **Baseline** - Pure RL without forecasting | `ppo_baseline_no_forecast` | `PPO_Without_Forecast` |
| 2 | PPO | ✅ LSTM | WITH_RISK | **Forecast Impact** - Test value of price prediction | `ppo_baseline_with_forecast` | `PPO_With_Forecast` |
| 3 | PPO | ❌ None | BASIC | **Reward Ablation** - Simple PnL - Cost (no risk penalty) | `ppo_basic_reward` | `PPO_Basic_Reward` |
| 4 | PPO | ❌ None | WITH_RISK | **Reward Ablation** - Standard (duplicate of Exp 1 for validation) | `ppo_with_risk` | `PPO_With_Risk` |
| 5 | PPO | ❌ None | WITH_SHARPE | **Reward Ablation** - Sharpe ratio optimization | `ppo_with_sharpe` | `PPO_With_Sharpe` |
| 6 | PPO | ❌ None | RISK_ADJUSTED | **Reward Ablation** - Risk-adjusted returns (PnL/Volatility) | `ppo_risk_adjusted` | `PPO_Risk_Adjusted` |

---

## Detailed Configuration Breakdown

### Experiment 1: PPO Without Forecast (Baseline)
```python
Config Name: PPO_Without_Forecast
Forecast: ForecastMode.NONE
Reward: RewardType.WITH_RISK
WandB Group: ppo_baseline_no_forecast
Purpose: Establish baseline RL agent performance without price forecasting
Reward Formula: PnL - Cost - Risk_Penalty
Key Feature: Pure market-driven decision making
```

### Experiment 2: PPO With Forecast
```python
Config Name: PPO_With_Forecast
Forecast: ForecastMode.LSTM
Reward: RewardType.WITH_RISK
WandB Group: ppo_baseline_with_forecast
Purpose: Measure impact of LSTM price direction forecasting
Reward Formula: PnL - Cost - Risk_Penalty (same as Exp 1)
Key Feature: LSTM model predicts price direction, agent uses forecast as input
```

### Experiment 3: PPO Basic Reward
```python
Config Name: PPO_Basic_Reward
Forecast: ForecastMode.NONE
Reward: RewardType.BASIC
WandB Group: ppo_basic_reward
Purpose: Test simple reward without risk penalty
Reward Formula: PnL - Cost (no kappa penalty)
Key Feature: Minimize costs only, no risk adjustment
```

### Experiment 4: PPO With Risk
```python
Config Name: PPO_With_Risk
Forecast: ForecastMode.NONE
Reward: RewardType.WITH_RISK
WandB Group: ppo_with_risk
Purpose: Validation control (repeat of Exp 1)
Reward Formula: PnL - Cost - Risk_Penalty (kappa = 0.01)
Key Feature: Same as Exp 1 for reproducibility check
```

### Experiment 5: PPO With Sharpe
```python
Config Name: PPO_With_Sharpe
Forecast: ForecastMode.NONE
Reward: RewardType.WITH_SHARPE
WandB Group: ppo_with_sharpe
Purpose: Test Sharpe ratio optimization
Reward Formula: (PnL - Cost) / Volatility - Risk_Penalty
Key Feature: Volatility-adjusted returns
```

### Experiment 6: PPO Risk-Adjusted
```python
Config Name: PPO_Risk_Adjusted
Forecast: ForecastMode.NONE
Reward: RewardType.RISK_ADJUSTED
WandB Group: ppo_risk_adjusted
Purpose: Test alternative risk adjustment method
Reward Formula: (PnL / Volatility) - Cost
Key Feature: Direct risk adjustment without penalty term
```

---

## Experiment Comparison Matrix

### Forecast Dimension
```
Experiments 1, 3, 4, 5, 6: NO FORECAST ❌
Experiment 2: LSTM FORECAST ✅
```

### Reward Function Dimension
```
Experiment 1, 4: WITH_RISK (PnL - Cost - Risk_Penalty)
Experiment 2: WITH_RISK (with Forecast)
Experiment 3: BASIC (PnL - Cost only)
Experiment 5: WITH_SHARPE (volatility-adjusted)
Experiment 6: RISK_ADJUSTED (alternative risk adjustment)
```

### Purpose Matrix
```
BASELINE:
  - Exp 1: Main baseline (no forecast, WITH_RISK)

FORECAST IMPACT:
  - Exp 2: Compare with forecast vs Exp 1

REWARD ABLATION:
  - Exp 3: Test simple reward (BASIC)
  - Exp 4: Validation control (repeat Exp 1)
  - Exp 5: Test Sharpe optimization
  - Exp 6: Test alternative risk adjustment
```

---

## Running the Experiments

### All 6 Experiments Together
```bash
python run_all_experiments.py
```
Runs experiments 1-6 in sequence.
Duration: ~2-3 hours
Output: 6 result directories in `./results/`

### Individual Experiment
```python
from trading_config import get_all_experiments
from trading_framework import ExperimentRunner

experiments = get_all_experiments()

# Run Experiment 2 (PPO With Forecast)
config = experiments['PPO_With_Forecast']
runner = ExperimentRunner(config)
results = runner.run()
```

---

## Expected Comparison Results

### What We'll Learn

**From Experiments 1 vs 2:**
- Impact of LSTM forecasting on trading performance
- Does price direction prediction improve returns?

**From Experiments 1, 3:**
- Impact of risk penalty (WITH_RISK vs BASIC)
- Does penalizing position volatility help?

**From Experiments 1, 4:**
- Reproducibility of baseline configuration
- Are results stable across runs?

**From Experiments 1, 5, 6:**
- Which reward formulation works best?
- Sharpe optimization vs Risk-adjusted vs Standard risk

---

## Output Organization

Each experiment creates:
```
./results/CONFIG_NAME_TIMESTAMP/
  ├── metrics.pkl                 # Full metrics (Python pickle)
  ├── metrics_summary.csv         # Metrics table
  ├── budget_report.csv           # Detailed budget tracking
  ├── 01_equity_curve.png         # Equity evolution from $100K
  ├── 02_budget_breakdown.png     # Position vs equity
  ├── 03_transaction_costs.png    # Cost analysis
  └── 04_returns_drawdown.png     # Returns with drawdown
```

Aggregated comparison:
```
./plots/
  ├── 01_metrics_comparison.png   # All 6 experiments
  ├── 02_equity_curves.png        # Overlaid equity curves
  ├── 03_returns_distribution.png # Daily returns
  ├── 04_metrics_heatmap.png      # Metrics heatmap
  └── metrics_report.txt          # Text summary
```

---

## Key Parameters (All Experiments)

| Parameter | Value |
|-----------|-------|
| Initial Equity | $100,000 |
| Transaction Fee | 0.0001 |
| Risk Penalty (Kappa) | 0.01 |
| Max Leverage | 1.0 |
| PPO Updates | 3,000 |
| Parallel Environments | 8 |
| Steps per Rollout | 256 |
| Learning Rate | 1e-4 |
| Seed | 42 |

---

## WandB Dashboard Organization

### Groups for Filtering
All 6 experiments available in WandB with distinct groups:

```
ppo_baseline_no_forecast        ← Experiment 1
ppo_baseline_with_forecast      ← Experiment 2
ppo_basic_reward                ← Experiment 3
ppo_with_risk                   ← Experiment 4
ppo_with_sharpe                 ← Experiment 5
ppo_risk_adjusted               ← Experiment 6
```

Filter by group in WandB UI to compare specific experiments.

---

## Summary: Experimental Design

| Dimension | Purpose | Experiments |
|-----------|---------|-------------|
| **Forecast Impact** | Test LSTM value | Exp 1 vs Exp 2 |
| **Risk Penalty** | Test WITH_RISK benefit | Exp 1 vs Exp 3 |
| **Reward Functions** | Ablate reward design | Exp 1, 5, 6 |
| **Reproducibility** | Validate setup | Exp 1 vs Exp 4 |

**Final Goal:** Determine optimal configuration for trading with PPO.


