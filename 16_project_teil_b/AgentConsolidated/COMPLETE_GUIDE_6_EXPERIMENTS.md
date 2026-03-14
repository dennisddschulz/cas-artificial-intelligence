
# Complete PPO Trading Framework - 6 Experiments Suite

## Overview

This framework runs **6 comprehensive PPO trading experiments** with:
- **Real-time console metrics** during training
- **Budget tracking with liquidity monitoring**
- **Detailed local plots** for each experiment
- **WandB integration** with distinct group IDs for comparison

## The 6 Experiments

### Experiment 1: PPO Without Forecast (Baseline)
- **WandB Group**: `ppo_baseline_no_forecast`
- **Configuration**: Pure RL agent without price forecasting
- **Reward Type**: WITH_RISK (PnL - Cost - Risk Penalty)
- **Purpose**: Establish baseline performance

### Experiment 2: PPO With Forecast
- **WandB Group**: `ppo_baseline_with_forecast`
- **Configuration**: RL agent with LSTM price prediction
- **Reward Type**: WITH_RISK
- **Purpose**: Measure value of forecast information

### Experiment 3: PPO Basic Reward
- **WandB Group**: `ppo_basic_reward`
- **Configuration**: Simple reward = PnL - Cost
- **Reward Type**: BASIC
- **Purpose**: Baseline without risk penalty

### Experiment 4: PPO With Risk
- **WandB Group**: `ppo_with_risk`
- **Configuration**: Standard setup with risk adjustment
- **Reward Type**: WITH_RISK
- **Purpose**: Control experiment (repeat Exp 1)

### Experiment 5: PPO With Sharpe Ratio
- **WandB Group**: `ppo_with_sharpe`
- **Configuration**: Reward scaled by volatility
- **Reward Type**: WITH_SHARPE
- **Purpose**: Test Sharpe-optimized reward

### Experiment 6: PPO Risk-Adjusted
- **WandB Group**: `ppo_risk_adjusted`
- **Configuration**: PnL divided by volatility
- **Reward Type**: RISK_ADJUSTED
- **Purpose**: Maximize risk-adjusted returns

## Running the Experiments

### Quick Start (All 6 Experiments)

```bash
python run_all_experiments.py
```

This will:
1. Run all 6 experiments sequentially
2. Print real-time metrics to console every 100 updates
3. Save metrics.pkl files in `./results/EXPERIMENT_NAME/`
4. Generate detailed plots in `./plots/`
5. Log metrics to WandB with group organization

### Run Single Experiment

```python
from trading_config import get_all_experiments
from trading_framework import ExperimentRunner

all_exp = get_all_experiments()
config = all_exp['PPO_Without_Forecast']

runner = ExperimentRunner(config)
results = runner.run()
```

## Real-Time Console Metrics

Every 100 updates, you'll see:

```
================================================================================
UPDATE   100 /  3000
================================================================================
Episode Returns (last 100): 123.45 ± 56.78
Total Episodes Trained:          1200
Log Std (exploration):         -0.523
Policy Loss:                   0.0234
Value Loss:                    0.1567
Entropy Loss:                  0.0432
Total Loss:                    0.3421
Approx KL:                     0.0125
================================================================================
```

### Metrics Explained

- **Episode Returns**: Average cumulative return from last 100 completed episodes
- **Log Std**: Exploration variance (lower = more deterministic)
- **Policy Loss**: Policy gradient loss
- **Value Loss**: Value function prediction error
- **Entropy Loss**: Entropy regularization for exploration
- **Total Loss**: Combined loss for optimization
- **Approx KL**: Kullback-Leibler divergence for early stopping

## Budget Tracking & Liquidity

### Initial Capital
All experiments start with **$100,000 USD**

### Tracked Metrics
For each step, the framework tracks:
- **Equity**: Current portfolio value
- **Position**: Current leverage/position size
- **Cost**: Transaction costs (fees)
- **PnL**: Profit and loss from position
- **Drawdown**: Peak-to-trough decline
- **Cumulative Return**: Total return percentage

### Equity Update Formula
```
equity = equity * exp(pnl - cost - slippage)
```

### Budget Constraints
- **Fee**: 0.0001 per unit position change
- **Kappa**: 0.01 risk penalty weight
- **Leverage Max**: 1.0 (no leverage)
- **Slippage**: 0.0 (no market impact)

## Output Files

### For Each Experiment

**In `./results/EXPERIMENT_NAME/`:**

```
├── metrics.pkl                 # Serialized metrics (load with pickle)
├── metrics_summary.csv         # CSV format metrics
├── budget_report.csv          # Detailed budget tracking
├── budget_summary.txt         # Text summary
├── 01_equity_curve.png        # Equity evolution from $100K
├── 02_budget_breakdown.png    # Position vs equity over time
├── 03_transaction_costs.png   # Daily and cumulative costs
└── 04_returns_drawdown.png    # Returns with drawdown overlay
```

### Aggregated Files

**In `./plots/`:**

```
├── 01_metrics_comparison.png  # Bar chart of all experiments
├── 02_equity_curves.png       # All equity curves overlaid
├── 03_returns_distribution.png # Histogram of daily returns
├── 04_metrics_heatmap.png     # Color-coded metrics table
└── metrics_report.txt         # Text comparison report
```

## Loading and Inspecting Metrics Locally

### Python Script

```bash
python load_and_inspect_metrics.py
```

This loads all metrics.pkl files and:
1. Prints summary for each experiment
2. Creates comparison table
3. Generates equity curve plots
4. Saves CSV comparison

### Python Code

```python
import pickle

# Load single experiment
with open('./results/PPO-Without-Forecast_20250312_120000/metrics.pkl', 'rb') as f:
    data = pickle.load(f)

# Access metrics
metrics = data['metrics']
print(f"Return: {metrics['total_return']*100:.2f}%")
print(f"Sharpe: {metrics['sharpe_ratio']:.4f}")
print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")

# Access equity curve
equity_curve = data['equity_curve']
print(f"Initial: {equity_curve[0]:,.2f}")
print(f"Final: {equity_curve[-1]:,.2f}")
```

## Metrics Available

### Risk-Adjusted Returns
- `total_return`: (Final - Initial) / Initial
- `annualized_return`: Daily mean * 252
- `sharpe_ratio`: Excess return / volatility
- `sortino_ratio`: Excess return / downside volatility
- `calmar_ratio`: Annualized return / max drawdown

### Risk Metrics
- `volatility`: Daily returns std
- `annualized_volatility`: Daily std * sqrt(252)
- `max_drawdown`: Maximum peak-to-trough
- `kurtosis`: Distribution tail heaviness
- `skewness`: Distribution asymmetry

### Trading Metrics
- `win_rate`: % of positive days
- `profit_factor`: Gains / Losses ratio
- `turnover`: Total position changes
- `total_costs`: Sum of fees paid
- `cost_ratio`: Costs / Initial equity

## WandB Integration

### Offline Mode
All metrics are logged to WandB in **offline mode** to avoid SSL issues:
```
./wandb/offline-run-*/
```

### Groups for Distinction

Each experiment logs to a unique group:
- `ppo_baseline_no_forecast`
- `ppo_baseline_with_forecast`
- `ppo_basic_reward`
- `ppo_with_risk`
- `ppo_with_sharpe`
- `ppo_risk_adjusted`

### Syncing to Cloud

After experiments complete:
```bash
wandb sync ./wandb/offline-run-*/
```

Then view in W&B online dashboard with filters by group.

## Expected Results

With default parameters, expect:

| Metric | Typical Range |
|--------|---------------|
| Total Return | -5% to +15% |
| Sharpe Ratio | -0.5 to 2.0 |
| Max Drawdown | -20% to -5% |
| Volatility | 0.5% to 2.0% |
| Win Rate | 40% to 55% |

**Variation is normal** - RL training is stochastic.

## Troubleshooting

### No Plots Generated
- Check `./results/` has metrics.pkl files
- Ensure matplotlib can save images
- Run: `python load_and_inspect_metrics.py`

### Negative Returns
- Expected when market is declining
- Check `sharpe_ratio` and `calmar_ratio` for risk-adjusted performance
- Compare with benchmark (buy-and-hold)

### WandB Not Logging
- Check `./wandb/offline-run-*/` exists
- Ensure `use_wandb=True` in config
- Review WandB initialization in console output

### Memory Issues
- Reduce `num_envs` (default 8)
- Reduce `n_steps` (default 256)
- Set `total_updates` lower for testing

## File Structure

```
AgentConsolidated/
├── trading_config.py              # 6 experiment configs
├── trading_framework.py            # PPO implementation
├── trading_metrics.py              # Metrics calculation
├── budget_tracker.py               # Budget tracking
├── visualize_metrics.py            # Visualization tools
├── load_and_inspect_metrics.py    # Metric inspection
├── run_all_experiments.py          # Master script
├── FRAMEWORK_GUIDE.md              # This file
├── requirements.txt                # Dependencies
│
├── results/                        # Experiment outputs
│   ├── PPO-Without-Forecast_20250312_120000/
│   ├── PPO-With-Forecast_20250312_121500/
│   ├── PPO-Basic-Reward_20250312_130000/
│   ├── PPO-With-Risk_20250312_135000/
│   ├── PPO-With-Sharpe_20250312_140000/
│   └── PPO-Risk-Adjusted_20250312_145000/
│
├── plots/                          # Aggregated visualizations
│   ├── 01_metrics_comparison.png
│   ├── 02_equity_curves.png
│   ├── 03_returns_distribution.png
│   ├── 04_metrics_heatmap.png
│   └── metrics_report.txt
│
└── wandb/                          # WandB offline runs
    ├── offline-run-*/
    └── ...
```

## Performance Tips

1. **Testing**: Set `total_updates=100` for quick validation
2. **GPU**: Automatically uses CUDA if available
3. **Speed**: Reduce `num_envs=2` on CPU-only systems
4. **Reproducibility**: Set `seed=42` (default)

## Next Steps After Running

1. **Compare Results**
   ```bash
   python load_and_inspect_metrics.py
   ```

2. **Review Plots**
   - Check `./plots/01_metrics_comparison.png`
   - View equity curves in `./plots/02_equity_curves.png`

3. **Analyze Metrics**
   - Open `./plots/metrics_report.txt`
   - Compare Sharpe ratios across experiments

4. **Sync to WandB**
   ```bash
   wandb sync ./wandb/offline-run-*/
   ```

5. **Modify & Re-run**
   - Adjust hyperparameters in `trading_config.py`
   - Run `python run_all_experiments.py` again

---

**Framework Version**: 2.0 Complete
**Last Updated**: 2025-03-12
**Status**: Production Ready for 6-Experiment Suite

