
# PPO Trading Experiments - Complete Framework

## 🚀 Quick Start

```bash
# 1. Validate setup
python validate_framework.py

# 2. Run all 6 experiments
python run_all_experiments.py

# 3. Inspect metrics locally
python load_and_inspect_metrics.py

# 4. View plots
ls ./plots/
```

## 📊 The 6 Experiments

| # | Experiment | Group | Purpose |
|---|-----------|-------|---------|
| 1 | PPO Without Forecast | `ppo_baseline_no_forecast` | Baseline RL agent |
| 2 | PPO With Forecast | `ppo_baseline_with_forecast` | RL + LSTM prediction |
| 3 | PPO Basic Reward | `ppo_basic_reward` | Simple PnL - Cost |
| 4 | PPO With Risk | `ppo_with_risk` | Standard WITH_RISK |
| 5 | PPO With Sharpe | `ppo_with_sharpe` | Sharpe-optimized |
| 6 | PPO Risk-Adjusted | `ppo_risk_adjusted` | PnL / Volatility |

## 💰 Budget & Liquidity Tracking

- **Initial Capital**: $100,000 USD
- **Real-time Tracking**: Equity, positions, costs, PnL, drawdown
- **Constraints**: Fee=0.0001, Kappa=0.01, MaxLeverage=1.0
- **Equity Update**: `equity = equity * exp(pnl - cost - slippage)`

## 📈 Real-Time Console Metrics

Every 100 updates, you'll see:
- Episode returns (last 100 episodes)
- Policy/Value/Entropy losses
- Exploration level (log_std)
- KL divergence (for early stopping)

Example output:
```
================================================================================
UPDATE   100 /  3000
================================================================================
Episode Returns (last 100): 123.45 ± 56.78
Policy Loss:                   0.0234
Value Loss:                    0.1567
Entropy Loss:                  0.0432
Total Loss:                    0.3421
```

## 📁 Output Files

### Per Experiment (in `./results/EXPERIMENT_NAME/`)
- `metrics.pkl` - Full metrics (load with pickle)
- `metrics_summary.csv` - CSV format
- `budget_report.csv` - Detailed tracking
- `01_equity_curve.png` - Equity evolution
- `02_budget_breakdown.png` - Position vs equity
- `03_transaction_costs.png` - Costs analysis
- `04_returns_drawdown.png` - Returns with drawdown

### Aggregated (in `./plots/`)
- `01_metrics_comparison.png` - All experiments comparison
- `02_equity_curves.png` - Overlaid equity curves
- `03_returns_distribution.png` - Daily returns histograms
- `04_metrics_heatmap.png` - Color-coded metrics
- `metrics_report.txt` - Text summary

## 📊 Metrics Calculated

### Per Experiment
- Total Return
- Sharpe Ratio
- Max Drawdown
- Volatility
- Win Rate
- Profit Factor
- Turnover
- **+ 8 more risk metrics**

All saved to:
1. `metrics.pkl` (for local analysis)
2. WandB (with group distinction)
3. CSV files
4. Plots (PNG)

## 🌐 WandB Integration

Each experiment logs to a **distinct group** for easy filtering:

```
ppo_baseline_no_forecast ─────┐
ppo_baseline_with_forecast ───┤
ppo_basic_reward ─────────────┼─→ WandB Dashboard
ppo_with_risk ────────────────┤
ppo_with_sharpe ──────────────┤
ppo_risk_adjusted ────────────┘
```

**Offline mode** (no SSL issues):
```bash
# After running, sync to cloud:
wandb sync ./wandb/offline-run-*/
```

## 🔧 Configuration

Edit `trading_config.py` to customize:

```python
# Example: Modify PPO settings
from trading_config import get_all_experiments

experiments = get_all_experiments()
config = experiments['PPO_Without_Forecast']

# Change parameters
config.ppo.total_updates = 1000  # Shorter training
config.ppo.num_envs = 4  # Fewer parallel envs
config.environment.fee = 0.0005  # Higher fees

# Run
runner = ExperimentRunner(config)
results = runner.run()
```

## 📚 Key Files

| File | Purpose |
|------|---------|
| `run_all_experiments.py` | Master script - runs all 6 |
| `trading_config.py` | 6 experiment configurations |
| `trading_framework.py` | PPO implementation |
| `trading_metrics.py` | Metrics calculation |
| `budget_tracker.py` | Budget & equity tracking |
| `visualize_metrics.py` | Plot generation |
| `load_and_inspect_metrics.py` | Local metric inspection |
| `validate_framework.py` | Setup validation |

## 🎯 Expected Output

### Console During Training
```
================================================================================
PPO TRAINING - REALTIME METRICS
================================================================================
Initial Equity: $100,000.00
Total Updates: 3000
Parallel Environments: 8
Fee: 0.0001 | Kappa: 0.01
================================================================================

[Progress updates every 100 steps...]

UPDATE   100 /  3000
Episode Returns (last 100): 456.78 ± 234.56
Policy Loss:                   0.0456
...
```

### After Completion
```
EXPERIMENTS COMPARISON TABLE
─────────────────────────────────────────────
Experiment              Group                Return   Sharpe
─────────────────────────────────────────────
PPO-Without-Forecast    baseline_no_forecast   3.43%   -0.0218
PPO-With-Forecast       baseline_with_forecast 5.12%    0.1234
PPO-Basic-Reward        basic_reward           2.15%   -0.1567
PPO-With-Risk           with_risk              4.56%    0.0789
PPO-With-Sharpe         with_sharpe            6.23%    0.2341
PPO-Risk-Adjusted       risk_adjusted          5.89%    0.1956
─────────────────────────────────────────────

✓ Results saved to: ./results/
✓ Plots saved to: ./plots/
✓ WandB offline runs saved to: ./wandb/
```

## 📖 How to Use

### 1. Validate
```bash
python validate_framework.py
```
Checks all dependencies, file structure, and 6 experiment configs.

### 2. Run
```bash
python run_all_experiments.py
```
Runs all 6 experiments in sequence with real-time metrics.

### 3. Analyze Locally
```bash
python load_and_inspect_metrics.py
```
Loads metrics.pkl files and generates local plots.

### 4. View Results
- **Plots**: Check `./plots/` folder
- **Metrics**: Review `./results/*/metrics_summary.csv`
- **Details**: Open `./results/*/budget_report.csv`

### 5. Sync to Cloud (Optional)
```bash
wandb sync ./wandb/offline-run-*/
```
Upload offline runs to WandB dashboard for online analysis.

## ⏱️ Execution Time

| Component | Time |
|-----------|------|
| Data download | 2-5 min |
| LSTM forecast | 5-10 min |
| PPO training (per exp) | 15-30 min |
| 6 Experiments Total | ~2-3 hours |
| Visualization | 2-5 min |

## 🐛 Troubleshooting

### Q: No metrics.pkl files
**A**: Check if experiments ran successfully:
```bash
ls ./results/
```

### Q: Memory error
**A**: Reduce parameters:
```python
config.ppo.num_envs = 2  # Was 8
config.ppo.n_steps = 64   # Was 256
```

### Q: Negative returns
**A**: Normal in market declines. Check Sharpe/Calmar ratios for risk-adjusted performance.

### Q: WandB not logging
**A**: Check offline mode is enabled:
```bash
ls ./wandb/offline-run-*/
```

## 📝 Example Metrics Output

```python
{
    'total_return': 0.0343,              # 3.43%
    'sharpe_ratio': -0.0218,
    'max_drawdown': -0.1390,             # -13.90%
    'volatility': 0.0054,                # 0.54%
    'annualized_return': 0.0876,         # 8.76%
    'annualized_volatility': 0.0857,
    'win_rate': 0.5234,                  # 52.34%
    'profit_factor': 1.1234,
    'turnover': 9.0910,
    'cost_ratio': 0.000345,              # 0.0345% of initial
    'calmar_ratio': 0.2456,
    'sortino_ratio': 0.1123,
}
```

## 🎓 Learning Resources

- **Framework Guide**: See `COMPLETE_GUIDE_6_EXPERIMENTS.md`
- **Budget Tracking**: See budget_tracker.py docstrings
- **Metrics**: See trading_metrics.py for all calculations
- **Configuration**: See trading_config.py for all options

## 🔐 Reproducibility

All experiments use:
- Fixed seed: 42
- Same ticker: BTC-USD
- Same date range: 2018-01-01 to present
- Same split: 60% train, 20% val, 20% test

For exact reproduction:
```python
config.seed = 42
config.data.start_date = "2018-01-01"
config.data.end_date = None  # Use all available data
```

## ✅ Success Checklist

- [ ] `python validate_framework.py` passes
- [ ] `python run_all_experiments.py` completes
- [ ] `./results/` has 6 subdirectories
- [ ] Each has `metrics.pkl` and PNG files
- [ ] `./plots/` has aggregated visualizations
- [ ] `./wandb/` has offline runs
- [ ] Can run `python load_and_inspect_metrics.py`
- [ ] Can view plots in `./plots/`

## 📞 Support

For issues or improvements:
1. Check `validate_framework.py` output
2. Review console error messages
3. Check file permissions in `./results/`
4. Ensure sufficient disk space (~5GB for all experiments)

---

**Framework Version**: 2.0 - Complete 6-Experiment Suite
**Status**: ✓ Production Ready
**Last Updated**: 2025-03-12
**Verified With**: Python 3.8+, PyTorch 1.9+, CUDA 11.0+

