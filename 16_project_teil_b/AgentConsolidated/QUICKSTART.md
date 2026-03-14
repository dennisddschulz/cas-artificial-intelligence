# Quick Start Guide

## 5-Minute Setup

### Step 1: Install Dependencies
```bash
pip install torch gymnasium yfinance scikit-learn pandas numpy matplotlib seaborn wandb
```

### Step 2: Navigate to Project
```bash
cd /home/isc-den/cas-artificial-intelligence/16_project_teil_b/AgentConsolidated/
```

### Step 3: Run Experiments (Choose One)

#### Option A: Jupyter Notebook (Recommended)
```bash
jupyter notebook Parameterized_Experiments_Master.ipynb
```
Then run cells sequentially to see live results and visualizations.

#### Option B: Command Line
```bash
python run_experiments.py
```
All experiments run automatically. Results saved to `./experiments/run_TIMESTAMP/`

#### Option C: Individual Experiment
```python
python3 << 'EOF'
from trading_config import get_ppo_without_forecast_config
from trading_framework import ExperimentRunner

config = get_ppo_without_forecast_config()
runner = ExperimentRunner(config)
results = runner.run()

print(f"Total Return: {results['metrics']['total_return']*100:.2f}%")
print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.4f}")
EOF
```

---

## What Gets Compared

| Experiment | Forecast | Reward Type | Purpose |
|------------|----------|------------|---------|
| PPO Without Forecast | ❌ | WITH_RISK | Baseline RL |
| PPO With Forecast | ✅ LSTM | WITH_RISK | Forecast impact |
| PPO Reward - BASIC | ❌ | BASIC | Ablation study |
| PPO Reward - WITH_RISK | ❌ | WITH_RISK | Ablation study |
| PPO Reward - WITH_SHARPE | ❌ | WITH_SHARPE | Ablation study |
| PPO Reward - RISK_ADJUSTED | ❌ | RISK_ADJUSTED | Ablation study |

---

## Key Metrics (6 Required Metrics)

✅ **Cumulative Return** - Total profit percentage
✅ **Sharpe Ratio** - Risk-adjusted return
✅ **Max Drawdown** - Largest peak-to-trough loss
✅ **Volatility** - Daily return standard deviation
✅ **Turnover** - Total position changes
+ **Win Rate** - % of profitable days (bonus)

---

## Expected Output

### 1. Console Output
```
================================================================================
EXPERIMENT 1: PPO WITHOUT FORECAST (BASELINE)
================================================================================

Loading Data...
✓ Data loaded: 2580 days
Adding features...
✓ Features added: 2577 rows remaining

PPO TRAINING
Initial Equity: $100,000
Total Updates: 3000

Update    0: Return=    -5.43±  45.22
Update  100: Return=    12.45±  38.90
Update  200: Return=    28.34±  32.15
...
Update 2900: Return=    35.67±  28.45

EVALUATION ON TEST SET
Total Return: 18.45%
Sharpe Ratio: 0.7234
Max Drawdown: -18.34%
Volatility: 2.45%
Turnover: 1.2345

✓ Experiment 1 completed
```

### 2. Generated Plots
- `equity_curves_comparison.png` - All strategies' equity growth
- `drawdown_comparison.png` - Maximum drawdown over time
- `returns_distribution.png` - Daily returns histograms
- `metrics_heatmap.png` - Color-coded metrics table

### 3. CSV Export
```
metrics_comparison.csv
───────────────────────────────────────────────────────────
Experiment,total_return,sharpe_ratio,max_drawdown,...
PPO_Without_Forecast,0.1845,0.7234,-0.1834,...
PPO_With_Forecast,0.2156,0.8342,-0.1567,...
PPO_Reward_Basic,0.1456,0.6234,-0.2134,...
...
```

---

## Customization Examples

### Example 1: Different Initial Budget
```python
from trading_config import get_ppo_without_forecast_config

config = get_ppo_without_forecast_config()
config.environment.initial_equity = 50000  # Change to $50,000
# or
config.environment.initial_equity = 250000  # Change to $250,000
```

### Example 2: Different Ticker
```python
config = get_ppo_without_forecast_config()
config.data.ticker = "ETH-USD"  # Trade Ethereum instead
```

### Example 3: More Training Steps
```python
config = get_ppo_without_forecast_config()
config.ppo.total_updates = 5000  # Train longer
```

### Example 4: Different Fee Structure
```python
config = get_ppo_without_forecast_config()
config.environment.fee = 0.0005  # 0.05% per trade
```

---

## Interpreting Results

### Good Results
- **Total Return**: > 15%
- **Sharpe Ratio**: > 0.5
- **Max Drawdown**: > -25%
- **Turnover**: < 5.0

### Excellent Results
- **Total Return**: > 30%
- **Sharpe Ratio**: > 1.0
- **Max Drawdown**: > -15%
- **Turnover**: < 2.0

### Comparing Forecast Impact
```
Forecast Impact = Return(With) - Return(Without)

Positive → Forecast helps
Negative → Forecast hurts
~0 → Forecast has no effect
```

---

## Troubleshooting

### Problem: "CUDA out of memory"
**Solution**: Use CPU instead
```python
# Force CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### Problem: Data download fails
**Solution**: Use shorter date range
```python
config.data.start_date = "2022-01-01"
config.data.end_date = "2024-01-01"
```

### Problem: Training is too slow
**Solution**: Reduce parallelization
```python
config.ppo.num_envs = 4  # Default: 8
config.ppo.n_steps = 128  # Default: 256
```

### Problem: Results look unrealistic
**Solution**: Check random seed
```python
config.seed = 42  # Default is deterministic
```

---

## File Structure After Running

```
AgentConsolidated/
├── Parameterized_Experiments_Master.ipynb  (INPUT: Jupyter notebook)
├── run_experiments.py                      (INPUT: Runner script)
├── trading_config.py                       (INPUT: Config module)
├── trading_framework.py                    (INPUT: Framework)
├── trading_metrics.py                      (INPUT: Metrics)
│
└── experiments/
    └── run_20260311_143025/
        ├── PPO_Without_Forecast/
        │   ├── metrics.csv
        │   ├── equity.npy
        │   └── ...
        ├── PPO_With_Forecast/
        │   └── ...
        ├── PPO_Reward_Basic/
        │   └── ...
        ├── all_metrics.csv
        ├── equity_curves.png
        ├── drawdown.png
        ├── metrics_heatmap.png
        └── report.html
```

---

## W&B Logging

### View Results Online
```bash
# After running with W&B
wandb.ai/btcprojekt2026-bfh/PPO_Bitcoin_Trading
```

### View Offline Results
```bash
# List offline runs
wandb offline-runs

# Sync to cloud
wandb sync ./wandb/offline-run-*/
```

---

## Next Steps

1. **Run baseline experiments** - Get baseline results
2. **Analyze plots** - Understand which strategy performs best
3. **Compare metrics** - Look at detailed performance table
4. **Customize config** - Try different parameters
5. **Share results** - Export to CSV/JSON/HTML

---

## Time Estimates

| Action | Time | System |
|--------|------|--------|
| Install deps | 2 min | Any |
| Run 1 experiment | 30-60 min | GPU: 10 min |
| Run all 6 experiments | 3-6 hours | GPU: 1 hour |
| Generate plots | 2-5 min | CPU |
| Full analysis | 5-10 min | CPU |

---

## Support Resources

- **Detailed README**: See `README.md`
- **Configuration Docs**: See `trading_config.py` docstrings
- **Metrics Docs**: See `trading_metrics.py` docstrings
- **Original Notebook**: `Project_Part_3_Final_Architecture.ipynb`

---

**Ready to trade? Run this command:**

```bash
jupyter notebook Parameterized_Experiments_Master.ipynb
```

Happy experimenting! 🚀📈

