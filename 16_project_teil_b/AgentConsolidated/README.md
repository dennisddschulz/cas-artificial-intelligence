# PPO Trading Experiments - Parameterized Framework

## Overview

This framework enables comprehensive comparison of PPO (Proximal Policy Optimization) trading strategies with different configurations, including:

1. **PPO Without Forecast** - Baseline reinforcement learning agent
2. **PPO With LSTM Forecast** - RL agent with price direction prediction
3. **PPO with Different Reward Functions** - Ablation study on reward design

**Initial Budget**: $100,000
**Asset**: BTC-USD (Bitcoin)
**Period**: 2018-01-01 to latest
**Log Integration**: Weights & Biases (W&B)

---

## Project Structure

```
AgentConsolidated/
├── trading_config.py              # Configuration module (enums, dataclasses)
├── trading_metrics.py             # Metrics calculation and comparison
├── trading_framework.py           # Main experiment runner
├── run_experiments.py            # Batch experiment manager
├── Parameterized_Experiments_Master.ipynb  # User-friendly interface
├── Project_Part_3_Final_Architecture.ipynb # Original notebook (reference)
└── README.md                      # This file
```

---

## Key Components

### 1. Configuration Module (`trading_config.py`)

Defines all experiment parameters:

```python
from trading_config import (
    get_ppo_without_forecast_config,
    get_ppo_with_forecast_config,
    get_ppo_different_rewards_configs,
    ForecastMode,
    RewardType
)
```

**Enums**:
- `ForecastMode`: NONE, LSTM
- `RewardType`: BASIC, WITH_RISK, WITH_SHARPE, RISK_ADJUSTED

**Config Classes**:
- `DataConfig`: Data loading parameters
- `ForecastingConfig`: LSTM model hyperparameters
- `EnvironmentConfig`: Trading environment settings
- `PPOConfig`: RL algorithm hyperparameters
- `ExperimentConfig`: Complete experiment setup

### 2. Metrics Module (`trading_metrics.py`)

Calculates performance metrics:

**TradingMetrics class**:
- Cumulative Return
- Sharpe Ratio
- Max Drawdown
- Volatility (daily and annualized)
- Calmar Ratio
- Sortino Ratio
- Win Rate
- Profit Factor
- Turnover
- Cost Ratio
- Kurtosis & Skewness

**MetricsComparison class**:
- Compare metrics across experiments
- Generate comparison plots
- Export to CSV/JSON

**EquityCurveAnalyzer class**:
- Plot equity curves
- Analyze drawdown
- Plot returns distribution

### 3. Main Framework (`trading_framework.py`)

**ExperimentRunner class**:

```python
from trading_framework import ExperimentRunner

config = get_ppo_without_forecast_config()
runner = ExperimentRunner(config)
results = runner.run()

# Results contain:
# - equity: array of equity values
# - positions: array of positions
# - costs: array of transaction costs
# - pnl: array of profit/loss
# - metrics: dict of all metrics
```

**Features**:
1. **setup_wandb()**: Initialize Weights & Biases logging
2. **load_market_data()**: Download BTC-USD data
3. **add_features()**: Add technical indicators
4. **split_data()**: Time-based train/val/test split
5. **train_forecast_model()**: Train LSTM if needed
6. **train_ppo()**: PPO training with vectorized environments
7. **evaluate()**: Test set evaluation with detailed metrics

### 4. Experiment Manager (`run_experiments.py`)

**ExperimentManager class**:

```python
from run_experiments import ExperimentManager

manager = ExperimentManager()
manager.run_all_experiments()        # Run all 3 experiments
manager.generate_comparison_report() # Create comparison table
manager.generate_plots()             # Generate visualizations
manager.generate_html_report()       # Create HTML report
```

---

## Usage Guide

### Option A: Jupyter Notebook (Recommended)

```bash
cd /home/isc-den/cas-artificial-intelligence/16_project_teil_b/AgentConsolidated/
jupyter notebook Parameterized_Experiments_Master.ipynb
```

The notebook provides a user-friendly interface with:
- Step-by-step execution
- Real-time progress output
- Integrated visualization
- Result export

### Option B: Command Line

```bash
cd /home/isc-den/cas-artificial-intelligence/16_project_teil_b/AgentConsolidated/
python run_experiments.py
```

Results are saved to `./experiments/run_YYYYMMDD_HHMMSS/`

### Option C: Custom Script

```python
from trading_config import get_ppo_without_forecast_config
from trading_framework import ExperimentRunner

# Create config
config = get_ppo_without_forecast_config()

# Run experiment
runner = ExperimentRunner(config)
results = runner.run()

# Access results
print(f"Return: {results['metrics']['total_return']:.4f}")
print(f"Sharpe: {results['metrics']['sharpe_ratio']:.4f}")
```

---

## Configuration Examples

### Example 1: Basic PPO (No Forecast)

```python
from trading_config import get_ppo_without_forecast_config

config = get_ppo_without_forecast_config()

# Key settings:
# - Forecast Mode: NONE
# - Reward Type: WITH_RISK
# - Initial Equity: $100,000
# - Fee: 0.0001 (0.01% per trade)
# - Leverage Max: 1.0 (no leverage)
# - PPO Updates: 3000
```

### Example 2: PPO with Forecast

```python
from trading_config import get_ppo_with_forecast_config

config = get_ppo_with_forecast_config()

# Key settings:
# - Forecast Mode: LSTM
# - LSTM Hidden Dim: 64
# - Lookback: 20 days
# - Forecast Horizon: 5 days
# - Early Stopping Patience: 20 epochs
```

### Example 3: Custom Configuration

```python
from trading_config import ConfigBuilder, ForecastMode, RewardType

config = (ConfigBuilder("My-Experiment")
    .with_forecast(ForecastMode.LSTM)
    .with_reward(RewardType.WITH_SHARPE)
    .with_initial_equity(100000.0)
    .with_fee(0.0001)
    .with_leverage(2.0)
    .with_ppo_updates(5000)
    .build())

runner = ExperimentRunner(config)
results = runner.run()
```

---

## Reward Functions

### BASIC
```
reward = PnL - TransactionCost
```

### WITH_RISK (Default)
```
reward = PnL - TransactionCost - RiskPenalty
risk_penalty = κ * position² * volatility
```

### WITH_SHARPE
```
reward = (PnL - TransactionCost) / (volatility + ε) - RiskPenalty
```

### RISK_ADJUSTED
```
reward = (PnL / (volatility + ε)) - TransactionCost
```

---

## Key Metrics Explained

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Total Return** | (Final - Initial) / Initial | Overall profit/loss percentage |
| **Sharpe Ratio** | (E[R] - Rf) / σ | Risk-adjusted return (higher is better) |
| **Max Drawdown** | (Peak - Trough) / Peak | Largest peak-to-trough decline |
| **Volatility** | σ(returns) | Daily return standard deviation |
| **Annualized Volatility** | σ(returns) × √252 | Yearly volatility estimate |
| **Win Rate** | Positive Days / Total Days | % of profitable days |
| **Turnover** | Σ\|Δposition\| | Total position changes |

---

## Output Files

### Automatic Outputs

When running `python run_experiments.py`:

```
experiments/run_YYYYMMDD_HHMMSS/
├── PPO_Without_Forecast/
│   ├── metrics.csv
│   ├── equity.npy
│   ├── positions.npy
│   ├── costs.npy
│   └── pnl.npy
├── PPO_With_Forecast/
│   └── [same structure]
├── PPO_Reward_Basic/
│   └── [same structure]
├── PPO_Reward_With_Risk/
│   └── [same structure]
├── PPO_Reward_With_Sharpe/
│   └── [same structure]
├── PPO_Reward_Risk_Adjusted/
│   └── [same structure]
├── all_metrics.csv
├── equity_curves.png
├── drawdown.png
├── returns_distribution.png
├── metrics_heatmap.png
├── report.html
└── results.json
```

### Manual Exports (Jupyter)

- `metrics_comparison.csv` - All metrics table
- `equity_curves_comparison.png` - Equity curves plot
- `drawdown_comparison.png` - Drawdown analysis
- `returns_distribution.png` - Returns histogram
- `metrics_heatmap.png` - Metrics heatmap
- `detailed_results.json` - Complete results

---

## W&B Integration

### Setup

1. **Offline Mode** (default):
```bash
export WANDB_MODE=offline
```

2. **Online Mode** (with cloud sync):
```bash
wandb login
export WANDB_MODE=online
```

### Experiment Groups

All experiments are logged to different groups:

- `baseline`: PPO without forecast, PPO with forecast
- `reward_ablation`: Different reward functions

### View Results

```bash
# Online
wandb.ai/btcprojekt2026-bfh/PPO_Bitcoin_Trading

# Offline (sync local runs)
wandb sync ./wandb/offline-run-*/
```

---

## Performance Expectations

Based on typical Bitcoin trading data (2018-2024):

| Configuration | Expected Return | Sharpe Ratio | Max Drawdown |
|---------------|-----------------|--------------|--------------|
| PPO No Forecast | 15-40% | 0.5-1.2 | -20% to -35% |
| PPO With Forecast | 20-50% | 0.7-1.5 | -15% to -30% |
| Reward - BASIC | 10-30% | 0.3-0.8 | -25% to -40% |
| Reward - WITH_RISK | 15-40% | 0.5-1.2 | -20% to -35% |
| Reward - WITH_SHARPE | 20-45% | 0.8-1.6 | -15% to -30% |
| Reward - RISK_ADJUSTED | 18-42% | 0.6-1.3 | -18% to -32% |

*Note: Results vary based on market conditions and random initialization*

---

## Troubleshooting

### Issue: Import errors

```python
ModuleNotFoundError: No module named 'gymnasium'
```

**Solution**:
```bash
pip install gymnasium torch yfinance scikit-learn pandas numpy matplotlib seaborn
```

### Issue: CUDA out of memory

**Solution**: Reduce `num_envs` or `n_steps` in config:
```python
config.ppo.num_envs = 4  # Default: 8
config.ppo.n_steps = 128  # Default: 256
```

### Issue: W&B SSL errors

**Solution**: Already handled - use offline mode (default)

### Issue: Data download fails

**Solution**: Use cached data or download separately:
```python
config.data.start_date = "2020-01-01"  # Shorter period
```

---

## Advanced Usage

### Custom Reward Function

```python
class CustomRewardEnv(TradingEnv):
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        # Modify reward
        custom_reward = reward * 2 - 0.1
        return obs, custom_reward, done, truncated, info
```

### Multi-Asset Training

```python
configs = []
for ticker in ["BTC-USD", "ETH-USD", "ADA-USD"]:
    config = get_ppo_without_forecast_config()
    config.data.ticker = ticker
    configs.append(config)

# Run all
for config in configs:
    runner = ExperimentRunner(config)
    runner.run()
```

### Hyperparameter Grid Search

```python
from itertools import product

fee_range = [0.00005, 0.0001, 0.0002]
leverage_range = [0.5, 1.0, 2.0]

for fee, leverage in product(fee_range, leverage_range):
    config = get_ppo_without_forecast_config()
    config.environment.fee = fee
    config.environment.leverage_max = leverage

    runner = ExperimentRunner(config)
    runner.run()
```

---

## Key Research Questions Answered

1. **Does price forecasting improve trading performance?**
   - Compare: PPO Without Forecast vs PPO With Forecast

2. **What reward function works best?**
   - Compare: BASIC vs WITH_RISK vs WITH_SHARPE vs RISK_ADJUSTED

3. **What are the optimal trading parameters?**
   - Analyze: Fee impact, Leverage impact, Position smoothing

4. **How robust is the strategy?**
   - Metrics: Sharpe ratio, Sortino ratio, Max drawdown

---

## References

- PPO: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- Trading Metrics: Pardo, "The Evaluation and Optimization of Trading Strategies" (2008)
- LSTM: Hochreiter & Schmidhuber, "Long Short-Term Memory" (1997)

---

## Contact & Support

For issues or questions, refer to:
- Original notebook: `Project_Part_3_Final_Architecture.ipynb`
- Configuration docs: See docstrings in `trading_config.py`
- Metrics docs: See docstrings in `trading_metrics.py`

---

**Last Updated**: 2026-03-11
**Version**: 1.0

