
# PPO Trading Experiments - Complete Framework

## Overview

This is a comprehensive parameterized framework for running and comparing PPO trading experiments with different configurations:

1. **PPO Without Forecast** (Baseline RL agent)
2. **PPO With Forecast** (RL agent with LSTM price prediction)
3. **PPO With Different Reward Functions** (Ablation study)

## Key Features

✅ **Initial Budget**: $100,000
✅ **Comprehensive Metrics**: Total Return, Sharpe Ratio, Max Drawdown, Volatility, Turnover
✅ **Budget Tracking**: Tracks equity evolution, transaction costs, and liquidity over time
✅ **WandB Integration**: Logs all metrics to WandB with proper group organization
✅ **Local Visualization**: Generates plots from saved metrics.pkl files
✅ **Detailed Reports**: CSV, JSON, and pickle file outputs

## Metrics Calculated

### Risk-Adjusted Returns
- **Total Return**: Overall percentage return from initial capital
- **Annualized Return**: Return scaled to 252 trading days
- **Sharpe Ratio**: Risk-adjusted return metric
- **Sortino Ratio**: Return adjusted for downside volatility only
- **Calmar Ratio**: Return / Max Drawdown

### Risk Metrics
- **Volatility**: Daily returns standard deviation
- **Annualized Volatility**: Volatility scaled to 252 trading days
- **Max Drawdown**: Maximum peak-to-trough decline
- **Kurtosis**: Distribution tail heaviness
- **Skewness**: Distribution asymmetry

### Trading Metrics
- **Win Rate**: Percentage of profitable days
- **Profit Factor**: Ratio of gains to losses
- **Turnover**: Total absolute position changes
- **Total Costs**: Sum of all transaction costs

## Project Structure

```
AgentConsolidated/
├── trading_config.py           # Configuration management
├── trading_framework.py         # PPO experiment runner
├── trading_metrics.py           # Metrics calculation
├── budget_tracker.py            # Budget & equity tracking
├── visualize_metrics.py         # Local plot generation
├── run_all_experiments.py       # Master experiment script
├── test_framework.py            # Framework validation
└── results/                     # Experiment outputs
    ├── PPO-Without-Forecast_YYYYMMDD_HHMMSS/
    │   ├── metrics.pkl          # Serialized metrics
    │   ├── metrics_summary.csv   # CSV metrics
    │   ├── 01_equity_curve.png
    │   ├── 02_budget_breakdown.png
    │   ├── 03_transaction_costs.png
    │   └── 04_returns_drawdown.png
    ├── PPO-With-Forecast_YYYYMMDD_HHMMSS/
    │   └── (same structure)
    └── PPO-Basic_YYYYMMDD_HHMMSS/
        └── (same structure)
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test Framework

```bash
python test_framework.py
```

### 3. Run All Experiments

```bash
# Run complete experiment suite with all three configurations
python run_all_experiments.py
```

This will:
- Train PPO Without Forecast
- Train PPO With Forecast
- Train PPO with different reward functions
- Log metrics to WandB
- Save metrics to pickle files
- Generate local visualizations

### 4. View Results

After experiments complete, check:
- **Metrics**: `./results/*/metrics.pkl` or `./results/*/metrics_summary.csv`
- **Plots**: `./plots/` (auto-generated)
- **WandB**: Review metrics in WandB offline runs

## Running Individual Experiments

### Example: PPO Without Forecast

```python
from trading_config import get_ppo_without_forecast_config
from trading_framework import ExperimentRunner

config = get_ppo_without_forecast_config(name="My-Experiment", group="test")
config.ppo.total_updates = 1000  # Customize training length
config.use_wandb = True

runner = ExperimentRunner(config)
results = runner.run()
```

### Example: PPO With Different Reward

```python
from trading_config import RewardType, ExperimentConfig, PPOConfig, EnvironmentConfig
from trading_framework import ExperimentRunner

config = ExperimentConfig(
    experiment_name="PPO-With-Sharpe",
    forecast_mode=ForecastMode.NONE,
    reward_type=RewardType.WITH_SHARPE,
    environment=EnvironmentConfig(reward_type=RewardType.WITH_SHARPE),
    ppo=PPOConfig(total_updates=1000),
)

runner = ExperimentRunner(config)
results = runner.run()
```

## Configuration Options

### Environment Configuration

```python
environment = EnvironmentConfig(
    initial_equity=100000.0,      # Starting capital
    fee=0.0001,                   # Transaction cost per position unit
    kappa=0.01,                   # Risk penalty weight
    leverage_max=1.0,             # Maximum position exposure
    slippage_coef=0.0,            # Market impact coefficient
    smoothing_alpha=1.0,          # Position execution delay
    reward_scale=1.0,             # Reward scaling factor
    reward_type=RewardType.WITH_RISK,
)
```

### PPO Configuration

```python
ppo = PPOConfig(
    num_envs=8,                   # Parallel environments
    n_steps=256,                  # Steps per rollout
    total_updates=3000,           # Training iterations
    gamma=0.99,                   # Discount factor
    gae_lambda=0.95,              # GAE parameter
    learning_rate=1e-4,           # Optimizer learning rate
    vf_coef=0.5,                  # Value loss coefficient
    ent_coef=0.01,                # Entropy bonus coefficient
    clip_eps=0.2,                 # PPO clip parameter
    ppo_epochs=20,                # Update epochs per rollout
)
```

### Reward Functions

- **BASIC**: `PnL - Cost`
- **WITH_RISK**: `PnL - Cost - Risk_Penalty`
- **WITH_SHARPE**: `(PnL - Cost) / Volatility - Risk_Penalty`
- **RISK_ADJUSTED**: `PnL / Volatility - Cost`

## Understanding the Metrics

### Budget Tracking Example

If you start with **$100,000**:

1. **Day 1-100**: Trading, positions change, costs accumulate
2. **WandB Logs**:
   - `evaluation/final_equity`: $103,430.00
   - `evaluation/total_return`: 0.0343 (3.43%)
   - `evaluation/total_costs`: $1,234.56
3. **Pickle File**: Contains full equity curve and daily metrics
4. **CSV Report**: Summarized metrics

### Reading Plot Files

- **01_equity_curve.png**: Shows equity evolution from $100K initial
- **02_budget_breakdown.png**: Position exposure vs total equity
- **03_transaction_costs.png**: Daily and cumulative costs
- **04_returns_drawdown.png**: Cumulative returns with drawdown

## WandB Integration

All metrics are logged to WandB with organized groups:

- **Group: `ppo_baseline`** - PPO Without Forecast
- **Group: `ppo_forecast`** - PPO With Forecast
- **Group: `reward_ablation`** - Different reward functions

To access offline W&B runs:
```bash
wandb sync ./wandb/offline-run-*/
```

## Troubleshooting

### No Data Downloaded
```python
# Check proxy settings in first cells
# Ensure yfinance can connect to the internet
```

### Metrics Not Saving
```python
# Check permissions in ./results/ directory
# Ensure pickle module is available
```

### WandB Issues
```python
# W&B runs in offline mode to avoid SSL issues
# Metrics saved locally in ./wandb/
# Sync later with: wandb sync ./wandb/offline-run-*/
```

## Performance Tips

1. **Reduce Updates**: Set `total_updates=100` for testing
2. **Fewer Environments**: Use `num_envs=2` for CPU-only systems
3. **Shorter Rollouts**: Set `n_steps=64` for faster iterations
4. **Disable WandB**: Set `use_wandb=False` for speed

## Expected Results

With default parameters (~$100K budget):

| Metric | Range | Status |
|--------|-------|--------|
| Total Return | -5% to +15% | Variable |
| Sharpe Ratio | -0.5 to 2.0 | Variable |
| Max Drawdown | -20% to -5% | Typical |
| Volatility | 0.5% to 2.0% | Expected |
| Win Rate | 40% to 55% | Typical |

## Citation & References

Framework based on:
- Proximal Policy Optimization (Schulman et al., 2017)
- Deep Reinforcement Learning for Trading (various papers)
- Technical Analysis Indicators

## Support

For issues or questions:
1. Check `test_framework.py` for validation
2. Review experiment logs in `./results/*/`
3. Check WandB offline runs with `wandb sync`

---

**Last Updated**: 2025-03-12
**Version**: 2.0 (Full Metrics & Visualization Support)

