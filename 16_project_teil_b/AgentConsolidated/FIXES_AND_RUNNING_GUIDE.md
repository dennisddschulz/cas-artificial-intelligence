# Fixes Applied and Execution Guide

## Date: 2026-03-13
## Version: 1.0

---

## Issues Fixed

### 1. **Hanging Issue After First Experiment**

**Problem:**
- Script would hang after completing the first experiment and trying to close WandB
- The issue was caused by SSL/network errors during `wandb.finish()` with offline mode
- The hang would occur due to retry loop in WandB trying to sync offline runs with corporate proxy

**Solution Applied:**
- Added timeout protection in `trading_framework.py` `run()` method
- Implemented signal-based timeout (30 seconds) for `wandb.finish()`
- Added fallback mechanism if timeout occurs
- Added proper exception handling without re-raising

**Changed Files:**
- `trading_framework.py`: Lines 506-564

```python
# Added timeout wrapper around wandb.finish()
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)
try:
    wandb.finish()
    signal.alarm(0)
except TimeoutError:
    print("⚠ WandB finish timed out (SSL/network issues), continuing...")
    signal.alarm(0)
```

### 2. **Environment Resource Leaks**

**Problem:**
- Training and evaluation environments were not being properly closed
- This could cause hanging, memory leaks, or resource contention between experiments

**Solution Applied:**
- Added explicit `env.close()` calls after PPO training completes
- Added explicit `env_test.close()` calls after evaluation completes
- Wrapped with try-except to prevent failures from blocking subsequent experiments

**Changed Files:**
- `trading_framework.py`: Lines 814-820 (train_ppo method)
- `trading_framework.py`: Lines 1159-1164 (evaluate method)

### 3. **Config Function Parameter Mismatch**

**Problem:**
- `run_all_experiments.py` was calling config functions with wrong parameter names
- Used `group="baseline"` instead of `experiment_type="ppo_baseline"`
- Functions expected `experiment_type` and `variant` parameters

**Solution Applied:**
- Fixed all three config function calls in `run_all_experiments.py`:
  1. Experiment 1: `get_ppo_without_forecast_config(experiment_type="ppo_baseline", variant="v1")`
  2. Experiment 2: `get_ppo_with_forecast_config(experiment_type="ppo_with_forecast", variant="v1")`
  3. Experiment 3: `get_ppo_different_rewards_configs(experiment_type="reward_ablation", variant="v1")`

**Changed Files:**
- `run_all_experiments.py`: Lines 663-688 (Exp 1)
- `run_all_experiments.py`: Lines 703-728 (Exp 2)
- `run_all_experiments.py`: Lines 751-788 (Exp 3)

### 4. **Unreachable Code in Visualization Function**

**Problem:**
- `generate_reward_comparison_analysis()` function had unreachable code after return statement
- Dead code trying to access undefined variables `all_results` and `comparison_df`
- This would cause issues if the function was modified

**Solution Applied:**
- Removed unreachable code block (~35 lines)
- Moved necessary export logic to proper `export_results()` function
- Enhanced `export_results()` with complete JSON and pickle exports

**Changed Files:**
- `run_all_experiments.py`: Lines 338-377 (export_results)
- `run_all_experiments.py`: Lines 450-470 (removed dead code from generate_reward_comparison_analysis)

### 5. **Duplicate Configuration in Script**

**Problem:**
- Duplicate `os.environ['MPLBACKEND'] = 'Agg'` setting

**Solution Applied:**
- Removed duplicate, kept single clean configuration

**Changed Files:**
- `run_all_experiments.py`: Lines 27-33

---

## How to Run Experiments

### Prerequisites

All required packages are already installed. The framework uses:
- PyTorch 2.9.1
- WandB (offline mode by default)
- Gymnasium for environment management
- yfinance for market data

### Running All Experiments

```bash
cd /home/isc-den/cas-artificial-intelligence/16_project_teil_b/AgentConsolidated
python run_all_experiments.py
```

**Expected Behavior:**

1. **Experiment 1** (~15-25 min): PPO Without Forecast (Baseline)
   - Trains PPO for 3000 updates
   - Evaluates on test set
   - Logs metrics to WandB group: `{date}/ppo_baseline/v1/seed_{seed}`
   - Properly closes environment and WandB connection

2. **Experiment 2** (~25-35 min): PPO With LSTM Forecast
   - Trains LSTM forecast model first
   - Then trains PPO for 3000 updates
   - Logs metrics to WandB group: `{date}/ppo_with_forecast/v1/seed_{seed}`

3. **Experiments 3A-3H** (~90-150 min total): Reward Function Ablation (8 variants)
   - BASIC reward
   - WITH_RISK reward
   - WITH_SHARPE reward
   - RISK_ADJUSTED reward
   - SORTINO reward
   - CALMAR reward
   - INFORMATION_RATIO reward
   - COMPOSITE reward
   - Each logs to: `{date}/reward_ablation/v1/seed_{seed}`

**Total Expected Runtime:** ~3-4 hours for all 10 experiments

### Running Single Experiments

If you want to test a single experiment before running all:

```bash
python -c "
from trading_config import get_ppo_without_forecast_config
from trading_framework import ExperimentRunner

config = get_ppo_without_forecast_config()
runner = ExperimentRunner(config)
results = runner.run()
"
```

### Running Only Reward Ablation

To skip baselines and test only reward functions:

```bash
python -c "
from trading_config import get_ppo_different_rewards_configs
from trading_framework import ExperimentRunner

configs = get_ppo_different_rewards_configs()
for config in configs[:3]:  # Run first 3
    print(f'Testing {config.reward_type.value}...')
    runner = ExperimentRunner(config)
    results = runner.run()
"
```

---

## Output Files

After running experiments, you'll have:

### CSV Files (Comparison Data)
- `metrics_comparison.csv` - All experiments comparison metrics
- `reward_comparison_detailed.csv` - Detailed reward ablation comparison

### JSON Files (Complete Results)
- `detailed_results.json` - Complete experiment results with equity curves

### PKL Files (Visualization Data)
- `metrics.pkl` - Pickled results for create_visualizations.py

### PNG Plots (Generated Automatically)
- `equity_curves_comparison.png` - All equity curves overlaid
- `drawdown_comparison.png` - Drawdown analysis
- `returns_distribution.png` - Daily returns histograms
- `metrics_heatmap.png` - Performance metrics heatmap
- `reward_ablation_analysis.png` - Reward function comparison

### WandB Data
- `./wandb/offline-run-**/` - Offline WandB runs
  - To sync later: `wandb sync ./wandb/offline-run-*/`

### Per-Experiment Results
- `./results/experiment_*/metrics.pkl` - Individual experiment metrics
- `./results/experiment_*/metrics_summary.csv` - Individual experiment summary

---

## Monitoring Execution

### Console Output
The script prints realtime updates:
```
UPDATE  100 /  3000
Episode Returns (last 100):        1.23 ±   0.45
Total Episodes Trained:            300
Log Std (exploration):            -0.987
...
EVALUATION ON TEST SET
...
Final Equity:          $117,691.28
Total Return:          17.70%
Sharpe Ratio:          0.3194
```

### Files to Monitor
- Check `metrics_comparison.csv` after all experiments complete
- Check generated PNG files for visualization
- Check `detailed_results.json` for complete data

---

## Troubleshooting

### If Script Hangs After Experiment Completes

**What's happening:** WandB timeout protection should kick in after 30 seconds

**Expected behavior:**
```
⚠ WandB finish timed out (SSL/network issues), continuing...
✓ EXPERIMENT 1 COMPLETED
```

**If it's still hanging (>2 minutes):**
1. Press Ctrl+C to interrupt
2. Check if experiment results are in `./results/` directory
3. Results may still be saved locally even if WandB sync fails

### If LSTM Forecast Fails

**Error:** `index 1754 is out of bounds for axis 0 with size 1754`

**Solution:** This is a known issue with LSTM sequence creation. The framework now has better error handling. If it occurs:
1. Check your data split ratios in `trading_config.py`
2. Ensure lookback window (20) is smaller than training set size
3. Try reducing forecast_horizon

### If Metrics Are Not Logged

**Check:**
1. Is WandB mode set correctly? (Should be 'offline' by default)
2. Are offline runs being created? Check `./wandb/offline-run-**/` directory
3. Run: `wandb offline-sync` to manually sync later

---

## WandB Configuration

### Current Setup
- **Mode:** OFFLINE (by default)
- **Project:** Configured in `trading_config.py`
- **Group:** Hierarchical - `{date}/{experiment_type}/{variant}/seed_{seed}`
- **Tags:** Includes date, experiment type, forecast mode, reward type, version, seed

### To Change to Online Mode

Edit `trading_config.py` and change:
```python
os.environ['WANDB_MODE'] = 'online'  # Instead of 'offline'
```

Or set environment variable before running:
```bash
export WANDB_MODE=online
python run_all_experiments.py
```

### To Sync Offline Runs Later

```bash
wandb sync ./wandb/offline-run-*/
```

---

## Key Metrics Tracked

For each experiment, the following metrics are logged:

### Budget & Liquidity
- Initial Equity: $100,000
- Final Equity
- Total Return (%)
- Total PnL
- Total Costs
- Avg Daily PnL

### Risk Metrics
- Max Drawdown (%)
- Volatility (%)
- Annualized Volatility (%)

### Performance Metrics
- Sharpe Ratio
- Calmar Ratio
- Sortino Ratio
- Annualized Return (%)
- Win Rate (%)

### Trading Metrics
- Turnover
- Cost Ratio
- Profit Factor

### Distribution Metrics
- Kurtosis
- Skewness
- Mean Daily Return
- Std Daily Return

---

## Expected Results

### Baseline Comparison (Exp 1 vs Exp 2)
- **Exp 1 (No Forecast)** typically shows steady learning
- **Exp 2 (With Forecast)** may show different risk profile due to signal incorporation

### Reward Ablation (Exp 3A-3H)
- **BASIC:** Simple PnL maximization - may be volatile
- **WITH_RISK:** Includes risk penalty - more stable
- **WITH_SHARPE:** Risk-adjusted returns - smoother equity curve
- **RISK_ADJUSTED:** Explicitly optimizes Sharpe ratio
- **SORTINO:** Focuses on downside risk
- **CALMAR:** Minimizes drawdown
- **INFORMATION_RATIO:** Seeks consistent alpha
- **COMPOSITE:** Balanced multi-objective approach

---

## Next Steps After Experiments Complete

1. **Review Results:**
   ```bash
   cat metrics_comparison.csv | column -t -s,
   ```

2. **Generate Comparison Report:**
   ```bash
   python create_visualizations.py
   ```

3. **Create Presentation:**
   ```bash
   python generate_presentation.py
   ```

4. **Upload to WandB (if online mode):**
   ```bash
   wandb sync ./wandb/offline-run-*/
   ```

---

## Verification Checklist

Before running experiments, verify:

- [ ] All imports work: `python -c "from run_all_experiments import *"`
- [ ] Data can be loaded: Check if BTC-USD data is accessible
- [ ] Results directory exists: `mkdir -p results`
- [ ] Sufficient disk space for results (estimated ~500MB total)
- [ ] Matplotlib backend is set to non-interactive (Agg)

---

## Contact & Support

If issues persist:
1. Check the console output for specific error messages
2. Review exception tracebacks
3. Verify data integrity and internet connectivity
4. Check WandB offline runs are created: `ls -la ./wandb/`

---

**Last Updated:** 2026-03-13
**Framework Version:** v1.0
**Status:** Ready for execution

