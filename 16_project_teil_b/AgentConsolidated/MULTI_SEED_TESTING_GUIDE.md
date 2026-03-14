# Multi-Seed Testing Guide

## Overview

You discovered an important issue: **getting identical results (43.63% return) across different runs indicates the random seed is fixed at seed=42**.

This guide explains:
1. How seeds work in the framework
2. How to test with different seeds
3. How to interpret seed stability results

## Current Seed Configuration

**Default seed**: `42` (set in `trading_config.py`)

This means every experiment run uses the same random initialization, resulting in identical results.

## Testing with Different Seeds

### Quick Test (Seeds: 10, 20, 30)

```bash
# Test reward ablation with 3 seeds
python multi_seed_testing.py --seeds 10 20 30 --mode rewards

# Test baseline experiments with 3 seeds
python multi_seed_testing.py --seeds 10 20 30 --mode baseline

# Test both
python multi_seed_testing.py --seeds 10 20 30 --mode full
```

### Extended Test (More Seeds)

```bash
# Test with 10 different seeds
python multi_seed_testing.py --seeds 1 5 10 15 20 25 30 35 40 50 --mode full

# Test with random seeds
python multi_seed_testing.py --seeds 42 100 999 2023 2024 --mode rewards
```

### Save Results to CSV

```bash
python multi_seed_testing.py \
  --seeds 10 20 30 \
  --mode full \
  --output_csv multi_seed_stability.csv \
  --output_dir ./multi_seed_results
```

## Understanding Seed Stability

The multi-seed tester analyzes three key stability metrics:

### 1. **Coefficient of Variation (CV %)**
```
CV = (Std Dev / Mean) × 100

- CV < 5%:   ✓ VERY STABLE (minimal seed effect)
- CV < 10%:  ✓ STABLE (reasonable stability)
- CV < 20%:  ⚠ MODERATE (some seed sensitivity)
- CV > 20%:  ✗ UNSTABLE (high seed sensitivity)
```

### 2. **Range (Max - Min)**
Shows absolute difference between best and worst seed result:
```
Low range = consistent results across seeds
High range = significant variance in results
```

### 3. **Mean & Std Dev**
- **Mean**: Average performance across seeds
- **Std Dev**: How much results vary

## Expected Results

### What You'll See When Running:

```
==================================================
SEED: 10
==================================================
  Testing basic               with seed 10... ✓ Return: 45.23%
  Testing with_risk           with seed 10... ✓ Return: 43.12%
  Testing with_sharpe         with seed 10... ✓ Return: 38.45%
  ...

==================================================
SEED: 20
==================================================
  Testing basic               with seed 20... ✓ Return: 46.89%
  Testing with_risk           with seed 20... ✓ Return: 42.67%
  Testing with_sharpe         with seed 20... ✓ Return: 39.12%
  ...

==================================================
SEED: 30
==================================================
  Testing basic               with seed 30... ✓ Return: 44.56%
  Testing with_risk           with seed 30... ✓ Return: 43.89%
  Testing with_sharpe         with seed 30... ✓ Return: 37.23%
  ...
```

Then analysis:

```
==================================================
ANALYSIS: REWARDS - MULTI SEED STABILITY
==================================================

BASIC
----
Total Return
  Mean:             0.450000
  Std Dev:          0.010000
  CV %:             2.22%
  Stability:        ✓ VERY STABLE

WITH_RISK
----
Total Return
  Mean:             0.438000
  Std Dev:          0.008000
  CV %:             1.83%
  Stability:        ✓ VERY STABLE
```

## How to Change Default Seed

If you want a different default seed for all experiments:

### Option 1: Edit trading_config.py

```python
@dataclass
class ExperimentConfig:
    # ... existing fields ...
    seed: int = 42  # Change to: seed: int = 10
```

### Option 2: Modify Config in Your Script

```python
from trading_config import get_ppo_without_forecast_config

config = get_ppo_without_forecast_config()
config.seed = 10  # Set custom seed
```

### Option 3: Use Environment Variable

You could add this to the ExperimentRunner:

```python
import os
seed = int(os.getenv('TRADING_SEED', '42'))
```

Then run:
```bash
TRADING_SEED=10 python run_all_experiments.py
```

## Interpreting Stability Results

### Scenario 1: Results Vary Significantly Across Seeds
```
Total Return:
  Mean:    0.45
  Std Dev: 0.10
  CV:      22%
  ✗ UNSTABLE

→ Strategy is sensitive to random initialization
→ Need to run multiple seeds and average
→ Consider increasing training steps
→ Check if model is overtrained on training data
```

### Scenario 2: Results Are Very Consistent
```
Total Return:
  Mean:    0.45
  Std Dev: 0.002
  CV:      0.4%
  ✓ VERY STABLE

→ Strategy is robust to seed variations
→ Single seed is representative
→ Results are reliable and reproducible
```

### Scenario 3: Some Metrics Stable, Others Not
```
Return:        CV = 3%   ✓ STABLE
Sharpe Ratio:  CV = 15%  ⚠ MODERATE
Max Drawdown:  CV = 25%  ✗ UNSTABLE

→ Strategy returns are consistent
→ But risk metrics vary significantly
→ Drawdown sensitive to random initialization
→ Need multiple seeds for risk analysis
```

## Output Files Generated

```
./multi_seed_results/
├── multi_seed_results.json          # Complete results
├── stability_reward_ablation.png    # Visualization
├── stability_baseline.png           # Baseline plot
└── multi_seed_stability.csv         # Stability metrics (if requested)
```

## Comparing Seed 42 vs Other Seeds

To understand why results change:

```bash
# Get baseline with seed 42 (original)
python run_reward_ablation.py  # Uses seed=42

# Compare with seed 10
python multi_seed_testing.py --seeds 42 10 --mode rewards

# Look at the differences
```

Expected differences:
- **Neural network initialization**: Different random weights → different training path
- **Action sampling**: Different exploration → different trades
- **Data shuffling**: Different batch ordering (if applicable)
- **Final performance**: 5-15% variance typical

## Best Practices

### 1. **For Robustness Analysis**
```bash
python multi_seed_testing.py --seeds 1 2 3 4 5 --mode full
# Run with 5+ seeds, report mean ± std
```

### 2. **For Final Evaluation**
```bash
python multi_seed_testing.py --seeds 42 100 999 --mode full
# Use standard seeds, compare to literature
```

### 3. **For Hyperparameter Tuning**
```bash
python multi_seed_testing.py --seeds 42 --mode rewards
# Tune on single seed first, validate on multiple
```

### 4. **For Publishing**
```bash
python multi_seed_testing.py --seeds 1 42 99 100 123 --mode full
# Report: "Mean: X%, CV: Y%, Range: Z%"
```

## Summary

| Question | Answer |
|----------|--------|
| **Current seed?** | 42 (fixed) |
| **Why same results?** | Same seed = same random numbers = same initialization |
| **How to test seeds?** | `python multi_seed_testing.py --seeds 10 20 30` |
| **How to interpret?** | Look at CV% - lower is more stable |
| **What's good CV%?** | <10% is stable, >20% is unstable |
| **Expected differences?** | 5-15% variation in returns is typical |

## Next Steps

1. **Run multi-seed test**:
   ```bash
   python multi_seed_testing.py --seeds 10 20 30 --mode full
   ```

2. **Check stability**:
   - Look at generated PNG files
   - Review stability metrics
   - Compare CV% across metrics

3. **Decide on approach**:
   - If stable: Use single seed (faster)
   - If unstable: Average multiple seeds (more robust)

4. **Update results**:
   - Report mean ± std from multiple seeds
   - Update your analysis with stability metrics
   - Adjust strategy if needed


o
