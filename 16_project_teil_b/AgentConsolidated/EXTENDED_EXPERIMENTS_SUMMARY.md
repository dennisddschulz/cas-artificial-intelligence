# Extended Reward Function Experiments - Implementation Summary

## Overview

The trading framework has been extended to support comprehensive ablation studies of 8 different reward function variations. This enables systematic comparison of how different reward signals affect PPO agent learning in trading environments.

## Files Modified/Created

### Modified Files

#### 1. `run_all_experiments.py`
**Changes**:
- Enhanced `run_experiment_3()` to show detailed metrics for each reward variant
- Added `generate_reward_comparison_analysis()` function for comprehensive reward ablation analysis
- Updated `main()` to orchestrate 10 experiments (2 baselines + 8 reward variants)
- Added detailed logging showing metrics for each reward function
- Added specialized visualization for reward function comparison

**New Features**:
- Shows total return, Sharpe ratio, and max drawdown for each reward variant
- Generates `reward_ablation_analysis.png` with 6 metric subplots
- Saves `reward_comparison_detailed.csv` with detailed metrics
- Identifies best performer for each metric

**Example Output**:
```
[3.1/8] Basic Reward
  Reward Type: basic
  ✓ COMPLETED
    - Total Return: 15.23%
    - Sharpe Ratio: 1.24
    - Max Drawdown: -8.52%
```

#### 2. `trading_framework.py`
**Changes in `TradingEnv.step()` method**:
- Added support for all 8 reward types in the reward calculation logic
- Implemented `SORTINO`, `CALMAR`, `INFORMATION_RATIO`, and `COMPOSITE` reward calculations
- Enhanced error handling with safe divisions (safe_sigma)
- Added reward type tracking in info dictionary

**New Reward Implementations**:

```python
# SORTINO: Downside-risk focused (1.2x multiplier for negative returns)
safe_sigma = max(sigma_t, 0.001)
downside_adjusted_pnl = pnl if pnl > 0 else pnl * 1.2
reward = (downside_adjusted_pnl - cost - slippage) / safe_sigma

# CALMAR: Drawdown-focused (κ × pos² × vol × 0.5 multiplier)
drawdown_estimate = self.kappa * (self.pos ** 2) * sigma_t
reward = true_reward - (drawdown_estimate * 0.5)

# INFORMATION_RATIO: Consistency bonus (+0.1 for positive returns)
consistency_bonus = 0.1 if pnl > 0 else 0
return_signal = pnl / safe_sigma if safe_sigma > 0 else 0
reward = return_signal - cost - slippage + consistency_bonus

# COMPOSITE: Multi-objective weighted blend (50-30-20)
signal_returns = true_reward
signal_sharpe = true_reward / safe_sigma
signal_risk = -risk_pen
reward = 0.5 * signal_returns + 0.3 * signal_sharpe + 0.2 * signal_risk
```

**Benefits**:
- All reward calculations use same true_reward for equity updates (no look-ahead bias)
- Risk-adjusted metrics don't artificially inflate/deflate equity
- Each reward type provides different learning incentives

### New Files Created

#### 1. `run_reward_ablation.py`
**Purpose**: Dedicated script for running and analyzing reward function ablation studies

**Features**:
- Runs all 8 reward function variants systematically
- Prints detailed explanation of each reward function
- Generates comparative analysis with best performer identification
- Creates visualization and CSV output
- Estimated runtime: 3-4 hours

**Usage**:
```bash
python run_reward_ablation.py
```

**Output Files**:
- `reward_ablation_comparison.csv` - Metrics table
- `reward_ablation_results.json` - Detailed results
- `reward_ablation_comparison.png` - 8-panel comparison plot

#### 2. `REWARD_ABLATION_GUIDE.md`
**Purpose**: Comprehensive guide to understanding and using reward functions

**Contents**:
- Detailed explanation of each of 8 reward functions
- Formula, behavior, and best-use-cases for each
- Parameter tuning guide
- Interpretation guidelines
- Troubleshooting section

---

## Reward Functions Implemented

### 1. BASIC
- **Formula**: `R = PnL - Cost - Slippage`
- **Purpose**: Pure return maximization
- **Characteristic**: No risk penalty

### 2. WITH_RISK (Original/Default)
- **Formula**: `R = PnL - Cost - κ × Position² × Volatility`
- **Purpose**: Balanced returns and risk
- **Characteristic**: Automatic risk scaling via volatility

### 3. WITH_SHARPE
- **Formula**: `R = (PnL - Cost - Slippage) / Volatility`
- **Purpose**: Explicit risk-adjusted optimization
- **Characteristic**: Penalizes returns more in volatile periods

### 4. RISK_ADJUSTED
- **Formula**: `R = (PnL / Volatility) - Cost - Slippage`
- **Purpose**: Normalized returns, full cost penalty
- **Characteristic**: More aggressive than WITH_SHARPE in calm periods

### 5. SORTINO
- **Formula**: `R = (PnL_adjusted / Volatility)` where losses have 1.2x multiplier
- **Purpose**: Downside-risk focus
- **Characteristic**: Conservative, penalizes losses more

### 6. CALMAR
- **Formula**: `R = PnL - (Drawdown_Estimate × 0.5)`
- **Purpose**: Maximum drawdown control
- **Characteristic**: Wealth preservation focus

### 7. INFORMATION_RATIO
- **Formula**: `R = (PnL / Volatility) + 0.1 × (1 if PnL > 0 else 0)`
- **Purpose**: Consistency emphasis
- **Characteristic**: Bonus for profitable steps

### 8. COMPOSITE
- **Formula**: `R = 0.5×Returns + 0.3×Sharpe + 0.2×Risk`
- **Purpose**: Multi-objective optimization
- **Characteristic**: Balanced signal blending

---

## How to Run Extended Experiments

### Option 1: Full Comprehensive Suite (10 Experiments)
```bash
python run_all_experiments.py
```
**Runs**:
- 2 baseline experiments (PPO without forecast, PPO with forecast)
- 8 reward ablation studies
- Full analysis and visualization
- **Duration**: ~5 hours
- **Output**: All comparison charts, metrics tables, and detailed results

### Option 2: Reward Ablation Only (8 Experiments)
```bash
python run_reward_ablation.py
```
**Runs**:
- 8 reward function variants
- Focused analysis on reward differences
- **Duration**: ~3-4 hours
- **Output**: Reward comparison charts and metrics

### Option 3: Single Experiment
```python
from trading_config import ConfigBuilder, RewardType
from trading_framework import ExperimentRunner

config = (ConfigBuilder("Custom-Reward-Test")
    .with_reward(RewardType.SORTINO)
    .with_ppo_updates(10000)
    .build())

runner = ExperimentRunner(config)
results = runner.run()
```

---

## Output Files Generated

### CSV Files
- **metrics_comparison.csv** - All 10 experiments metrics in one table
- **reward_comparison_detailed.csv** - 8 reward variants detailed metrics
- **reward_ablation_comparison.csv** - Reward-only comparison

### PNG Visualizations
- **equity_curves_comparison.png** - Final equity curves for all experiments
- **drawdown_comparison.png** - Maximum drawdown analysis
- **returns_distribution.png** - Daily returns histograms
- **metrics_heatmap.png** - Performance metrics heatmap
- **reward_ablation_analysis.png** - 8-panel reward function comparison
- **reward_ablation_comparison.png** - Reward-only 8-panel comparison

### JSON Files
- **detailed_results.json** - Complete results with all metrics
- **reward_ablation_results.json** - Reward-focused detailed results

### Pickle Files
- **metrics.pkl** - Binary format for Python analysis tools

---

## Key Design Decisions

### 1. True Reward vs Learning Reward
```
true_reward = pnl - cost - slippage  # Update equity with this
learning_reward = modified(true_reward)  # Train policy with this
```
**Why**: Prevents risk penalties from artificially reducing equity growth

### 2. Safe Division (sigma_t)
```python
safe_sigma = max(sigma_t, 0.001)  # Prevent division by zero
```
**Why**: Ensures stability when volatility approaches zero

### 3. Configurable Parameters
```python
reward_params = {
    'kappa': 0.01,  # Risk aversion
    'downside_scale': 1.2,  # Sortino emphasis
    'consistency_bonus': 0.1,  # Info ratio bonus
}
```
**Why**: Allows easy tuning without code changes

### 4. Deterministic Evaluation
Uses `model.forward().mean()` for deterministic action selection during testing

### 5. Separate WandB Groups
Each reward type gets its own WandB group for easy filtering

---

## Interpreting Results

### Example: Comparing Results

If you see:
```
Reward Type        Total Return   Sharpe Ratio   Max Drawdown
basic                   18.5%        0.95           -12.3%
with_risk               12.8%        1.45            -6.2%
with_sharpe              8.2%        2.10            -4.1%
composite               15.2%        1.68            -7.8%
```

**Analysis**:
- BASIC: Greedy, highest returns but risky
- WITH_RISK: Balanced, good Sharpe
- WITH_SHARPE: Conservative, best risk-adjusted
- COMPOSITE: Middle ground with diversified signals

**Choice depends on your objective**:
- Maximum returns → BASIC
- Best risk-adjusted → WITH_SHARPE
- Balanced → COMPOSITE

### Common Patterns

1. **Sharpe-based rewards** → Lower volatility, lower returns
2. **Return-focused rewards** → Higher volatility, higher returns
3. **Drawdown-focused rewards** → Less aggressive positioning
4. **Composite rewards** → Tend to be balanced

---

## Performance Expectations

### Baseline Performance (BTC-USD, 2020-2024)
- Typical Sharpe Ratio: 0.5 - 2.0
- Typical Return: -10% to +50% depending on period
- Typical Max Drawdown: 5% - 30%

### Reward Function Impact
- Different reward functions typically:
  - Vary Sharpe ratio by ±0.3 to ±0.8
  - Vary returns by ±5% to ±15%
  - Vary max drawdown by ±3% to ±10%

### Market Dependency
- Bull markets: Most rewards succeed
- Bear markets: Only risk-aware rewards succeed
- Sideways markets: Turnover becomes critical

---

## Troubleshooting

### Problem: One Reward Type Fails
**Check**:
1. RewardType enum defined in `trading_config.py`
2. Corresponding elif/else branch in `TradingEnv.step()`
3. Safe divisions and NaN checks

### Problem: Inconsistent Results
**Check**:
1. Same random seed across experiments
2. Same market data period
3. Same PPO hyperparameters (lr, num_envs, etc.)

### Problem: Poor Performance All Rewards
**Check**:
1. Market is in down trend (BTC or SPY down period)
2. Initial equity is too small (minimum $1000)
3. Fee too high (should be < 0.001)
4. Training updates too few (minimum 5000)

---

## Code Structure

```
TradingEnv.step():
  ├─ Calculate raw PnL, cost, slippage
  ├─ Select reward type (if/elif)
  │   ├─ BASIC: true_reward
  │   ├─ WITH_RISK: true_reward - κ×pos²×σ
  │   ├─ WITH_SHARPE: true_reward / σ
  │   ├─ RISK_ADJUSTED: (pnl/σ) - cost
  │   ├─ SORTINO: (pnl_adj / σ) with downside penalty
  │   ├─ CALMAR: pnl - drawdown_est
  │   ├─ INFORMATION_RATIO: (pnl/σ) + consistency_bonus
  │   └─ COMPOSITE: 0.5×ret + 0.3×sharpe + 0.2×risk
  ├─ Apply reward_scale multiplier
  ├─ Update equity with true_reward (NOT learning reward)
  └─ Return observation, learning_reward, done, info
```

---

## Next Steps

1. **Run basic test**:
   ```bash
   python run_reward_ablation.py
   ```

2. **Analyze results**:
   - Open `reward_ablation_comparison.csv`
   - View `reward_ablation_comparison.png`

3. **Interpret findings**:
   - Read `REWARD_ABLATION_GUIDE.md`
   - Compare metrics across reward types

4. **Experiment further**:
   - Modify `reward_params` in `trading_config.py`
   - Try different markets (SPY, GLD, etc.)
   - Adjust leverage limits

---

## References

- **Original Framework**: Based on PPO trading agent from project_teil_b
- **Reward Functions**: Standard finance metrics (Sharpe, Sortino, Calmar ratios)
- **Implementation**: Custom PyTorch + Gymnasium framework

---

## Contact & Questions

For issues or questions:
1. Check `REWARD_ABLATION_GUIDE.md` for detailed explanations
2. Review `trading_framework.py` TradingEnv.step() for implementation details
3. Check `trading_config.py` for configuration options
4. Run individual experiments with debug output enabled


