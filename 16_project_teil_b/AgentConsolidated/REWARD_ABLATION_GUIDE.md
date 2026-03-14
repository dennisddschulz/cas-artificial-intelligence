# Comprehensive Reward Function Ablation Study

This guide explains the extended experiments framework for systematically testing 8 different reward functions with PPO trading agents.

## Overview

The extended experiment suite tests:
- **2 Baseline Experiments**: PPO without forecast, PPO with LSTM forecast
- **8 Reward Function Variants**: Systematic ablation study of reward function design
- **Total: 10 Experiments** with comprehensive comparison and analysis

## Reward Functions Explained

### 1. BASIC Reward
**Formula**: `R = PnL - TransactionCost - Slippage`

**Behavior**:
- Pure return maximization without risk constraints
- No position size penalty
- Encourages large leveraged bets in any market
- **Best for**: High-conviction trend following

**Use when**: You want maximum raw returns, willing to accept volatility

---

### 2. WITH_RISK Reward
**Formula**: `R = PnL - Cost - Slippage - κ × (Position²) × Volatility`

**Behavior**:
- Adds quadratic position penalty scaled by volatility
- Smaller positions in volatile periods (automatic risk scaling)
- Balanced returns and risk control
- κ (kappa) parameter controls risk aversion
- **Default in original experiments**

**Use when**: You want reasonable returns with built-in risk management

---

### 3. WITH_SHARPE Reward
**Formula**: `R = (PnL - Cost - Slippage) / Volatility`

**Behavior**:
- Explicitly optimizes risk-adjusted returns
- Divides returns by volatility
- Small returns in volatile periods count less
- Large returns in calm periods count more
- **Best for**: Consistent, low-volatility strategies

**Use when**: You prioritize Sharpe ratio above all else

---

### 4. RISK_ADJUSTED Reward
**Formula**: `R = (PnL / Volatility) - Cost - Slippage`

**Behavior**:
- Returns normalized by volatility, costs not normalized
- More aggressive than WITH_SHARPE in calm periods
- Similar to Sharpe but with asymmetric cost treatment
- **Best for**: Adaptive position sizing based on risk regime

**Use when**: You want normalized returns but full cost penalties

---

### 5. SORTINO Reward
**Formula**: `R = (PnL_adjusted / Volatility)`
where PnL_adjusted has downside multiplier (1.2x) for negative returns

**Behavior**:
- Penalizes losses 20% more than gains
- Focuses on downside risk, not total volatility
- More conservative than Sharpe ratio
- **Best for**: Risk-averse investors

**Use when**: Drawdowns are more important than volatility

---

### 6. CALMAR Reward
**Formula**: `R = PnL - (DrawdownEstimate × 0.5)`
where DrawdownEstimate = κ × Position² × Volatility

**Behavior**:
- Uses estimated drawdown as risk metric
- Combines returns with maximum drawdown penalty
- Conservative: strongly penalizes large positions
- **Best for**: Wealth preservation focus

**Use when**: You want maximum drawdown control

---

### 7. INFORMATION_RATIO Reward
**Formula**: `R = (PnL / Volatility) - Cost + 0.1 × (1 if PnL > 0 else 0)`

**Behavior**:
- Risk-adjusted returns with consistency bonus
- 10% bonus for profitable steps
- Encourages consistent positive returns
- **Best for**: Building consistent alpha

**Use when**: Consistency matters more than raw magnitude

---

### 8. COMPOSITE Reward
**Formula**: `R = 0.5×SignalReturns + 0.3×SignalSharpe + 0.2×SignalRisk`

**Behavior**:
- Multi-objective weighted blend
- 50% weight: raw returns (PnL - costs)
- 30% weight: Sharpe-like (returns / volatility)
- 20% weight: risk penalty (-κ × pos² × vol)
- **Best for**: Balanced optimization

**Use when**: You want balanced return-risk profile

---

## Running the Extended Experiments

### Quick Start

```bash
# Run all 10 experiments with comprehensive analysis
python run_all_experiments.py

# Expected output:
# - 2 baseline experiments (20-30 min each)
# - 8 reward ablation studies (20-30 min each)
# - Total runtime: ~4-5 hours
```

### What Gets Generated

After running `run_all_experiments.py`:

**CSV Files**:
- `metrics_comparison.csv` - All experiments metrics in one table
- `reward_comparison_detailed.csv` - Detailed reward ablation results

**PNG Visualizations**:
- `equity_curves_comparison.png` - Final equity curves
- `drawdown_comparison.png` - Maximum drawdown over time
- `returns_distribution.png` - Daily returns histograms
- `metrics_heatmap.png` - Performance metrics heatmap
- `reward_ablation_analysis.png` - Reward function comparison charts

**Data Files**:
- `detailed_results.json` - Complete experiment results
- `metrics.pkl` - Pickle files for further analysis

**WandB Offline Runs**:
- `./wandb/offline-run-*/` - Can be synced later with `wandb sync`

---

## Interpreting Results

### Key Metrics to Compare

1. **Total Return**
   - What: Final portfolio gain/loss percentage
   - Higher is better
   - Use to evaluate absolute performance

2. **Sharpe Ratio**
   - What: Return per unit of risk (volatility)
   - Higher is better (>1.0 is good)
   - Use to compare risk-adjusted performance

3. **Max Drawdown**
   - What: Largest peak-to-trough decline
   - Higher is better (less negative)
   - Use to assess worst-case scenario

4. **Volatility**
   - What: Standard deviation of daily returns
   - Lower is better
   - Use to assess stability

5. **Turnover**
   - What: Average position change magnitude
   - Lower is better (saves costs)
   - Use to assess trading frequency

### Interpretation Examples

**Scenario 1**: Reward A has highest Sharpe (2.5) but Reward B has highest return (15%)
- → Reward A is more efficient (less risky)
- → Reward B is greedier (higher risk)
- Choose based on your risk tolerance

**Scenario 2**: Reward C has low volatility (5%) but terrible Sharpe (-0.5)
- → Strategy is losing consistently
- → Low volatility means losses are steady
- → Reject this reward function

**Scenario 3**: All rewards have similar Sharpe but different turnover
- → Choose lowest turnover (saves transaction costs)

---

## Parameter Tuning

### Adjusting Reward Parameters

Edit `trading_config.py` to modify reward parameters:

```python
# Example: Increase risk aversion in WITH_RISK
env_config.reward_params['kappa'] = 0.02  # was 0.01

# Example: Change COMPOSITE weights
env_config.reward_params['weight_returns'] = 0.6  # increase returns focus
env_config.reward_params['weight_sharpe'] = 0.2   # decrease Sharpe focus
env_config.reward_params['weight_risk'] = 0.2     # keep risk same
```

### Common Parameter Adjustments

| Parameter | Default | Conservative | Aggressive |
|-----------|---------|--------------|-----------|
| kappa | 0.01 | 0.02 | 0.005 |
| downside_scale | 1.2 | 1.5 | 1.0 |
| consistency_bonus | 0.1 | 0.2 | 0.05 |

---

## Advanced Analysis

### Creating Custom Comparisons

```python
import pandas as pd

# Load results
reward_df = pd.read_csv('reward_comparison_detailed.csv', index_col=0)

# Compare specific reward types
selected = reward_df.loc[['PPO_basic', 'PPO_with_sharpe', 'PPO_composite']]
print(selected)

# Find best reward for your objective
print("Best Sharpe:", reward_df['sharpe_ratio'].idxmax())
print("Best Return:", reward_df['total_return'].idxmax())
print("Best Drawdown:", reward_df['max_drawdown'].idxmax())
```

### Analyzing Training Dynamics

```python
import pickle

# Load detailed results
with open('detailed_results.json', 'r') as f:
    results = json.load(f)

# Compare training progression across reward types
for exp_name, exp_data in results.items():
    metrics = exp_data['metrics']
    print(f"{exp_name}: Sharpe={metrics['sharpe_ratio']:.2f}, Return={metrics['total_return']*100:.1f}%")
```

---

## Best Practices

### 1. Always Run Baselines First
Run PPO-Without-Forecast and PPO-With-Forecast before reward ablation.
These establish your baseline performance.

### 2. Compare Like-with-Like
- Compare all reward types under same market data
- Use same random seed for reproducibility
- Use same initial equity and leverage limits

### 3. Look at Multiple Metrics
Don't optimize for single metric:
- High Sharpe + High Turnover = unstable strategy
- High Return + High Drawdown = risky strategy
- Low Volatility + Negative Return = broken strategy

### 4. Check Training Stability
- Look at equity curve shape (smooth > jumpy)
- Check if strategy learned or just lucky (randomness test)
- Verify results don't depend on random seed

### 5. Validate Out-of-Sample
- Results shown are on test set
- But train with different periods to validate robustness

---

## Troubleshooting

### Problem: All Reward Functions Have Negative Returns
**Solution**:
- Market data might be in down trend
- Try different ticker (BTC-USD, SPY, etc.)
- Increase training updates (total_updates in config)
- Check feature engineering (check add_features function)

### Problem: High Variance Between Runs
**Solution**:
- Set seed explicitly in config
- Increase PPO training updates
- Use more parallel environments (num_envs)
- Run multiple seeds and average results

### Problem: Reward Type X Failed to Run
**Solution**:
- Check trading_framework.py step() function for that reward type
- Verify RewardType enum is defined in trading_config.py
- Check for NaN/Inf in reward calculations (safe_sigma, safe divisions)

### Problem: WandB Connection Issues
**Solution**:
- Scripts run in OFFLINE mode by default
- Sync later with: `wandb sync ./wandb/offline-run-*/`
- Or set `WANDB_MODE=online` in environment

---

## File Structure

```
AgentConsolidated/
├── run_all_experiments.py          # Main entry point
├── trading_framework.py             # Core framework + TradingEnv
├── trading_config.py                # Configurations + RewardType enum
├── reward_calculators.py            # Reward calculation utilities (reference)
├── trading_metrics.py               # Metrics calculation
├── REWARD_ABLATION_GUIDE.md        # This file
│
├── results/                         # Generated results directory
│   ├── metrics_comparison.csv
│   ├── reward_comparison_detailed.csv
│   ├── detailed_results.json
│   └── PPO-*-*/                    # Individual experiment results
│
└── wandb/                           # WandB offline runs
    └── offline-run-*/
```

---

## Citation & References

This ablation study implements reward function variations based on:
- Original baseline: WITH_RISK (quadratic position penalty)
- Sharpe-based: Risk-adjusted returns focus
- Drawdown-based: Conservative wealth preservation
- Information ratio: Consistency emphasis
- Composite: Multi-objective optimization

All implementations follow standard financial engineering practices.

---

## Questions?

Check these files for more details:
- `trading_config.py` - How reward types are configured
- `trading_framework.py` - How reward types are calculated in step()
- `reward_calculators.py` - Reference implementations (not used in RL, for reference only)
- Example outputs in `results/` directory after first run


