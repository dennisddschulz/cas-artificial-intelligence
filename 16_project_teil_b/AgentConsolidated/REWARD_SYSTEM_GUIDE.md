# Modular Reward System - Implementation Guide

## Overview

The reward function system has been refactored into a **modular, testable, and extensible architecture**. This document explains the improvements and how to use them.

## What Changed

### Before: Hardcoded if/elif in TradingEnv
```python
# OLD - Tightly coupled
if self.reward_type == RewardType.BASIC:
    reward = true_reward
elif self.reward_type == RewardType.WITH_RISK:
    reward = true_reward - risk_pen
# ... more conditions ...
```

### After: Pluggable RewardCalculator Classes
```python
# NEW - Modular and extensible
calculator = create_reward_calculator(reward_type, **reward_params)
reward = calculator.calculate(components)
```

## Files Created

### 1. `reward_calculators.py` (Main Implementation)
**Purpose**: Defines the modular reward calculation system

**Key Classes**:
- `RewardComponents`: Data container for reward inputs
- `RewardCalculator`: Abstract base class
- Concrete implementations:
  - `BasicReward`
  - `WithRiskReward`
  - `WithSharpeReward`
  - `RiskAdjustedReward`
  - `SortinoReward` ✨ NEW
  - `CalmarReward` ✨ NEW
  - `InformationRatioReward` ✨ NEW
  - `CompositeReward` ✨ NEW
- `create_reward_calculator()`: Factory function

**Example Usage**:
```python
from reward_calculators import create_reward_calculator, RewardCalculatorType, RewardComponents

# Create a Sortino reward calculator
calc = create_reward_calculator(
    RewardCalculatorType.SORTINO,
    epsilon=0.001,
    downside_scale=1.2
)

# Calculate reward
components = RewardComponents(
    pnl=0.001,
    cost=0.0001,
    slippage=0.00005,
    risk_penalty=0.0,
    volatility=0.01,
    position=0.5,
    true_reward=0.00085
)
reward = calc.calculate(components)
print(f"Reward: {reward:.6f}")

# Get statistics
stats = calc.get_stats()
```

### 2. `test_reward_calculators.py` (Comprehensive Tests)
**Purpose**: Unit tests for all reward calculators

**Coverage**:
- ✓ Basic functionality for each reward type
- ✓ Parameter validation
- ✓ Edge cases (zero volatility, large positions, etc.)
- ✓ Numerical stability
- ✓ Factory function
- ✓ History tracking and statistics

**Run Tests**:
```bash
pytest test_reward_calculators.py -v
```

### 3. `REWARD_ANALYSIS_COMPREHENSIVE.md` (Documentation)
**Purpose**: Detailed analysis of each reward type

**Contents**:
- Formulas for each reward type
- Pros and cons
- When to use each
- Numerical stability concerns
- Configuration best practices
- Validation recommendations

### 4. Updated `trading_config.py`
**Changes**:
- Added 4 new `RewardType` enum values
- Extended `EnvironmentConfig` with `reward_params` dictionary
- Updated `get_ppo_different_rewards_configs()` to support all 8 reward types
- Each config has customizable parameters per reward type

---

## The 8 Reward Types Explained

### 1. BASIC
```
R = PnL - Cost - Slippage
```
**Best for**: Baseline comparisons, very stable markets

### 2. WITH_RISK
```
R = (PnL - Cost - Slippage) - κ * pos² * σ
```
**Best for**: Balanced return-risk strategies (RECOMMENDED for most uses)
**Parameters**: `kappa` (default 0.01)

### 3. WITH_SHARPE
```
R = (PnL - Cost - Slippage) / (σ + ε)
```
**Best for**: Explicit Sharpe ratio optimization
**Parameters**: `epsilon` (default 0.001)
**Warning**: Can produce extreme values when σ ≈ 0

### 4. RISK_ADJUSTED
```
R = (PnL / (σ + ε)) - Cost - Slippage
```
**Best for**: Research/experimentation (asymmetric cost treatment)
**Parameters**: `epsilon` (default 0.001)

### 5. SORTINO ✨ NEW
```
R = (PnL - Cost) / (σ_downside_adjusted + ε)
```
**Best for**: Downside-risk focused trading, capital preservation
**Parameters**: `epsilon` (0.001), `downside_scale` (1.2)
**Differs from WITH_SHARPE**: Emphasizes downside volatility more

### 6. CALMAR ✨ NEW
```
R = (PnL - Cost) / (Drawdown_estimate + ε)
```
**Best for**: Drawdown-constrained portfolios
**Parameters**: `epsilon` (0.001), `drawdown_multiplier` (0.5)
**Differs from others**: Directly penalizes drawdown risk

### 7. INFORMATION_RATIO ✨ NEW
```
R = (PnL - Cost) / (σ + ε) + consistency_bonus
```
**Best for**: Alpha generation and consistency
**Parameters**: `epsilon` (0.001), `consistency_bonus` (0.1)
**Differs from WITH_SHARPE**: Bonus for positive returns

### 8. COMPOSITE ✨ NEW
```
R = w₁*R_returns + w₂*R_sharpe + w₃*R_risk
where: w₁ + w₂ + w₃ = 1
```
**Best for**: Fine-tuned multi-objective optimization
**Parameters**: `weight_returns` (0.5), `weight_sharpe` (0.3), `weight_risk` (0.2)
**Differs from others**: Explicitly balances multiple signals

---

## Configuration Examples

### Example 1: Run with SORTINO Reward

```python
from trading_config import ExperimentConfig, ForecastMode, RewardType, DataConfig, \
    ForecastingConfig, EnvironmentConfig, PPOConfig

config = ExperimentConfig(
    experiment_name="PPO-Sortino-Downside-Focus",
    forecast_mode=ForecastMode.NONE,
    reward_type=RewardType.SORTINO,
    data=DataConfig(),
    forecasting=ForecastingConfig(),
    environment=EnvironmentConfig(
        reward_type=RewardType.SORTINO,
        reward_params={
            'epsilon': 0.001,
            'downside_scale': 1.5,  # More aggressive downside penalty
        }
    ),
    ppo=PPOConfig(total_updates=3000),
    wandb_group="sortino_experiments",
)

from trading_framework import ExperimentRunner
runner = ExperimentRunner(config)
results = runner.run()
```

### Example 2: Run with COMPOSITE Reward

```python
config = ExperimentConfig(
    experiment_name="PPO-Composite-Balanced",
    forecast_mode=ForecastMode.NONE,
    reward_type=RewardType.COMPOSITE,
    # ... other configs ...
    environment=EnvironmentConfig(
        reward_type=RewardType.COMPOSITE,
        reward_params={
            'weight_returns': 0.4,   # Less focus on raw returns
            'weight_sharpe': 0.4,    # More focus on risk-adjusted
            'weight_risk': 0.2,      # Risk penalty
            'kappa': 0.01,
            'epsilon': 0.001,
        }
    ),
    # ...
)
```

### Example 3: Batch Experiment (All 8 Reward Types)

```python
from trading_config import get_ppo_different_rewards_configs
from trading_framework import ExperimentRunner

configs = get_ppo_different_rewards_configs()
results_by_reward = {}

for config in configs:
    print(f"\nRunning: {config.experiment_name}")
    print(f"  Reward Type: {config.reward_type.value}")
    print(f"  Params: {config.environment.reward_params}")

    runner = ExperimentRunner(config)
    results = runner.run()
    results_by_reward[config.reward_type.value] = results

# Compare results across reward types
```

---

## Integration with TradingEnv (Future)

Once you're ready to use the new modular system, update `TradingEnv.step()`:

```python
# In TradingEnv.__init__()
from reward_calculators import create_reward_calculator, RewardCalculatorType

# Map old RewardType to new RewardCalculatorType
reward_type_map = {
    RewardType.BASIC: RewardCalculatorType.BASIC,
    RewardType.WITH_RISK: RewardCalculatorType.WITH_RISK,
    # ... etc
}

self.reward_calculator = create_reward_calculator(
    reward_type_map[reward_type],
    **environment.reward_params
)

# In TradingEnv.step()
components = RewardComponents(
    pnl=pnl,
    cost=cost,
    slippage=slippage,
    risk_penalty=0.0,  # Not used with new system
    volatility=sigma_t,
    position=self.pos,
    true_reward=true_reward
)

reward = self.reward_calculator.calculate(components)
reward *= self.reward_scale
```

---

## Testing Your Configuration

### Quick Validation Script

```python
from reward_calculators import create_reward_calculator, RewardCalculatorType, RewardComponents
import numpy as np

# Test configuration
reward_type = RewardCalculatorType.SORTINO
params = {'epsilon': 0.001, 'downside_scale': 1.2}
calc = create_reward_calculator(reward_type, **params)

# Generate synthetic data
np.random.seed(42)
for i in range(100):
    pnl = np.random.normal(0.0005, 0.0002)
    cost = 0.0001
    slippage = 0.00005
    vol = np.random.uniform(0.005, 0.02)
    pos = np.random.uniform(-1.0, 1.0)

    components = RewardComponents(
        pnl=pnl, cost=cost, slippage=slippage, risk_penalty=0.0,
        volatility=vol, position=pos,
        true_reward=pnl - cost - slippage
    )

    reward = calc.calculate(components)
    calc.record_reward(reward)

# Validate
stats = calc.get_stats()
print(f"{calc.name} Reward Statistics:")
print(f"  Mean: {stats['mean']:.6f}")
print(f"  Std:  {stats['std']:.6f}")
print(f"  Min:  {stats['min']:.6f}")
print(f"  Max:  {stats['max']:.6f}")
```

---

## Key Improvements

| Aspect | Before | After |
|--------|--------|-------|
| **Modularity** | Hardcoded if/elif | Pluggable classes |
| **Testability** | Difficult (integration only) | Easy (unit tests) |
| **Extensibility** | Requires code changes | Just add new class |
| **Configurability** | Limited to kappa | Full per-reward params |
| **Clarity** | Mixed logic | Clear separation of concerns |
| **Reward Types** | 4 types | 8 types |
| **Documentation** | Minimal | Comprehensive |

---

## Next Steps

1. **Run Tests**:
   ```bash
   pytest test_reward_calculators.py -v
   ```

2. **Review Documentation**:
   - Read `REWARD_ANALYSIS_COMPREHENSIVE.md` for detailed analysis

3. **Try Configurations**:
   - Run experiments with different reward types
   - Compare results in WandB

4. **Integrate with TradingEnv** (when ready):
   - Update `TradingEnv.step()` to use `RewardCalculator`
   - Remove hardcoded if/elif logic

5. **Analyze Results**:
   - Which reward type works best for Bitcoin?
   - What parameter values are optimal?
   - Any insights on risk management?

---

## FAQ

**Q: Should I use the new system immediately?**
A: The new reward calculators are standalone and work independently. The integration with TradingEnv is optional. You can test the new system without breaking existing code.

**Q: Which reward type should I start with?**
A: `WITH_RISK` (the current default) is a good balanced choice. Start there, then compare with `WITH_SHARPE` for risk-adjusted focus and `SORTINO` for downside protection.

**Q: Can I create my own reward type?**
A: Yes! Subclass `RewardCalculator` and implement the `calculate()` method. Then use it with:
```python
class MyCustomReward(RewardCalculator):
    def calculate(self, components: RewardComponents) -> float:
        # Your logic here
        return reward
```

**Q: What if I want to test multiple parameter values for one reward type?**
A: Use the `ConfigBuilder` or modify `reward_params`:
```python
for downside_scale in [1.0, 1.2, 1.5]:
    config = ExperimentConfig(...)
    config.environment.reward_params['downside_scale'] = downside_scale
    runner = ExperimentRunner(config)
    results = runner.run()
```

---

## Support & Questions

For detailed formulas and analysis, see `REWARD_ANALYSIS_COMPREHENSIVE.md`

For unit test examples, see `test_reward_calculators.py`

For implementation details, see `reward_calculators.py`

