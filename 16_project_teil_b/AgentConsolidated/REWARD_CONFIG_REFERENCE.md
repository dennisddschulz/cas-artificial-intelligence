# Reward Configuration & Kappa Analysis

## Location of Reward Configurations

**File**: `/home/isc-den/cas-artificial-intelligence/16_project_teil_b/AgentConsolidated/trading_config.py`

**Function**: `get_ppo_different_rewards_configs()` (Line 370-425)

---

## Reward Ablation Configurations (8 Variants)

```python
reward_configs = [
    (RewardType.BASIC, "basic_reward", {
        "epsilon": 0.001,
    }),
    (RewardType.WITH_RISK, "with_risk_reward", {
        "kappa": 0.01,           # ← EXPLICIT KAPPA
        "epsilon": 0.001,
    }),
    (RewardType.WITH_SHARPE, "with_sharpe_reward", {
        "epsilon": 0.001,
    }),
    (RewardType.RISK_ADJUSTED, "risk_adjusted_reward", {
        "epsilon": 0.001,
    }),
    (RewardType.SORTINO, "sortino_reward", {
        "epsilon": 0.001,
        "downside_scale": 1.2,
    }),
    (RewardType.CALMAR, "calmar_reward", {
        "epsilon": 0.001,
        "drawdown_multiplier": 0.5,
    }),
    (RewardType.INFORMATION_RATIO, "information_ratio_reward", {
        "epsilon": 0.001,
        "consistency_bonus": 0.1,
    }),
    (RewardType.COMPOSITE, "composite_reward", {
        "weight_returns": 0.5,
        "weight_sharpe": 0.3,
        "weight_risk": 0.2,
        "kappa": 0.01,           # ← EXPLICIT KAPPA
        "epsilon": 0.001,
    }),
]
```

---

## Kappa Usage Analysis

### **Default Kappa Value**
Located in `EnvironmentConfig` class (Line 140):
```python
@dataclass
class EnvironmentConfig:
    """Trading Environment Configuration"""
    kappa: float = 0.01  # Risk penalty weight (for WITH_RISK reward)
```

### **Kappa Across Reward Types**

| Reward Type | Kappa Used? | Value | Purpose |
|------------|-------------|-------|---------|
| **BASIC** | ❌ No | - | Simple PnL maximization |
| **WITH_RISK** | ✅ Yes | 0.01 | Risk penalty on leverage |
| **WITH_SHARPE** | ❌ No | - | Uses volatility internally |
| **RISK_ADJUSTED** | ❌ No | - | Uses volatility internally |
| **SORTINO** | ❌ No | - | Uses downside_scale=1.2 |
| **CALMAR** | ❌ No | - | Uses drawdown_multiplier=0.5 |
| **INFORMATION_RATIO** | ❌ No | - | Uses consistency_bonus=0.1 |
| **COMPOSITE** | ✅ Yes | 0.01 | Multi-objective blend |

---

## Key Findings

### ✅ Same Kappa Across Variants
- **WITH_RISK** uses `kappa=0.01`
- **COMPOSITE** (which includes risk component) uses `kappa=0.01`
- Both explicitly configured in reward_configs dict

### ⚠️ Other Variants DON'T Use Kappa
- Instead, they use **different risk metrics**:
  - `WITH_SHARPE`: Volatility-based (σ)
  - `RISK_ADJUSTED`: Return / Volatility ratio
  - `SORTINO`: Downside volatility (σ_down)
  - `CALMAR`: Return / Max Drawdown
  - `INFORMATION_RATIO`: Alpha / Tracking Error

### 📊 Parameters Applied Per Reward Type

Each reward config is passed to `EnvironmentConfig.reward_params`:

```python
env_config = EnvironmentConfig(reward_type=reward_type)
env_config.reward_params.update(reward_params)  # ← Merges with defaults
```

The `reward_params` dict contains **reward-specific parameters**:
- Not global settings
- Applied only to that reward function
- Merged with defaults from EnvironmentConfig

---

## How It Works in Training

In `trading_framework.py`, rewards are computed using these parameters:

```python
def compute_reward(self, ...):
    if self.config.reward_type == RewardType.WITH_RISK:
        # Uses config.environment.reward_params['kappa']
        leverage_penalty = kappa * leverage^2 * volatility
        reward = pnl - cost - leverage_penalty

    elif self.config.reward_type == RewardType.COMPOSITE:
        # Uses weighted combination + kappa
        sharpe_signal = pnl / volatility
        reward_composite = (
            w_return * pnl +
            w_sharpe * sharpe_signal +
            w_risk * (-kappa * leverage^2 * volatility)
        )
```

---

## Summary

| Question | Answer |
|----------|--------|
| **Where are reward configs?** | `trading_config.py` line 370-425, function `get_ppo_different_rewards_configs()` |
| **Is same kappa used?** | ✅ Yes, `kappa=0.01` for WITH_RISK and COMPOSITE; other variants use different risk metrics |
| **Why not all use kappa?** | Each reward type optimizes a different objective (return, Sharpe, Sortino, Calmar, etc.) |
| **Are parameters separate?** | ✅ Yes, each reward has its own `reward_params` dict merged with defaults |
| **Where are defaults?** | `EnvironmentConfig.reward_params` (lines 165-182) |

---

## Comparison: Ablation Study Design

The ablation study tests **8 different risk management approaches**:

```
BASIC              → No risk penalty
├─ WITH_RISK       → Leverage penalty: -κ×L²×σ
├─ WITH_SHARPE     → Return/Volatility ratio
├─ RISK_ADJUSTED   → Direct Return/σ
├─ SORTINO         → Downside volatility focus
├─ CALMAR          → Return/Drawdown focus
├─ INFO_RATIO      → Alpha/Tracking Error
└─ COMPOSITE       → Weighted blend (includes κ)
```

All use `epsilon=0.001` for stability, but each has unique risk component.


