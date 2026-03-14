
# ⚠️ CRITICAL FINDING: Reward Functions Not Implemented

## User Question
"Explain the different reward functions for experiments 3-6. Show all configurations and explain why all seem to have the same results?"

## Answer
✅ **THE CONFIGURATIONS ARE DEFINED BUT NOT IMPLEMENTED IN CODE**

**This explains why all experiments have identical results ($143,611.60 final equity)**

═══════════════════════════════════════════════════════════════════════════════
## EXPERIMENTS 3-6: CONFIGURATION DEFINITIONS
═══════════════════════════════════════════════════════════════════════════════

### Experiment 3: PPO-Basic-Reward
```python
ExperimentConfig(
    experiment_name="PPO-Basic-Reward",
    forecast_mode=ForecastMode.NONE,
    reward_type=RewardType.BASIC,
    environment=EnvironmentConfig(reward_type=RewardType.BASIC),
)
```

**Configuration:**
- Forecast: ❌ NO (ForecastMode.NONE)
- Reward Type: BASIC
- Initial Equity: $100,000
- Fee: 0.0001
- Kappa: 0.01
- Max Leverage: 1.0

**Intended Reward Function:**
```
reward = PnL - Cost
       = (position × return) - (fee × turnover)
```

---

### Experiment 4: PPO-With-Risk
```python
ExperimentConfig(
    experiment_name="PPO-With-Risk",
    forecast_mode=ForecastMode.NONE,
    reward_type=RewardType.WITH_RISK,
    environment=EnvironmentConfig(reward_type=RewardType.WITH_RISK),
)
```

**Configuration:**
- Forecast: ❌ NO (ForecastMode.NONE)
- Reward Type: WITH_RISK
- Initial Equity: $100,000
- Fee: 0.0001
- Kappa: 0.01
- Max Leverage: 1.0

**Intended Reward Function:**
```
reward = PnL - Cost - Risk_Penalty
       = (position × return) - (fee × turnover) - (kappa × position² × volatility)
```

---

### Experiment 5: PPO-With-Sharpe
```python
ExperimentConfig(
    experiment_name="PPO-With-Sharpe",
    forecast_mode=ForecastMode.NONE,
    reward_type=RewardType.WITH_SHARPE,
    environment=EnvironmentConfig(reward_type=RewardType.WITH_SHARPE),
)
```

**Configuration:**
- Forecast: ❌ NO (ForecastMode.NONE)
- Reward Type: WITH_SHARPE
- Initial Equity: $100,000
- Fee: 0.0001
- Kappa: 0.01
- Max Leverage: 1.0

**Intended Reward Function:**
```
reward = (PnL - Cost) / Volatility
       = ((position × return) - (fee × turnover)) / volatility
```

---

### Experiment 6: PPO-Risk-Adjusted
```python
ExperimentConfig(
    experiment_name="PPO-Risk-Adjusted",
    forecast_mode=ForecastMode.NONE,
    reward_type=RewardType.RISK_ADJUSTED,
    environment=EnvironmentConfig(reward_type=RewardType.RISK_ADJUSTED),
)
```

**Configuration:**
- Forecast: ❌ NO (ForecastMode.NONE)
- Reward Type: RISK_ADJUSTED
- Initial Equity: $100,000
- Fee: 0.0001
- Kappa: 0.01
- Max Leverage: 1.0

**Intended Reward Function:**
```
reward = (PnL / Volatility) - Cost
       = ((position × return) / volatility) - (fee × turnover)
```

═══════════════════════════════════════════════════════════════════════════════
## WHY ALL CONFIGURATIONS HAVE SAME RESULTS
═══════════════════════════════════════════════════════════════════════════════

### ⚠️ THE BUG: Reward Type Is Stored But NOT USED

**Location: trading_framework.py, TradingEnv.step() method (Lines 1130-1155)**

Current Implementation (STATIC):
```python
# 8) Risk penalty (ALWAYS applied)
risk_pen = self.kappa * (self.pos ** 2) * sigma_t

# 9) Final reward (ALWAYS same formula)
true_reward = pnl - cost - slippage
reward = true_reward - risk_pen          # ← Always WITH_RISK reward!
reward *= self.reward_scale
```

**Problem:**
- ✗ reward_type is stored (Line 1057): `self.reward_type = reward_type`
- ✗ reward_type is passed to TradingEnv (Line 561)
- ✗ reward_type is logged to WandB (Lines 146, 183, 905)
- ✅ BUT it's NEVER USED to calculate different rewards!

**All experiments use the SAME reward formula:**
```
reward = (pnl - cost - slippage) - risk_penalty
```

This is exactly the **WITH_RISK** reward (Experiment 4's intended function).

All other reward types (BASIC, WITH_SHARPE, RISK_ADJUSTED) are **never implemented**.

═══════════════════════════════════════════════════════════════════════════════
## EXPECTED vs ACTUAL RESULTS
═══════════════════════════════════════════════════════════════════════════════

### Expected Results (If Implemented Correctly):

| Experiment | Reward Type | Expected Effect | Expected Equity |
|---|---|---|---|
| 3 | BASIC | Less reward shaping, more trading | ~$130K-$150K |
| 4 | WITH_RISK | Risk penalty discourages large positions | ~$140K-$160K |
| 5 | WITH_SHARPE | Normalize by volatility, prefer stability | ~$120K-$140K |
| 6 | RISK_ADJUSTED | PnL/volatility, aggressive in calm times | ~$130K-$150K |

### Actual Results (All Using WITH_RISK):

| Experiment | Actual Equity | Why |
|---|---|---|
| 3 | $143,611.60 | Using WITH_RISK (not BASIC) |
| 4 | $143,611.60 | Using WITH_RISK (correct) |
| 5 | $143,611.60 | Using WITH_RISK (not WITH_SHARPE) |
| 6 | $143,611.60 | Using WITH_RISK (not RISK_ADJUSTED) |

**Conclusion:** All experiments produce identical results because they all use the same underlying reward function.

═══════════════════════════════════════════════════════════════════════════════
## DETAILED EXPLANATION OF EACH REWARD FUNCTION
═══════════════════════════════════════════════════════════════════════════════

### 1. BASIC Reward Function
**Formula:**
```
reward_t = pnl_t - cost_t
         = (position_t × return_t) - (fee × |position_change_t|)
```

**Interpretation:**
- Simplest reward: only PnL minus trading costs
- No explicit risk penalty
- Agent learns to maximize raw returns
- May take excessive risks (overleveraging)

**Pros:**
- Simple, direct signal
- Encourages profitable trades

**Cons:**
- Ignores volatility/risk
- Can lead to unstable strategies
- High drawdowns possible

---

### 2. WITH_RISK Reward Function (CURRENTLY IMPLEMENTED)
**Formula:**
```
reward_t = (pnl_t - cost_t - slippage_t) - risk_penalty_t
         = (pos_t × r_t - fee × turnover_t) - (kappa × pos_t² × sigma_t)

where:
  kappa = 0.01 (risk penalty coefficient)
  sigma_t = rolling volatility
```

**Interpretation:**
- Base reward: PnL - costs (like BASIC)
- Plus: Risk penalty term that discourages large positions
- Penalty is quadratic in position (pos²), so doubles position → 4× penalty
- Penalty scales with volatility (high vol → higher penalty)

**Pros:**
- Encourages reasonable position sizing
- More stable than BASIC
- Risk-aware without explicit Sharpe optimization

**Cons:**
- Penalizes positions even if profitable
- Fixed penalty may not match market regimes
- Doesn't explicitly optimize Sharpe ratio

---

### 3. WITH_SHARPE Reward Function (NOT IMPLEMENTED - BUG)
**Formula (Should Be):**
```
reward_t = (pnl_t - cost_t) / (sigma_t + epsilon)
         = (pos_t × r_t - fee × turnover_t) / (rolling_volatility_t + 1e-8)
```

**Interpretation:**
- Normalize returns by volatility
- High returns with low volatility = high reward
- Low returns with high volatility = low reward
- Directly optimizes risk-adjusted returns

**Pros:**
- Explicitly targets Sharpe ratio improvement
- Penalizes volatility more than WITH_RISK
- Produces more stable, smoother equity curves

**Cons:**
- May under-reward when vol is low
- Division by small numbers can be noisy
- May lead to under-trading (too conservative)

**Why It's Better Than WITH_RISK:**
```
WITH_RISK:
  Example: pnl=0.001, cost=0.0001, risk_pen=0.0005
  reward = (0.001 - 0.0001 - 0.0005) = 0.0004

WITH_SHARPE:
  Example: pnl=0.001, cost=0.0001, vol=0.02
  reward = (0.001 - 0.0001) / 0.02 = 0.045
  (Much higher reward signal!)
```

---

### 4. RISK_ADJUSTED Reward Function (NOT IMPLEMENTED - BUG)
**Formula (Should Be):**
```
reward_t = (pnl_t / volatility_t) - cost_t
         = (pos_t × r_t / sigma_t) - (fee × turnover_t)
```

**Interpretation:**
- Risk-adjust the PnL component only
- Still penalize costs directly (not relative to volatility)
- Return/volatility is like Sharpe but without the subtraction
- Aggressive in calm periods (low vol → high return/vol)

**Pros:**
- Adaptive to market regimes
- High returns when market is calm
- Better capital utilization

**Cons:**
- Can lead to leverage-seeking behavior
- May over-trade in low-volatility periods
- Riskier than WITH_SHARPE

**Difference from WITH_SHARPE:**
```
WITH_SHARPE:   reward = (pnl - cost) / vol    ← Everything divided by vol
RISK_ADJUSTED: reward = (pnl / vol) - cost    ← Only pnl divided, costs separate
```

With RISK_ADJUSTED, costs have more impact, encouraging less trading.

═══════════════════════════════════════════════════════════════════════════════
## WHY THESE COMBINATIONS MAKE SENSE (ABLATION STUDY)
═══════════════════════════════════════════════════════════════════════════════

**Experiments 3-6 are a Reward Function Ablation Study:**

1. **BASIC** (Exp 3): Baseline reward - just PnL - costs
   → Test: How much does risk-awareness help?

2. **WITH_RISK** (Exp 4): Add quadratic position penalty
   → Test: Does explicit risk penalty improve stability?

3. **WITH_SHARPE** (Exp 5): Normalize by volatility
   → Test: Does Sharpe optimization beat WITH_RISK?

4. **RISK_ADJUSTED** (Exp 6): Risk-adjust PnL only
   → Test: Does adaptive risk adjustment help?

**Progression of Risk Awareness:**
```
BASIC (none)
  ↓
WITH_RISK (quadratic penalty on position)
  ↓
WITH_SHARPE (normalize everything by vol)
  ↓
RISK_ADJUSTED (normalize PnL, fixed costs)
```

Each level adds more sophistication in handling risk.

═══════════════════════════════════════════════════════════════════════════════
## THE FIX REQUIRED
═══════════════════════════════════════════════════════════════════════════════

To implement the intended functionality, modify TradingEnv.step():

```python
# Current code (WRONG - all rewards are WITH_RISK):
reward = true_reward - risk_pen

# Should be (CORRECT - different rewards):
if self.reward_type == RewardType.BASIC:
    reward = true_reward

elif self.reward_type == RewardType.WITH_RISK:
    reward = true_reward - risk_pen

elif self.reward_type == RewardType.WITH_SHARPE:
    safe_sigma = max(sigma_t, 0.001)  # Avoid division by zero
    reward = true_reward / safe_sigma

elif self.reward_type == RewardType.RISK_ADJUSTED:
    safe_sigma = max(sigma_t, 0.001)
    risk_adjusted_pnl = pnl / safe_sigma
    reward = risk_adjusted_pnl - cost - slippage

reward *= self.reward_scale
```

═══════════════════════════════════════════════════════════════════════════════
## SUMMARY
═══════════════════════════════════════════════════════════════════════════════

**Question: Why do all experiments (3-6) have the same results?**

**Answer:** Because the reward functions are defined in the configuration but NOT implemented in the actual trading environment code.

**Current State:**
- ✓ Config files define 4 different reward types
- ✓ reward_type is passed to TradingEnv
- ✓ reward_type is logged to WandB
- ✗ reward_type is NEVER USED in reward calculation
- ✗ All experiments use WITH_RISK reward (hardcoded)

**Impact:**
- All 4 reward function experiments produce identical results
- The ablation study doesn't work (no variation to compare)
- $143,611.60 is the result of WITH_RISK reward for all

**Solution:**
Implement the if/elif chain in TradingEnv.step() to actually use self.reward_type


