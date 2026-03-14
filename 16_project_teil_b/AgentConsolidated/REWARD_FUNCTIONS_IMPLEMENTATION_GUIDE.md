
# ✅ REWARD FUNCTIONS FIXED - IMPLEMENTATION GUIDE

## The Problem (Now Fixed)
All experiments 3-6 had identical results ($143,611.60) because the reward functions were defined in configuration but NOT actually used in the code.

## The Solution Applied
Added if/elif logic in `TradingEnv.step()` (Lines 1156-1185) to implement 4 distinct reward functions.

═══════════════════════════════════════════════════════════════════════════════
## EXPERIMENTS 3-6: REWARD FUNCTION IMPLEMENTATIONS
═══════════════════════════════════════════════════════════════════════════════

### Experiment 1: PPO-Without-Forecast (Baseline)
- Forecast: ❌ NO
- Reward: WITH_RISK (standard)
- Purpose: Baseline for comparison

### Experiment 2: PPO-With-Forecast (Baseline with Forecast)
- Forecast: ✅ YES (LSTM)
- Reward: WITH_RISK (standard)
- Purpose: Test forecast impact

---

### Experiment 3: PPO-Basic-Reward
**Code:** Lines 1159-1162
```python
if self.reward_type == RewardType.BASIC:
    # Basic reward: Just PnL minus costs
    reward = true_reward
```

**Formula:**
```
reward_t = PnL_t - Cost_t
         = (position_t × return_t) - (fee × turnover_t)
```

**Characteristics:**
- ✓ Simplest reward signal
- ✓ No risk penalty
- ✓ Encourages maximum returns
- ✗ Can lead to risky behaviors
- ✗ Likely higher volatility/drawdowns

**Expected Results:**
- Higher returns (less risk penalty)
- Higher drawdowns
- More aggressive trading
- Possibly unstable equity curve
- Example: Final equity $150K-$160K (more aggressive than WITH_RISK)

**Use Case:**
- Test: How much does risk awareness help?
- Baseline: Pure profit maximization

---

### Experiment 4: PPO-With-Risk (The Implemented Standard)
**Code:** Lines 1164-1167
```python
elif self.reward_type == RewardType.WITH_RISK:
    # With risk penalty: PnL - costs - quadratic position penalty
    reward = true_reward - risk_pen
```

**Formula:**
```
reward_t = (PnL_t - Cost_t - Slippage_t) - Risk_Penalty_t
         = (pos_t × r_t - fee × turnover) - (kappa × pos_t² × sigma_t)

where:
  kappa = 0.01 (risk penalty coefficient)
  sigma_t = rolling volatility (20-day)
```

**Components Explained:**

1. **PnL Component:** `pos_t × r_t`
   - Position × daily return
   - If position=0.5, return=0.01: PnL = 0.005

2. **Cost Component:** `fee × turnover`
   - Trading fee per unit change
   - If turnover=0.1, fee=0.0001: Cost = 0.00001

3. **Risk Penalty:** `kappa × pos_t² × sigma_t`
   - Quadratic in position (doubles position → 4× penalty)
   - Scales with volatility (high vol → higher penalty)
   - Example: pos=0.5, vol=0.02, kappa=0.01
     - Penalty = 0.01 × 0.25 × 0.02 = 0.00005

**Characteristics:**
- ✓ Balances returns with risk
- ✓ Penalizes large positions
- ✓ Adaptive to market volatility
- ✓ More stable than BASIC
- ✗ May under-utilize capital
- ✗ Fixed penalty coefficient may not adapt well

**Expected Results:**
- Moderate returns
- Lower volatility than BASIC
- Reasonable position sizing
- Stable equity curve
- Example: Final equity $140K-$150K

**Use Case:**
- Standard approach
- Good balance of risk/return
- Baseline for risk-aware trading

---

### Experiment 5: PPO-With-Sharpe
**Code:** Lines 1169-1173
```python
elif self.reward_type == RewardType.WITH_SHARPE:
    # Sharpe-like reward: normalize by volatility
    safe_sigma = max(sigma_t, 0.001)
    reward = true_reward / safe_sigma
```

**Formula:**
```
reward_t = (PnL_t - Cost_t - Slippage_t) / Volatility_t
         = (pos_t × r_t - fee × turnover) / sigma_t
```

**Example Calculation:**
```
Scenario 1 (Calm market):
  PnL: 0.001, Cost: 0.00001, Vol: 0.005
  reward = 0.00099 / 0.005 = 0.198

Scenario 2 (Turbulent market):
  PnL: 0.001, Cost: 0.00001, Vol: 0.050
  reward = 0.00099 / 0.050 = 0.0198
  (Same PnL → 10× lower reward due to higher vol!)
```

**Characteristics:**
- ✓ Explicitly optimizes risk-adjusted returns (Sharpe-like)
- ✓ Penalizes volatility heavily
- ✓ Encourages smooth, stable returns
- ✓ More sophisticated than WITH_RISK
- ✗ May underutilize opportunities in calm periods (low vol)
- ✗ Very conservative in low volatility
- ✗ Can create numerical instability if vol approaches zero

**Expected Results:**
- Lower returns overall
- Much lower volatility
- Very smooth equity curve (lowest drawdowns)
- Conservative position sizing
- More consistent daily returns
- Example: Final equity $120K-$140K

**Why Better Than WITH_RISK:**
- WITH_RISK: Fixed penalty regardless of reward magnitude
- WITH_SHARPE: Penalty scales with actual profitability
- WITH_SHARPE penalizes both bad trades AND trades in volatile markets

**Use Case:**
- Stability-focused strategy
- Institutional mandate: maximize Sharpe ratio
- Risk-averse investors
- "Smooth returns" objective

---

### Experiment 6: PPO-Risk-Adjusted
**Code:** Lines 1175-1180
```python
elif self.reward_type == RewardType.RISK_ADJUSTED:
    # Risk-adjusted reward: (PnL / volatility) - cost
    safe_sigma = max(sigma_t, 0.001)
    risk_adjusted_pnl = pnl / safe_sigma
    reward = risk_adjusted_pnl - cost - slippage
```

**Formula:**
```
reward_t = (PnL_t / Volatility_t) - Cost_t - Slippage_t
         = (pos_t × r_t / sigma_t) - (fee × turnover)
```

**Example Calculation:**
```
Scenario 1 (Calm market, profitable):
  PnL: 0.001, Cost: 0.00001, Vol: 0.005
  reward = (0.001 / 0.005) - 0.00001 = 0.2 - 0.00001 ≈ 0.199

Scenario 2 (Turbulent market, same PnL):
  PnL: 0.001, Cost: 0.00001, Vol: 0.050
  reward = (0.001 / 0.050) - 0.00001 = 0.02 - 0.00001 ≈ 0.0199

Scenario 3 (Calm market, loss):
  PnL: -0.001, Cost: 0.00001, Vol: 0.005
  reward = (-0.001 / 0.005) - 0.00001 = -0.2 - 0.00001 ≈ -0.200
```

**Key Difference from WITH_SHARPE:**
```
WITH_SHARPE:   reward = (pnl - cost) / vol     ← Everything normalized
               = (0.001 - 0.00001) / vol = 0.00099 / vol

RISK_ADJUSTED: reward = (pnl / vol) - cost     ← Only PnL normalized
               = (0.001 / vol) - 0.00001
```

The difference: in WITH_SHARPE, costs are included in the numerator (reduced before division).
In RISK_ADJUSTED, costs are subtracted after division (less impact).

**Characteristics:**
- ✓ Adaptive to market regimes
- ✓ Aggressive in calm periods (low vol → high reward)
- ✓ More trading when market is stable
- ✓ Good capital utilization
- ✗ Can lead to over-trading in calm markets
- ✗ More risk than WITH_SHARPE
- ✗ Leverage-seeking behavior possible

**Expected Results:**
- Higher returns than WITH_SHARPE
- Moderate volatility
- More trading than WITH_SHARPE
- Varies by market regime
- Example: Final equity $130K-$150K

**Use Case:**
- Active trading strategy
- Regime-aware optimization
- "Trade when calm, reduce when turbulent"
- Higher return target with moderate risk

═══════════════════════════════════════════════════════════════════════════════
## EXPECTED PERFORMANCE COMPARISON
═══════════════════════════════════════════════════════════════════════════════

Based on reward theory, expected results after re-running with fixes:

| Exp | Name | Reward Type | Return | Volatility | Sharpe | DD | Risk Level |
|---|---|---|---|---|---|---|---|
| 3 | Basic | BASIC | HIGH | HIGH | LOW | HIGH | AGGRESSIVE |
| 4 | With-Risk | WITH_RISK | MODERATE | MODERATE | MODERATE | MODERATE | BALANCED |
| 5 | With-Sharpe | WITH_SHARPE | LOW | LOW | HIGH | LOW | CONSERVATIVE |
| 6 | Risk-Adjusted | RISK_ADJUSTED | MODERATE-HIGH | MODERATE | MODERATE | MODERATE | ACTIVE |

**Expected Ranges (vs current $143,611.60):**
- Exp 3 (BASIC): $150K - $170K (more aggressive)
- Exp 4 (WITH_RISK): $140K - $150K (similar to current)
- Exp 5 (WITH_SHARPE): $120K - $140K (conservative)
- Exp 6 (RISK_ADJUSTED): $135K - $155K (adaptive)

═══════════════════════════════════════════════════════════════════════════════
## WHY THIS ABLATION MAKES SENSE
═══════════════════════════════════════════════════════════════════════════════

The 4 reward functions form a progression of risk sophistication:

```
Sophistication Level:  1         2            3            4
                      BASIC  → WITH_RISK → WITH_SHARPE → RISK_ADJUSTED

Risk Management:      None   Linear      Normalized   Regime-Adaptive
                             Penalty     Penalty      Penalty

Formula:             pnl-c  (pnl-c)      (pnl-c)/vol  (pnl/vol)-c
                            -kappa*p²

Risk Focus:          Return Focus    Stability Focus         Adaptive
```

**Research Questions Answered:**

1. **BASIC vs WITH_RISK:**
   "How much does explicit risk penalty help?"
   → Compare BASIC vs WITH_RISK returns/Sharpe

2. **WITH_RISK vs WITH_SHARPE:**
   "Is normalization by vol better than fixed penalty?"
   → Compare stability and Sharpe ratio

3. **WITH_SHARPE vs RISK_ADJUSTED:**
   "Is regime-adaptive better than static normalization?"
   → Compare performance in different market conditions

═══════════════════════════════════════════════════════════════════════════════
## CODE CHANGES VERIFICATION
═══════════════════════════════════════════════════════════════════════════════

✅ File: trading_framework.py
✅ Location: TradingEnv.step() method, Lines 1156-1185
✅ Changes: Added if/elif block for 4 reward types
✅ Imports: RewardType already imported (Line 98)
✅ Testing: Ready to run experiments with different results

## Next Steps

1. **Run experiments again:**
   ```bash
   python run_all_experiments.py
   ```

2. **Expected console output:** Different metrics for each reward type

3. **Expected WandB results:** Different reward/equity curves for each experiment

4. **Comparison table:** You'll see variation between experiments (no longer all identical)

═══════════════════════════════════════════════════════════════════════════════
## SUMMARY
═══════════════════════════════════════════════════════════════════════════════

**Before Fix:**
- All 4 reward experiments produced identical results
- Reward type was stored but not used
- Ablation study didn't work

**After Fix:**
- Each reward type has distinct implementation
- Formulas match intended behavior
- Ablation study will show variation
- Can compare risk/return tradeoffs

**Now you can measure:**
- Impact of explicit risk penalty (BASIC vs WITH_RISK)
- Value of Sharpe optimization (WITH_RISK vs WITH_SHARPE)
- Benefit of regime adaptation (WITH_SHARPE vs RISK_ADJUSTED)


