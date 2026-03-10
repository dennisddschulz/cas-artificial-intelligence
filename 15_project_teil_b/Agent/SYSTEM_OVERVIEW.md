# Budget & Liquidity System - Visual Overview

## 📊 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRADING ENVIRONMENT                          │
│                   (EnhancedTradingEnv)                          │
└─────────────────────────────────────────────────────────────────┘

START:
┌─────────────────────────────────────────────────────────────────┐
│  INITIAL STATE (reset)                                          │
│  ├─ budget_initial = 100,000 USD      ← FIXED REFERENCE        │
│  ├─ budget = 100,000 USD              ← Current total           │
│  ├─ equity = 100,000 USD              ← Portfolio value         │
│  ├─ cash = 100,000 USD                ← All cash, no position   │
│  ├─ pos = 0.0                         ← No position             │
│  └─ peak_equity = 100,000 USD         ← For drawdown tracking   │
└─────────────────────────────────────────────────────────────────┘

STEP t:
┌─────────────────────────────────────────────────────────────────┐
│  ACTION FROM POLICY                                             │
│  a_t ∈ [-1.0, 1.0]  (target position as capital fraction)      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  MARKET DATA                                                    │
│  ├─ r_t: log return                                            │
│  ├─ sigma_t: volatility                                        │
│  └─ mu_hat: forecasted return                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PnL CALCULATION                                                │
│  pnl = pos_{t-1} * r_t                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  COST CALCULATION                                               │
│  cost = fee * |a_t - pos_{t-1}|                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  UPDATE EQUITY                                                  │
│  equity_t = equity_{t-1} * exp(pnl - cost)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  UPDATE POSITION                                                │
│  pos = a_t                                                      │
│  peak_equity = max(peak_equity, equity_t)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  LIQUIDITY MANAGEMENT                                           │
│  ├─ position_value = |pos| * equity                            │
│  ├─ cash = equity - position_value                             │
│  └─ cash_ratio = cash / equity                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  LIQUIDITY CONSTRAINT CHECK                                     │
│  min_cash_required = 0.05 * equity                             │
│                                                                 │
│  IF cash < min_cash_required:                                  │
│    ├─ violation = TRUE                                         │
│    ├─ Scale down position                                      │
│    └─ reward -= 0.05                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  REWARD SHAPING                                                 │
│  base_reward = pnl - cost - risk_penalty                       │
│  alignment = a_t * mu_hat                                      │
│  reward = base_reward + alignment * 0.1                        │
│  reward = clip(reward, -0.1, 0.1)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  BUILD OBSERVATION (14 features)                                │
│                                                                 │
│  Market Features (8):                                           │
│  ├─ r, r_lag1, mu_hat, sigma_hat                              │
│  └─ rsi, macd_diff, bb_width, ema_ratio                        │
│                                                                 │
│  Portfolio Features (6):                                        │
│  ├─ current_position = tanh(pos/0.5)                           │
│  ├─ liquidity_ratio = cash / equity  ← KEY!                   │
│  ├─ current_leverage = |pos|                                   │
│  ├─ drawdown                                                   │
│  ├─ cumulative_pnl                                             │
│  └─ recent_return                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  RETURN (obs, reward, done, info)                              │
│                                                                 │
│  obs: 14-dimensional normalized feature vector                │
│                                                                 │
│  info: {                                                        │
│    'pnl': float,                                               │
│    'cost': float,                                              │
│    'equity': float,         ← BUDGET INFO                      │
│    'cash': float,           ← BUDGET INFO                      │
│    'cash_ratio': float,     ← LIQUIDITY INFO                   │
│    'position_value': float, ← BUDGET INFO                      │
│    'liquidity_violation': bool,  ← CONSTRAINT                  │
│    'min_cash_required': float,   ← CONSTRAINT                  │
│    'cumulative_return': float    ← PERFORMANCE                 │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 💰 Budget Flow Example

```
DAY 1 (Start)
─────────────
Initial: $100,000

DAY 2 (Agent buys 50% long)
────────────────────────────
Action: a = 0.5
PnL (from day 1): pnl = 0 * r = 0
Cost: fee * |0.5 - 0.0| = 0.0001 * 0.5 = 0.00005
New Equity: 100,000 * exp(-0.00005) = 99,995
Position Value: 0.5 * 99,995 = 49,997
Cash: 99,995 - 49,997 = 49,998
Cash Ratio: 49,998 / 99,995 = 0.500

✓ Liquidity OK: 0.500 > 0.05

DAY 3 (Market up 2%)
────────────────────
r = 0.02
PnL: 0.5 * 0.02 = 0.01 (1% profit!)
Cost: 0 (no position change)
New Equity: 99,995 * exp(0.01) = 101,000
Position Value: 0.5 * 101,000 = 50,500
Cash: 101,000 - 50,500 = 50,500
Cash Ratio: 50,500 / 101,000 = 0.500

✓ Liquidity OK

DAY 4 (Agent reverses to 50% short)
────────────────────────────────────
Action: a = -0.5
PnL: 0.5 * (-0.02) = -0.01 (1% loss! but market was down)
Cost: 0.0001 * |-0.5 - 0.5| = 0.0001
New Equity: 101,000 * exp(-0.01 - 0.0001) = 99,900
Position Value: 0.5 * 99,900 = 49,950
Cash: 99,900 - 49,950 = 49,950
Cash Ratio: 49,950 / 99,900 = 0.500

✓ Liquidity OK: Full reversal cost 1 basis point
```

---

## 📏 Liquidity Constraint Examples

```
SCENARIO 1: Normal Operation
────────────────────────────
Equity: $100,000
Min Cash (5%): $5,000
Max Position Value: $95,000
Max Position: 0.95

Position: 0.50
Position Value: $50,000
Cash: $50,000
Cash Ratio: 0.50

✓ VALID: 50,000 > 5,000

SCENARIO 2: Large Position
──────────────────────────
Position: 0.98
Position Value: $98,000
Cash: $2,000
Cash Ratio: 0.02

✗ VIOLATION: 2,000 < 5,000

ACTION:
├─ Max allowed position = (100,000 - 5,000) / 100,000 = 0.95
├─ Scale down: 0.98 → 0.95
├─ New position value: $95,000
├─ New cash: $5,000
└─ Reward penalty: -0.05

SCENARIO 3: Edge Case (Extreme Volatility)
───────────────────────────────────────────
After big loss: Equity = $50,000
Min Cash (5%): $2,500
Position: 0.95
Position Value: $47,500
Cash: $2,500
Cash Ratio: 0.05

✓ VALID: At minimum threshold
```

---

## 🎁 Reward Shaping Flow

```
┌─────────────────────────────────────────┐
│  Raw Components (all in [-0.1, 0.1])   │
└─────────────────────────────────────────┘

PnL
  0.5 long * 0.02 return = 0.01
  OR
  0.5 short * -0.02 return = 0.01
  RANGE: [-0.01, 0.01]

Cost
  fee * position_change
  fee = 0.0001
  RANGE: [0, 0.0001]

Risk Penalty
  kappa * pos^2 * sigma
  kappa = 0.01
  RANGE: [0, 0.001]

Position Alignment
  clip(a * mu_hat, -0.01, 0.01)
  RANGE: [-0.01, 0.01]

                    ↓

base_reward = pnl - cost - risk_penalty
  RANGE: [-0.0111, 0.01]

alignment_bonus = position_alignment * 0.1
  RANGE: [-0.001, 0.001]

reward = base_reward + alignment_bonus
  RANGE: [-0.0121, 0.011]

                    ↓

FINAL: clip(reward, -0.1, 0.1)
  RANGE: [-0.1, 0.1]

                    ↓

IF liquidity_violation:
  reward -= 0.05
  RANGE: [-0.15, 0.05]
  BUT clipped to [-0.1, 0.1]
  FINAL: [-0.1, 0.05]
```

---

## 🔍 State Space Visualization

```
OBSERVATION VECTOR (14 components)
───────────────────────────────────

[market_features (8)]  [portfolio_features (6)]
        │                        │
        ↓                        ↓
    r=0.01              current_position=0.5
    r_lag1=-0.02        liquidity_ratio=0.50
    mu_hat=0.001        current_leverage=0.50
    sigma_hat=0.02      drawdown=0.05
    rsi=0.3             cumulative_pnl=0.02
    macd_diff=-0.1      recent_return=0.01
    bb_width=1.2
    ema_ratio=0.5

         │
         ↓
    NEURAL NETWORK
    ├─ Dense(14, 256)
    ├─ LayerNorm
    ├─ Dense(256, 256)
    ├─ LayerNorm
    └─ Dense(256, 128)
         │
         ├─→ Actor Head:    μ ∈ ℝ¹  (mean position)
         └─→ Critic Head:   V ∈ ℝ   (value estimate)
```

---

## 📈 Training Convergence Pattern

```
EPISODE RETURNS
───────────────

Update 0-500:    Returns ~ N(-0.5, 1.5)   [HIGH VARIANCE]
│
│  ████░░░░░░░░░░░  High volatility
│  ░░░░████░░░░░░░  Exploration phase
│  ░░░░░░░░░░░░░░░

Update 500-1500: Returns ~ N(0.0, 0.8)   [MEDIUM VARIANCE]
│
│  ░░░░░░████░░░░░  Learning signal
│  ░░░░░░░░░░████░  Convergence starts
│  ░░░░░░░░░░░░░░░

Update 1500-3000: Returns ~ N(0.1, 0.4) [LOW VARIANCE]
│
│  ░░░░░░░░░░████░  Stable strategy
│  ░░░░░░░░░░░░████ Positive returns
│  ░░░░░░░░░░░░░░░░ Exploiting patterns
```

---

## 🎯 Key Metrics to Monitor

```
TRAINING:
─────────
Every 100 updates:
├─ Mean Return (last 100): Should trend upward
├─ log_std: Should stay in [-2.0, -0.5]
├─ Avg|Advantage|: Should be > 0.01 (learning signal)
└─ KL Divergence: Should not exceed 0.05

EVALUATION:
──────────
After training:
├─ Test Episode Return: Should be > 0
├─ Win Rate: Should be > 50%
├─ Liquidity Violations: Should be minimal
└─ Sharpe Ratio: Should be > 0.5
```

---

**Budget & Liquidity System: FULLY INTEGRATED ✅**

Alle Komponenten sind jetzt zusammengebunden und ready for training!

