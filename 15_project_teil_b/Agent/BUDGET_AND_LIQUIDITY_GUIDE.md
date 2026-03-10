# Budget & Liquidity Management System

## 📊 System Overview

Das Trading Environment implementiert ein vollständiges **Budget- und Liquidity-Management-System** mit folgenden Komponenten:

### Initial Budget (Initialization)
```python
INITIAL_EQUITY = 100,000 USD

# State variables:
self.budget_initial = 100,000.0   # Never changes (reference point)
self.budget = 100,000.0           # Current total budget (= equity)
self.equity = 100,000.0           # Total portfolio value
self.cash = 100,000.0             # Cash on hand (not invested)
self.pos = 0.0                    # Target position as fraction of equity
```

---

## 💰 Budget Components

### 1. **Equity** (Gesamtwert)
```
Equity_t = Equity_{t-1} * (1 + pnl - transaction_cost)

Interpretation:
- Total value of portfolio
- Includes both cash and position value
- Subject to PnL and trading costs
```

### 2. **Cash** (Liquide Mittel)
```
Cash_t = Equity_t - Position_Value

Position_Value = |position| * Equity_t

Example:
- Equity = 100,000 USD
- Position = 0.5 (50% long)
- Position_Value = 0.5 * 100,000 = 50,000 USD
- Cash = 100,000 - 50,000 = 50,000 USD
```

### 3. **Position** (Exposure)
```
Position ∈ [-LEVERAGE_MAX, LEVERAGE_MAX] = [-1.0, 1.0]

Interpretation:
- Positive: Long exposure
- Negative: Short exposure
- Magnitude: how much capital is deployed

Examples:
- pos = 0.0  →  0% exposed, 100% cash
- pos = 0.5  →  50% long, 50% cash
- pos = -0.5 →  50% short, 50% cash
- pos = 1.0  →  100% long, 0% cash
```

---

## 📏 Liquidity Constraints

### Minimum Cash Ratio
```python
MIN_CASH_RATIO = 0.05  # 5% minimum

Constraint:
Cash >= MIN_CASH_RATIO * Equity
OR
Position_Value <= (1 - MIN_CASH_RATIO) * Equity = 0.95 * Equity

Example:
- Equity = 100,000
- Min Cash = 5,000
- Max Investment = 95,000
- Max Position = 0.95
```

### Enforcement
```python
# If position is too large:
if cash < min_cash_required:
    # Reduce position to meet constraint
    max_position = (equity - min_cash_required) / equity
    position = clip(position, -max_position, max_position)

    # Apply penalty
    reward -= 0.05
```

**Intuition**: Imitiert **Margin Requirements** im echten Trading
- Broker verlangt Mindest-Cash als Sicherheit
- Verhindert zu aggressive Positionen
- Erzeugt Trading Discipline

---

## 🎯 PnL Calculation

### Daily PnL (skalierungsinvariant)
```
pnl_t = position_{t-1} * r_t

where:
- position_{t-1}: target position from previous day
- r_t = log(price_t / price_{t-1}): log return

Examples:

Example 1: Long position with positive return
- position = 0.5 (50% long)
- return = 0.02 (2% market move up)
- pnl = 0.5 * 0.02 = 0.01 = +1% portfolio return

Example 2: Short position with positive return
- position = -0.5 (50% short)
- return = 0.02 (2% market move up)
- pnl = -0.5 * 0.02 = -0.01 = -1% portfolio loss

Example 3: Short position with negative return
- position = -0.5 (50% short)
- return = -0.02 (2% market move down)
- pnl = -0.5 * (-0.02) = 0.01 = +1% portfolio gain
```

### Equity Update
```
Equity_t = Equity_{t-1} * exp(pnl_t - cost_t)

Why exp()?
- Compound returns correctly
- Prevents negative equity
- Matches log-return framework
```

---

## 💸 Transaction Costs

### Cost Calculation
```
cost_t = fee * |action_t - position_{t-1}|

where:
- fee = 0.0001 (10 basis points per unit change)
- action_t: new target position
- position_{t-1}: old position

Examples:

Example 1: No change
- old pos = 0.3
- new action = 0.3
- cost = 0.0001 * |0.3 - 0.3| = 0

Example 2: Small adjustment
- old pos = 0.3
- new action = 0.4
- cost = 0.0001 * |0.4 - 0.3| = 0.00001

Example 3: Full reversal
- old pos = 0.5 (50% long)
- new action = -0.5 (50% short)
- cost = 0.0001 * |-0.5 - 0.5| = 0.0001 = 10 basis points
```

**Impact on Equity**:
```
Equity *= exp(-cost)

Example:
- Equity = 100,000
- cost = 0.0001
- Equity *= exp(-0.0001) ≈ 0.99999 * Equity
- Loss ≈ 0.01% of equity (very small!)
```

---

## 🎁 Reward Structure (mit Budget Awareness)

```python
reward = pnl - transaction_cost - risk_penalty + position_alignment_bonus

where:

1. pnl: Profit from current position
2. cost: Penalty for trading frequency
3. risk_penalty: Discourage big positions in high volatility
4. alignment_bonus: Bonus for positions aligned with momentum

# Clipping für Stabilität
reward = clip(reward, -0.1, 0.1)

# Liquidity violation
if cash < min_cash_required:
    reward -= 0.05
```

---

## 📈 State Representation

### Market Features (8)
- `r`: current log return
- `r_lag1`: lagged return
- `mu_hat`: forecasted return (EWMA)
- `sigma_hat`: volatility estimate
- `rsi`: momentum indicator
- `macd_diff`: trend indicator
- `bb_width`: volatility signal
- `ema_ratio`: momentum ratio

### Portfolio Features (6) - **Mit Budget Awareness**
```python
1. current_position = tanh(pos / 0.5)
   # Current exposure as target position

2. liquidity_ratio = cash / equity
   # How much is in cash (1.0 = all cash, 0.0 = fully invested)

3. current_leverage = |pos| / LEVERAGE_MAX
   # How much leverage is being used

4. drawdown = (peak_equity - equity) / peak_equity
   # Risk metric: how far below peak

5. cumulative_pnl = tanh((equity - budget_initial) / budget_initial)
   # Total profit/loss from start

6. recent_return = tanh((equity - last_equity) / last_equity)
   # Last step's change
```

**Alle Features sind bounded und normalized!**

---

## 🔍 Diagnostics Output

Nach dem Training zeigt das Notebook:

### Budget State
```
Initial Budget: $100,000.00
Initial Equity: $100,000.00
Initial Cash: $100,000.00

Final Equity: $XX,XXX.XX
Absolute Gain/Loss: $X,XXX.XX
Total Return: X.XX%
```

### Liquidity Tracking
```
Days with violations: X
Average cash ratio: X.XX
Min/Max cash: $X,XXX / $X,XXX
Position utilization: X.XX%
```

### Trading Activity
```
Positive PnL days: XXX / XXXX (XX.X%)
Negative PnL days: XXX / XXXX (XX.X%)
Total transaction costs: $X,XXX (X.XX% of initial)
Mean daily PnL: +X.XXX%
```

---

## 📊 Info Dictionary (per Step)

```python
info = {
    # Core PnL
    "pnl": float,
    "cost": float,
    "risk_penalty": float,

    # Budget & Liquidity
    "equity": float,
    "budget": float,
    "cash": float,
    "position": float,
    "position_value": float,
    "cash_ratio": float,  # cash / equity
    "liquidity_violation": bool,
    "min_cash_required": float,

    # Performance
    "peak_equity": float,
    "drawdown": float,
    "cumulative_return": float
}
```

---

## 🎯 Example Trading Episode

**Day 1 (Start)**
```
Budget: $100,000
Equity: $100,000
Cash: $100,000
Position: 0.0 (no position)
```

**Day 2: Agent takes 50% long position**
```
Action: a = 0.5
Position Value: 0.5 * $100,000 = $50,000
Cash: $100,000 - $50,000 = $50,000
Cost: fee * |0.5 - 0.0| = 0.0001 * 0.5 = 0.00005
PnL: 0 (no position yesterday)
Reward: -0.00005 (only cost, no PnL yet)
New Equity: $100,000 * exp(-0.00005) ≈ $99,995
```

**Day 3: Market goes up 2%, position stays long**
```
Return: 0.02
PnL: 0.5 * 0.02 = 0.01 = +1%
Cost: 0.0001 * |0.5 - 0.5| = 0 (no position change)
New Equity: $99,995 * exp(0.01) ≈ $101,000
```

**Day 4: Market crashes 5%, agent goes short**
```
Return: -0.05
PnL: 0.5 * (-0.05) = -0.025 = -2.5%
Cost: 0.0001 * |-0.5 - 0.5| = 0.0001
New Equity: $101,000 * exp(-0.025 - 0.0001) ≈ $98,450
```

**Day 5: Market continues down 2%, short position profits**
```
Return: -0.02
PnL: -0.5 * (-0.02) = 0.01 = +1%
Cost: 0.0001 * |-0.5 - (-0.5)| = 0
New Equity: $98,450 * exp(0.01) ≈ $99,450
```

---

## 🚀 Key Features

✅ **Realistic Budget Constraints**
- Minimum cash requirement (liquidity buffer)
- Position sizing based on equity
- Margin-like enforcement

✅ **Scalable Positions**
- Position as fraction of equity (not absolute shares)
- Adapts to portfolio size
- Prevents unbounded exposure

✅ **Detailed Tracking**
- PnL, costs, penalties separated
- Budget state observable in every step
- Liquidity violations detectable

✅ **Training Stability**
- Bounded rewards: [-0.1, 0.1]
- Normalized state features
- Penalty for constraint violations

---

**Das Budget-System ist jetzt vollständig implementiert und ready for training! 🎯**

