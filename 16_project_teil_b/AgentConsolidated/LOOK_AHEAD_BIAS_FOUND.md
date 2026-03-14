
# 🚨 CRITICAL LOOK-AHEAD BIAS FOUND IN LSTM

## THE PROBLEM

**Location:** Lines 305-307 in train_forecast_model()

```python
# Simple direction label - NO [:-1] slice, shift(-1) handles alignment
y_train = (df_train['r'].shift(-1) > 0).astype(int).fillna(0).values
```

## WHAT'S HAPPENING

### Input Features (X):
```
Time t:  X = [r_t, sigma_hat_t, rsi_t, macd_diff_t, signal_strength_t]
         where r_t is the CURRENT return (already observed)
```

### Target Label (y):
```
y_t = (r_t+1 > 0) ? 1 : 0
      └─ NEXT DAY's return (shift(-1) = forward looking)
```

### Over 20-day Lookback Window:
```
Input sequence:
  Day t:     [r_t, sigma_t, rsi_t, macd_t, signal_t]
  Day t+1:   [r_t+1, sigma_t+1, rsi_t+1, macd_t+1, signal_t+1]
  ...
  Day t+19:  [r_t+19, sigma_t+19, rsi_t+19, macd_t+19, signal_t+19]

Target:
  y = (r_t+20 > 0) ? 1 : 0
      └─ Predicting 20+ days ahead!
```

## THE LOOK-AHEAD BIAS

**Problem:** The LSTM features contain current/recent returns (r_t, r_t+1, ... r_t+19)
but predicts even further future returns (r_t+20).

Since returns have strong autocorrelation in a 20-day window:
- If returns have been positive, next return is likely positive
- If returns have been negative, next return is likely negative

**Result:** The LSTM is just learning patterns within the recent return window,
not actually forecasting new information!

---

## PROPER IMPLEMENTATION

To fix this and create a true forecast (not just copying past values):

### Option 1: Exclude Current Return from Features

```python
# WRONG (current version):
feature_cols = ['r', 'sigma_hat', 'rsi', 'macd_diff', 'signal_strength']
                 ↑ Contains today's return!
y_train = df_train['r'].shift(-1) > 0  # Predicts tomorrow

# RIGHT (should be):
feature_cols = ['sigma_hat', 'rsi', 'macd_diff', 'signal_strength']
                 ↑ No current return - only indicators
y_train = df_train['r'].shift(-1) > 0  # Still predicts tomorrow
```

### Option 2: Use Lagged Returns

```python
# Include only PAST returns, not current
feature_cols = ['r_lag1', 'r_lag2', 'r_lag3', 'sigma_hat', 'rsi', 'macd_diff', 'signal_strength']
                 ↑ Lags only, no current return
```

---

## IMPACT ON RESULTS

This explains why:

1. **LSTM Forecast is Useless**
   - SMAPE/MAPE will be artificially low
   - But it's just copying recent patterns
   - No real predictive power

2. **PPO Performance Drops With Forecast**
   - Wrong forecast signal confuses the agent
   - Agent learns to ignore the signal
   - But observation space is polluted
   - Result: Worse performance

3. **Without Forecast, PPO Does Better**
   - Cleaner observation space
   - Agent focuses on meaningful indicators
   - Better convergence
   - Result: +43.61% return

---

## VERIFICATION

Check the feature columns used:

**Current (WRONG):**
```python
feature_cols = ['r', 'sigma_hat', 'rsi', 'macd_diff', 'signal_strength']
                ↑ This is the problem!
```

**Notebook likely had same issue** - you copied it correctly

---

## FIX NEEDED

Remove 'r' (current return) from LSTM feature columns:

```python
# OLD (Lines 299-300):
feature_cols = ['r', 'volatility', 'rsi', 'macd_diff', 'signal_strength']

# NEW:
feature_cols = ['sigma_hat', 'rsi', 'macd_diff', 'signal_strength']
# Note: removed 'r' (current return) to avoid look-ahead bias
```

---

## WHAT HAPPENS WITH THIS FIX

1. **LSTM Train**: Learn from indicators only (clean features)
2. **LSTM Forecast**: True forecast of future direction (not just auto-correlation)
3. **PPO With Forecast**: Gets meaningful signal, performance similar to baseline
4. **PPO Without Forecast**: Still works well

Expected Results:
- Both should have similar performance (~40-50% return)
- Forecast might help or hurt slightly (depends on forecast quality)
- But no look-ahead bias, proper experiment

---

## STATUS

🚨 **CRITICAL ISSUE** - Current return included in features
📍 **LOCATION** - Lines 299-300, feature_cols definition
✅ **FIX READY** - Remove 'r' from feature columns

Should I apply this fix?


