### Plan: Feature Engineering Extensions for a stable index (SMI)

#### Goal
Enhance predictive signal for a relatively stable, low‑volatility equity index by prioritizing smooth, low‑variance, leakage‑safe covariates that capture trend, slow seasonality, regime shifts, and long‑horizon momentum/volatility — while avoiding overly reactive, noisy indicators.

---

### 1) Audit current features ✓
- Already present: `log_return`, `rolling_vol` (short), `sma_fast`, `sma_slow`, `bb_mid/upper/lower`.
- Baseline: keep these as core; tune windows toward longer horizons suitable for indices.

### 2) Add low‑variance calendar/seasonality encoders (recommended first)
- Dummies or cyclical encodings:
  - Day of week: `dow_sin`, `dow_cos`
  - Month of year: `moy_sin`, `moy_cos`
  - Turn‑of‑month/end‑of‑quarter flags: `is_tom`, `is_eoq`, `is_eoy`
  - Trading day in month (1..N) as cyclical: `tdim_sin`, `tdim_cos`
  - Holiday proximity (Swiss market): `pre_holiday`, `post_holiday` (requires a Swiss holiday calendar)
- Rationale: captures modest seasonality effects common in broad indices without adding noise.

### 3) Trend and mean‑reversion features (long horizons)
- Longer SMAs/EMAs: `sma_100`, `sma_200`, `ema_63`, `ema_126`
- Distance to average: `dist_sma_200 = close/sma_200 - 1` and its z‑score
- Rolling linear trend slope: `slope_63` via rolling OLS on price or log‑price
- Price channel/Donchian: `donchian_mid_55`, `donchian_width_55`
- Bollinger derivatives: `%b` (position inside band), `bb_bw` (bandwidth)
- Rationale: indices exhibit slow trends; distance and slope convey over/under‑extension and mean‑reversion potential.

### 4) Momentum (use longer windows; avoid noisy short ones)
- Cumulative returns: `mom_21`, `mom_63`, `mom_126` (on log returns)
- RSI with longer periods: `rsi_21`, optionally `rsi_63`
- MACD with de‑noised params: `macd = ema_12-ema_26`, `macd_signal = ema_9(macd)` (or slower 24/52/18 for indices)
- Streak length: consecutive up/down days count (capped) over last 10–20 days
- Rationale: smoother momentum tends to be more robust on indices.

### 5) Volatility and risk features (emphasize long windows)
- Realized volatility long/short: `rv_21`, `rv_63`, `rv_ratio = rv_21/rv_63`
- Vol of vol: stdev of `rolling_vol` over 63 days
- Drawdown metrics: `drawdown`, `max_dd_126`, `time_since_high`
- Ulcer Index (captures depth and duration below peak): `ulcer_63`
- If OHLC available: ATR `atr_14`, Parkinson volatility `parkinson_20`
- Rationale: regimes (quiet vs. somewhat volatile) affect forecastability and error structure.

### 6) Autocorrelation/statistics features
- Lagged target/returns: `lag_ret_1`, `lag_ret_5`, `lag_ret_10`
- Rolling autocorrelation of returns at lag 1 and 5 over 63‑day window: `acf1_63`, `acf5_63`
- Rolling z‑scores: `z_close_63`, `z_ret_63`
- Optional: Hurst exponent (coarse) over 126 days, if compute budget allows
- Rationale: captures persistence vs. mean‑reversion.

### 7) Volume/flow features (if `volume` present)
- Volume z‑score: `z_vol_63`
- OBV and its EMA: `obv`, `ema_obv_21`
- Price–volume trend (PVT)
- Rationale: broad indices volume can reflect participation; use long windows to reduce noise.

### 8) External covariates (optional, high value if available)
- Related markets: DAX, EUROSTOXX50, S&P500 as past covariates (aligned and scaled)
- FX: `USDCHF`, `EURCHF` (CHF strength vs. global risk appetite)
- Rates: Swiss 10y yield; credit spread proxy (e.g., EUR IG spread)
- Rationale: indices co‑move with global factors; these often improve stability.

### 9) Implementation plan in this repo
- Config additions in `FeatureConfig` (as booleans and window ints):
  - `use_calendar`, `use_long_sma`, `use_ema`, `use_distance_to_sma`, `use_trend_slope`, `use_momentum`, `use_rsi`, `use_macd`, `use_vol_long`, `use_vol_ratio`, `use_drawdown`, `use_autocorr`, `use_volume_feats`
  - Windows: `sma_long_window=200`, `ema_windows=(21,63)`, `mom_windows=(21,63,126)`, `rsi_window=21`, `trend_window=63`, `vol_long_window=63`, etc.
- `feature_engineering.py` extensions:
  - Compute calendar features from index safely; avoid leakage
  - Add long‑window stats before calling `dropna()`
  - Guard OHLC/volume‑only features with column checks
- Keep features as past_covariates only; ensure all use strictly past information.

### 10) Validation and selection
- Run ablations: start with calendar + long SMA distance + long RV; add groups incrementally
- Monitor RMSE and sMAPE; prefer simpler sets that improve both train and test
- Use TFT variable importances to prune redundant indicators (e.g., SMA vs. EMA distance)
- Check multicollinearity: avoid many overlapping windows of similar indicators

### 11) Parameter guidance for a stable index
- Favor longer windows: 63/126/200 over 5/10/14
- Normalize via z‑scores and ratios to keep scales consistent with Scaler
- Prefer binary/categorical regime flags when possible (e.g., high‑vol regime)

### 12) Quality and leakage checks
- All features must be computed with `.shift()` or rolling windows that don’t peek into the future
- Re‑run your existing `check_timeseries()` to ensure no NaN/Inf after transforms
- Keep feature counts moderate to avoid overfitting (start with 6–12 strongest)

---

### Concrete feature list (suggested first batch)
- Calendar: `dow_sin`, `dow_cos`, `moy_sin`, `moy_cos`, `is_tom`
- Trend: `sma_200`, `dist_sma_200`, `slope_63`, `z_close_63`
- Momentum: `mom_63`
- Volatility: `rv_63`, `rv_ratio = rv_21/rv_63`
- Risk: `drawdown`, `time_since_high`
- If volume exists: `z_vol_63`

This set is compact, low‑variance, and well‑suited to SMI’s stability. You can expand with RSI(21), %b, and MACD later if helpful.

If you want, I can draft the exact column formulas and a minimal code patch to add the first batch behind new `FeatureConfig` flags.