# Technical Report: Forecast-Augmented RL Trading System
## Does LSTM Price Movement Forecasts Improve PPO Trading Performance?

**Author:** CAS AI Research Group
**Date:** {DATE}
**Duration:** Experimental Study (10-15 pages)

---

## Executive Summary

This technical report documents a comprehensive experimental study investigating whether integrating LSTM price movement forecasts into a PPO (Proximal Policy Optimization) deep reinforcement learning agent improves trading performance.

### Key Research Question
**Does incorporating LSTM-generated forecast signals into the RL agent's state representation improve trading returns, risk-adjusted returns, and overall strategy performance?**

### Hypotheses
1. **H1 (Positive):** Forecast signals provide valuable information that PPO can leverage for improved returns
2. **H2 (Neutral):** Forecast signals are redundant; RL learns equivalent patterns from market features
3. **H3 (Negative):** Forecast signals introduce noise/overconfidence, degrading RL performance

### Key Finding
[PLACEHOLDER: Based on experimental results]
- Forecast Impact: {FORECAST_IMPROVEMENT}%
- Best Strategy: {BEST_STRATEGY}
- Sharpe Improvement: {SHARPE_DIFF}

---

## 1. Introduction

### 1.1 Problem Statement
Predicting asset prices and making profitable trades is one of the most challenging problems in finance. Recent advances in deep learning have shown promise in:
- **Price prediction:** LSTM and transformer models can learn temporal patterns
- **Trading decisions:** Deep reinforcement learning (PPO, A3C, etc.) can optimize trading policies

However, the question remains: **Do these technologies complement each other?** Should we give the RL agent access to price forecasts?

### 1.2 Research Objectives
1. Develop a forecast-augmented RL trading system
2. Compare three strategies systematically:
   - Forecast-Only (rule-based baseline)
   - PPO without forecast (RL baseline)
   - PPO with forecast (proposed method)
3. Evaluate impact of different reward definitions
4. Provide critical analysis of results

### 1.3 Significance
Understanding whether forecasts improve RL has practical implications for:
- **Algorithmic trading:** Should systems use ensemble approaches?
- **Architecture design:** What information should RL agents receive?
- **Model complexity:** Is added complexity justified by improved returns?

---

## 2. Literature Review

### 2.1 LSTM for Price Forecasting
- **Foundational work:** Hochreiter & Schmidhuber (1997) - LSTM architecture
- **Application to finance:** [Key papers on financial LSTM models]
- **Accuracy limitations:** Even accurate forecasts may not be profitable when trading costs are considered

### 2.2 Deep Reinforcement Learning for Trading
- **PPO algorithm:** Schulman et al. (2017) - Proximal Policy Optimization
- **Trading applications:** [Key papers on RL trading agents]
- **Challenges:** Data non-stationarity, reward function design, transaction costs

### 2.3 Ensemble and Hybrid Approaches
- Benefits and challenges of combining prediction + RL
- When ensemble methods work vs. when single approaches are better

---

## 3. Methodology

### 3.1 Data & Experimental Setup

#### Dataset
- **Asset:** S&P 500 Index (^GSPC)
- **Period:** 2020-2023 (Training/Validation)
- **Test Period:** Last 20% of data
- **Frequency:** Daily OHLCV data
- **Features:** Close price (primary), volatility indicators

#### Train/Test Split
- **Training:** First 80% of data for model development
- **Testing:** Last 20% of data for evaluation
- **Validation:** Cross-validation during LSTM training

#### Trading Assumptions
- **Initial Equity:** $100,000
- **Position Limits:** ±1.0x leverage (no short-selling restrictions)
- **Trading Costs:**
  - Fee (commission): 0.05% per transaction
  - Kappa penalty: 0.1x per unit position change
- **Slippage:** Not modeled (simplified)

### 3.2 System Architecture

#### Component 1: LSTM Price Forecaster
```
Architecture:
  Input → (LSTM layer 1: 64 units) → Dropout(0.2)
       → (LSTM layer 2: 32 units) → Dropout(0.2)
       → (Dense: 16 units, ReLU) → Dropout(0.1)
       → (Dense: 1, Sigmoid) → Binary Classification (Up/Down)

Training:
  - Lookback window: 30 days
  - Forecast horizon: 5 days ahead
  - Target: Binary classification (1 = price will go up, 0 = down)
  - Loss: Binary crossentropy
  - Optimizer: Adam
  - Epochs: 50
  - Batch size: 32

Output:
  Probability of upward price movement
```

#### Component 2: Trading Environment
```
State Space:
  - Current position (±1.0)
  - Cash ratio (cash / total equity)
  - Price momentum (5-day)
  - Volatility (20-day rolling std)
  - [Optional] Forecast probability from LSTM

Action Space:
  - Continuous leverage: [-1.0, +1.0]

Reward Function (configurable):
  reward = alpha * PnL + beta * (-drawdown) + gamma * (-transaction_cost)

  Configurations:
    1. Simple PnL: α=1.0, β=0, γ=0
    2. PnL + Risk: α=1.0, β=0.1, γ=0
    3. PnL + Costs: α=1.0, β=0, γ=0.5
    4. Balanced: α=1.0, β=0.1, γ=0.1
```

#### Component 3: PPO Agent
```
Algorithm: Proximal Policy Optimization (PPO)

Hyperparameters:
  - Learning rate: 3e-4
  - N steps: 2048 (rollout length)
  - Batch size: 64
  - N epochs: 10 (per update)
  - Total timesteps: 100,000
  - Entropy coefficient: 0.01
  - Gamma (discount): 0.99
  - GAE lambda: 0.95

Network Architecture:
  - Policy network: [64, 64] hidden layers
  - Value network: [64, 64] hidden layers
  - Activation: ReLU
  - Output: Tanh (action scaling)
```

### 3.3 Experimental Design

#### Experiment 1: Forecast-Only Baseline
- **Strategy:** Simple rule-based
- **Logic:**
  ```
  position = +1.0 if forecast_prob > 0.5 else -1.0
  ```
- **Purpose:** Establish baseline; test forecast quality independently
- **Expected:** Shows forecast skill (or lack thereof)

#### Experiment 2: PPO Without Forecast (RL Baseline)
- **Architecture:** PPO trained on market features only
- **State:** [position, cash_ratio, momentum, volatility]
- **Purpose:** Show what RL can achieve without forecast
- **Expected:** Moderate performance; RL learns from market structure

#### Experiment 3: PPO With Forecast
- **Architecture:** PPO with forecast in state
- **State:** [position, cash_ratio, momentum, volatility, forecast_prob]
- **Purpose:** Show impact of adding forecast signal
- **Expected:** Outcome depends on forecast quality + RL learning

#### Experiment 4: Reward Function Variations
- **Test 4 different reward configurations:**
  1. Simple PnL (baseline reward)
  2. PnL with risk penalty (penalize drawdowns)
  3. PnL with transaction cost penalty (discourage overtrading)
  4. Balanced (all three components)
- **Purpose:** Isolate impact of reward function design
- **Expected:** Different rewards → different trading behaviors

### 3.4 Evaluation Metrics

#### Primary Metrics
| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Total Return** | (Final Equity - Initial) / Initial | Absolute profit percentage |
| **Sharpe Ratio** | Mean Return / Std Dev × √252 | Risk-adjusted return (higher better) |
| **Max Drawdown** | Min(Peak-to-Trough) / Peak | Maximum capital loss (lower better) |

#### Secondary Metrics
| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Volatility** | Std Dev of Daily Returns × √252 | Annualized volatility |
| **Turnover** | Σ\|position change\| / n_days | Trading activity level |
| **Win Rate** | n_profitable_days / n_total_days | % of profitable days |
| **Profit Factor** | Gross Profit / Gross Loss | Reward/risk ratio |

---

## 4. Results

### 4.1 Experimental Results

#### Strategy Comparison Table
| Strategy | Final Equity | Return | Sharpe | Volatility | Max DD | Turnover |
|----------|-------------|--------|--------|-----------|--------|----------|
| Forecast-Only | ${FO_EQUITY} | {FO_RETURN}% | {FO_SHARPE} | {FO_VOL}% | {FO_DD}% | {FO_TURN} |
| PPO (No Forecast) | ${PPO_NF_EQUITY} | {PPO_NF_RETURN}% | {PPO_NF_SHARPE} | {PPO_NF_VOL}% | {PPO_NF_DD}% | {PPO_NF_TURN} |
| PPO (With Forecast) | ${PPO_WF_EQUITY} | {PPO_WF_RETURN}% | {PPO_WF_SHARPE} | {PPO_WF_VOL}% | {PPO_WF_DD}% | {PPO_WF_TURN} |

#### Key Findings

**Q1: Does forecast improve RL performance?**
Answer: **{FORECAST_HELPS}**
- Improvement: {FORECAST_IMPROVEMENT}%
- Sharpe improvement: {SHARPE_IMPROVEMENT}
- Statistical significance: {SIGNIFICANCE}

**Q2: Which reward function works best?**
Answer: **{BEST_REWARD}**
- Balanced reward performs {REWARD_COMPARISON}
- Risk penalty helps with drawdown control: {RISK_CONTROL}
- Cost penalty reduces overtrading: {COST_REDUCTION}

**Q3: How does RL compare to simple forecast rule?**
Answer: **{RL_VS_FORECAST}**
- PPO advantage: {PPO_ADVANTAGE}%
- Risk-adjusted comparison: {SHARPE_COMPARISON}

### 4.2 Equity Curve Analysis

[PLACEHOLDER FOR CHARTS]
- Equity curves overlay (3 strategies)
- Cumulative returns comparison
- Drawdown periods analysis

### 4.3 Statistical Analysis

#### Return Distribution
- Forecast-Only: Mean = {}, Std = {}, Skew = {}
- PPO No Forecast: Mean = {}, Std = {}, Skew = {}
- PPO With Forecast: Mean = {}, Std = {}, Skew = {}

#### Statistical Tests
- **T-test** (PPO with vs without forecast): p-value = {}
- **Mann-Whitney U test** (non-parametric alternative): p-value = {}
- **Conclusion:** Results are {SIGNIFICANT/NOT SIGNIFICANT} at 5% level

---

## 5. Analysis & Interpretation

### 5.1 Why Did Forecast Help/Hurt?

[DETAILED ANALYSIS BASED ON RESULTS]

### 5.2 Reward Function Impact

[ANALYSIS OF REWARD CONFIGURATIONS]

### 5.3 Failure Modes & Challenges

1. **Data snooping:** Model may overfit to historical patterns
2. **Forecast quality:** LSTM forecast accuracy vs. trading profitability gap
3. **Non-stationary market:** Strategies optimal in past may fail in future
4. **Transaction costs:** Even small fees significantly reduce profits
5. **Overfitting:** PPO may overfit to training period

### 5.4 Lessons Learned

1. Forecast signals help/don't help because...
2. The importance of proper reward function design
3. Trade-off between complexity and performance
4. Challenges in applying RL to real trading

---

## 6. Conclusions & Recommendations

### 6.1 Main Findings

**Primary Question: Does forecast improve RL performance?**
✓ **Answer:** {ANSWER}

**Supporting Evidence:**
- Return comparison: {RETURN_EVIDENCE}
- Risk-adjusted comparison: {SHARPE_EVIDENCE}
- Consistency across reward functions: {CONSISTENCY}

### 6.2 Practical Recommendations

**For practitioners implementing similar systems:**
1. {RECOMMENDATION_1}
2. {RECOMMENDATION_2}
3. {RECOMMENDATION_3}

**For future research:**
1. Test on additional assets and time periods
2. Incorporate more sophisticated forecast signals
3. Explore ensemble methods combining all approaches
4. Investigate forecast-based reward shaping
5. Implement live trading with risk controls

### 6.3 Limitations

- **Limited scope:** Single asset, single time period
- **Transaction costs:** Simplified model; real costs vary
- **Forecast quality:** Binary classification; could be improved
- **Market conditions:** Bull market; untested in bear market
- **Statistical power:** Limited test period for significance testing

### 6.4 Final Thoughts

[CRITICAL REFLECTION ON THE WORK]

---

## 7. References

[ACADEMIC CITATIONS]

---

## Appendix A: Hyperparameter Sensitivity Analysis

[ANALYSIS OF KEY HYPERPARAMETER CHOICES]

## Appendix B: Code Repository

**Location:** `/home/isc-den/cas-artificial-intelligence/16_project_teil_b/Agent/`

**Key Files:**
- `experiment_framework.py` - Core experimental framework
- `Project_Part_2_Final_Architecture.ipynb` - Main notebook with all experiments
- `results_comparison_table.csv` - Results for easy inclusion

**Running Experiments:**
```bash
cd /home/isc-den/cas-artificial-intelligence/16_project_teil_b/Agent/
python3 -m jupyter notebook Project_Part_2_Final_Architecture.ipynb
# Or use the experiment_framework directly
python3 -c "from experiment_framework import ExperimentRunner; runner = ExperimentRunner(); runner.run_all_experiments(...)"
```

## Appendix C: Architecture Diagram

[ASCII/Visual diagram of system components]

```
┌─────────────────────────────────────────────────────────┐
│              FORECAST-AUGMENTED RL TRADING              │
└─────────────────────────────────────────────────────────┘

                    Market Data (OHLCV)
                          ↓
                ┌─────────┴─────────┐
                ↓                   ↓
        ┌──────────────┐    ┌──────────────┐
        │    LSTM      │    │   Features   │
        │  Forecaster  │    │  Extractor   │
        │  (Bi-LSTM)   │    │  (Momentum,  │
        │              │    │   Volatility)│
        └──────┬───────┘    └──────┬───────┘
               │                   │
               │ Forecast Prob     │ Market Features
               │                   │
               └─────────┬─────────┘
                         ↓
                  ┌──────────────┐
                  │  State Space │
                  │ [pos, cash,  │
                  │  mom, vol,   │
                  │  forecast]   │
                  └──────┬───────┘
                         ↓
                  ┌──────────────┐
                  │  PPO Agent   │
                  │  (Policy NN) │
                  │  (Value NN)  │
                  └──────┬───────┘
                         ↓
                  ┌──────────────┐
                  │   Action     │
                  │  (Leverage)  │
                  │  [-1.0, +1.0]│
                  └──────┬───────┘
                         ↓
                  ┌──────────────┐
                  │  Trading Env │
                  │  (Execute,   │
                  │   Costs,     │
                  │   Reward)    │
                  └──────┬───────┘
                         ↓
                      Equity
                    & Reward
```

---

**Document Version:** 1.0
**Last Updated:** {DATE}
**Status:** Final


