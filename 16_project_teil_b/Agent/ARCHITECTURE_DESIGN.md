# Architecture & System Design Documentation

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│        FORECAST-AUGMENTED RL TRADING SYSTEM                 │
│                                                             │
│  Integrates: LSTM Forecasting + PPO Deep RL + Trading      │
└─────────────────────────────────────────────────────────────┘

                    ▼
        ┌───────────────────────┐
        │   Input: Daily OHLCV  │
        │   S&P 500 (20+ years) │
        └───────────┬───────────┘
                    ▼
    ┌───────────────────────────────────┐
    │     DATA PREPROCESSING             │
    │ • Feature engineering              │
    │ • Normalization                    │
    │ • Train/test split (80/20)        │
    └───────────────┬───────────────────┘
                    ▼
        ┌───────────────────────┐
        │  COMPONENT 1: LSTM    │
        │   FORECASTER          │
        │                       │
        │ Input: 30-day window  │
        │ Output: Prob(up move) │
        │        in next 5 days │
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────────────┐
        │  COMPONENT 2: FEATURES        │
        │   Momentum                    │
        │   Volatility                  │
        │   Technical Indicators        │
        └───────────┬───────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
    FORECAST              MARKET FEATURES
    (Binary Prob)         (4-dimensional)
        │                       │
        └───────────┬───────────┘
                    ▼
    ┌──────────────────────────────────┐
    │  STATE SPACE (5-dimensional)     │
    │                                  │
    │  • Current position: [-1, +1]    │
    │  • Cash ratio: [0, 1]            │
    │  • Momentum: ℝ                   │
    │  • Volatility: ℝ+                │
    │  • LSTM forecast: [0, 1]         │
    └──────────────┬───────────────────┘
                    ▼
    ┌──────────────────────────────────┐
    │  COMPONENT 3: PPO AGENT          │
    │  (Deep RL Decision Maker)        │
    │                                  │
    │  Policy Network:  64→64→1 (tanh) │
    │  Value Network:   64→64→1 (none) │
    │                                  │
    │  Learns to maximize reward:      │
    │  R = α*PnL - β*DD - γ*Costs      │
    └──────────────┬───────────────────┘
                    ▼
    ┌──────────────────────────────────┐
    │  ACTION: Leverage Position       │
    │                                  │
    │  • Continuous [-1.0, +1.0]       │
    │  • -1.0 = 100% short             │
    │  •  0.0 = flat                   │
    │  • +1.0 = 100% long              │
    └──────────────┬───────────────────┘
                    ▼
    ┌──────────────────────────────────┐
    │  COMPONENT 4: TRADING ENV        │
    │  (Execute, Cost, Reward)         │
    │                                  │
    │  • Position execution             │
    │  • Fee: 0.05%                     │
    │  • Kappa penalty: 0.1x            │
    │  • Reward signal generation       │
    └──────────────┬───────────────────┘
                    ▼
    ┌──────────────────────────────────┐
    │  OUTPUT: Trading Results         │
    │                                  │
    │  • Equity curves                  │
    │  • Performance metrics            │
    │  • Risk statistics                │
    └──────────────────────────────────┘
```

## Component Specifications

### Component 1: LSTM Price Forecaster

#### Purpose
Predict binary price movement (up/down) over a 5-day horizon.

#### Architecture
```
Input Layer (30 features)
    │
    ├→ LSTM Cell (64 units)
    │  ├─ forget gate
    │  ├─ input gate
    │  └─ output gate
    │
    ├→ Dropout(0.2)
    │
    ├→ LSTM Cell (32 units)
    │
    ├→ Dropout(0.2)
    │
    ├→ Dense(16, ReLU)
    │
    ├→ Dropout(0.1)
    │
    └→ Dense(1, Sigmoid) → Output: [0, 1]
```

#### Key Features
- **Bidirectional:** Processes sequence forward and backward
- **Dropout:** Prevents overfitting during training
- **Output:** Probability of upward movement

#### Training Configuration
```
Loss: Binary Crossentropy
Optimizer: Adam(lr=0.001)
Batch size: 32
Epochs: 50
Early stopping: Patience=5 on validation loss
Validation split: 20% of training data
```

#### Performance Metrics
```
Expected Accuracy: 50-55% (slightly above random)
→ Small edge exploited by RL agent
Precision/Recall: Balanced

Note: Perfect forecasting is impossible due to:
- Market efficiency (prices reflect known info)
- Latent factors (geopolitics, sentiment, etc.)
- Noise (market microstructure)
```

---

### Component 2: Feature Extraction

#### Market Features Used
```
1. Current Position
   - What we currently own
   - Range: [-1, +1]
   - Informs: risk exposure, forced liquidation risk

2. Cash Ratio
   - Available cash / Total equity
   - Range: [0, 1]
   - Informs: liquidity, ability to increase position

3. Price Momentum
   - (Price[t] - Price[t-5]) / Price[t-5]
   - Range: ℝ (typically ±0.05)
   - Informs: short-term trend following opportunity

4. Volatility
   - Std dev of 20-day returns
   - Range: (0, ∞) (typically 0.01-0.05)
   - Informs: market risk level, bet sizing
```

#### Why These Features?
- **Market-standard:** Used in professional trading
- **Non-redundant:** Each provides unique information
- **Normalized appropriately:** Ready for neural networks
- **Interpretable:** Easy to understand agent behavior

---

### Component 3: PPO Agent (Proximal Policy Optimization)

#### Algorithm Overview
```
Classical RL Problem:
  max E[reward]

PPO Solution:
  1. Collect trajectory rollout (T=2048 steps)
  2. Compute advantages: A(t) = R(t) + γV(t+1) - V(t)
  3. Optimize policy using clipped surrogate:
     L^CLIP = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
     where r_t = π_new(a|s) / π_old(a|s)
  4. Update value function
  5. Repeat
```

#### Network Architecture
```
POLICY NETWORK
───────────────
Input Layer:  5 nodes (state)
              │
Hidden Layer: 64 nodes, ReLU activation
              │
              Batch Norm / Dropout?
              │
Hidden Layer: 64 nodes, ReLU activation
              │
Output Layer: 1 node, Tanh activation → [-1, +1]

Total parameters: ~5,500

VALUE NETWORK
──────────────
Same architecture, but outputs single scalar value estimate
Total parameters: ~5,500
```

#### Hyperparameters

```
Learning:
  learning_rate: 3e-4 (Adam optimizer)
  entropy coefficient: 0.01 (encourage exploration)

Sampling:
  n_steps: 2048 (rollout length before update)
  batch_size: 64 (mini-batch for gradient descent)
  n_epochs: 10 (number of passes through data)

Discounting:
  gamma: 0.99 (discount factor; future reward importance)
  gae_lambda: 0.95 (GAE parameter; bias-variance tradeoff)

Training:
  total_timesteps: 100,000 (total environment interactions)
  max_grad_norm: 0.5 (gradient clipping for stability)
  clip_range: 0.2 (PPO clipping parameter ε)
```

#### Why PPO?
✓ Sample efficient
✓ Stable training (clipping prevents large updates)
✓ Good empirical performance
✓ Supported by Stable-Baselines3

---

### Component 4: Trading Environment

#### State Transition Dynamics
```
At each time step t:

INPUT:
  • Current state S_t
  • Action A_t (target position)

COMPUTATION:
  1. Get current price P_t
  2. Position size change: ΔPos = A_t - Pos_{t-1}
  3. Transaction cost: C_t = |ΔPos| * P_t * (fee + kappa)
  4. Update cash: Cash_t -= C_t
  5. Move to next day
  6. Get new price P_{t+1}
  7. Mark-to-market: PnL_t = Pos_t * (P_{t+1} - P_t)
  8. Update equity: Equity_t = Cash_t + Pos_t * P_{t+1}
  9. Calculate reward R_t
  10. Return (S_{t+1}, R_t, done)

OUTPUT:
  • New state S_{t+1}
  • Reward R_t
  • Done flag (terminal or not)
```

#### Reward Function Configuration

The framework supports 4 configurable reward definitions:

```
1. SIMPLE PnL
   R_t = (Equity_t - Equity_0) / Equity_0

   Use case: Maximum profit seeking
   Drawback: Ignores risk

2. PnL WITH RISK PENALTY
   R_t = PnL - 0.1 * Drawdown

   Use case: Encourage stable growth
   Drawback: May be too conservative

3. PnL WITH COST PENALTY
   R_t = PnL - 0.5 * TransactionCost

   Use case: Discourage overtrading
   Drawback: May underestimate value of rebalancing

4. BALANCED (RECOMMENDED)
   R_t = PnL - 0.1*Drawdown - 0.1*Cost

   Use case: Practical balanced approach
   Benefit: Considers multiple objectives
```

#### Cost Model
```
Fee: 0.05% per transaction
  → Realistic for institutional trading
  → Example: $100K position change costs $50

Kappa Penalty: 0.1x position change
  → Linearized market impact model
  → Discourages rapid/large rebalancing
  → Example: $100K short→long swing has $20K penalty
```

---

## Data Flow Diagram

```
Historical Data
     │
     ▼
┌──────────────────────────┐
│  Feature Computation     │
│  (Momentum, Vol, etc.)   │
└────────┬─────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
  LSTM    Market Features
    │         │
    └────┬────┘
         ▼
    State Vector
    (5 features)
         │
         ▼
    ┌──────────────┐
    │  PPO Agent   │◄──── Trained Model
    │  Policy Net  │
    └────┬─────────┘
         │
    Action (Leverage)
         │
         ▼
    ┌──────────────────┐
    │  Trading Engine  │
    │  • Execute       │
    │  • Apply costs   │
    │  • Calc reward   │
    └────┬─────────────┘
         │
    ┌────┴─────────┐
    ▼              ▼
  Equity        Reward
   Curve        Signal
    │              │
    └──────┬───────┘
           ▼
    Performance Metrics
```

---

## Experiment Flow

```
START
  │
  ├─→ [1] Prepare Data
  │   • Load historical data
  │   • Create features
  │   • Split train/test
  │   │
  │   └─→ LSTM Training
  │       • Train forecaster
  │       • Generate probabilities
  │
  ├─→ [2] Forecast-Only Baseline
  │   • Simple rule: pos = 1 if prob>0.5 else -1
  │   • Simulate trading
  │   • Calculate metrics
  │   • Log to W&B
  │
  ├─→ [3] PPO Without Forecast
  │   • Create env (no forecast signal)
  │   • Train PPO agent
  │   • Evaluate on test set
  │   • Calculate metrics
  │   • Log to W&B
  │
  ├─→ [4] PPO With Forecast
  │   • Create env (with forecast signal)
  │   • Train PPO agent
  │   • Evaluate on test set
  │   • Calculate metrics
  │   • Log to W&B
  │
  ├─→ [5] Reward Variations
  │   • For each reward config:
  │     - Train PPO
  │     - Evaluate
  │     - Log results
  │
  ├─→ [6] Analysis & Comparison
  │   • Create comparison table
  │   • Generate plots
  │   • Calculate statistics
  │   • Write report
  │
  └─→ [7] Export & Report Generation
      • Save results CSV
      • Generate markdown report
      • Create visualizations
      • Push to W&B
END
```

---

## W&B Experiment Structure

```
Project: forecast-rl-trading
├── Group: baseline
│   └── Run: forecast-only
│       ├── Config: {strategy: "forecast-only", ...}
│       └── Metrics: {return, sharpe, maxdd, ...}
│
├── Group: ppo-variants
│   ├── Run: ppo-without-forecast
│   │   ├── Config: {include_forecast: false, reward: "simple", ...}
│   │   └── Metrics: {...}
│   │
│   ├── Run: ppo-with-forecast
│   │   ├── Config: {include_forecast: true, reward: "simple", ...}
│   │   └── Metrics: {...}
│   │
│   ├── Run: ppo-with-forecast-reward-pnl_with_risk
│   │   └── Metrics: {...}
│   │
│   ├── Run: ppo-with-forecast-reward-pnl_with_costs
│   │   └── Metrics: {...}
│   │
│   └── Run: ppo-with-forecast-reward-balanced
│       └── Metrics: {...}
│
└── Comparison Table
    ├── Strategy names
    ├── Final equities
    ├── Returns
    ├── Sharpe ratios
    └── Risk metrics
```

---

## Model Persistence

```
Checkpoints saved:
  /path/to/models/
  ├── lstm_forecaster.h5
  ├── ppo_no_forecast.zip
  ├── ppo_with_forecast.zip
  ├── ppo_reward_pnl_with_risk.zip
  └── ...

Can be loaded and:
  • Used for live trading
  • Fine-tuned on new data
  • Analyzed for interpretability
```

---

## Performance Expectations

### LSTM Forecaster
- **Accuracy:** 51-54% (slightly above random 50%)
- **Justification:** Markets are hard to predict; 51-54% is realistic edge
- **Time to train:** 2-5 minutes

### PPO Without Forecast
- **Return:** +5% to +15% (depends on market conditions)
- **Sharpe:** 0.3-0.8
- **Time to train:** 10-20 minutes

### PPO With Forecast
- **Scenario A (Forecast helps):** +20-40% (better than baseline)
- **Scenario B (No difference):** ±5% (forecast is noise)
- **Scenario C (Forecast hurts):** -10% to -20% (overconfidence)

---

## Files & Usage

### Core Files
```
experiment_framework.py          - Main framework class
Project_Part_2_Final_Architecture.ipynb - Notebook with full pipeline
```

### Usage Example
```python
from experiment_framework import ExperimentRunner, ExperimentConfig

config = ExperimentConfig()
runner = ExperimentRunner(config=config)
results = runner.run_all_experiments(df_test, forecast_signal)

# Access results
print(results['ppo-with-forecast']['total_return'])
```


