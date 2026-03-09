# Forecast-Augmented Reinforcement Learning for Cryptocurrency Trading

A complete implementation of a trading agent that integrates time-series forecasting with deep reinforcement learning (PPO) to optimize trading strategies on cryptocurrency markets.

## 📋 Project Overview

### Objective
Build and evaluate a trading agent that uses LSTM-based price forecasts to improve decision-making in a reinforcement learning framework. The project compares performance with and without forecast signals to quantify their value.

### Key Questions
1. **Does forecast help?** Can predictive models improve RL trading performance?
2. **Why or why not?** What are the mechanisms and failure modes?
3. **How do we measure success?** Financial metrics (Sharpe, Calmar, drawdown)

## 🏗️ Architecture

```
Data Loading (yfinance)
    ↓
Feature Engineering
    ├─→ LSTM Forecaster (20-day lookback → 5-day forecast)
    │
    ├─→ Trading Environment
    │   ├─ State: Market + Position + Portfolio + Forecast
    │   ├─ Action: Target position [-2, 2] (long/short with leverage)
    │   └─ Reward: PnL - Cost - Risk Penalty
    │
    ├─→ PPO Agent (With Forecast)
    │   ├─ Actor: Policy π(a|s)
    │   └─ Critic: Value V(s)
    │
    └─→ PPO Agent (Without Forecast)
        ├─ Actor: Policy π(a|s)
        └─ Critic: Value V(s)

    ↓
Evaluation & Metrics
    ├─ Cumulative return
    ├─ Sharpe ratio
    ├─ Max drawdown
    ├─ Volatility
    ├─ Win rate
    └─ Turnover

    ↓
Comparison & Analysis
```

## 📁 File Structure

```
14_project_teil_b/
├── 01_Complete_Solution.ipynb        # Main notebook (executable)
├── trading_env.py                    # Enhanced trading environment
├── forecasting.py                    # LSTM forecasting module
├── ppo_trainer.py                    # PPO training logic
├── evaluation.py                     # Metrics and evaluation
├── TECHNICAL_REPORT.md               # Detailed 15-page report
├── PRESENTATION_OUTLINE.md           # 20-minute presentation structure
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd /home/isc-den/cas-artificial-intelligence/14_project_teil_b
pip install -r requirements.txt
```

### 2. Run the Main Notebook

```bash
jupyter notebook 01_Complete_Solution.ipynb
```

The notebook is fully self-contained and will:
- Load Bitcoin data (2022-2024)
- Train LSTM forecasting model (~10 min)
- Train PPO with forecast (~30 min)
- Train PPO without forecast (~25 min)
- Evaluate both on test set (~5 min)
- Generate comparison visualizations

### 3. Expected Output

The notebook produces:
- **Models**: `ppo_with_forecast.pt`, `ppo_without_forecast.pt`, `lstm_forecaster.pt`
- **Metrics**: `performance_comparison.csv`, `results_summary.json`
- **Plots**: `training_curves.png`, `equity_curves.png`, `return_distributions.png`

## 🔧 Core Components

### Trading Environment (`trading_env.py`)

**Key Features:**
- State dimension: 19 (with forecast) or 18 (without)
- Action space: Continuous [-2, 2] (position size with leverage)
- Realistic constraints: Long/short, leverage limits, liquidity
- Clean reward: PnL - transaction cost - risk penalty

**State Representation:**
```
Market Features (4):
  - r: Daily log return
  - r_lag1: Lagged return
  - μ̂: Exponential moving average (forecast signal)
  - σ̂: Rolling volatility

Position State (4):
  - position: Current target position
  - leverage_used: Normalized exposure
  - long_exposure: Positive position component
  - short_exposure: Negative position component

Portfolio Metrics (6):
  - equity_norm: Equity / initial capital
  - drawdown: Peak-to-current decline
  - cash_ratio: Available cash / equity
  - sharpe_20d: 20-day Sharpe ratio
  - volatility_20d: 20-day rolling volatility
  - max_position_change: Leverage limit

Optional Forecast (1):
  - forecast: LSTM prediction signal (with/without)
```

### Forecasting Module (`forecasting.py`)

**LSTM Forecaster:**
- Input: 20-day sequence of [r, μ, σ]
- Architecture: 2-layer LSTM (64 hidden units) with dropout
- Output: 5-step ahead return predictions
- Training: MSE loss on validation split

**Signal Generation:**
```python
signal = tanh(forecast_mean / threshold) → [-1, 1]
```

### PPO Trainer (`ppo_trainer.py`)

**Architecture:**
- Actor-Critic network with shared feature layers
- Gaussian policy for continuous actions
- Value function for advantage estimation
- GAE (λ=0.95) for advantage computation

**Training:**
- Vectorized: 8 parallel environments
- Rollout: 128 steps per update
- PPO epochs: 10 per update
- Total updates: 500

### Evaluation Module (`evaluation.py`)

**Metrics Computed:**
1. **Returns**: Cumulative return, daily returns
2. **Risk-Adjusted**: Sharpe ratio, Sortino ratio, Calmar ratio
3. **Risk**: Max drawdown, volatility
4. **Profitability**: Win rate, profit factor
5. **Activity**: Turnover (position changes)

## 📊 Key Results Format

Expected output from evaluation:

```
=== Performance Comparison ===

Metric          | With Forecast | Without Forecast | Difference
─────────────────┼───────────────┼──────────────────┼──────────
Return          | [%]           | [%]              | [%]
Sharpe          | [ratio]       | [ratio]          | [ratio]
Max DD          | [%]           | [%]              | [%]
Volatility      | [%]           | [%]              | [%]
Win Rate        | [%]           | [%]              | [%]
Turnover        | [value]       | [value]          | [value]
Calmar          | [ratio]       | [ratio]          | [ratio]
```

## 🎯 How to Interpret Results

### Scenario 1: Forecast Helps ✅
- **Sharpe ratio**: Improved (e.g., 0.5 → 0.8)
- **Max drawdown**: Reduced (e.g., -20% → -15%)
- **Interpretation**: Forward-looking signal improves risk management
- **Conclusion**: Deploy with monitoring for regime changes

### Scenario 2: Forecast Hurts ❌
- **Sharpe ratio**: Degraded
- **Max drawdown**: Increased
- **Interpretation**: LSTM predictions too noisy or outdated
- **Conclusion**: Use baseline RL without forecast

### Scenario 3: Mixed Results ⚠️
- **Returns**: Similar or slightly better
- **Risk metrics**: Improved
- **Interpretation**: Forecast helps with risk but not absolute returns
- **Conclusion**: Valuable for risk management, less clear for profits

## 📈 Visualization Guide

The notebook generates 3 main plots:

### 1. Training Curves (`training_curves.png`)
- **Left**: Episode returns over 500 updates (100-episode moving average)
- **Right**: KL divergence (should stay below target)
- **Interpretation**: Both models should converge smoothly without divergence

### 2. Equity Curves (`equity_curves.png`)
- **Top-left**: With forecast - mean equity ± 1 std dev
- **Top-right**: Without forecast - mean equity ± 1 std dev
- **Bottom**: Position distributions (how aggressive is each agent?)
- **Interpretation**: Higher/more stable equity curves indicate better performance

### 3. Return Distributions (`return_distributions.png`)
- **Left**: Daily returns histogram (with forecast)
- **Right**: Daily returns histogram (without forecast)
- **Interpretation**: Wider vs. narrower distributions indicate different risk/return profiles

## 💡 Customization

### Change Parameters

Edit these in the notebook:

```python
# Data
TICKER = "BTC-USD"                # Change to ETH-USD, etc.
START = "2022-01-01"
END = "2024-01-01"
TRAIN_FRAC = 0.8

# Trading
FEE = 0.0005                      # Transaction fee
KAPPA = 0.1                       # Risk penalty weight
INITIAL_CASH = 100000.0           # Starting capital
MAX_LEVERAGE = 2.0                # Maximum leverage

# Training
NUM_ENVS = 8                      # Parallel environments
TOTAL_UPDATES = 500               # PPO updates
LR = 3e-4                        # Learning rate
```

### Add More Features

In the data preprocessing section:

```python
def add_features(df):
    # Add your indicators
    df['rsi'] = compute_rsi(df['close'], 14)
    df['macd'] = compute_macd(df['close'])
    df['bb_width'] = compute_bollinger_bands(df['close'])
    return df
```

### Change Forecast Horizon

In forecasting.py:

```python
forecaster = TimeSeriesForecaster(
    forecast_horizon=10  # Change from 5 to 10 days
)
```

## ⚠️ Important Considerations

### Limitations

1. **Data Snooping**: Test set performance may not generalize to future data
2. **Regime Changes**: Market dynamics change; model may not adapt
3. **Simplifications**:
   - Assumes perfect execution (no slippage)
   - Assumes infinite liquidity
   - No funding rates for leverage
4. **Forecast Accuracy**: LSTM is only as good as training data allows

### Risk Management

**DO NOT deploy this system without:**
- ✅ Out-of-sample validation on truly held-out data
- ✅ Live monitoring for distribution shifts
- ✅ Position limits and circuit breakers
- ✅ Regular model retraining
- ✅ Realistic slippage/friction modeling

### Reproducibility

- Seeds set to 42 for reproducibility
- CUDA/CPU auto-detection
- All dependencies pinned in requirements.txt
- Full code provided (no black boxes)

## 📖 Documentation

### Technical Report (`TECHNICAL_REPORT.md`)
- 15-page detailed analysis
- Architecture diagrams
- Mathematical formulations
- Results interpretation
- Failure mode analysis

### Presentation Outline (`PRESENTATION_OUTLINE.md`)
- 20-minute presentation structure
- 20 slides with notes
- Speaking points
- Q&A preparation
- Visual design guidelines

## 🔬 Experimental Design

### Controlled Experiment

**Factor Varied**: Forecast signal in state
**Fixed**: Architecture, hyperparameters, data split

**Treatment A**: State includes forecast
```
obs = [market_features, position_state, portfolio_metrics, forecast]
```

**Control B**: State without forecast
```
obs = [market_features, position_state, portfolio_metrics]
```

**Measurement**: 10 episodes each on test set

**Metrics**: Sharpe ratio (primary), Calmar ratio, max drawdown

### Statistical Rigor

- Multiple episodes (10) to reduce variance
- Reported means and standard deviations
- Visual distributions of returns
- Consistent random seed for reproducibility

## 🚢 Deployment Considerations

### For Research
- ✅ Great for academic publication
- ✅ Clear experimental design
- ✅ Reproducible results
- ✅ Open-source code

### For Trading
- ⚠️ Requires live monitoring and adaptation
- ⚠️ Add realistic market impact model
- ⚠️ Implement position limits and risk controls
- ⚠️ Daily/weekly model retraining

### For Learning
- ✅ Excellent learning resource
- ✅ Combines forecasting + RL
- ✅ Clean code structure
- ✅ Well-documented

## 📚 References

### Core Papers
- Schulman et al. (2017): "Proximal Policy Optimization Algorithms" (PPO)
- Hochreiter & Schmidhuber (1997): "Long Short-Term Memory" (LSTM)
- Mnih et al. (2016): "Asynchronous Methods for Deep RL" (A3C)

### Textbooks
- Sutton & Barto (2018): "Reinforcement Learning: An Introduction"
- Goodfellow et al. (2016): "Deep Learning"
- Markowitz (1952): "Portfolio Selection"

### Useful Resources
- Gymnasium documentation: https://gymnasium.farama.org
- PyTorch documentation: https://pytorch.org
- yfinance: https://github.com/ranaroussi/yfinance

## ❓ FAQ

### Q: How long does training take?
**A:** ~1.5 hours total:
- LSTM: 10 minutes
- PPO with forecast: 30 minutes
- PPO without forecast: 25 minutes
- Evaluation: 10 minutes

### Q: What hardware do I need?
**A:** Works on CPU and GPU:
- GPU (6GB VRAM): ~30 minutes total
- CPU (modern): ~2 hours total

### Q: Can I use a different asset?
**A:** Yes! Just change `TICKER` in the notebook (e.g., "ETH-USD", "AAPL")

### Q: How do I know if results are good?
**A:** Compare to benchmarks:
- Random trading: ~0.0 Sharpe ratio
- Buy and hold: Market Sharpe ratio (~0.3-0.5 for crypto)
- Good RL strategy: 0.5-1.5 Sharpe ratio

### Q: Can I use this for real trading?
**A:** Potentially, but:
1. Add realistic transaction costs and slippage
2. Implement position limits and risk controls
3. Monitor for distribution shifts
4. Backtest on longer time periods
5. Paper trade before live deployment

## 📝 Citation

If you use this code in research, please cite:

```bibtex
@project{forecast_rl_trading_2024,
  title={Forecast-Augmented Reinforcement Learning for Cryptocurrency Trading},
  author={[Your Name]},
  year={2024},
  institution={CAS AI}
}
```

## 📄 License

MIT License - Free for academic and commercial use

## 🤝 Contributing

Found a bug or want to improve? You can:
1. Report issues
2. Submit improvements
3. Extend to multi-asset trading
4. Add uncertainty quantification

## 📧 Contact

For questions, reach out or open an issue on the repository.

---

**Last Updated**: March 2024
**Status**: Complete and tested ✅

