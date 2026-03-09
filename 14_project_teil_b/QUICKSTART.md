# Quick Start Guide

## 🎯 5-Minute Setup

### Step 1: Install Dependencies (2 minutes)

```bash
cd /home/isc-den/cas-artificial-intelligence/14_project_teil_b

# Option A: Using pip
pip install -r requirements.txt

# Option B: Using conda (if you prefer)
# conda create -n trading-rl python=3.9
# conda activate trading-rl
# conda install pytorch::pytorch pytorch::torchvision pytorch::torchaudio -c pytorch
# pip install gymnasium yfinance pandas numpy matplotlib jupyter
```

### Step 2: Launch Jupyter (1 minute)

```bash
jupyter notebook 01_Complete_Solution.ipynb
```

The browser will open automatically. If not, navigate to `http://localhost:8888`

### Step 3: Run the Notebook (Can take 1-2 hours)

The notebook is fully self-contained. Just press **Shift+Enter** on each cell sequentially, or use **Cell → Run All**.

Expected execution time:
- Data loading: 1 minute
- LSTM training: 10 minutes
- PPO training (with forecast): 30 minutes
- PPO training (without forecast): 25 minutes
- Evaluation: 10 minutes
- **Total**: ~75 minutes

## 📊 What You Get

After running the full notebook, you'll have:

### Outputs
```
✓ Models trained and saved:
  - ppo_with_forecast.pt (PPO agent that uses forecast)
  - ppo_without_forecast.pt (baseline PPO agent)
  - lstm_forecaster.pt (price prediction model)

✓ Metrics CSV:
  - performance_comparison.csv (results table)
  - results_summary.json (summary statistics)

✓ Visualizations:
  - training_curves.png (convergence analysis)
  - equity_curves.png (performance curves)
  - return_distributions.png (risk analysis)

✓ Analysis:
  - Did forecast help? (answer in notebook output)
  - Why/why not? (detailed explanation)
  - What failed? (failure modes)
  - What improved? (success factors)
```

### Key Outputs Explained

**Performance Comparison CSV:**
```
Metric              With Forecast    Without Forecast
Return              [value]          [value]
Sharpe              [value]          [value]
Max Drawdown        [value]          [value]
Volatility          [value]          [value]
Win Rate            [value]          [value]
Profit Factor       [value]          [value]
Calmar Ratio        [value]          [value]
Sortino Ratio       [value]          [value]
Turnover            [value]          [value]
```

**Training Curves Plot:**
- Left panel: Episode returns over training (should increase)
- Right panel: KL divergence (should stabilize below target)

**Equity Curves Plot:**
- Top: Final wealth trajectory on test set
- Bottom: Position distributions (how aggressive)

## 🔍 Understanding the Results

### Metrics Explained

| Metric | Good Value | Interpretation |
|--------|-----------|-----------------|
| **Return** | > 0 | Profitable strategy |
| **Sharpe** | > 1.0 | Good risk-adjusted returns |
| **Max DD** | > -10% | Limited downside risk |
| **Volatility** | < 20% | Stable returns |
| **Win Rate** | > 50% | More winning days |
| **Turnover** | < 0.5 | Not overtrading |
| **Calmar** | > 0.5 | Return per unit drawdown |

### Interpreting Comparison

**Forecast Helps If:**
- ✅ With forecast Sharpe > Without forecast Sharpe
- ✅ With forecast Max DD is better (less negative)
- ✅ With forecast Calmar ratio is higher

**Forecast Hurts If:**
- ❌ With forecast metrics are worse than baseline
- ❌ With forecast has higher volatility
- ❌ Return is lower despite complexity

## 💻 Code Structure

### Main Files

| File | Purpose | Key Classes |
|------|---------|------------|
| `01_Complete_Solution.ipynb` | Main executable | None (script format) |
| `trading_env.py` | Trading environment | `EnhancedTradingEnv` |
| `forecasting.py` | LSTM model | `TimeSeriesForecaster` |
| `ppo_trainer.py` | PPO training | `PPOTrainer`, `ActorCritic` |
| `evaluation.py` | Metrics | `TradingMetrics`, `StrategyEvaluator` |

### Using the Modules Directly

```python
# Load the modules
from trading_env import EnhancedTradingEnv
from ppo_trainer import PPOTrainer

# Create environment
env = EnhancedTradingEnv(df_train, fee=0.0005)

# Create trainer
trainer = PPOTrainer(
    obs_dim=env.single_observation_space.shape[0],
    act_dim=env.single_action_space.shape[0],
    device=device
)

# Train
for update in range(500):
    rollout, _ = trainer.rollout(env, n_steps=128)
    metrics = trainer.update(rollout, epochs=10, minibatch_size=64)
```

## ⚙️ Configuration Options

Edit these parameters in the notebook to customize:

```python
# Data selection
TICKER = "BTC-USD"           # Bitcoin (default), or "ETH-USD" for Ethereum
START = "2022-01-01"         # Training start date
END = "2024-01-01"           # Training end date
TRAIN_FRAC = 0.8             # 80% train, 20% test

# Trading constraints
FEE = 0.0005                 # 0.05% transaction fee
KAPPA = 0.1                  # Risk penalty weight (higher = less risky)
INITIAL_CASH = 100000.0      # Starting capital
MAX_LEVERAGE = 2.0           # Max 2x leverage (reduce for more conservative)

# PPO training
NUM_ENVS = 8                 # Parallel environments (reduce if memory limited)
N_STEPS = 128                # Rollout length (more = more stable but slower)
TOTAL_UPDATES = 500          # Number of PPO updates (500-2000 typical)
LR = 3e-4                    # Learning rate (3e-4 to 1e-3 typical)
```

## 🐛 Troubleshooting

### Issue: Out of Memory
**Solution:**
```python
NUM_ENVS = 4  # Reduce from 8 to 4
MINIBATCH_SIZE = 32  # Reduce from 64 to 32
```

### Issue: Training is Slow
**Solution:**
```python
TOTAL_UPDATES = 200  # Reduce from 500 to 200
N_STEPS = 64  # Reduce from 128 to 64
```

### Issue: yfinance download fails
**Solution:**
```python
# Manually download and save CSV instead:
df = pd.read_csv('your_btc_data.csv', parse_dates=True, index_col=0)
```

### Issue: Models don't converge
**Solution:**
```python
LR = 1e-4  # Reduce learning rate
ENT_COEF = 0.01  # Increase exploration
PPO_EPOCHS = 5  # Reduce epochs to prevent overfitting
```

## 🎓 Learning Path

### For First-Time Users
1. **Read**: Start with README.md and PRESENTATION_OUTLINE.md
2. **Watch**: Conceptually understand the slides
3. **Run**: Execute the notebook with default parameters
4. **Analyze**: Examine the generated plots and metrics
5. **Modify**: Change TICKER or parameters and re-run

### For Deep Dive
1. **Study**: Read TECHNICAL_REPORT.md for equations
2. **Code Review**: Read and understand each Python module
3. **Trace Execution**: Add print statements, set breakpoints
4. **Experiment**: Test different architectures
5. **Extend**: Add features or modify reward function

### For Production
1. **Validate**: Test on multiple assets and time periods
2. **Monitor**: Implement distribution shift detection
3. **Optimize**: Hyperparameter tuning with Bayesian optimization
4. **Deploy**: With proper risk controls and circuit breakers
5. **Maintain**: Regular retraining and performance monitoring

## 📞 Getting Help

### Issue Checklist
Before asking for help:
- [ ] Read error message carefully
- [ ] Check if all requirements installed (`pip list`)
- [ ] Try reducing NUM_ENVS and TOTAL_UPDATES
- [ ] Check disk space (need ~1GB for data)
- [ ] Check GPU memory if using CUDA (`nvidia-smi`)

### Common Error Messages

**`ModuleNotFoundError: No module named 'gymnasium'`**
→ `pip install gymnasium`

**`RuntimeError: CUDA out of memory`**
→ Reduce `NUM_ENVS` or use CPU

**`HTTPError: HTTP Error 404 when downloading`**
→ yfinance has issues; try downloading later

## 📈 Next Steps

After running the basic notebook:

1. **Try Different Assets**
   - Change `TICKER = "ETH-USD"` (Ethereum)
   - Change `TICKER = "AAPL"` (Apple stock)

2. **Experiment with Parameters**
   - Increase `MAX_LEVERAGE` to 3.0
   - Increase `TOTAL_UPDATES` to 1000
   - Reduce `FEE` to 0.0001 for better markets

3. **Extend the Architecture**
   - Add more technical indicators
   - Implement uncertainty quantification
   - Add multi-asset portfolio optimization

4. **Deploy**
   - Implement live trading with backtest validation
   - Add position sizing based on risk
   - Monitor for distribution shifts

## 📚 Additional Resources

### Papers to Read
- [PPO Paper](https://arxiv.org/abs/1707.06347): Proximal Policy Optimization
- [LSTM Paper](https://ieeexplore.ieee.org/document/6795925): Long Short-Term Memory
- [Deep RL Survey](https://arxiv.org/abs/1811.12560): Deep Reinforcement Learning

### Helpful Tutorials
- Gymnasium Tutorial: https://gymnasium.farama.org/content/basic_usage/
- PyTorch RL: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
- Trading with ML: https://github.com/kthouz/RL_trading

---

## ✅ Checklist Before Presentation

- [ ] Run notebook completely (no errors)
- [ ] Review generated plots and metrics
- [ ] Read TECHNICAL_REPORT.md thoroughly
- [ ] Practice explaining the architecture
- [ ] Prepare slides (use PRESENTATION_OUTLINE.md as template)
- [ ] Have backup plans for questions
- [ ] Test live demo if planning to show code

## 🎯 Success Criteria

Your implementation is successful if:

✅ **Code runs** without errors
✅ **Models train** and converge
✅ **Metrics computed** (Sharpe, drawdown, etc.)
✅ **Comparison made** between with/without forecast
✅ **Plots generated** showing results
✅ **Analysis provided** explaining findings
✅ **Report written** with conclusions
✅ **Presentation prepared** for discussion

---

**Ready?** Start with: `jupyter notebook 01_Complete_Solution.ipynb`

