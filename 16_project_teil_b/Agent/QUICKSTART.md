# QUICK START GUIDE

## What's Working

✅ **Complete experiment suite in notebook**
- Cells are self-contained and executable
- All working code from `run_experiments.py` integrated
- W&B logging included

## How to Run

### Option 1: Jupyter Notebook (Recommended)
```bash
cd /home/isc-den/cas-artificial-intelligence/16_project_teil_b/Agent/
jupyter notebook Project_Part_2_Final_Architecture.ipynb
```
Then execute cells in order from top to bottom.

### Option 2: Python Script
```bash
cd /home/isc-den/cas-artificial-intelligence/16_project_teil_b/Agent/
python3 run_experiments.py
```

## Notebook Cell Structure

1. **Setup & Imports** - Load all libraries
2. **Data Loading** - Download S&P 500 data (2020-2023)
3. **LSTM Forecaster Model** - Define neural network
4. **Train LSTM** - Train price prediction model (30 epochs)
5. **Generate Forecasts** - Create forecast signals for all data
6. **Trading Environment** - Gym environment for trading
7. **Experiment 1: Forecast-Only** - Simple rule-based baseline
8. **Experiment 2 & 3: PPO** - Train PPO with/without forecast
9. **Experiment 4: Reward Variations** - Test 3 reward functions
10. **Results Summary** - Compare all strategies and answer research question

## Expected Output

After running all cells:

```
========================================================================
FINAL RESULTS SUMMARY
========================================================================

                                Return   Sharpe  Volatility  Max DD  Turnover
Forecast-Only                    X.XX%   X.XXXX      X.XX%    X.XX%    X.XXXX
PPO-Without-Forecast             X.XX%   X.XXXX      X.XX%    X.XX%    X.XXXX
PPO-With-Forecast                X.XX%   X.XXXX      X.XX%    X.XX%    X.XXXX
PPO-pnl-risk                      X.XX%   X.XXXX      X.XX%    X.XX%    X.XXXX
PPO-pnl-cost                      X.XX%   X.XXXX      X.XX%    X.XX%    X.XXXX
PPO-balanced                      X.XX%   X.XXXX      X.XX%    X.XX%    X.XXXX

RESEARCH QUESTION: Does forecast improve RL performance?

PPO with Forecast:      X.XX%
PPO without Forecast:   X.XX%
Improvement:           +X.X%

✓ YES - Forecast HELPS
Explanation: RL agent benefits from price movement predictions
```

## Files Generated

- `results_comparison.csv` - Results table
- `./wandb/` - W&B offline logs

## Runtime

- LSTM training: ~5-10 minutes
- PPO training: ~40-60 minutes
- **Total: 60-90 minutes**

## Troubleshooting

### Issue: Out of memory
**Solution:** Reduce PPO timesteps in cell 8
```python
model.learn(total_timesteps=10000)  # Instead of 20000
```

### Issue: Slow training
**Solution:** Use GPU
```python
device = torch.device('cuda')  # Automatically detected
```

### Issue: Data not downloading
**Solution:** Check internet connection, or provide local CSV file
```python
df = pd.read_csv('your_data.csv')
df = df.rename(columns={'Close': 'close'})
```

## Key Code Components

### 1. LSTM Forecaster
- Input: 30-day price history
- Output: Probability of price going up in next 5 days
- Task: Binary classification

### 2. Trading Environment
- State: [position, cash_ratio, momentum, volatility, (forecast)]
- Action: Leverage position [-1, +1]
- Cost model: Fee (0.05%) + Kappa penalty (0.1x)
- Reward: Configurable (PnL, risk, costs)

### 3. Experiments
1. **Forecast-Only**: Simple rule, pos = 1 if forecast > 0.5 else -1
2. **PPO No Forecast**: RL agent with 4-dim state
3. **PPO With Forecast**: RL agent with 5-dim state (adds forecast)
4. **Reward Variations**: 3 different reward functions

## Metrics

For each strategy:
- **Return %** - Absolute profit
- **Sharpe Ratio** - Return per unit volatility
- **Volatility** - Annualized std dev
- **Max Drawdown** - Largest peak-to-trough loss
- **Turnover** - Trading activity level

## Understanding Results

### Forecast HELPS (+5%+)
- PPO with forecast > PPO without forecast
- RL agent successfully uses price predictions
- Recommendation: Keep forecast in state

### Forecast NEUTRAL (±5%)
- Performance about the same
- Forecast signal is redundant
- RL learns patterns from market features alone
- Recommendation: Use simpler model without forecast

### Forecast HURTS (-5%+)
- PPO with forecast < PPO without forecast
- Forecast causes overconfidence/overtrading
- Possible issues: forecast noise, overfitting, fees
- Recommendation: Remove forecast, add constraints

## Next Steps

1. **Run all cells** from top to bottom
2. **Review results** - Do you see forecast help?
3. **Write report** - Document findings using TECHNICAL_REPORT_TEMPLATE.md
4. **Iterate** - Modify hyperparameters, reward functions, etc.

---

**Ready to run!** Start with Jupyter notebook or Python script.

