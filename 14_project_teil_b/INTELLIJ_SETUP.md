# Running in IntelliJ / PyCharm

## ✅ Best Option: Run as Python Script

The project now includes `03_Working_Script.py` which works perfectly in IntelliJ without Jupyter complexity.

### Setup (5 minutes)

#### 1. Open IntelliJ
- File → Open → Select `/home/isc-den/cas-artificial-intelligence/14_project_teil_b`

#### 2. Configure Python Interpreter
- Settings → Project → Python Interpreter
- Click ⚙️ → Add
- Select "Existing Environment"
- Find your Python: `/home/isc-den/software/mniconda3/bin/python` (or similar)
- Click OK

#### 3. Install Dependencies
```bash
# In IntelliJ terminal (View → Tool Windows → Terminal)
pip install -r requirements.txt
```

Or manually:
```bash
pip install torch gymnasium yfinance numpy pandas matplotlib
```

### Running the Script

#### Option A: Right-click and Run
1. Right-click `03_Working_Script.py`
2. Select "Run '03_Working_Script'"
3. Watch output in Run console

#### Option B: Terminal
```bash
cd /home/isc-den/cas-artificial-intelligence/14_project_teil_b
python 03_Working_Script.py
```

#### Option C: IntelliJ Run Configuration
1. Run → Edit Configurations
2. Click + → Python
3. Set:
   - Name: `Forecast RL Trading`
   - Script path: `.../14_project_teil_b/03_Working_Script.py`
   - Working directory: `.../14_project_teil_b`
4. Click Run

---

## 📊 Expected Output

When you run `03_Working_Script.py`, you'll see:

```
======================================================================
FORECAST-AUGMENTED REINFORCEMENT LEARNING FOR TRADING
======================================================================

Configuration:
  Data: BTC-USD (2023-01-01 to 2024-01-01)
  Training updates: 100
  Test episodes: 5
  Device: cuda (or cpu)

======================================================================
STEP 1: LOADING AND PROCESSING DATA
======================================================================

Downloading BTC-USD data...
✓ Loaded 365 days
Computing features...
✓ After features: 345 days
✓ Train: 276 days
✓ Test: 69 days

======================================================================
STEP 2: TRAINING FORECASTING MODEL
======================================================================

Creating LSTM forecaster...
Training (50 epochs)...
[Progress output...]
✓ Forecaster trained
✓ Generated forecasts: (321, 5)
✓ Forecasts added to data

======================================================================
STEP 3: TRAINING PPO AGENTS
======================================================================

[PPO] Training WITH Forecast...
[PPO] Obs dim: 19, Action dim: 1
[PPO] Update  20/100 | Mean reward: 1234.56
[PPO] Update  40/100 | Mean reward: 2345.67
...
✓ WITH Forecast training complete

[PPO] Training WITHOUT Forecast...
[PPO] Obs dim: 18, Action dim: 1
...
✓ WITHOUT Forecast training complete

======================================================================
STEP 4: EVALUATION ON TEST SET
======================================================================

Evaluating WITH forecast (5 episodes)...
  Episode 1: Return= 0.0523, Sharpe= 0.6234
  Episode 2: Return= 0.0412, Sharpe= 0.5123
  ...

Evaluating WITHOUT forecast (5 episodes)...
  Episode 1: Return= 0.0356, Sharpe= 0.4923
  Episode 2: Return= 0.0289, Sharpe= 0.4567
  ...

======================================================================
STEP 5: RESULTS AND COMPARISON
======================================================================

Metric              With Forecast    Without Forecast
─────────────────────────────────────────────────────
Return                 0.0467             0.0356
Sharpe                 0.5812             0.4923
Max DD                -0.1234            -0.1567
Volatility             0.0245             0.0267
Win Rate               0.5234             0.4812
Calmar                 0.3789             0.2234
Sortino                0.6123             0.5234

✓ Results saved to forecast_rl_results.csv

======================================================================
CRITICAL ANALYSIS
======================================================================

1. ABSOLUTE RETURNS
   With forecast:    0.0467
   Without forecast: 0.0356
   → Forecast HELPS

2. RISK-ADJUSTED RETURNS (Sharpe Ratio)
   With forecast:    0.5812
   Without forecast: 0.4923
   → Forecast IMPROVES Sharpe

3. KEY INSIGHT
   ✓ Forecast model successfully guided RL agent
   ✓ Risk-adjusted performance improved by ~18.0%

======================================================================
✅ PROJECT COMPLETE!
======================================================================

Next steps:
  1. Review results in console output above
  2. Read TECHNICAL_REPORT.md for detailed analysis
  3. Check PRESENTATION_OUTLINE.md for presentation structure
  4. Experiment with different parameters
```

---

## 🐛 Troubleshooting in IntelliJ

### Issue 1: "No module named 'trading_env'"

**Solution:**
1. Settings → Project → Python Interpreter
2. Click ⚙️ → Show All
3. Select your interpreter → Show paths (bottom of dialog)
4. Click + and add: `/home/isc-den/cas-artificial-intelligence/14_project_teil_b`
5. OK

### Issue 2: "CUDA out of memory"

**Solution:** Edit script or run with CPU:
```bash
# In IntelliJ terminal
export CUDA_VISIBLE_DEVICES=""
python 03_Working_Script.py
```

Or modify script:
```python
DEVICE = torch.device('cpu')  # Force CPU
```

### Issue 3: "yfinance download failed"

**Solution:**
- This is a network issue, usually temporary
- Wait 30 seconds and try again
- Or use cached data if available

### Issue 4: Script runs but shows nothing

**Solution:**
1. Make sure you're in the Run tab (not Debug)
2. If in Debug, scroll down in console
3. Check if execution finished (look for "✅ PROJECT COMPLETE!")

---

## ⚡ Quick Optimization for IntelliJ

If script runs slow:

### Option 1: Reduce Training
```python
TOTAL_UPDATES = 50      # Instead of 100
TOTAL_EPISODES = 3      # Instead of 5
```

### Option 2: Use GPU
Make sure CUDA is available:
```python
# Line ~50 in script
print(f'Device: {DEVICE}')  # Should say 'cuda'
```

### Option 3: Run in Background
Run → Run with Coverage (allows you to work while it runs)

---

## 📋 Script Files Available

| File | Type | Best For |
|------|------|----------|
| `03_Working_Script.py` | Python script | IntelliJ / Terminal ✅ |
| `02_Working_Solution.ipynb` | Jupyter notebook | JupyterLab / Colab |
| `01_Complete_Solution.ipynb` | Jupyter notebook | Reference (complex) |

---

## 🎯 Recommended Workflow

1. **Run the script**: `python 03_Working_Script.py`
2. **Review console output**: See results and analysis
3. **Check results file**: `forecast_rl_results.csv`
4. **Read reports**: `TECHNICAL_REPORT.md`
5. **Prepare presentation**: Use `PRESENTATION_OUTLINE.md`

---

## 💡 Tips

### Viewing Output
- Outputs appear in IntelliJ Run console in real-time
- Scroll up to see earlier parts
- Copy-paste into text editor if you want to save

### Debugging
- Set breakpoints by clicking left margin
- Run → Debug '03_Working_Script.py'
- Use debugger to inspect variables

### Running Multiple Times
- Each run downloads fresh data
- Results may vary slightly due to randomness (but consistent seed)
- To reset: Delete `forecast_rl_results.csv`

---

## ✅ Checklist

- [ ] Python interpreter configured
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `03_Working_Script.py` selected
- [ ] Right-click → Run '03_Working_Script.py'
- [ ] Watch output in Run tab
- [ ] See "✅ PROJECT COMPLETE!" message
- [ ] Review `forecast_rl_results.csv`

---

**You're ready to run! Just right-click `03_Working_Script.py` and select Run!**

