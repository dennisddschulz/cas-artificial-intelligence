# Extended Reward Function Experiments - Master Index

## 📋 Documentation Files (Start Here!)

### For Quick Start (5-10 minutes)
1. **QUICKSTART_REWARD_EXPERIMENTS.md** ⭐
   - Overview of what was done
   - How to run experiments (3 options)
   - Expected outputs
   - Common questions answered
   - **Start here if you want to run experiments quickly**

### For Understanding Reward Functions (20-30 minutes)
2. **REWARD_ABLATION_GUIDE.md** 📚
   - Detailed explanation of each 8 reward functions
   - Formula, behavior, and best use cases
   - Parameter tuning guide
   - How to interpret results
   - Troubleshooting section
   - **Start here if you want to understand rewards deeply**

### For Technical Details (30-45 minutes)
3. **EXTENDED_EXPERIMENTS_SUMMARY.md** 🔧
   - What was modified and created
   - Key design decisions explained
   - Code structure and implementation
   - How to use new features
   - Performance expectations
   - **Start here if you want to understand technical implementation**

### For Implementation Confirmation
4. **IMPLEMENTATION_COMPLETE.md** ✅
   - Summary of all changes made
   - Files modified and created
   - Reward functions implemented
   - Usage examples
   - Testing and validation
   - **Reference this to verify all changes**

---

## 🚀 Quick Start

### Option 1: Run All Experiments (Recommended First Time)
```bash
cd AgentConsolidated/
python run_all_experiments.py
```
**Duration**: 5 hours | **Experiments**: 10 (2 baseline + 8 reward variants)

### Option 2: Run Reward Ablation Only (Faster)
```bash
cd AgentConsolidated/
python run_reward_ablation.py
```
**Duration**: 3-4 hours | **Experiments**: 8 reward variants only

### Option 3: Run Single Experiment (Testing)
```bash
cd AgentConsolidated/
python -c "
from trading_config import ConfigBuilder, RewardType
from trading_framework import ExperimentRunner

config = ConfigBuilder('Quick-Test').with_reward(RewardType.SORTINO).build()
runner = ExperimentRunner(config)
results = runner.run()
print(f'Final Return: {results[\"metrics\"][\"total_return\"]*100:.2f}%')
"
```
**Duration**: 20-30 minutes | **Experiments**: 1

---

## 📊 8 Reward Functions at a Glance

| # | Name | Formula Essence | Incentivizes | Best For |
|---|------|---|---|---|
| 1 | **BASIC** | `PnL - Cost` | Maximum returns | Trend followers |
| 2 | **WITH_RISK** | `PnL - κ×Pos²×σ` | Balanced return-risk | General trading |
| 3 | **WITH_SHARPE** | `(PnL) / σ` | Risk-adjusted returns | Sharpe optimization |
| 4 | **RISK_ADJUSTED** | `(PnL/σ) - Cost` | Volatility normalization | Adaptive sizing |
| 5 | **SORTINO** | `(PnL_adj) / σ` | Downside control | Conservative |
| 6 | **CALMAR** | `PnL - DD_est` | Drawdown minimization | Wealth preservation |
| 7 | **INFO_RATIO** | `(PnL/σ) + Bonus` | Consistent profits | Alpha building |
| 8 | **COMPOSITE** | `0.5R + 0.3S + 0.2Risk` | Multiple objectives | Balanced approach |

**Learn more**: Read REWARD_ABLATION_GUIDE.md for detailed explanations

---

## 📁 Files Modified/Created

### Modified Files
- ✏️ **run_all_experiments.py** - Enhanced to 10 experiments + reward analysis
- ✏️ **trading_framework.py** - Added 4 new reward types (SORTINO, CALMAR, INFORMATION_RATIO, COMPOSITE)

### New Scripts
- 🆕 **run_reward_ablation.py** - Dedicated reward ablation runner

### New Documentation
- 📖 **QUICKSTART_REWARD_EXPERIMENTS.md** - Start here!
- 📖 **REWARD_ABLATION_GUIDE.md** - Deep dive into rewards
- 📖 **EXTENDED_EXPERIMENTS_SUMMARY.md** - Technical reference
- 📖 **IMPLEMENTATION_COMPLETE.md** - Change summary
- 📖 **EXPERIMENTS_INDEX.md** - This file

---

## 📈 Output Files Generated

After running experiments, you'll get:

### CSV Metrics
```
metrics_comparison.csv                 - All 10 experiments
reward_comparison_detailed.csv        - 8 reward variants
reward_ablation_comparison.csv        - Reward-only focused
```

### Visualizations (PNG)
```
equity_curves_comparison.png          - Final equity curves
drawdown_comparison.png               - Drawdown analysis
returns_distribution.png              - Return histograms
metrics_heatmap.png                   - Performance heatmap
reward_ablation_analysis.png          - 6-metric reward comparison
reward_ablation_comparison.png        - 8-metric reward comparison
```

### Raw Data
```
detailed_results.json                 - Complete results
reward_ablation_results.json          - Reward-focused
metrics.pkl                           - Python pickle format
./wandb/offline-run-*/               - W&B logs (sync-able)
```

---

## 💡 How to Use Results

### Step 1: Run Experiments
```bash
python run_reward_ablation.py  # 3-4 hours for 8 reward variants
```

### Step 2: Review Metrics
```bash
# Open these in your favorite spreadsheet program:
reward_ablation_comparison.csv
```

**Key columns to check**:
- `total_return` - Final wealth change (% higher is better)
- `sharpe_ratio` - Return per unit of risk (higher is better)
- `max_drawdown` - Worst loss (higher/less negative is better)
- `annualized_volatility` - Risk level (lower is better)
- `turnover` - Trading frequency (lower = fewer costs)

### Step 3: View Visualizations
```bash
# Open these in any image viewer:
reward_ablation_comparison.png        # Visual comparison
equity_curves_comparison.png          # Equity growth paths
drawdown_comparison.png               # Risk analysis
```

### Step 4: Make Decision
Choose reward function based on your goal:
- **Highest return?** → BASIC
- **Best risk-adjusted?** → WITH_SHARPE
- **Conservative?** → SORTINO or CALMAR
- **Balanced?** → COMPOSITE or WITH_RISK

---

## ❓ Common Questions

### Q: How long will this take?
- **Single experiment**: 20-30 min
- **8 reward variants**: 3-4 hours
- **Full 10 experiments**: 5 hours

### Q: Which should I choose?
Start with COMPOSITE (balanced) or WITH_SHARPE (risk-adjusted). See REWARD_ABLATION_GUIDE.md for details.

### Q: Can I modify parameters?
Yes! Edit `trading_config.py`:
```python
env_config.reward_params['kappa'] = 0.02  # Increase risk aversion
```

### Q: Why different returns?
Each reward type optimizes different objectives. See EXTENDED_EXPERIMENTS_SUMMARY.md for details.

### Q: WandB not working?
Scripts use offline mode by default. Sync with:
```bash
wandb sync ./wandb/offline-run-*/
```

### Q: My results look weird?
Check REWARD_ABLATION_GUIDE.md "Troubleshooting" section or review trading_framework.py step() function.

---

## 🎯 Recommended Reading Order

### For Practitioners (Want to Run & Get Results)
1. This file (you are here!) ← **5 min**
2. QUICKSTART_REWARD_EXPERIMENTS.md ← **10 min**
3. Run `python run_reward_ablation.py` ← **3-4 hours**
4. Review CSV and PNG outputs ← **30 min**
5. REWARD_ABLATION_GUIDE.md (interpretation section) ← **10 min**

### For Researchers (Want Deep Understanding)
1. REWARD_ABLATION_GUIDE.md (all 8 rewards) ← **30 min**
2. EXTENDED_EXPERIMENTS_SUMMARY.md (implementation) ← **30 min**
3. trading_framework.py (lines 1123-1228, reward calcs) ← **20 min**
4. trading_config.py (RewardType enum) ← **10 min**
5. Run `python run_all_experiments.py` ← **5 hours**
6. IMPLEMENTATION_COMPLETE.md (summary) ← **10 min**

### For Developers (Want to Modify Code)
1. EXTENDED_EXPERIMENTS_SUMMARY.md (what changed) ← **30 min**
2. trading_framework.py (TradingEnv.step method) ← **30 min**
3. trading_config.py (RewardType enum) ← **10 min**
4. IMPLEMENTATION_COMPLETE.md (design decisions) ← **10 min**
5. Modify and test ← **varies**

---

## 🔧 Code References

### Where Reward Types Are Defined
```python
# trading_config.py
class RewardType(Enum):
    BASIC = "basic"
    WITH_RISK = "with_risk"
    WITH_SHARPE = "with_sharpe"
    RISK_ADJUSTED = "risk_adjusted"
    SORTINO = "sortino"
    CALMAR = "calmar"
    INFORMATION_RATIO = "information_ratio"
    COMPOSITE = "composite"
```

### Where Rewards Are Calculated
```python
# trading_framework.py, TradingEnv.step() method (lines 1123-1228)
if self.reward_type == RewardType.BASIC:
    reward = true_reward
elif self.reward_type == RewardType.WITH_SHARPE:
    reward = true_reward / safe_sigma
# ... etc for all 8 types
```

### Where Experiments Are Configured
```python
# trading_config.py
def get_ppo_different_rewards_configs(group="reward_ablation"):
    # Returns list of ExperimentConfig objects
    # One for each of the 8 reward types
```

### Where Experiments Are Run
```python
# run_all_experiments.py
def main():
    # Runs 10 experiments:
    # - 2 baselines (without forecast, with forecast)
    # - 8 reward variants
```

---

## 📚 Full File List

### Core Files
- `trading_framework.py` - Main framework (1268 lines)
- `trading_config.py` - Configuration and enums
- `run_all_experiments.py` - Full experiment suite (716 lines)
- `run_reward_ablation.py` - Reward ablation script (310 lines)
- `trading_metrics.py` - Metrics calculation

### Documentation (This Section)
- `EXPERIMENTS_INDEX.md` ← **You are here**
- `QUICKSTART_REWARD_EXPERIMENTS.md` ← Start here for quick start
- `REWARD_ABLATION_GUIDE.md` ← Start here for deep understanding
- `EXTENDED_EXPERIMENTS_SUMMARY.md` ← Start here for technical details
- `IMPLEMENTATION_COMPLETE.md` ← Verification of changes

### Output Directory (After Running)
- `results/` - Generated results directory
- `wandb/` - WandB offline logs

---

## ✅ Verification Checklist

- ✅ run_all_experiments.py - Enhanced with reward analysis
- ✅ trading_framework.py - TradingEnv supports all 8 reward types
- ✅ run_reward_ablation.py - New dedicated ablation script
- ✅ REWARD_ABLATION_GUIDE.md - Comprehensive reward documentation
- ✅ EXTENDED_EXPERIMENTS_SUMMARY.md - Technical implementation guide
- ✅ QUICKSTART_REWARD_EXPERIMENTS.md - Quick reference guide
- ✅ IMPLEMENTATION_COMPLETE.md - Change verification summary
- ✅ EXPERIMENTS_INDEX.md - This master index file

---

## 🎯 Next Action

**Pick one**:

### 1️⃣ Just Want to Run It?
→ Go to QUICKSTART_REWARD_EXPERIMENTS.md and follow "Quick Start" section

### 2️⃣ Want to Understand Rewards?
→ Go to REWARD_ABLATION_GUIDE.md and read the reward explanations

### 3️⃣ Want Technical Details?
→ Go to EXTENDED_EXPERIMENTS_SUMMARY.md for implementation overview

### 4️⃣ Want to Get Started NOW?
→ Run this command:
```bash
cd AgentConsolidated/
python run_reward_ablation.py
```

---

**Version**: 1.0
**Status**: ✅ Complete and ready to use
**Last Updated**: 2024-03-12
**Framework**: PyTorch + Gymnasium + PPO Trading

---

*Questions? Check the documentation files above or review the code directly.*


