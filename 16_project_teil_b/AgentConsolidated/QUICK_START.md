# QUICK START GUIDE

## Three-Step Execution

### **Step 1: Run All Experiments** (3-4 hours)
```bash
cd /home/isc-den/cas-artificial-intelligence/16_project_teil_b/AgentConsolidated
python run_all_experiments.py
```

**What happens:**
- 10 experiments run sequentially
- Each experiment trains PPO for 3000 updates
- Metrics saved to `results/experiment_*/metrics.pkl`
- All results logged to WandB (offline mode by default)
- Real-time console output shows training progress

**Expected output:**
```
✓ EXPERIMENT 1/10: PPO-WITHOUT-FORECAST
  [Training progress... 100 → 200 → ... → 3000 updates]
  ✓ Training completed
  ✓ Evaluation completed
  Metrics: Return=17.69%, Sharpe=0.32, Max DD=-30.09%, Vol=1.77%
  ✓ Results saved

✓ EXPERIMENT 2/10: PPO-WITH-FORECAST
  [Training with LSTM forecast...]
  ...

[Continues for experiments 3-10]

✓ ALL EXPERIMENTS COMPLETED
```

---

### **Step 2: Generate Visualizations** (5 minutes)
```bash
python create_visualizations.py
```

**What happens:**
- Loads metrics from all `results/*/metrics.pkl` files
- Generates 12 professional PNG plots (300 DPI)
- Creates comparison tables and heatmaps
- Saves all visualizations to `visualizations/` folder

**Output files:**
```
visualizations/
├── 01_equity_curves.png           # All equity curves comparison
├── 02_risk_metrics.png            # Risk analysis (Sharpe, Calmar, Sortino)
├── 03_returns_distribution.png    # Daily returns histogram & box plot
├── 04_drawdown.png                # Drawdown evolution & severity
├── 05_heatmap.png                 # Performance metrics heatmap
├── 06_table.png                   # Comparison table (all metrics)
├── 07_forecast_impact.png         # Forecast impact analysis
├── 08_reward_comparison.png       # Reward function comparison
├── 09_summary.png                 # Key findings summary
├── 10_architecture.png            # Agent architecture diagram
├── 11_overview.png                # Experiment overview
└── 12_checklist.png               # Requirements fulfillment
```

---

### **Step 3: Create Presentation** (1 minute)
```bash
python generate_presentation.py
```

**What happens:**
- Creates PowerPoint presentation from visualizations
- Includes 25+ professionally formatted slides
- Embeds all PNG plots with proper titles
- Ready for 20-minute presentation

**Output:**
```
PPO_Trading_Agent_Presentation.pptx (25+ slides)
├── Slides 1-3: Introduction & Overview
├── Slides 4-6: Methodology & Architecture
├── Slides 7-13: Results & Visualizations
├── Slides 14-17: Analysis (Forecast & Rewards)
├── Slides 18-22: Findings & Critical Reflection
├── Slides 23-25: Deliverables & Conclusion
```

---

## What You Get

### **Code**
- ✅ `run_all_experiments.py` - Execute all 10 experiments
- ✅ `create_visualizations.py` - Generate 12 plots
- ✅ `generate_presentation.py` - Create PowerPoint
- ✅ `trading_framework.py` - Core agent implementation
- ✅ `trading_config.py` - Parameter configurations

### **Results**
- ✅ 10 experiment metric files (results/*/metrics.pkl)
- ✅ Comparison CSV (metrics_comparison.csv)
- ✅ 12 professional PNG visualizations (300 DPI)

### **Presentation**
- ✅ 25+ slide PowerPoint presentation
- ✅ All visualizations embedded
- ✅ Architecture diagrams
- ✅ Comprehensive analysis
- ✅ Critical findings & recommendations

---

## Key Metrics Provided

For **every experiment**, you get:

**Primary Metrics** (Required):
- ✓ Cumulative Return (%)
- ✓ Sharpe Ratio
- ✓ Maximum Drawdown (%)
- ✓ Annualized Volatility (%)
- ✓ Turnover

**Secondary Metrics** (Bonus):
- ✓ Calmar Ratio
- ✓ Sortino Ratio
- ✓ Win Rate (%)
- ✓ Profit Factor
- ✓ Kurtosis & Skewness
- ✓ Cost metrics

---

## Experiments Breakdown

| Exp | Type | Forecast | Reward | Purpose |
|-----|------|----------|--------|---------|
| 1 | PPO | None | WITH_RISK | **Baseline** |
| 2 | PPO | LSTM | WITH_RISK | **Forecast impact** |
| 3 | PPO | None | BASIC | Reward ablation |
| 4 | PPO | None | WITH_SHARPE | Reward ablation |
| 5 | PPO | None | RISK_ADJUSTED | Reward ablation |
| 6 | PPO | None | SORTINO | Reward ablation |
| 7 | PPO | None | CALMAR | Reward ablation |
| 8 | PPO | None | COMPOSITE | Reward ablation |

---

## Troubleshooting

**If experiment hangs:**
- ✓ 30-second timeout on WandB.finish() prevents infinite hangs
- ✓ If still hanging (rare), press Ctrl+C - results are already saved

**If visualization script fails:**
- Check that `results/` directory exists
- Ensure `metrics.pkl` files exist in subdirectories
- Run with: `python create_visualizations.py 2>&1`

**If presentation creation fails:**
- `python-pptx` will be auto-installed if missing
- Check visualization PNG files exist
- Re-run: `python generate_presentation.py`

---

## File Locations

After execution:

```
AgentConsolidated/
├── results/                        # Experiment results
│   ├── experiment_1/metrics.pkl
│   ├── experiment_2/metrics.pkl
│   └── ... (10 total)
│
├── visualizations/                 # Generated plots
│   ├── 01_equity_curves.png
│   ├── ... (12 total)
│
├── metrics_comparison.csv          # Overall comparison
├── PPO_Trading_Agent_Presentation.pptx  # Final presentation
└── [documentation files]
```

---

## Total Runtime

| Step | Time |
|------|------|
| Experiments | 3-4 hours |
| Visualizations | 5 minutes |
| Presentation | 1 minute |
| **Total** | **~3-4.25 hours** |

---

## For Presentation (20 minutes)

**What to present:**
1. Architecture (Slide 5)
2. Key findings (Slides 14-17)
3. Visualizations (Slides 7-13)
4. Critical analysis (Slides 18-22)
5. Recommendations (Slide 21)

**Talking points:**
- "We tested 10 different configurations..."
- "Forecast helps with risk metrics but not absolute returns..."
- "Risk-aware reward functions produce stable, practical strategies..."
- "Here are the detailed performance comparisons..."
- "Our recommendations for production trading..."

---

## Questions?

Check documentation:
- `COMPLETE_SOLUTION_SUMMARY.md` - Full overview
- `HANGING_ISSUE_FIXED.md` - Technical details
- `FIXES_AND_RUNNING_GUIDE.md` - Comprehensive guide

---

**Status**: ✅ **READY TO EXECUTE**

You have everything needed to:
1. ✅ Run comprehensive experiments
2. ✅ Generate professional visualizations
3. ✅ Create presentation-ready materials
4. ✅ Meet all assignment requirements
5. ✅ Present findings clearly

**Next action**: Execute `python run_all_experiments.py`

