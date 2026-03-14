#!/usr/bin/env python3
"""
Quick Reference: Run Experiments & Generate Visualizations

Two-Step Process:
1. Run experiments and save metrics to pickle file
2. Generate visualizations from saved metrics (can be run anytime)
"""

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                    PPO TRADING FRAMEWORK - QUICK START                     ║
╚════════════════════════════════════════════════════════════════════════════╝

STEP 1: INSTALL DEPENDENCIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    pip install -r requirements.txt


STEP 2: RUN ALL EXPERIMENTS & SAVE METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    python run_all_experiments.py

    This will:
    ✓ Run 6 experiments (PPO without forecast, with forecast, 4 reward types)
    ✓ Calculate all metrics (16 total)
    ✓ Save results to:
      - metrics.pkl (for visualization script)
      - metrics_comparison.csv (Excel compatible)
      - detailed_results.json (raw data)
      - equity_curves_comparison.png
      - drawdown_comparison.png
      - returns_distribution.png
      - metrics_heatmap.png

    Time: 3-6 hours (CPU) or 1-2 hours (GPU)


STEP 3: GENERATE VISUALIZATIONS (Optional - Any Time Later)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    If you already have metrics.pkl, you can regenerate visualizations anytime:

    python create_visualizations.py

    This will:
    ✓ Load metrics.pkl (fast - no re-running experiments)
    ✓ Regenerate all plots
    ✓ Create new analysis and reports

    Time: 1-2 minutes


WORKFLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Option A: Full Run (First Time)
────────────────────────────────
    1. pip install -r requirements.txt
    2. python run_all_experiments.py          (creates metrics.pkl)
    3. Check generated files and plots


Option B: Regenerate Visualizations (Later)
────────────────────────────────────────────
    1. python create_visualizations.py        (loads metrics.pkl)
    2. Check newly generated plots


Option C: Custom Experiments
────────────────────────────
    1. Edit trading_config.py to change parameters
    2. python run_all_experiments.py          (creates new metrics.pkl)
    3. python create_visualizations.py        (generates plots from new data)


FILES CREATED BY run_all_experiments.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Essential:
  ✓ metrics.pkl                    - Pickle file with all results (for create_visualizations.py)

Data Export:
  ✓ metrics_comparison.csv         - All metrics in CSV format
  ✓ detailed_results.json          - Complete results as JSON

Visualizations:
  ✓ equity_curves_comparison.png   - All strategies' equity curves overlaid
  ✓ drawdown_comparison.png        - Drawdown analysis over time
  ✓ returns_distribution.png       - Daily returns histograms
  ✓ metrics_heatmap.png            - Color-coded metrics table


6 EXPERIMENTS INCLUDED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. PPO Without Forecast          → Baseline RL agent
2. PPO With LSTM Forecast        → RL + price predictor
3. PPO Reward - BASIC            → Minimal reward function
4. PPO Reward - WITH_RISK        → Risk-penalized reward
5. PPO Reward - WITH_SHARPE      → Sharpe-based reward
6. PPO Reward - RISK_ADJUSTED    → Risk-adjusted return

All metrics calculated:
  ✓ Cumulative Return
  ✓ Sharpe Ratio
  ✓ Max Drawdown
  ✓ Volatility
  ✓ Turnover
  ✓ Win Rate, Profit Factor, Costs, etc. (16 total)


CUSTOMIZATION EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Change Initial Budget:
    Edit trading_config.py:
    config.environment.initial_equity = 250000  # $250k instead of $100k

Change Asset:
    Edit trading_config.py:
    config.data.ticker = "ETH-USD"  # Trade Ethereum

Faster Testing (fewer updates):
    Edit trading_config.py:
    config.ppo.total_updates = 500  # Instead of 3000
    config.ppo.num_envs = 4  # Instead of 8


TROUBLESHOOTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Problem: "ModuleNotFoundError"
Solution: pip install -r requirements.txt

Problem: "CUDA out of memory"
Solution: In trading_config.py, set config.ppo.num_envs = 4

Problem: "Can't import TradingMetrics"
Solution: Make sure trading_metrics.py exists and has content

Problem: "metrics.pkl not found" (when running create_visualizations.py)
Solution: First run: python run_all_experiments.py

Problem: Script too slow
Solution: In trading_config.py, reduce total_updates and n_steps


KEY FEATURES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Fully parameterized - Easy to customize
✓ 6 experiments - All auto-run
✓ 16 metrics - Comprehensive analysis
✓ Pickle saving - Fast visualization regeneration
✓ No Jupyter - Pure Python scripts
✓ W&B logging - Optional cloud tracking
✓ CSV/JSON export - Share results easily
✓ Production ready - Error handling included


NEXT STEPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Run: python run_all_experiments.py
2. Wait for completion (3-6 hours)
3. Check generated PNG files
4. Review metrics_comparison.csv
5. Later: python create_visualizations.py (regenerate plots anytime)


═════════════════════════════════════════════════════════════════════════════
For detailed documentation, see:
  - README.md (full documentation)
  - RUN_SCRIPT_GUIDE.md (script guide)
  - START_HERE.md (quick start)
═════════════════════════════════════════════════════════════════════════════
""")

