#!/usr/bin/env python3
"""
EXECUTION GUIDE - Step-by-Step Instructions
Run this file to get interactive execution guidance
"""

import os
import sys

def print_header(title):
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")

def print_section(title):
    print("\n" + "-"*80)
    print(f"  {title}")
    print("-"*80 + "\n")

def main():
    print_header("PPO TRADING EXPERIMENTS - EXECUTION GUIDE")
    
    print("""
This framework runs 6 comprehensive PPO trading experiments:
  1. PPO Without Forecast (Baseline)
  2. PPO With Forecast
  3. PPO Basic Reward
  4. PPO With Risk
  5. PPO With Sharpe
  6. PPO Risk-Adjusted

Each experiment includes:
  ✓ Real-time console metrics (every 100 updates)
  ✓ Budget tracking from initial $100K
  ✓ Transaction costs monitoring
  ✓ Detailed local plots
  ✓ WandB logging with distinct groups
""")
    
    print_section("STEP 1: VALIDATE FRAMEWORK")
    
    print("Run the validation script to check setup:")
    print("  python validate_framework.py")
    print("""
This checks:
  ✓ Python version (3.7+)
  ✓ All required packages installed
  ✓ GPU availability
  ✓ File structure
  ✓ 6 experiments configured
  ✓ All imports work
    """)
    
    print_section("STEP 2: RUN ALL 6 EXPERIMENTS")
    
    print("Execute the master script:")
    print("  python run_all_experiments.py")
    print("""
This will:
  1. Run PPO Without Forecast (15-30 min)
  2. Run PPO With Forecast (20-35 min, includes LSTM training)
  3. Run PPO Basic Reward (15-30 min)
  4. Run PPO With Risk (15-30 min)
  5. Run PPO With Sharpe (15-30 min)
  6. Run PPO Risk-Adjusted (15-30 min)
  
  Total time: 2-3 hours depending on hardware
  
Real-time metrics will print every 100 updates showing:
  • Episode returns (last 100 episodes)
  • Policy, Value, Entropy losses
  • Exploration level
  • Training health metrics
    """)
    
    print_section("STEP 3: MONITOR DURING EXECUTION")
    
    print("""
While experiments run, you'll see:

================================================================================
UPDATE   100 /  3000
================================================================================
Episode Returns (last 100): 123.45 ± 56.78
Total Episodes Trained:          1200
Log Std (exploration):         -0.523
Policy Loss:                   0.0234
Value Loss:                    0.1567
Entropy Loss:                  0.0432
Total Loss:                    0.3421
Approx KL:                     0.0125
================================================================================

• Episode Returns: Average cumulative return from last 100 episodes
• Log Std: Exploration variance (lower = more deterministic)
• Losses: Training metrics for policy, value, and entropy
    """)
    
    print_section("STEP 4: INSPECT METRICS LOCALLY")
    
    print("After experiments complete, run:")
    print("  python load_and_inspect_metrics.py")
    print("""
This script:
  ✓ Loads all metrics.pkl files
  ✓ Prints summary for each experiment
  ✓ Creates comparison table
  ✓ Generates equity curve plots
  ✓ Saves CSV comparison

Output includes:
  • Individual experiment metrics
  • Comparison table (Return, Sharpe, Drawdown, etc.)
  • Equity curves overlaid
  • Returns distribution histograms
    """)
    
    print_section("STEP 5: VIEW GENERATED PLOTS")
    
    print("Check the ./plots/ directory:")
    print("  01_metrics_comparison.png - All experiments compared")
    print("  02_equity_curves.png - All equity curves overlaid")
    print("  03_returns_distribution.png - Daily returns histograms")
    print("  04_metrics_heatmap.png - Color-coded metrics table")
    print("  metrics_report.txt - Text comparison report")
    print("""
These plots show:
  • Which experiment performed best
  • Risk-adjusted returns (Sharpe, Calmar)
  • Budget tracking from $100K initial
  • Transaction costs impact
  • Equity evolution over time
    """)
    
    print_section("STEP 6: REVIEW DETAILED RESULTS")
    
    print("Each experiment folder (./results/EXPERIMENT_NAME/) contains:")
    print("  metrics.pkl - Load with Python pickle")
    print("  metrics_summary.csv - Spreadsheet format")
    print("  budget_report.csv - Detailed tracking")
    print("  budget_summary.txt - Text summary")
    print("  01_equity_curve.png - Equity from $100K")
    print("  02_budget_breakdown.png - Position vs equity")
    print("  03_transaction_costs.png - Daily and cumulative costs")
    print("  04_returns_drawdown.png - Returns with drawdown")
    
    print_section("STEP 7 (OPTIONAL): SYNC TO WANDB CLOUD")
    
    print("Upload offline runs to WandB online dashboard:")
    print("  wandb sync ./wandb/offline-run-*/")
    print("""
Then:
  1. Go to wandb.ai
  2. Filter by group to compare experiments:
     - ppo_baseline_no_forecast
     - ppo_baseline_with_forecast
     - ppo_basic_reward
     - ppo_with_risk
     - ppo_with_sharpe
     - ppo_risk_adjusted
     
  3. Use WandB tools to:
     - Compare metrics across experiments
     - Plot training curves
     - Export data for analysis
    """)
    
    print_section("CUSTOMIZE & RE-RUN")
    
    print("""
To modify settings:
  1. Edit trading_config.py
  2. Change hyperparameters in get_all_experiments()
  3. Run python run_all_experiments.py again

Common modifications:
  • Shorter training: total_updates = 100 (testing)
  • Fewer envs: num_envs = 2 (CPU-only systems)
  • Higher fees: fee = 0.0005 (different market)
  • Different dates: start_date = "2023-01-01"
    """)
    
    print_section("EXPECTED RESULTS")
    
    print("""
Example output (varies with market conditions):
    
    Experiment              Return    Sharpe    Drawdown  Volatility
    ─────────────────────────────────────────────────────────────
    PPO Without Forecast    3.43%    -0.0218   -13.90%     0.54%
    PPO With Forecast       5.12%     0.1234   -11.20%     0.62%
    PPO Basic Reward        2.15%    -0.1567   -15.40%     0.48%
    PPO With Risk           4.56%     0.0789   -12.80%     0.58%
    PPO With Sharpe         6.23%     0.2341    -9.60%     0.71%
    PPO Risk-Adjusted       5.89%     0.1956   -10.20%     0.65%
    
Note: Returns vary based on market conditions and RL randomness
      Focus on Sharpe Ratio for risk-adjusted performance
    """)
    
    print_section("TROUBLESHOOTING")
    
    print("""
Issue: validate_framework.py fails
  → Check Python version: python --version
  → Install packages: pip install -r requirements.txt
  
Issue: run_all_experiments.py hangs
  → Check CPU/GPU usage: nvidia-smi or top
  → Reduce num_envs if memory error
  
Issue: No metrics.pkl files created
  → Check ./results/ directory
  → Review experiment output for errors
  → May need longer training time
  
Issue: Negative returns everywhere
  → Normal if market is declining
  → Check Sharpe/Calmar ratios for risk-adjusted performance
  → Try different reward functions
  
Issue: WandB metrics not appearing
  → Check ./wandb/offline-run-*/ exists
  → Run: wandb sync ./wandb/offline-run-*/
  → Check WandB initialization in console output
    """)
    
    print_section("KEY FILES & THEIR PURPOSE")
    
    print("""
Framework Files:
  trading_config.py - 6 experiment configurations with parameters
  trading_framework.py - PPO algorithm implementation
  trading_metrics.py - Metrics calculation (Sharpe, Drawdown, etc.)
  budget_tracker.py - Budget and equity tracking over time
  
Execution Files:
  run_all_experiments.py - Master script (run this!)
  validate_framework.py - Setup validation
  load_and_inspect_metrics.py - Metric inspection tool
  visualize_metrics.py - Advanced visualization generation
  
Output Files (automatically generated):
  ./results/EXPERIMENT_NAME/metrics.pkl - Full metrics
  ./results/EXPERIMENT_NAME/*.png - Experiment plots
  ./results/EXPERIMENT_NAME/*.csv - Data in spreadsheet format
  ./plots/ - Aggregated comparison plots
  ./wandb/ - WandB offline runs
    """)
    
    print_section("QUICK COMMANDS SUMMARY")
    
    print("""
# Validate setup
python validate_framework.py

# Run all 6 experiments
python run_all_experiments.py

# Inspect metrics after completion
python load_and_inspect_metrics.py

# Sync to WandB cloud (optional)
wandb sync ./wandb/offline-run-*/

# View results
cat ./plots/metrics_report.txt
ls ./plots/*.png

# Load metrics in Python
import pickle
with open('./results/PPO-Without-Forecast_XXXXX/metrics.pkl', 'rb') as f:
    data = pickle.load(f)
print(data['metrics'])
    """)
    
    print_header("YOU'RE READY TO START!")
    
    print("""
Next action:
  
  1. Run validation:
     python validate_framework.py
  
  2. Run all experiments:
     python run_all_experiments.py
  
  3. Wait for completion (2-3 hours)
  
  4. Inspect results:
     python load_and_inspect_metrics.py
  
Good luck! 🚀
    """)


if __name__ == "__main__":
    main()

