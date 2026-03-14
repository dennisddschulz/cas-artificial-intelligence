#!/usr/bin/env python3
"""
COMPLETE WORKFLOW SCRIPT
Run this to execute everything:
    1. Run all experiments
    2. Generate comprehensive analysis
    3. Create all plots
    
Usage: python3 complete_workflow.py
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and report status"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"✗ Error running: {cmd}")
        return False
    return True

def main():
    print("\n" + "="*100)
    print("FORECAST-AUGMENTED RL TRADING SYSTEM - COMPLETE WORKFLOW")
    print("="*100)
    
    # Create directories
    Path('plots').mkdir(exist_ok=True)
    
    # Step 1: Install dependencies
    print("\nStep 1: Installing dependencies...")
    run_command(
        "pip install -q numpy pandas torch gymnasium stable-baselines3 yfinance scikit-learn wandb matplotlib seaborn 2>/dev/null",
        "Installing Python packages"
    )
    
    # Step 2: Run experiments
    if not run_command(
        "python3 main.py",
        "Step 2: Running all experiments (This takes 60-90 minutes)"
    ):
        print("✗ Experiments failed!")
        sys.exit(1)
    
    # Step 3: Generate analysis and plots
    if not run_command(
        "python3 analyze_results.py",
        "Step 3: Generating comprehensive analysis and plots"
    ):
        print("✗ Analysis failed!")
        sys.exit(1)
    
    # Step 4: Summary
    print("\n" + "="*100)
    print("✓ COMPLETE WORKFLOW FINISHED SUCCESSFULLY!")
    print("="*100)
    
    print("\n📊 OUTPUT FILES GENERATED:")
    print("\n1. Results Data:")
    print("   - results_comparison.csv (comparison table)")
    print("   - experiment_results.json (complete results)")
    print("   - ./wandb/ (W&B offline logs)")
    
    print("\n2. Visualizations (in plots/ directory):")
    print("   - comprehensive_analysis.png (main dashboard)")
    print("   - 01_performance_comparison.png")
    print("   - 06_model_summary.png")
    print("   - 10_returns_analysis.png")
    print("   - 11_risk_metrics.png")
    
    print("\n3. Analysis Ready For Report:")
    print("   - All metrics calculated")
    print("   - All comparisons generated")
    print("   - Research question answered")
    print("   - Visualizations ready for technical report")
    
    print("\n" + "="*100)
    print("NEXT STEPS:")
    print("="*100)
    print("\n1. Review the comprehensive_analysis.png dashboard")
    print("2. Check results_comparison.csv for detailed metrics")
    print("3. Use experiment_results.json for report generation")
    print("4. All visualizations are in the plots/ folder")
    print("\nEstimated total runtime: 60-90 minutes")
    print("="*100 + "\n")

if __name__ == "__main__":
    main()

