#!/usr/bin/env python3
"""
Monitor the progress of running experiments
"""

import os
import time
from pathlib import Path
from datetime import datetime

def check_status():
    """Check if experiments are running and print status"""
    
    log_file = Path("experiments_run.log")
    metrics_file = Path("metrics.pkl")
    
    print("\n" + "="*80)
    print(f"EXPERIMENT STATUS CHECK - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Check log file
    if log_file.exists():
        size = log_file.stat().st_size
        print(f"✓ Log file exists: {size:,} bytes")
        
        # Show last few lines
        with open(log_file, 'r') as f:
            lines = f.readlines()
            if lines:
                print(f"\nLast 10 lines of log:")
                print("-" * 80)
                for line in lines[-10:]:
                    print(line.rstrip())
                print("-" * 80)
    else:
        print("✗ Log file not found yet")
    
    # Check metrics file
    if metrics_file.exists():
        size = metrics_file.stat().st_size
        print(f"\n✓ metrics.pkl created: {size:,} bytes")
        print("  → Experiments are complete!")
    else:
        print(f"\n⏳ metrics.pkl not created yet")
        print("   → Experiments still running...")
    
    # Check for output plots
    plots = ["equity_curves_comparison.png", "drawdown_comparison.png", 
             "returns_distribution.png", "metrics_heatmap.png"]
    
    print(f"\nGenerated plots:")
    for plot in plots:
        if Path(plot).exists():
            print(f"  ✓ {plot}")
        else:
            print(f"  ⏳ {plot}")
    
    # Check for CSV/JSON
    if Path("metrics_comparison.csv").exists():
        print(f"\n✓ metrics_comparison.csv created")
    else:
        print(f"\n⏳ metrics_comparison.csv pending")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    check_status()
    print("\nTo check again: python check_progress.py")

