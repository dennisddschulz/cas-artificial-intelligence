#!/usr/bin/env python3
"""
Real-time Experiment Monitoring Dashboard
Run this in a separate terminal to track progress
"""

import os
import subprocess
import time
from pathlib import Path
from datetime import datetime

def get_line_count(filename):
    """Get number of lines in a file"""
    try:
        result = subprocess.run(['wc', '-l', filename], capture_output=True, text=True)
        return int(result.stdout.split()[0])
    except:
        return 0

def tail_file(filename, n=20):
    """Get last n lines of file"""
    try:
        result = subprocess.run(['tail', '-n', str(n), filename], capture_output=True, text=True)
        return result.stdout
    except:
        return "File not found\n"

def monitor():
    """Monitor experiments in real-time"""
    
    while True:
        os.system('clear')  # Clear screen
        
        print("\n" + "="*80)
        print(f"📊 PPO TRADING EXPERIMENTS - REAL-TIME MONITOR")
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
        
        # Check log file
        log_file = "experiments_run.log"
        if Path(log_file).exists():
            lines = get_line_count(log_file)
            size = Path(log_file).stat().st_size / 1024  # KB
            print(f"📝 Log File: {lines:,} lines | {size:.1f} KB")
            
            # Show last 20 lines
            print("\n📌 Latest Output:")
            print("-" * 80)
            print(tail_file(log_file, 20))
            print("-" * 80)
        else:
            print("⏳ Waiting for log file to be created...")
        
        # Check generated files
        print("\n📁 Generated Files:")
        files = {
            "metrics.pkl": "Metrics data (for visualization)",
            "metrics_comparison.csv": "Metrics table",
            "detailed_results.json": "Detailed results",
            "equity_curves_comparison.png": "Equity curves plot",
            "drawdown_comparison.png": "Drawdown analysis",
            "returns_distribution.png": "Returns distribution",
            "metrics_heatmap.png": "Metrics heatmap",
        }
        
        for filename, description in files.items():
            if Path(filename).exists():
                size = Path(filename).stat().st_size
                if size > 1024*1024:
                    size_str = f"{size/(1024*1024):.1f} MB"
                elif size > 1024:
                    size_str = f"{size/1024:.1f} KB"
                else:
                    size_str = f"{size} B"
                print(f"  ✓ {filename:35s} ({size_str})")
            else:
                print(f"  ⏳ {filename:35s}")
        
        # Status summary
        print("\n" + "="*80)
        if Path("metrics.pkl").exists():
            print("✅ EXPERIMENTS COMPLETE!")
            print("   Run: python create_visualizations.py")
            break
        else:
            print("🔄 EXPERIMENTS RUNNING...")
            print("   (Check back in a few minutes)")
            print("   Press Ctrl+C to exit monitor")
        print("="*80 + "\n")
        
        # Wait before refreshing
        time.sleep(30)

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\n\n👋 Monitor stopped\n")

