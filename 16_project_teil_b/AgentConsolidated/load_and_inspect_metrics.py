#!/usr/bin/env python3
"""
Example: Load and Inspect Metrics from Pickle Files

This script demonstrates how to:
1. Load metrics.pkl files from completed experiments
2. Inspect individual metrics
3. Compare experiments locally
4. Generate custom visualizations
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


def load_experiment_metrics(pkl_path: str) -> dict:
    """Load metrics from pickle file"""
    if not os.path.exists(pkl_path):
        print(f"❌ File not found: {pkl_path}")
        return None
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Loaded metrics from: {pkl_path}")
        return data
    except Exception as e:
        print(f"❌ Error loading pickle file: {e}")
        return None


def print_metrics_summary(data: dict):
    """Print a summary of metrics"""
    if data is None:
        return
    
    print("\n" + "="*70)
    print("METRICS SUMMARY")
    print("="*70)
    
    print(f"\nExperiment: {data.get('experiment_name', 'Unknown')}")
    print(f"Forecast Mode: {data.get('forecast_mode', 'Unknown')}")
    print(f"Reward Type: {data.get('reward_type', 'Unknown')}")
    print(f"Timestamp: {data.get('timestamp', 'Unknown')}")
    
    if 'config' in data:
        print(f"\nConfiguration:")
        print(f"  Initial Equity: ${data['config'].get('initial_equity', 0):,.2f}")
        print(f"  Fee: {data['config'].get('fee', 0)}")
        print(f"  Kappa: {data['config'].get('kappa', 0)}")
        print(f"  Leverage Max: {data['config'].get('leverage_max', 0)}")
        print(f"  Total PPO Updates: {data['config'].get('total_updates', 0)}")
    
    if 'metrics' in data:
        print(f"\nPerformance Metrics:")
        print("-" * 70)
        metrics = data['metrics']
        
        # Key metrics with better formatting
        print(f"  Total Return:          {metrics.get('total_return', 0)*100:>10.2f}%")
        print(f"  Annualized Return:     {metrics.get('annualized_return', 0)*100:>10.2f}%")
        print(f"  Sharpe Ratio:          {metrics.get('sharpe_ratio', 0):>10.4f}")
        print(f"  Sortino Ratio:         {metrics.get('sortino_ratio', 0):>10.4f}")
        print(f"  Calmar Ratio:          {metrics.get('calmar_ratio', 0):>10.4f}")
        print(f"  Max Drawdown:          {metrics.get('max_drawdown', 0)*100:>10.2f}%")
        print(f"  Volatility:            {metrics.get('volatility', 0)*100:>10.2f}%")
        print(f"  Annualized Volatility: {metrics.get('annualized_volatility', 0)*100:>10.2f}%")
        print(f"  Win Rate:              {metrics.get('win_rate', 0)*100:>10.2f}%")
        print(f"  Profit Factor:         {metrics.get('profit_factor', 0):>10.4f}")
        print(f"  Turnover:              {metrics.get('turnover', 0):>10.4f}")
        print(f"  Cost Ratio:            {metrics.get('cost_ratio', 0):>10.4f}")
        print(f"  Kurtosis:              {metrics.get('kurtosis', 0):>10.4f}")
        print(f"  Skewness:              {metrics.get('skewness', 0):>10.4f}")
    
    if 'tracker_summary' in data:
        print(f"\nBudget Summary:")
        print("-" * 70)
        summary = data['tracker_summary']
        print(f"  Initial Equity:        ${summary.get('initial_equity', 0):>10,.2f}")
        print(f"  Final Equity:          ${summary.get('final_equity', 0):>10,.2f}")
        print(f"  Total Costs:           ${summary.get('total_costs', 0):>10,.2f}")
        print(f"  Avg Daily PnL:         ${summary.get('avg_daily_pnl', 0):>10,.2f}")
        print(f"  Avg Position Size:     {summary.get('avg_position', 0):>10.4f}")
        print(f"  Number of Steps:       {summary.get('num_steps', 0):>10d}")
    
    print("\n" + "="*70 + "\n")


def plot_equity_curve(data: dict, title: str = "Equity Curve"):
    """Plot equity curve"""
    if data is None or 'equity_curve' not in data:
        print("❌ No equity curve data available")
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    equity = np.array(data['equity_curve'])
    initial = data['config'].get('initial_equity', 100000)
    returns = ((equity - initial) / initial) * 100
    
    ax.plot(returns, linewidth=2, color='#2E86AB', label='Equity Curve')
    ax.fill_between(range(len(returns)), 0, returns, 
                    where=(returns >= 0), alpha=0.2, color='green', label='Profit')
    ax.fill_between(range(len(returns)), 0, returns, 
                    where=(returns < 0), alpha=0.2, color='red', label='Loss')
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Return (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def compare_metrics(data_list: list):
    """Compare metrics from multiple experiments"""
    comparison_data = []
    
    for data in data_list:
        if data is None:
            continue
        
        row = {
            'Experiment': data.get('experiment_name', 'Unknown'),
            'Forecast': data.get('forecast_mode', 'N/A'),
            'Reward': data.get('reward_type', 'N/A'),
        }
        
        if 'metrics' in data:
            metrics = data['metrics']
            row['Return (%)'] = metrics.get('total_return', 0) * 100
            row['Sharpe'] = metrics.get('sharpe_ratio', 0)
            row['Drawdown (%)'] = metrics.get('max_drawdown', 0) * 100
            row['Volatility (%)'] = metrics.get('volatility', 0) * 100
            row['Win Rate (%)'] = metrics.get('win_rate', 0) * 100
            row['Turnover'] = metrics.get('turnover', 0)
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    print("\n" + "="*100)
    print("EXPERIMENTS COMPARISON TABLE")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100 + "\n")
    
    return df


def main():
    """Main example function"""
    print("\n" + "="*70)
    print("METRICS LOADING & INSPECTION EXAMPLE")
    print("="*70 + "\n")
    
    # Find all metrics.pkl files
    results_dir = "./results"
    pkl_files = list(Path(results_dir).glob("**/metrics.pkl"))
    
    if not pkl_files:
        print(f"❌ No metrics.pkl files found in {results_dir}")
        print("\nPlace completed experiment results in:")
        print(f"  {results_dir}/EXPERIMENT_NAME/metrics.pkl")
        return
    
    print(f"Found {len(pkl_files)} experiment(s):\n")
    
    # Load all experiments
    experiments = []
    for pkl_file in pkl_files:
        print(f"Loading: {pkl_file}")
        data = load_experiment_metrics(str(pkl_file))
        experiments.append(data)
    
    print(f"\n✓ Loaded {len(experiments)} experiments\n")
    
    # Print individual summaries
    for data in experiments:
        print_metrics_summary(data)
    
    # Compare experiments if more than one
    if len(experiments) > 1:
        comparison_df = compare_metrics(experiments)
        
        # Optional: Save comparison to CSV
        comparison_df.to_csv("./results/experiments_comparison.csv", index=False)
        print(f"✓ Comparison saved to: ./results/experiments_comparison.csv\n")
    
    # Plot equity curves
    print("Generating plots...")
    fig, ax = plt.subplots(figsize=(15, 8))
    
    for idx, data in enumerate(experiments):
        if data and 'equity_curve' in data:
            equity = np.array(data['equity_curve'])
            initial = data['config'].get('initial_equity', 100000)
            returns = ((equity - initial) / initial) * 100
            
            label = f"{data.get('experiment_name', f'Exp {idx}')} ({data.get('forecast_mode', 'N/A')})"
            ax.plot(returns, linewidth=2, label=label, alpha=0.8)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Return (%)', fontsize=12, fontweight='bold')
    ax.set_title('Equity Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("./results/equity_curves_comparison.png", dpi=300, bbox_inches='tight')
    print("✓ Plot saved: ./results/equity_curves_comparison.png\n")
    
    print("✓ Example completed!")
    print("\nNext steps:")
    print("  1. Review metrics in: ./results/")
    print("  2. Check generated plots")
    print("  3. Modify config and re-run experiments")
    print("  4. Use visualize_metrics.py for advanced analysis")


if __name__ == "__main__":
    main()

