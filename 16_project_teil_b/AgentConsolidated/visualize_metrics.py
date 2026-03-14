#!/usr/bin/env python3
"""
Local Metrics Visualization and Comparison
Load metrics from pickle files and generate comparison plots
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob


class MetricsVisualizer:
    """Load and visualize metrics from pickle files"""
    
    def __init__(self, results_base_dir: str = "./results"):
        self.results_base_dir = results_base_dir
        self.experiments = {}
        self.metrics_df = None
    
    def load_experiment(self, experiment_name: str, metrics_path: str):
        """Load a single experiment's metrics"""
        if not os.path.exists(metrics_path):
            print(f"⚠ File not found: {metrics_path}")
            return False
        
        try:
            with open(metrics_path, 'rb') as f:
                data = pickle.load(f)
            self.experiments[experiment_name] = data
            print(f"✓ Loaded: {experiment_name}")
            return True
        except Exception as e:
            print(f"✗ Error loading {experiment_name}: {e}")
            return False
    
    def load_all_from_directory(self, base_dir: str = "./results"):
        """Auto-load all metrics.pkl files from subdirectories"""
        pkl_files = glob.glob(f"{base_dir}/**/metrics.pkl", recursive=True)
        
        for pkl_file in pkl_files:
            # Extract experiment name from path
            rel_path = os.path.relpath(pkl_file, base_dir)
            exp_name = os.path.dirname(rel_path)
            if not exp_name:
                exp_name = "experiment_0"
            
            self.load_experiment(exp_name, pkl_file)
        
        if self.experiments:
            print(f"\n✓ Loaded {len(self.experiments)} experiments")
        else:
            print(f"\n⚠ No experiments found in {base_dir}")
    
    def create_comparison_dataframe(self) -> pd.DataFrame:
        """Create a comparison DataFrame from all experiments"""
        rows = []
        for exp_name, data in self.experiments.items():
            row = {
                'Experiment': exp_name,
                'Forecast Mode': data.get('forecast_mode', 'N/A'),
                'Reward Type': data.get('reward_type', 'N/A'),
            }
            
            if 'metrics' in data:
                metrics = data['metrics']
                row.update({
                    'Total Return (%)': metrics.get('total_return', 0) * 100,
                    'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                    'Max Drawdown (%)': metrics.get('max_drawdown', 0) * 100,
                    'Volatility (%)': metrics.get('volatility', 0) * 100,
                    'Annualized Return (%)': metrics.get('annualized_return', 0) * 100,
                    'Annualized Volatility (%)': metrics.get('annualized_volatility', 0) * 100,
                    'Calmar Ratio': metrics.get('calmar_ratio', 0),
                    'Sortino Ratio': metrics.get('sortino_ratio', 0),
                    'Win Rate (%)': metrics.get('win_rate', 0) * 100,
                    'Profit Factor': metrics.get('profit_factor', 0),
                    'Turnover': metrics.get('turnover', 0),
                    'Total Costs': data.get('config', {}).get('initial_equity', 100000) * metrics.get('cost_ratio', 0),
                    'Cost Ratio': metrics.get('cost_ratio', 0),
                })
            
            if 'tracker_summary' in data:
                summary = data['tracker_summary']
                row['Final Equity'] = summary.get('final_equity', 0)
            
            rows.append(row)
        
        self.metrics_df = pd.DataFrame(rows)
        return self.metrics_df
    
    def print_comparison_table(self):
        """Print comparison table"""
        if self.metrics_df is None:
            self.create_comparison_dataframe()
        
        print("\n" + "="*150)
        print("EXPERIMENTS COMPARISON TABLE")
        print("="*150)
        print(self.metrics_df.to_string())
        print("="*150 + "\n")
    
    def plot_metrics_comparison(self, metrics_to_plot=None, save_dir: str = "./plots"):
        """Create comparison plots"""
        if self.metrics_df is None:
            self.create_comparison_dataframe()
        
        if metrics_to_plot is None:
            metrics_to_plot = [
                'Total Return (%)',
                'Sharpe Ratio',
                'Max Drawdown (%)',
                'Volatility (%)',
                'Turnover'
            ]
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Filter to available metrics
        available_metrics = [m for m in metrics_to_plot if m in self.metrics_df.columns]
        
        if not available_metrics:
            print("⚠ No metrics available for plotting")
            return
        
        # Create comparison bar plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(available_metrics[:6]):
            ax = axes[idx]
            
            data = self.metrics_df.sort_values(metric, ascending=False)
            colors = ['green' if x > 0 else 'red' for x in data[metric]]
            
            ax.barh(data['Experiment'], data[metric], color=colors, alpha=0.7)
            ax.set_xlabel(metric, fontsize=11, fontweight='bold')
            ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, v in enumerate(data[metric]):
                ax.text(v, i, f' {v:.2f}', va='center', fontsize=9)
        
        # Hide unused subplots
        for idx in range(len(available_metrics), 6):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, '01_metrics_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_equity_curves(self, save_dir: str = "./plots"):
        """Plot equity curves for all experiments"""
        os.makedirs(save_dir, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        for exp_name, data in self.experiments.items():
            if 'equity_curve' in data:
                equity = np.array(data['equity_curve'])
                initial = data['config'].get('initial_equity', 100000)
                
                # Normalize to percentage return
                returns = ((equity - initial) / initial) * 100
                
                ax.plot(returns, label=exp_name, linewidth=2, alpha=0.8)
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Return (%)', fontsize=12, fontweight='bold')
        ax.set_title('Equity Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        save_path = os.path.join(save_dir, '02_equity_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_returns_distribution(self, save_dir: str = "./plots"):
        """Plot daily returns distribution"""
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, len(self.experiments), figsize=(15, 5),
                                sharex=True, sharey=True)
        
        if len(self.experiments) == 1:
            axes = [axes]
        
        for idx, (exp_name, data) in enumerate(self.experiments.items()):
            if 'daily_returns' in data:
                returns = np.array(data['daily_returns']) * 100
                
                axes[idx].hist(returns, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
                axes[idx].set_title(f'{exp_name}', fontsize=11, fontweight='bold')
                axes[idx].axvline(x=np.mean(returns), color='red', linestyle='--', 
                                 linewidth=2, label=f'Mean: {np.mean(returns):.2f}%')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3, axis='y')
        
        axes[0].set_xlabel('Daily Return (%)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, '03_returns_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_metrics_heatmap(self, save_dir: str = "./plots"):
        """Create heatmap of all metrics"""
        os.makedirs(save_dir, exist_ok=True)
        
        if self.metrics_df is None:
            self.create_comparison_dataframe()
        
        # Select numeric columns
        numeric_cols = self.metrics_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print("⚠ No numeric columns for heatmap")
            return
        
        # Normalize to [-1, 1] range for better visualization
        data_for_heatmap = self.metrics_df[numeric_cols].copy()
        for col in data_for_heatmap.columns:
            max_val = data_for_heatmap[col].abs().max()
            if max_val > 0:
                data_for_heatmap[col] = data_for_heatmap[col] / max_val
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        sns.heatmap(data_for_heatmap.T, annot=True, fmt='.2f', cmap='RdYlGn',
                   center=0, cbar_kws={'label': 'Normalized Value'},
                   xticklabels=self.metrics_df['Experiment'], ax=ax)
        
        ax.set_title('Metrics Heatmap (Normalized)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, '04_metrics_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def generate_report(self, output_file: str = "metrics_report.txt"):
        """Generate a text report"""
        if self.metrics_df is None:
            self.create_comparison_dataframe()
        
        with open(output_file, 'w') as f:
            f.write("="*150 + "\n")
            f.write("TRADING EXPERIMENTS COMPARISON REPORT\n")
            f.write("="*150 + "\n\n")
            
            f.write(self.metrics_df.to_string() + "\n\n")
            
            f.write("="*150 + "\n")
            f.write("DETAILED METRICS\n")
            f.write("="*150 + "\n\n")
            
            for exp_name, data in self.experiments.items():
                f.write(f"\n{'='*100}\n")
                f.write(f"EXPERIMENT: {exp_name}\n")
                f.write(f"{'='*100}\n")
                
                f.write(f"Forecast Mode: {data.get('forecast_mode', 'N/A')}\n")
                f.write(f"Reward Type: {data.get('reward_type', 'N/A')}\n")
                f.write(f"Timestamp: {data.get('timestamp', 'N/A')}\n\n")
                
                if 'metrics' in data:
                    f.write("Performance Metrics:\n")
                    f.write("-" * 50 + "\n")
                    for key, value in data['metrics'].items():
                        if isinstance(value, float):
                            f.write(f"  {key:30s}: {value:>15.4f}\n")
                        else:
                            f.write(f"  {key:30s}: {value:>15}\n")
                
                if 'config' in data:
                    f.write("\nConfiguration:\n")
                    f.write("-" * 50 + "\n")
                    for key, value in data['config'].items():
                        f.write(f"  {key:30s}: {value}\n")
        
        print(f"✓ Report saved to: {output_file}\n")
    
    def create_all_visualizations(self, plots_dir: str = "./plots"):
        """Create all visualizations"""
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70 + "\n")
        
        self.create_comparison_dataframe()
        self.print_comparison_table()
        
        self.plot_metrics_comparison(save_dir=plots_dir)
        self.plot_equity_curves(save_dir=plots_dir)
        self.plot_returns_distribution(save_dir=plots_dir)
        self.plot_metrics_heatmap(save_dir=plots_dir)
        self.generate_report(output_file=os.path.join(plots_dir, 'metrics_report.txt'))
        
        print("\n" + "="*70)
        print(f"✓ All visualizations saved to: {plots_dir}/")
        print("="*70 + "\n")


def main():
    """Main function"""
    import sys
    
    print("\n" + "="*70)
    print("LOCAL METRICS VISUALIZATION TOOL")
    print("="*70 + "\n")
    
    # Check if custom directory provided
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "./results"
    plots_dir = sys.argv[2] if len(sys.argv) > 2 else "./plots"
    
    print(f"Results directory: {results_dir}")
    print(f"Plots directory: {plots_dir}\n")
    
    # Load experiments
    visualizer = MetricsVisualizer()
    visualizer.load_all_from_directory(results_dir)
    
    if not visualizer.experiments:
        print("❌ No experiments found!")
        sys.exit(1)
    
    # Generate visualizations
    visualizer.create_all_visualizations(plots_dir)


if __name__ == "__main__":
    main()

