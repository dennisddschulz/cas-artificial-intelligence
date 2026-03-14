#!/usr/bin/env python3
"""
create_visualizations_extended.py

Generate professional trading analysis visualizations with reward function ablation support.
Creates high-quality PNG plots for presentation and reports, including reward comparisons.

Usage:
    python create_visualizations_extended.py --metrics metrics.pkl --output_dir ./plots
    python create_visualizations_extended.py --reward_csv reward_ablation_comparison.csv --output_dir ./plots
"""

import argparse
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


class RewardVisualizationGenerator:
    """Generate visualizations for reward function ablation studies"""
    
    def __init__(self, output_dir='./plots'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def load_reward_comparison(self, csv_path):
        """Load reward comparison CSV"""
        return pd.read_csv(csv_path, index_col=0)
    
    def create_reward_metrics_comparison(self, df, output_path=None):
        """Create side-by-side comparison of reward functions across key metrics"""
        if output_path is None:
            output_path = self.output_dir / 'reward_metrics_comparison.png'
        
        # Select key metrics
        key_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'annualized_volatility']
        available_metrics = [m for m in key_metrics if m in df.columns]
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Reward Function Ablation - Key Metrics Comparison', 
                     fontsize=16, fontweight='bold')
        
        axes_flat = axes.flatten()
        colors = plt.cm.tab10(np.linspace(0, 1, len(df)))
        
        for idx, metric in enumerate(available_metrics):
            ax = axes_flat[idx]
            data = df[metric].sort_values(ascending=False)
            
            bars = ax.barh(range(len(data)), data.values, color=colors)
            ax.set_yticks(range(len(data)))
            ax.set_yticklabels([x.replace('PPO_', '').replace('_', ' ').title() for x in data.index])
            ax.set_xlabel('Value')
            
            # Format title
            metric_display = metric.replace('_', ' ').title()
            if 'return' in metric.lower():
                ax.set_title(f'{metric_display} (%)', fontweight='bold')
                # Add percentage signs
                for i, v in enumerate(data.values):
                    ax.text(v, i, f' {v*100:.1f}%', va='center', fontsize=9)
            else:
                ax.set_title(f'{metric_display}', fontweight='bold')
                # Add value labels
                for i, v in enumerate(data.values):
                    ax.text(v, i, f' {v:.3f}', va='center', fontsize=9)
            
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
        return output_path
    
    def create_reward_heatmap(self, df, output_path=None):
        """Create heatmap of normalized metrics across rewards"""
        if output_path is None:
            output_path = self.output_dir / 'reward_heatmap.png'
        
        # Normalize metrics
        df_norm = df.copy()
        for col in df_norm.columns:
            col_min = df_norm[col].min()
            col_max = df_norm[col].max()
            if col_max - col_min != 0:
                df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        sns.heatmap(df_norm.T, annot=df.T.round(3), fmt='g', cmap='RdYlGn',
                   cbar_kws={'label': 'Normalized Score'}, ax=ax, linewidths=0.5)
        
        ax.set_title('Reward Function Ablation - Normalized Performance Heatmap\n(Color: Normalized, Values: Actual)',
                    fontweight='bold', fontsize=14)
        ax.set_xlabel('Reward Function')
        ax.set_ylabel('Metric')
        
        # Improve labels
        ax.set_xticklabels([x.get_text().replace('PPO_', '').replace('_', ' ').title() 
                           for x in ax.get_xticklabels()], rotation=45, ha='right')
        ax.set_yticklabels([y.get_text().replace('_', ' ').title() 
                           for y in ax.get_yticklabels()], rotation=0)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
        return output_path
    
    def create_reward_scatter(self, df, output_path=None):
        """Create scatter plot: Return vs Sharpe vs Volatility"""
        if output_path is None:
            output_path = self.output_dir / 'reward_scatter.png'
        
        if not all(m in df.columns for m in ['total_return', 'sharpe_ratio', 'annualized_volatility']):
            print("⚠ Missing required columns for scatter plot")
            return None
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create scatter with bubble size as volatility
        scatter = ax.scatter(
            df['total_return'] * 100,
            df['sharpe_ratio'],
            s=df['annualized_volatility'] * 500,  # Size represents volatility
            c=range(len(df)),
            cmap='tab10',
            alpha=0.6,
            edgecolors='black',
            linewidth=2
        )
        
        # Add labels for each point
        for idx, row in df.iterrows():
            label = idx.replace('PPO_', '').replace('_', ' ').title()
            ax.annotate(label, (row['total_return']*100, row['sharpe_ratio']),
                       fontsize=10, fontweight='bold',
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Total Return (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
        ax.set_title('Reward Function Comparison: Risk vs Return\n(Bubble size = Volatility)',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add risk-free zone reference
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Target Sharpe=1.0')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
        return output_path
    
    def create_reward_ranking(self, df, output_path=None):
        """Create ranking visualization for each metric"""
        if output_path is None:
            output_path = self.output_dir / 'reward_ranking.png'
        
        n_metrics = min(6, len(df.columns))
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows))
        fig.suptitle('Reward Function Rankings by Metric', fontsize=16, fontweight='bold')
        
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        metrics = df.columns[:n_metrics]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            data = df[metric].sort_values(ascending=False)
            
            # Color by rank
            colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(data)))
            
            bars = ax.barh(range(len(data)), data.values, color=colors, edgecolor='black', linewidth=1.5)
            ax.set_yticks(range(len(data)))
            ax.set_yticklabels([x.replace('PPO_', '').replace('_', ' ').title() for x in data.index])
            
            # Add value and rank
            for i, (idx_val, val) in enumerate(data.items()):
                rank_symbol = ['🥇', '🥈', '🥉'] + [''] * (len(data) - 3)
                medal = rank_symbol[i] if i < len(rank_symbol) else ''
                ax.text(val, i, f' {medal} {val:.4f}', va='center', fontweight='bold', fontsize=9)
            
            ax.set_xlabel('Value')
            ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
        return output_path
    
    def create_baseline_vs_reward_comparison(self, baseline_metrics, reward_df, output_path=None):
        """Compare baseline (WITH_RISK) to other reward functions"""
        if output_path is None:
            output_path = self.output_dir / 'baseline_vs_rewards.png'
        
        if 'PPO_with_risk' not in reward_df.index:
            print("⚠ Baseline (PPO_with_risk) not found in reward comparison")
            return None
        
        baseline = reward_df.loc['PPO_with_risk']
        
        # Calculate differences
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Baseline (WITH_RISK) vs Other Reward Functions',
                    fontsize=16, fontweight='bold')
        
        metrics_to_compare = ['total_return', 'sharpe_ratio', 'max_drawdown', 'annualized_volatility']
        available_metrics = [m for m in metrics_to_compare if m in reward_df.columns]
        
        for idx, metric in enumerate(available_metrics[:4]):
            ax = axes.flatten()[idx]
            
            # Calculate differences from baseline
            differences = (reward_df[metric] - baseline[metric])
            differences = differences.sort_values(ascending=False)
            
            # Color: positive = green, negative = red
            colors = ['green' if x > 0 else 'red' for x in differences.values]
            
            bars = ax.barh(range(len(differences)), differences.values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.set_yticks(range(len(differences)))
            ax.set_yticklabels([x.replace('PPO_', '').replace('_', ' ').title() for x in differences.index])
            
            # Add baseline reference line
            ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Baseline')
            
            ax.set_xlabel(f'Difference from Baseline')
            ax.set_title(f'{metric.replace("_", " ").title()} vs Baseline', fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
        return output_path


def create_comparison_chart_extended(metrics_dict, output_path='./plots/performance_comparison.png'):
    """
    Create extended comparison chart: All reward functions + baselines
    
    Parameters
    ----------
    metrics_dict : dict
        Dictionary containing metrics for different experiments
    output_path : str
        Where to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Performance Comparison: All Experiments',
                 fontsize=16, fontweight='bold', y=0.995)

    # Prepare data from metrics_dict
    exp_names = []
    returns = []
    sharpe = []
    drawdown = []
    volatility = []
    
    for exp_key, metrics in metrics_dict.items():
        if isinstance(metrics, dict) and 'total_return' in metrics:
            exp_names.append(exp_key.replace('_', ' ').replace('PPO', '').strip())
            returns.append(metrics.get('total_return', 0) * 100)
            sharpe.append(metrics.get('sharpe_ratio', 0))
            drawdown.append(metrics.get('max_drawdown', 0) * 100)
            volatility.append(metrics.get('annualized_volatility', 0) * 100)
    
    if not returns:
        print("⚠ No valid metrics found in metrics_dict")
        return None
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(returns)))
    
    # Plot 1: Returns
    ax = axes[0, 0]
    bars = ax.bar(range(len(returns)), returns, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(exp_names, rotation=45, ha='right')
    ax.set_ylabel('Total Return (%)', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Return Comparison', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars, returns)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.2, f'{val:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Plot 2: Sharpe Ratio
    ax = axes[0, 1]
    bars = ax.bar(range(len(sharpe)), sharpe, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(exp_names, rotation=45, ha='right')
    ax.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Risk-Adjusted Returns', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, sharpe):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Plot 3: Max Drawdown
    ax = axes[1, 0]
    bars = ax.bar(range(len(drawdown)), drawdown, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(exp_names, rotation=45, ha='right')
    ax.set_ylabel('Max Drawdown (%)', fontsize=12, fontweight='bold')
    ax.set_title('Maximum Drawdown (Lower is Better)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, drawdown):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Plot 4: Volatility
    ax = axes[1, 1]
    bars = ax.bar(range(len(volatility)), volatility, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(exp_names, rotation=45, ha='right')
    ax.set_ylabel('Annual Volatility (%)', fontsize=12, fontweight='bold')
    ax.set_title('Volatility Comparison (Lower is Better)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, volatility):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.3, f'{val:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Generate trading visualizations with reward ablation support')
    parser.add_argument('--reward_csv', type=str, help='Path to reward_ablation_comparison.csv')
    parser.add_argument('--metrics_csv', type=str, help='Path to metrics_comparison.csv')
    parser.add_argument('--metrics_pkl', type=str, help='Path to metrics.pkl')
    parser.add_argument('--output_dir', type=str, default='./plots', help='Output directory for plots')
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(exist_ok=True)
    
    gen = RewardVisualizationGenerator(args.output_dir)
    
    # Load and visualize reward comparison
    if args.reward_csv:
        print(f"\nLoading reward comparison from {args.reward_csv}...")
        try:
            df_rewards = gen.load_reward_comparison(args.reward_csv)
            
            print("Generating reward visualizations...")
            gen.create_reward_metrics_comparison(df_rewards)
            gen.create_reward_heatmap(df_rewards)
            gen.create_reward_scatter(df_rewards)
            gen.create_reward_ranking(df_rewards)
            
            print("✓ Reward visualizations generated successfully!")
        except Exception as e:
            print(f"✗ Error generating reward visualizations: {e}")
    
    # Load and visualize all metrics
    if args.metrics_csv:
        print(f"\nLoading metrics comparison from {args.metrics_csv}...")
        try:
            df_metrics = gen.load_reward_comparison(args.metrics_csv)
            create_comparison_chart_extended(df_metrics.to_dict('index'), 
                                            f'{args.output_dir}/all_experiments_comparison.png')
            print("✓ Metrics visualizations generated successfully!")
        except Exception as e:
            print(f"✗ Error generating metrics visualizations: {e}")
    
    # Load and visualize from pickle
    if args.metrics_pkl:
        print(f"\nLoading metrics from {args.metrics_pkl}...")
        try:
            with open(args.metrics_pkl, 'rb') as f:
                metrics_data = pickle.load(f)
            
            if isinstance(metrics_data, dict) and 'results' in metrics_data:
                metrics_dict = {k: v.get('metrics', {}) for k, v in metrics_data['results'].items() if isinstance(v, dict)}
                create_comparison_chart_extended(metrics_dict, f'{args.output_dir}/all_experiments_comparison.png')
            
            print("✓ Pickle-based visualizations generated successfully!")
        except Exception as e:
            print(f"✗ Error loading pickle file: {e}")
    
    print(f"\n✓ All visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

