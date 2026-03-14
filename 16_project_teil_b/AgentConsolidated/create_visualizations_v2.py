#!/usr/bin/env python3
"""
STEP 2: Create Comprehensive Visualizations from Metrics

Loads metrics.pkl from all experiments and generates publication-quality plots
addressing all course requirements.

Usage:
    python create_visualizations_v2.py
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import warnings

warnings.filterwarnings('ignore')

# Professional styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ComprehensiveVisualizationGenerator:
    """Generate all required plots for presentation"""
    
    def __init__(self, results_dir: str = "./results", plots_dir: str = "./plots"):
        self.results_dir = results_dir
        self.plots_dir = plots_dir
        self.experiments = {}
        self.metrics_df = None
        os.makedirs(plots_dir, exist_ok=True)
    
    def load_all_metrics(self):
        """Load metrics.pkl from all experiment directories"""
        results_path = Path(self.results_dir)
        pkl_files = sorted(list(results_path.glob("**/metrics.pkl")))
        
        print(f"\n{'='*80}")
        print(f"LOADING METRICS FROM EXPERIMENTS")
        print(f"{'='*80}\n")
        
        if not pkl_files:
            raise ValueError(f"No metrics.pkl files found in {self.results_dir}")
        
        for pkl_file in pkl_files:
            exp_dir = pkl_file.parent.name
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                self.experiments[exp_dir] = data
                exp_name = data.get('experiment_name', 'Unknown')
                print(f"✓ {exp_name}")
            except Exception as e:
                print(f"✗ Failed: {exp_dir} - {e}")
        
        print(f"\n✓ Loaded {len(self.experiments)} experiments\n")
    
    def create_metrics_dataframe(self):
        """Create comparison dataframe from all experiments"""
        rows = []
        
        for exp_dir, data in self.experiments.items():
            forecast_mode = data.get('forecast_mode', 'unknown')
            reward_type = data.get('reward_type', 'unknown')
            
            row = {
                'Experiment': data.get('experiment_name', 'Unknown'),
                'Forecast': 'LSTM' if forecast_mode == 'lstm' else 'None',
                'Reward': reward_type.replace('_', ' ').title() if reward_type else 'Unknown',
            }
            
            if 'metrics' in data:
                m = data['metrics']
                row.update({
                    'Return (%)': m.get('total_return', 0) * 100,
                    'Sharpe': m.get('sharpe_ratio', 0),
                    'Max DD (%)': m.get('max_drawdown', 0) * 100,
                    'Volatility (%)': m.get('volatility', 0) * 100,
                    'Win Rate (%)': m.get('win_rate', 0) * 100,
                    'Turnover': m.get('turnover', 0),
                    'Sortino': m.get('sortino_ratio', 0),
                    'Calmar': m.get('calmar_ratio', 0),
                })
            
            rows.append(row)
        
        self.metrics_df = pd.DataFrame(rows).sort_values('Return (%)', ascending=False)
        return self.metrics_df
    
    def plot_1_metrics_comparison(self):
        """Bar chart comparison of key metrics across all experiments"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Experiment Metrics Comparison (All 6 Experiments)', 
                    fontsize=20, fontweight='bold', y=0.995)
        
        metrics_to_plot = [
            ('Return (%)', 'green', True),
            ('Sharpe', 'blue', True),
            ('Max DD (%)', 'red', False),
            ('Volatility (%)', 'orange', False),
            ('Win Rate (%)', 'purple', True),
            ('Turnover', 'brown', False),
        ]
        
        for idx, (metric, color, ascending) in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]
            
            if metric in self.metrics_df.columns:
                data = self.metrics_df.sort_values(metric, ascending=not ascending)
                
                bars = ax.barh(data['Experiment'], data[metric], color=color, 
                              alpha=0.75, edgecolor='black', linewidth=1.5)
                
                # Add value labels on bars
                for i, (bar, val) in enumerate(zip(bars, data[metric])):
                    ax.text(val, i, f' {val:.2f}', va='center', 
                           fontsize=10, fontweight='bold')
                
                ax.set_xlabel(metric, fontsize=12, fontweight='bold')
                ax.set_title(f'{metric}', fontsize=13, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
                ax.invert_yaxis()
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, '01_metrics_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: 01_metrics_comparison.png")
        plt.close()
    
    def plot_2_equity_curves(self):
        """Overlay equity curves for all experiments"""
        fig, ax = plt.subplots(figsize=(16, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.experiments)))
        
        for (exp_dir, data), color in zip(sorted(self.experiments.items()), colors):
            if 'equity_curve' in data:
                equity = np.array(data['equity_curve'])
                initial = data['config'].get('initial_equity', 100000)
                returns_pct = ((equity - initial) / initial) * 100
                
                exp_name = data.get('experiment_name', exp_dir)
                ax.plot(returns_pct, linewidth=2.5, label=exp_name, 
                       color=color, alpha=0.85, marker='o', markersize=3, markevery=50)
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.fill_between(range(len(returns_pct)), 0, ax.get_ylim()[1], 
                       alpha=0.05, color='green')
        
        ax.set_xlabel('Trading Days', fontsize=13, fontweight='bold')
        ax.set_ylabel('Cumulative Return (%)', fontsize=13, fontweight='bold')
        ax.set_title('Equity Curves: All 6 Experiments (Starting Capital: $100,000)', 
                    fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, '02_equity_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: 02_equity_curves.png")
        plt.close()
    
    def plot_3_forecast_impact(self):
        """Compare PPO with and without forecast"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Find experiments
        with_forecast = None
        without_forecast = None
        
        for exp_dir, data in self.experiments.items():
            if data.get('forecast_mode') == 'lstm':
                with_forecast = data
            elif (data.get('reward_type') == 'with_risk' and 
                  data.get('forecast_mode') != 'lstm'):
                without_forecast = data
        
        if not with_forecast or not without_forecast:
            print("⚠ Cannot create forecast impact plot (missing experiments)")
            return
        
        # Plot 1: Equity curves
        ax = axes[0]
        for data, label in [(without_forecast, "PPO WITHOUT Forecast"), 
                           (with_forecast, "PPO WITH Forecast (LSTM)")]:
            equity = np.array(data['equity_curve'])
            initial = data['config'].get('initial_equity', 100000)
            returns = ((equity - initial) / initial) * 100
            color = '#E57373' if 'WITHOUT' in label else '#81C784'
            ax.plot(returns, linewidth=3, label=label, color=color, alpha=0.8)
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Trading Days', fontsize=12, fontweight='bold')
        ax.set_ylabel('Return (%)', fontsize=12, fontweight='bold')
        ax.set_title('Impact of LSTM Forecast on Trading Performance', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Metrics comparison
        ax = axes[1]
        metrics_names = ['Return (%)', 'Sharpe', 'Max DD (%)']
        x = np.arange(len(metrics_names))
        width = 0.35
        
        without_vals = []
        with_vals = []
        
        for metric in metrics_names:
            if metric in self.metrics_df.columns:
                without_row = self.metrics_df[
                    self.metrics_df['Forecast'] == 'None'
                ]
                with_row = self.metrics_df[
                    self.metrics_df['Forecast'] == 'LSTM'
                ]
                
                if not without_row.empty:
                    without_vals.append(without_row[metric].iloc[0])
                if not with_row.empty:
                    with_vals.append(with_row[metric].iloc[0])
        
        if without_vals and with_vals:
            ax.bar(x - width/2, without_vals, width, label='Without Forecast', 
                  color='#E57373', alpha=0.8, edgecolor='black')
            ax.bar(x + width/2, with_vals, width, label='With Forecast', 
                  color='#81C784', alpha=0.8, edgecolor='black')
            
            ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
            ax.set_title('Key Metrics: Forecast Impact', fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_names)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, '03_forecast_impact.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: 03_forecast_impact.png")
        plt.close()
    
    def plot_4_reward_ablation(self):
        """Compare different reward functions"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Reward Function Ablation Study', fontsize=18, fontweight='bold')
        
        reward_rewards = {}
        for exp_dir, data in self.experiments.items():
            reward_type = data.get('reward_type', 'unknown')
            if reward_type not in reward_rewards:
                reward_rewards[reward_type] = (exp_dir, data)
        
        for idx, (reward_type, (exp_dir, data)) in enumerate(reward_rewards.items()):
            if idx >= 4:
                break
            
            ax = axes[idx // 2, idx % 2]
            
            if 'equity_curve' in data:
                equity = np.array(data['equity_curve'])
                initial = data['config'].get('initial_equity', 100000)
                returns = ((equity - initial) / initial) * 100
                
                ax.fill_between(range(len(returns)), 0, returns,
                               where=(returns >= 0), alpha=0.3, color='green', label='Profit')
                ax.fill_between(range(len(returns)), 0, returns,
                               where=(returns < 0), alpha=0.3, color='red', label='Loss')
                ax.plot(returns, linewidth=2.5, color='darkblue', alpha=0.85)
                ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
                
                reward_label = reward_type.replace('_', ' ').title()
                ax.set_title(f'Reward: {reward_label}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Trading Days')
                ax.set_ylabel('Return (%)')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=10, loc='best')
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, '04_reward_ablation.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: 04_reward_ablation.png")
        plt.close()
    
    def plot_5_risk_heatmap(self):
        """Risk metrics heatmap"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        heatmap_cols = ['Return (%)', 'Sharpe', 'Max DD (%)', 'Volatility (%)', 
                       'Win Rate (%)', 'Sortino', 'Calmar']
        available_cols = [col for col in heatmap_cols if col in self.metrics_df.columns]
        
        # Normalize
        data_norm = self.metrics_df[available_cols].copy()
        for col in data_norm.columns:
            max_val = data_norm[col].abs().max()
            if max_val > 0:
                data_norm[col] = data_norm[col] / max_val
        
        sns.heatmap(data_norm.T, annot=self.metrics_df[available_cols].T.values,
                   fmt='.2f', cmap='RdYlGn', center=0,
                   xticklabels=self.metrics_df['Experiment'],
                   cbar_kws={'label': 'Normalized Value'},
                   ax=ax, linewidths=1.5, linecolor='gray')
        
        ax.set_title('Risk Metrics Heatmap (Normalized)', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        save_path = os.path.join(self.plots_dir, '05_risk_metrics_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: 05_risk_metrics_heatmap.png")
        plt.close()
    
    def plot_6_metrics_table(self):
        """High-quality metrics table image"""
        fig, ax = plt.subplots(figsize=(18, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_cols = ['Experiment', 'Forecast', 'Reward', 'Return (%)', 
                     'Sharpe', 'Max DD (%)', 'Volatility (%)']
        table_data = self.metrics_df[table_cols].copy()
        
        # Format
        for col in ['Return (%)', 'Sharpe', 'Max DD (%)', 'Volatility (%)']:
            if col in table_data.columns:
                table_data[col] = table_data[col].apply(lambda x: f'{x:.2f}')
        
        table = ax.table(cellText=table_data.values,
                        colLabels=table_data.columns,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.18, 0.12, 0.18, 0.12, 0.12, 0.12, 0.12])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        # Style
        for i in range(len(table_data.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white', size=12)
        
        for i in range(1, len(table_data) + 1):
            for j in range(len(table_data.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.title('Experiment Results: Summary Metrics', 
                 fontsize=18, fontweight='bold', pad=20)
        
        save_path = os.path.join(self.plots_dir, '06_metrics_table.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: 06_metrics_table.png")
        plt.close()
    
    def create_architecture_diagram(self):
        """Create system architecture diagram"""
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Title
        ax.text(5, 9.5, 'Forecast-Aware Trading Agent Architecture', 
               fontsize=18, fontweight='bold', ha='center')
        
        # Components with boxes
        boxes = [
            (1.5, 8, 2, 0.8, 'Market Data\n(BTC-USD)', '#E3F2FD', '#1976D2'),
            (0.2, 6.5, 2.5, 1, 'Technical Features\nRSI, MACD, EMA\nReturns, Volatility', '#F3E5F5', '#7B1FA2'),
            (3.5, 6.5, 2.5, 1, 'LSTM Forecast\nPrice Direction\nLookback: 20 days', '#E8F5E9', '#388E3C'),
            (1.5, 4.8, 4.5, 1.2, 'State Representation\nMarket + Forecast + Position + Equity', '#FFF3E0', '#F57C00'),
            (1.5, 3, 4.5, 1.3, 'PPO Agent\nActor-Critic, Tanh Squashing\nAction: Leverage [-1, 1]', '#FCE4EC', '#C2185B'),
            (0.5, 1, 3, 1.2, 'Trading Action\nBuy/Hold/Sell\nPosition Sizing', '#E1F5FE', '#0288D1'),
            (5.5, 1, 3, 1.2, 'Reward Function\nPnL - Cost\nRisk Penalty', '#F1F8E9', '#558B2F'),
        ]
        
        for x, y, w, h, text, bg_color, border_color in boxes:
            rect = plt.Rectangle((x, y), w, h, fill=True, facecolor=bg_color,
                               edgecolor=border_color, linewidth=2)
            ax.add_patch(rect)
            ax.text(x + w/2, y + h/2, text, fontsize=9, ha='center', va='center',
                   fontweight='bold')
        
        # Arrows
        arrows = [
            (1.5, 8, 1.5, 7.5),  # Market to Features
            (4.75, 7.5, 4.75, 7.5),  # Features to Forecast
            (3.75, 5.8, 3.75, 6),  # All to State
            (3.75, 4.8, 3.75, 4.3),  # State to PPO
            (2.8, 3, -0.8, -1.2),  # PPO to Action
            (4.8, 3, 0.8, -1.2),  # PPO to Reward
        ]
        
        for x1, y1, dx, dy in arrows:
            ax.arrow(x1, y1, dx, dy, head_width=0.15, head_length=0.1,
                    fc='gray', ec='gray', alpha=0.6)
        
        # Environment box
        env_box = plt.Rectangle((0.2, -0.5), 8.6, 0.8, fill=True, 
                               facecolor='#ECEFF1', edgecolor='#455A64', 
                               linewidth=2, linestyle='--')
        ax.add_patch(env_box)
        ax.text(4.5, -0.1, 'Trading Environment (Gymnasium)', 
               fontsize=11, ha='center', fontweight='bold', style='italic')
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, '07_architecture_diagram.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: 07_architecture_diagram.png")
        plt.close()
    
    def generate_summary_report(self):
        """Generate text summary report"""
        report_path = os.path.join(self.plots_dir, 'visualization_summary.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("VISUALIZATION GENERATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write("EXPERIMENTS ANALYZED:\n")
            f.write("-" * 80 + "\n")
            for exp_name in self.metrics_df['Experiment'].values:
                f.write(f"  • {exp_name}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("METRICS COMPARISON TABLE\n")
            f.write("="*80 + "\n\n")
            f.write(self.metrics_df.to_string())
            
            f.write("\n\n" + "="*80 + "\n")
            f.write("KEY FINDINGS\n")
            f.write("="*80 + "\n\n")
            
            best_return = self.metrics_df.loc[self.metrics_df['Return (%)'].idxmax()]
            best_sharpe = self.metrics_df.loc[self.metrics_df['Sharpe'].idxmax()]
            
            f.write(f"Best Return: {best_return['Experiment']} ({best_return['Return (%)']:.2f}%)\n")
            f.write(f"Best Sharpe: {best_sharpe['Experiment']} ({best_sharpe['Sharpe']:.4f})\n")
            
            f.write("\n\nForecast Impact:\n")
            with_forecast = self.metrics_df[self.metrics_df['Forecast'] == 'LSTM']
            without_forecast = self.metrics_df[self.metrics_df['Forecast'] == 'None']
            
            if not with_forecast.empty and not without_forecast.empty:
                avg_return_with = with_forecast['Return (%)'].mean()
                avg_return_without = without_forecast['Return (%)'].mean()
                
                f.write(f"  With Forecast Avg Return: {avg_return_with:.2f}%\n")
                f.write(f"  Without Forecast Avg Return: {avg_return_without:.2f}%\n")
                f.write(f"  Improvement: {avg_return_with - avg_return_without:.2f}%\n")
        
        print(f"✓ Saved: visualization_summary.txt")
    
    def run(self):
        """Generate all visualizations"""
        print("\n" + "="*80)
        print("STEP 2: CREATING COMPREHENSIVE VISUALIZATIONS")
        print("="*80 + "\n")
        
        self.load_all_metrics()
        self.create_metrics_dataframe()
        
        print("Generating plots:\n")
        
        self.plot_1_metrics_comparison()
        self.plot_2_equity_curves()
        self.plot_3_forecast_impact()
        self.plot_4_reward_ablation()
        self.plot_5_risk_heatmap()
        self.plot_6_metrics_table()
        self.create_architecture_diagram()
        self.generate_summary_report()
        
        print(f"\n{'='*80}")
        print(f"✓ ALL VISUALIZATIONS CREATED SUCCESSFULLY")
        print(f"{'='*80}\n")
        print(f"Output Directory: {self.plots_dir}/\n")
        print("Generated Files:")
        print("  01_metrics_comparison.png - All metrics bar charts")
        print("  02_equity_curves.png - Overlaid equity curves")
        print("  03_forecast_impact.png - Forecast value analysis")
        print("  04_reward_ablation.png - Reward function comparison")
        print("  05_risk_metrics_heatmap.png - Risk metrics visualization")
        print("  06_metrics_table.png - Summary results table")
        print("  07_architecture_diagram.png - System architecture")
        print("  visualization_summary.txt - Text summary\n")
        print("✓ Ready for STEP 3: Generate PowerPoint Presentation\n")


if __name__ == "__main__":
    viz = ComprehensiveVisualizationGenerator()
    viz.run()

