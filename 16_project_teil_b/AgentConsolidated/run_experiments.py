"""
Parameterized Experiment Runner
Executes all required experiments and comparisons
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

# Import experiment components
from trading_config import (
    get_ppo_without_forecast_config,
    get_ppo_with_forecast_config,
    get_ppo_different_rewards_configs,
    ForecastMode, RewardType
)
from trading_framework import ExperimentRunner
from trading_metrics import MetricsComparison, EquityCurveAnalyzer

# W&B setup
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Plotting setup
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)


class ExperimentManager:
    """Manage and run multiple trading experiments"""
    
    def __init__(self, base_dir: str = "./experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.metrics_comparison = MetricsComparison()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = self.base_dir / f"run_{self.timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def run_experiment(self, config_name: str, config) -> dict:
        """Run single experiment"""
        print(f"\n{'='*80}")
        print(f"STARTING: {config_name}")
        print(f"{'='*80}\n")
        
        try:
            runner = ExperimentRunner(config)
            results = runner.run()
            
            # Store results
            self.results[config_name] = results
            self.metrics_comparison.add_experiment(config_name, results['metrics'])
            
            # Save results
            self._save_results(config_name, results)
            
            print(f"\n✓ {config_name} completed successfully")
            return results
        except Exception as e:
            print(f"\n✗ {config_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_results(self, name: str, results: dict):
        """Save experiment results"""
        exp_dir = self.results_dir / name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_df = pd.DataFrame([results['metrics']])
        metrics_df.to_csv(exp_dir / "metrics.csv", index=False)
        
        # Save arrays
        np.save(exp_dir / "equity.npy", results['equity'])
        np.save(exp_dir / "positions.npy", results['positions'])
        np.save(exp_dir / "costs.npy", results['costs'])
        np.save(exp_dir / "pnl.npy", results['pnl'])
    
    def run_ppo_without_forecast(self) -> dict:
        """Run PPO trading without forecast"""
        config = get_ppo_without_forecast_config(
            name="PPO-Without-Forecast",
            group="baseline"
        )
        return self.run_experiment("PPO_Without_Forecast", config)
    
    def run_ppo_with_forecast(self) -> dict:
        """Run PPO trading with LSTM forecast"""
        config = get_ppo_with_forecast_config(
            name="PPO-With-Forecast",
            group="baseline"
        )
        return self.run_experiment("PPO_With_Forecast", config)
    
    def run_ppo_reward_ablation(self) -> dict:
        """Run PPO with different reward definitions"""
        configs = get_ppo_different_rewards_configs(group="reward_ablation")
        results = {}
        
        for config in configs:
            name = f"PPO_Reward_{config.reward_type.value.capitalize()}"
            result = self.run_experiment(name, config)
            if result is not None:
                results[name] = result
        
        return results
    
    def run_all_experiments(self):
        """Run all required experiments"""
        print(f"\n{'='*80}")
        print(f"RUNNING ALL EXPERIMENTS")
        print(f"Results directory: {self.results_dir}")
        print(f"{'='*80}\n")
        
        # 1. PPO without forecast
        print("\n[1/3] PPO Without Forecast")
        self.run_ppo_without_forecast()
        
        # 2. PPO with forecast
        print("\n[2/3] PPO With Forecast")
        self.run_ppo_with_forecast()
        
        # 3. PPO with different rewards
        print("\n[3/3] PPO With Different Reward Functions")
        self.run_ppo_reward_ablation()
        
        print(f"\n{'='*80}")
        print(f"ALL EXPERIMENTS COMPLETED")
        print(f"{'='*80}")
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        print(f"\n{'='*80}")
        print(f"GENERATING COMPARISON REPORT")
        print(f"{'='*80}\n")
        
        # Create comparison dataframe
        comparison_df = self.metrics_comparison.to_dataframe()
        comparison_df.to_csv(self.results_dir / "all_metrics.csv")
        
        print("\n=== METRICS COMPARISON ===\n")
        print(comparison_df.to_string())
        print()
        
        # Key metrics summary
        print("\n=== KEY METRICS SUMMARY ===\n")
        key_metrics = [
            'total_return',
            'sharpe_ratio',
            'max_drawdown',
            'annualized_volatility',
            'turnover'
        ]
        
        for metric in key_metrics:
            if metric in comparison_df.columns:
                print(f"\n{metric.upper()}:")
                print(comparison_df[metric].sort_values(ascending=False))
        
        # Rankings
        self._create_rankings(comparison_df)
        
        return comparison_df
    
    def _create_rankings(self, comparison_df: pd.DataFrame):
        """Create performance rankings"""
        print("\n\n=== PERFORMANCE RANKINGS ===\n")
        
        rankings = {}
        
        # Return ranking (higher is better)
        rankings['Total Return'] = comparison_df['total_return'].sort_values(ascending=False)
        
        # Sharpe ratio ranking (higher is better)
        rankings['Sharpe Ratio'] = comparison_df['sharpe_ratio'].sort_values(ascending=False)
        
        # Max drawdown ranking (higher/less negative is better)
        rankings['Max Drawdown (Less Negative)'] = comparison_df['max_drawdown'].sort_values(ascending=False)
        
        # Volatility ranking (lower is better)
        rankings['Lowest Volatility'] = comparison_df['annualized_volatility'].sort_values(ascending=True)
        
        # Turnover ranking (lower is better for efficiency)
        rankings['Lowest Turnover'] = comparison_df['turnover'].sort_values(ascending=True)
        
        for category, ranking in rankings.items():
            print(f"\n{category}:")
            for i, (exp, value) in enumerate(ranking.items(), 1):
                print(f"  {i}. {exp}: {value:.6f}")
    
    def generate_plots(self):
        """Generate all comparison plots"""
        print(f"\n{'='*80}")
        print(f"GENERATING PLOTS")
        print(f"{'='*80}\n")
        
        # 1. Metrics comparison
        comparison_df = self.metrics_comparison.to_dataframe()
        fig = self.metrics_comparison.plot_comparison()
        fig.savefig(self.results_dir / "metrics_comparison.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("✓ Saved: metrics_comparison.png")
        
        # 2. Equity curves
        equity_curves = {
            name: results['equity']
            for name, results in self.results.items()
            if results is not None
        }
        if equity_curves:
            fig = EquityCurveAnalyzer.plot_equity_curve(equity_curves)
            fig.savefig(self.results_dir / "equity_curves.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            print("✓ Saved: equity_curves.png")
        
        # 3. Drawdown comparison
        if equity_curves:
            fig = EquityCurveAnalyzer.plot_drawdown(equity_curves)
            fig.savefig(self.results_dir / "drawdown.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            print("✓ Saved: drawdown.png")
        
        # 4. Returns distribution
        returns = {
            name: np.diff(results['equity']) / results['equity'][:-1]
            for name, results in self.results.items()
            if results is not None
        }
        if returns:
            fig = EquityCurveAnalyzer.plot_returns_distribution(returns)
            fig.savefig(self.results_dir / "returns_distribution.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            print("✓ Saved: returns_distribution.png")
        
        # 5. Custom comparison heatmap
        self._create_heatmap(comparison_df)
    
    def _create_heatmap(self, comparison_df: pd.DataFrame):
        """Create metrics heatmap"""
        # Normalize metrics for heatmap
        normalized = comparison_df.copy()
        for col in normalized.columns:
            col_min = normalized[col].min()
            col_max = normalized[col].max()
            if col_max - col_min != 0:
                normalized[col] = (normalized[col] - col_min) / (col_max - col_min)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(normalized.T, annot=comparison_df.T, fmt='.4f', cmap='RdYlGn',
                    cbar_kws={'label': 'Normalized Score'}, ax=ax)
        ax.set_title('Metrics Heatmap (Color: Normalized, Values: Actual)', fontweight='bold')
        ax.set_xlabel('Experiment')
        fig.savefig(self.results_dir / "metrics_heatmap.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("✓ Saved: metrics_heatmap.png")
    
    def generate_html_report(self):
        """Generate HTML summary report"""
        comparison_df = self.metrics_comparison.to_dataframe()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>PPO Trading Experiments Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>PPO Trading Experiments Report</h1>
            <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>Results directory: {self.results_dir}</p>
            
            <h2>Experiments Run</h2>
            <ul>
                <li>PPO Without Forecast</li>
                <li>PPO With Forecast (LSTM)</li>
                <li>PPO With Different Reward Functions</li>
            </ul>
            
            <h2>Key Metrics Comparison</h2>
            <table>
                <tr>
                    <th>Experiment</th>
                    <th>Total Return</th>
                    <th>Sharpe Ratio</th>
                    <th>Max Drawdown</th>
                    <th>Volatility</th>
                    <th>Turnover</th>
                </tr>
        """
        
        for idx, row in comparison_df.iterrows():
            html += f"<tr><td><b>{idx}</b></td>"
            html += f"<td class='{'positive' if row['total_return'] > 0 else 'negative'}'>{row['total_return']:.4f}</td>"
            html += f"<td class='{'positive' if row['sharpe_ratio'] > 0 else 'negative'}'>{row['sharpe_ratio']:.4f}</td>"
            html += f"<td class='{'positive' if row['max_drawdown'] > -0.1 else 'negative'}'>{row['max_drawdown']:.4f}</td>"
            html += f"<td>{row['annualized_volatility']:.4f}</td>"
            html += f"<td>{row['turnover']:.4f}</td>"
            html += "</tr>"
        
        html += """
            </table>
            
            <h2>Visualizations</h2>
            <h3>Equity Curves</h3>
            <img src="equity_curves.png" alt="Equity Curves">
            
            <h3>Metrics Comparison</h3>
            <img src="metrics_comparison.png" alt="Metrics Comparison">
            
            <h3>Drawdown Analysis</h3>
            <img src="drawdown.png" alt="Drawdown">
            
            <h3>Metrics Heatmap</h3>
            <img src="metrics_heatmap.png" alt="Heatmap">
            
        </body>
        </html>
        """
        
        with open(self.results_dir / "report.html", "w") as f:
            f.write(html)
        print("✓ Saved: report.html")
    
    def export_all_results(self):
        """Export all results to JSON"""
        export_data = {
            'timestamp': self.timestamp,
            'metrics': self.metrics_comparison.to_dataframe().to_dict(),
            'experiments': {
                name: {
                    'metrics': results['metrics'],
                    'equity_stats': {
                        'initial': float(results['equity'][0]),
                        'final': float(results['equity'][-1]),
                        'max': float(np.max(results['equity'])),
                        'min': float(np.min(results['equity'])),
                    }
                }
                for name, results in self.results.items()
                if results is not None
            }
        }
        
        with open(self.results_dir / "results.json", "w") as f:
            json.dump(export_data, f, indent=2)
        print("✓ Saved: results.json")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("PPO TRADING EXPERIMENTS - PARAMETERIZED FRAMEWORK")
    print("="*80 + "\n")
    
    # Create manager
    manager = ExperimentManager()
    
    # Run all experiments
    manager.run_all_experiments()
    
    # Generate reports and plots
    manager.generate_comparison_report()
    manager.generate_plots()
    manager.generate_html_report()
    manager.export_all_results()
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT SUITE COMPLETED")
    print(f"Results saved to: {manager.results_dir}")
    print(f"Open report.html for full report")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

