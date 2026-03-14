#!/usr/bin/env python3
"""
multi_seed_testing.py

Test reward functions and experiments with multiple random seeds to assess 
robustness and consistency of results across different random initializations.

Usage:
    python multi_seed_testing.py --seeds 10 20 30 --mode reward_only
    python multi_seed_testing.py --seeds 10 20 30 --mode full
    python multi_seed_testing.py --seeds 10 20 30 --output results_multi_seed.csv
"""

import argparse
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from trading_config import (
        RewardType,
        get_ppo_different_rewards_configs,
        get_ppo_without_forecast_config,
        get_ppo_with_forecast_config,
        ExperimentConfig
    )
    from trading_framework import ExperimentRunner
    IMPORTS_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not import trading modules: {e}")
    IMPORTS_AVAILABLE = False


def print_header(title):
    """Print formatted section header"""
    print("\n" + "="*100)
    print(title)
    print("="*100 + "\n")


class MultiSeedTester:
    """Orchestrate multi-seed testing of reward functions"""
    
    def __init__(self, seeds: List[int], output_dir: str = './multi_seed_results'):
        self.seeds = seeds
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        
        print(f"Multi-Seed Tester initialized:")
        print(f"  Seeds to test: {seeds}")
        print(f"  Output directory: {self.output_dir}")
    
    def run_reward_ablation_multi_seed(self) -> Dict[str, pd.DataFrame]:
        """
        Run reward ablation study with multiple seeds
        
        Returns:
            Dict mapping reward type to results DataFrame
        """
        print_header(f"REWARD ABLATION WITH MULTIPLE SEEDS")
        print(f"Testing {len(self.seeds)} seeds: {self.seeds}\n")
        
        if not IMPORTS_AVAILABLE:
            print("✗ Cannot run: Trading modules not available")
            return {}
        
        all_results = {}  # {reward_type: {seed: metrics}}
        
        for reward_seed in self.seeds:
            print(f"\n{'='*100}")
            print(f"SEED: {reward_seed}")
            print(f"{'='*100}\n")
            
            # Get reward configs
            reward_configs = get_ppo_different_rewards_configs(group="reward_ablation")
            
            for config in reward_configs:
                reward_type = config.reward_type.value
                
                # Set seed
                config.seed = reward_seed
                
                print(f"  Testing {reward_type:20s} with seed {reward_seed}...", end=" ", flush=True)
                
                try:
                    runner = ExperimentRunner(config)
                    result = runner.run()
                    
                    if result and 'metrics' in result:
                        metrics = result['metrics']
                        
                        # Store results
                        if reward_type not in all_results:
                            all_results[reward_type] = {}
                        all_results[reward_type][reward_seed] = metrics
                        
                        print(f"✓ Return: {metrics.get('total_return', 0)*100:6.2f}%")
                    else:
                        print("✗ Failed to get metrics")
                
                except Exception as e:
                    print(f"✗ Error: {str(e)[:50]}")
        
        # Convert to DataFrames
        results_dfs = {}
        for reward_type, seed_results in all_results.items():
            results_dfs[reward_type] = pd.DataFrame(seed_results).T
        
        self.results['reward_ablation'] = all_results
        return results_dfs
    
    def run_baseline_multi_seed(self) -> pd.DataFrame:
        """
        Run baseline experiments (with/without forecast) with multiple seeds
        
        Returns:
            DataFrame with results for each seed
        """
        print_header("BASELINE EXPERIMENTS WITH MULTIPLE SEEDS")
        print(f"Testing {len(self.seeds)} seeds: {self.seeds}\n")
        
        if not IMPORTS_AVAILABLE:
            print("✗ Cannot run: Trading modules not available")
            return pd.DataFrame()
        
        baseline_results = {}  # {seed: {experiment: metrics}}
        
        for seed in self.seeds:
            print(f"\n{'='*100}")
            print(f"SEED: {seed}")
            print(f"{'='*100}\n")
            
            baseline_results[seed] = {}
            
            # Experiment 1: Without forecast
            print(f"  Testing PPO-Without-Forecast with seed {seed}...", end=" ", flush=True)
            try:
                config1 = get_ppo_without_forecast_config()
                config1.seed = seed
                runner1 = ExperimentRunner(config1)
                result1 = runner1.run()
                
                if result1 and 'metrics' in result1:
                    baseline_results[seed]['without_forecast'] = result1['metrics']
                    print(f"✓ Return: {result1['metrics'].get('total_return', 0)*100:6.2f}%")
                else:
                    print("✗ Failed")
            except Exception as e:
                print(f"✗ Error: {str(e)[:50]}")
            
            # Experiment 2: With forecast
            print(f"  Testing PPO-With-Forecast with seed {seed}...", end=" ", flush=True)
            try:
                config2 = get_ppo_with_forecast_config()
                config2.seed = seed
                runner2 = ExperimentRunner(config2)
                result2 = runner2.run()
                
                if result2 and 'metrics' in result2:
                    baseline_results[seed]['with_forecast'] = result2['metrics']
                    print(f"✓ Return: {result2['metrics'].get('total_return', 0)*100:6.2f}%")
                else:
                    print("✗ Failed")
            except Exception as e:
                print(f"✗ Error: {str(e)[:50]}")
        
        self.results['baseline'] = baseline_results
        return pd.DataFrame(baseline_results).T
    
    def analyze_results(self, results: Dict[str, pd.DataFrame], analysis_type: str = "rewards"):
        """
        Analyze and display results from multi-seed testing
        
        Parameters:
        -----------
        results : Dict[str, pd.DataFrame]
            Dictionary of results DataFrames
        analysis_type : str
            "rewards" for reward ablation, "baseline" for baseline experiments
        """
        print_header(f"ANALYSIS: {analysis_type.upper()} - MULTI SEED STABILITY")
        
        if not results:
            print("⚠ No results to analyze")
            return
        
        # Key metrics to analyze
        key_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'annualized_volatility']
        
        print(f"\nTesting consistency across seeds: {self.seeds}\n")
        
        for reward_or_exp, df in results.items():
            print(f"\n{reward_or_exp.replace('_', ' ').upper()}")
            print("-" * 80)
            
            # Only analyze available metrics
            available_metrics = [m for m in key_metrics if m in df.columns]
            
            if available_metrics:
                stats = df[available_metrics].describe().T
                
                for metric in available_metrics:
                    values = df[metric].values
                    mean = values.mean()
                    std = values.std()
                    cv = (std / abs(mean)) * 100 if mean != 0 else 0  # Coefficient of variation
                    
                    metric_name = metric.replace('_', ' ').title()
                    print(f"\n{metric_name:30s}")
                    print(f"  Mean:             {mean:>10.6f}")
                    print(f"  Std Dev:          {std:>10.6f}")
                    print(f"  Min:              {values.min():>10.6f}")
                    print(f"  Max:              {values.max():>10.6f}")
                    print(f"  Range:            {values.max() - values.min():>10.6f}")
                    print(f"  Coeff. of Var:    {cv:>10.2f}%")
                    
                    # Interpret stability
                    if cv < 5:
                        stability = "✓ VERY STABLE"
                    elif cv < 10:
                        stability = "✓ STABLE"
                    elif cv < 20:
                        stability = "⚠ MODERATE"
                    else:
                        stability = "✗ UNSTABLE"
                    print(f"  Stability:        {stability}")
    
    def create_stability_report(self, results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create a stability report showing variance across seeds
        
        Returns:
            DataFrame with stability metrics
        """
        print_header("STABILITY REPORT: VARIANCE ACROSS SEEDS")
        
        stability_report = {}
        
        for exp_name, df in results.items():
            key_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'annualized_volatility']
            available_metrics = [m for m in key_metrics if m in df.columns]
            
            for metric in available_metrics:
                values = df[metric].values
                mean = values.mean()
                std = values.std()
                cv = (std / abs(mean)) * 100 if mean != 0 else 0
                
                key = f"{exp_name}_{metric}"
                stability_report[key] = {
                    'Mean': mean,
                    'Std Dev': std,
                    'CV %': cv,
                    'Min': values.min(),
                    'Max': values.max(),
                    'Range': values.max() - values.min()
                }
        
        report_df = pd.DataFrame(stability_report).T
        
        print("\nStability Metrics (Lower CV % = More Stable):\n")
        print(report_df.round(4).to_string())
        
        return report_df
    
    def plot_seed_stability(self, results: Dict[str, pd.DataFrame], output_file: str = None):
        """
        Create visualization of stability across seeds
        
        Parameters:
        -----------
        results : Dict[str, pd.DataFrame]
            Results to plot
        output_file : str, optional
            Where to save the plot
        """
        if not results:
            print("⚠ No results to plot")
            return
        
        print("\nGenerating stability plots...")
        
        # Create figure for each result set
        for result_name, df in results.items():
            key_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'annualized_volatility']
            available_metrics = [m for m in key_metrics if m in df.columns]
            
            if not available_metrics:
                continue
            
            n_cols = min(2, len(available_metrics))
            n_rows = (len(available_metrics) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
            fig.suptitle(f'Seed Stability Analysis: {result_name.replace("_", " ").title()}',
                        fontsize=14, fontweight='bold')
            
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for idx, metric in enumerate(available_metrics):
                ax = axes[idx]
                
                values = df[metric].values
                means = [values.mean()] * len(self.seeds)
                
                ax.plot(self.seeds, values, 'o-', linewidth=2, markersize=8, label='Values')
                ax.plot(self.seeds, means, '--', linewidth=2, color='red', label='Mean')
                ax.fill_between(self.seeds, 
                               values.min() - 0.01*abs(values.mean()),
                               values.max() + 0.01*abs(values.mean()),
                               alpha=0.2)
                
                ax.set_xlabel('Random Seed')
                ax.set_ylabel('Value')
                ax.set_title(metric.replace('_', ' ').title())
                ax.grid(True, alpha=0.3)
                ax.legend()
                ax.set_xticks(self.seeds)
            
            # Hide unused subplots
            for idx in range(len(available_metrics), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            
            # Save plot
            if output_file is None:
                output_file = self.output_dir / f'stability_{result_name}.png'
            
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {output_file}")
            plt.close(fig)
    
    def save_results(self, results_dict: Dict, filename: str = "multi_seed_results.json"):
        """Save results to JSON file"""
        output_path = self.output_dir / filename
        
        # Convert to serializable format
        serializable = {}
        for key, value in results_dict.items():
            if isinstance(value, dict):
                serializable[key] = {}
                for k, v in value.items():
                    if isinstance(v, (pd.DataFrame, dict)):
                        if isinstance(v, pd.DataFrame):
                            serializable[key][k] = v.to_dict()
                        else:
                            serializable[key][k] = v
                    else:
                        serializable[key][k] = str(v)
        
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)
        
        print(f"\n✓ Results saved to: {output_path}")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Test reward functions and experiments with multiple random seeds'
    )
    parser.add_argument('--seeds', type=int, nargs='+', default=[10, 20, 30],
                       help='Random seeds to test (default: [10, 20, 30])')
    parser.add_argument('--mode', type=str, choices=['rewards', 'baseline', 'full'],
                       default='rewards',
                       help='Mode: rewards (ablation), baseline, or full (both)')
    parser.add_argument('--output_dir', type=str, default='./multi_seed_results',
                       help='Output directory for results')
    parser.add_argument('--output_csv', type=str, default=None,
                       help='Save results to CSV file')
    
    args = parser.parse_args()
    
    print("\n" + "="*100)
    print("MULTI-SEED TESTING FRAMEWORK FOR REWARD FUNCTION EXPERIMENTS")
    print("="*100)
    print(f"\nConfiguration:")
    print(f"  Random Seeds: {args.seeds}")
    print(f"  Mode: {args.mode}")
    print(f"  Output Directory: {args.output_dir}")
    print("="*100 + "\n")
    
    tester = MultiSeedTester(args.seeds, args.output_dir)
    
    # Run tests based on mode
    if args.mode in ['rewards', 'full']:
        print("\nPhase 1: Testing reward function ablation with multiple seeds...")
        reward_results = tester.run_reward_ablation_multi_seed()
        
        if reward_results:
            # Analyze
            tester.analyze_results(reward_results, "rewards")
            
            # Create stability report
            stability_df = tester.create_stability_report(reward_results)
            
            # Plot
            tester.plot_seed_stability(reward_results)
            
            # Save CSV if requested
            if args.output_csv:
                output_csv = Path(args.output_csv)
                stability_df.to_csv(output_csv)
                print(f"✓ Stability report saved to: {output_csv}")
    
    if args.mode in ['baseline', 'full']:
        print("\nPhase 2: Testing baseline experiments with multiple seeds...")
        baseline_results = tester.run_baseline_multi_seed()
        
        if not baseline_results.empty:
            # Analyze
            baseline_dict = {}
            for exp in baseline_results.columns:
                baseline_dict[exp] = baseline_results[exp].to_frame()
            tester.analyze_results(baseline_dict, "baseline")
            
            # Plot
            tester.plot_seed_stability(baseline_dict)
    
    # Save all results
    tester.save_results(tester.results)
    
    print("\n" + "="*100)
    print("MULTI-SEED TESTING COMPLETED")
    print("="*100)
    print(f"Results saved to: {tester.output_dir}")


if __name__ == "__main__":
    main()

