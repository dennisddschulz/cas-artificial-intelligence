#!/usr/bin/env python3
"""
Reward Function Ablation Study
Comprehensive testing of 8 different reward functions with PPO trading agents

Run this script to execute all reward function variants and generate comparison reports.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Configure environment
os.environ['MPLBACKEND'] = 'Agg'

# Import custom modules
try:
    from trading_config import (
        RewardType,
        get_ppo_different_rewards_configs
    )
    from trading_framework import ExperimentRunner
    
    print("✓ Custom modules imported successfully")
except Exception as e:
    print(f"✗ Error importing modules: {e}")
    sys.exit(1)


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*100)
    print(title)
    print("="*100 + "\n")


def print_reward_function_info():
    """Print information about each reward function"""
    print_header("REWARD FUNCTION EXPLANATIONS")
    
    reward_info = {
        RewardType.BASIC: {
            "name": "BASIC",
            "formula": "R = PnL - Cost - Slippage",
            "description": "Pure return maximization without risk constraints",
            "best_for": "High-conviction trend following",
        },
        RewardType.WITH_RISK: {
            "name": "WITH_RISK",
            "formula": "R = PnL - Cost - κ × (Position²) × Volatility",
            "description": "Adds quadratic position penalty scaled by volatility (DEFAULT)",
            "best_for": "Balanced returns and risk control",
        },
        RewardType.WITH_SHARPE: {
            "name": "WITH_SHARPE",
            "formula": "R = (PnL - Cost) / Volatility",
            "description": "Explicitly optimizes risk-adjusted returns",
            "best_for": "Consistent, low-volatility strategies",
        },
        RewardType.RISK_ADJUSTED: {
            "name": "RISK_ADJUSTED",
            "formula": "R = (PnL / Volatility) - Cost",
            "description": "Returns normalized by volatility, costs not normalized",
            "best_for": "Adaptive position sizing based on risk regime",
        },
        RewardType.SORTINO: {
            "name": "SORTINO",
            "formula": "R = (PnL_adjusted / Volatility)",
            "description": "Penalizes losses 20% more than gains (downside focus)",
            "best_for": "Risk-averse investors",
        },
        RewardType.CALMAR: {
            "name": "CALMAR",
            "formula": "R = PnL - (DrawdownEstimate × 0.5)",
            "description": "Uses estimated drawdown as risk metric",
            "best_for": "Wealth preservation focus",
        },
        RewardType.INFORMATION_RATIO: {
            "name": "INFORMATION_RATIO",
            "formula": "R = (PnL / Volatility) + Consistency_Bonus",
            "description": "Risk-adjusted returns with consistency bonus",
            "best_for": "Building consistent alpha",
        },
        RewardType.COMPOSITE: {
            "name": "COMPOSITE",
            "formula": "R = 0.5×Returns + 0.3×Sharpe + 0.2×Risk",
            "description": "Multi-objective weighted blend of signals",
            "best_for": "Balanced return-risk profile",
        },
    }
    
    for i, (reward_type, info) in enumerate(reward_info.items(), 1):
        print(f"{i}. {info['name']}")
        print(f"   Formula: {info['formula']}")
        print(f"   Description: {info['description']}")
        print(f"   Best for: {info['best_for']}")
        print()


def run_reward_ablation():
    """Run comprehensive reward function ablation study"""
    print_header("REWARD FUNCTION ABLATION STUDY")
    print("Testing 8 different reward functions with PPO trading agents\n")
    
    # Print reward function explanations
    print_reward_function_info()
    
    # Get all reward configurations
    reward_configs = get_ppo_different_rewards_configs(group="reward_ablation")
    print(f"Running {len(reward_configs)} reward function variants...\n")
    
    start_time = datetime.now()
    results = {}
    
    for i, config in enumerate(reward_configs, 1):
        reward_name = config.reward_type.value.replace('_', ' ').title()
        print(f"\n{'='*100}")
        print(f"[{i}/{len(reward_configs)}] TESTING: {reward_name}")
        print(f"{'='*100}")
        print(f"Reward Type: {config.reward_type.value}")
        print(f"Experiment Name: {config.experiment_name}")
        print(f"WandB Group: {config.wandb_group}")
        print(f"Initial Equity: ${config.environment.initial_equity:,.0f}")
        print(f"Leverage Max: {config.environment.leverage_max}")
        print(f"Fee: {config.environment.fee}")
        print(f"PPO Updates: {config.ppo.total_updates}")
        print(f"Parallel Envs: {config.ppo.num_envs}")
        print()
        
        try:
            # Run experiment
            runner = ExperimentRunner(config)
            result = runner.run()
            
            if result is not None:
                exp_key = f"PPO-{config.reward_type.value}"
                results[exp_key] = result
                
                # Extract and display key metrics
                metrics = result.get('metrics', {})
                
                print(f"\n✓ EXPERIMENT COMPLETED")
                print(f"{'─'*100}")
                print(f"Results for {reward_name}:")
                print(f"  • Total Return:          {metrics.get('total_return', 0)*100:>8.2f}%")
                print(f"  • Sharpe Ratio:          {metrics.get('sharpe_ratio', 0):>8.4f}")
                print(f"  • Max Drawdown:          {metrics.get('max_drawdown', 0)*100:>8.2f}%")
                print(f"  • Volatility:            {metrics.get('annualized_volatility', 0)*100:>8.2f}%")
                print(f"  • Turnover:              {metrics.get('turnover', 0):>8.4f}")
                print(f"  • Win Rate:              {metrics.get('win_rate', 0)*100:>8.2f}%")
                print(f"  • Profit Factor:         {metrics.get('profit_factor', 0):>8.4f}")
                print(f"  • Calmar Ratio:          {metrics.get('calmar_ratio', 0):>8.4f}")
                print(f"  • Sortino Ratio:         {metrics.get('sortino_ratio', 0):>8.4f}")
                
        except Exception as e:
            print(f"✗ EXPERIMENT FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================================================
    # ANALYSIS & COMPARISON
    # ========================================================================
    if not results:
        print("\n✗ No experiments completed successfully")
        return
    
    print_header("REWARD FUNCTION COMPARISON ANALYSIS")
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        exp_name: result.get('metrics', {})
        for exp_name, result in results.items()
    }).T
    
    print("\nAll Reward Functions Comparison:")
    print("="*100)
    print(comparison_df.to_string())
    print("="*100)
    
    # Find best performers
    print("\n\nBEST PERFORMERS BY METRIC:")
    print("─"*100)
    
    metrics_ranking = {
        'total_return': ('Total Return %', lambda x: x.max(), True),
        'sharpe_ratio': ('Sharpe Ratio', lambda x: x.max(), True),
        'max_drawdown': ('Max Drawdown %', lambda x: x.max(), True),
        'annualized_volatility': ('Volatility %', lambda x: x.min(), False),
        'turnover': ('Turnover', lambda x: x.min(), False),
        'win_rate': ('Win Rate %', lambda x: x.max(), True),
        'calmar_ratio': ('Calmar Ratio', lambda x: x.max(), True),
        'sortino_ratio': ('Sortino Ratio', lambda x: x.max(), True),
    }
    
    for metric_key, (metric_name, func, _) in metrics_ranking.items():
        if metric_key in comparison_df.columns:
            best_idx = func(comparison_df[metric_key])
            best_name = comparison_df[metric_key].idxmax() if metric_key not in ['annualized_volatility', 'turnover'] else comparison_df[metric_key].idxmin()
            best_val = comparison_df.loc[best_name, metric_key]
            reward_type = best_name.replace('PPO-', '')
            print(f"{metric_name:30s}: {reward_type:30s} = {best_val:>10.4f}")
    
    # Save results
    print("\n")
    print_header("SAVING RESULTS")
    
    try:
        comparison_df.to_csv('reward_ablation_comparison.csv')
        print("✓ Saved: reward_ablation_comparison.csv")
    except Exception as e:
        print(f"✗ Error saving CSV: {e}")
    
    try:
        import json
        summary = {
            'timestamp': datetime.now().isoformat(),
            'num_experiments': len(results),
            'results': {
                exp_name: {
                    'metrics': result.get('metrics', {}),
                    'final_equity': float(result.get('equity', [0])[-1]) if result.get('equity') else 0,
                }
                for exp_name, result in results.items()
            }
        }
        with open('reward_ablation_results.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print("✓ Saved: reward_ablation_results.json")
    except Exception as e:
        print(f"✗ Error saving JSON: {e}")
    
    # Generate visualization
    try:
        print("\nGenerating visualization...")
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Reward Function Ablation Study - Performance Comparison', 
                     fontsize=16, fontweight='bold')
        
        plot_metrics = [
            ('total_return', 'Total Return (%)'),
            ('sharpe_ratio', 'Sharpe Ratio'),
            ('max_drawdown', 'Max Drawdown (%)'),
            ('annualized_volatility', 'Volatility (%)'),
            ('turnover', 'Turnover'),
            ('win_rate', 'Win Rate (%)'),
            ('calmar_ratio', 'Calmar Ratio'),
            ('sortino_ratio', 'Sortino Ratio'),
        ]
        
        axes_flat = axes.flatten()
        
        for idx, (metric, label) in enumerate(plot_metrics):
            ax = axes_flat[idx]
            
            if metric in comparison_df.columns:
                data = comparison_df[metric].sort_values(ascending=False)
                colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(data)))
                
                bars = ax.barh(range(len(data)), data.values, color=colors)
                ax.set_yticks(range(len(data)))
                ax.set_yticklabels([x.replace('PPO-', '') for x in data.index], fontsize=9)
                ax.set_xlabel('Value')
                ax.set_title(label, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                
                # Add value labels
                for i, (bar, val) in enumerate(zip(bars, data.values)):
                    ax.text(val, i, f' {val:.4f}', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('reward_ablation_comparison.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("✓ Saved: reward_ablation_comparison.png")
    except Exception as e:
        print(f"✗ Error generating visualization: {e}")
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n")
    print_header("REWARD ABLATION STUDY COMPLETED")
    print(f"\nTotal Reward Functions Tested: {len(results)}")
    print(f"Total Duration: {duration}")
    print(f"\n✓ Results saved to:")
    print(f"  - reward_ablation_comparison.csv")
    print(f"  - reward_ablation_results.json")
    print(f"  - reward_ablation_comparison.png")
    print(f"\n✓ Next Steps:")
    print(f"  1. Review reward_ablation_comparison.csv for metrics table")
    print(f"  2. View reward_ablation_comparison.png for visual comparison")
    print(f"  3. Check reward_ablation_results.json for detailed results")
    print(f"  4. Read REWARD_ABLATION_GUIDE.md for interpretation guidance")
    print("="*100 + "\n")


if __name__ == "__main__":
    print("\n" + "="*100)
    print("REWARD FUNCTION ABLATION STUDY - COMPREHENSIVE TESTING")
    print("="*100)
    print("\nThis script tests 8 different reward functions with PPO trading agents")
    print("Expected runtime: ~3-4 hours (20-30 min per experiment)")
    print("Results will be saved to: reward_ablation_*")
    print("="*100 + "\n")
    
    run_reward_ablation()

