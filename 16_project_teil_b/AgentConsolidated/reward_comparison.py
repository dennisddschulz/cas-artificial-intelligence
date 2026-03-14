"""
Reward Comparison Utility
Helps visualize and compare different reward functions
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from reward_calculators import (
    RewardCalculatorType, create_reward_calculator, RewardComponents
)


class RewardComparator:
    """Compare different reward functions side-by-side"""
    
    def __init__(self):
        self.results = {}
        self.sns_style = sns.set_style("whitegrid")
    
    def generate_test_scenarios(self, n_scenarios: int = 1000) -> pd.DataFrame:
        """Generate realistic market/portfolio scenarios"""
        np.random.seed(42)
        
        scenarios = {
            'pnl': np.random.normal(0.0005, 0.0003, n_scenarios),
            'volatility': np.random.uniform(0.005, 0.03, n_scenarios),
            'position': np.random.uniform(-1.0, 1.0, n_scenarios),
            'cost': np.full(n_scenarios, 0.0001),
            'slippage': np.full(n_scenarios, 0.00005),
        }
        
        # Ensure true_reward = pnl - cost - slippage
        scenarios['true_reward'] = (
            scenarios['pnl'] - scenarios['cost'] - scenarios['slippage']
        )
        
        return pd.DataFrame(scenarios)
    
    def calculate_all_rewards(self, scenarios: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate rewards for all types across scenarios"""
        rewards = {}
        
        for reward_type in RewardCalculatorType:
            rewards[reward_type.value] = []
            calc = create_reward_calculator(reward_type)
            
            for _, scenario in scenarios.iterrows():
                components = RewardComponents(
                    pnl=float(scenario['pnl']),
                    cost=float(scenario['cost']),
                    slippage=float(scenario['slippage']),
                    risk_penalty=0.0,
                    volatility=float(scenario['volatility']),
                    position=float(scenario['position']),
                    true_reward=float(scenario['true_reward'])
                )
                
                reward = calc.calculate(components)
                rewards[reward_type.value].append(reward)
            
            rewards[reward_type.value] = np.array(rewards[reward_type.value])
        
        return rewards
    
    def create_comparison_report(self, scenarios: pd.DataFrame) -> pd.DataFrame:
        """Create detailed comparison report"""
        rewards = self.calculate_all_rewards(scenarios)
        
        report = {}
        for reward_type, reward_values in rewards.items():
            report[reward_type] = {
                'Mean': np.mean(reward_values),
                'Std': np.std(reward_values),
                'Min': np.min(reward_values),
                'Max': np.max(reward_values),
                'Median': np.median(reward_values),
                'Q25': np.percentile(reward_values, 25),
                'Q75': np.percentile(reward_values, 75),
                'Skewness': float(pd.Series(reward_values).skew()),
                'Kurtosis': float(pd.Series(reward_values).kurtosis()),
                '% Positive': 100 * np.mean(reward_values > 0),
                'Max Abs': np.max(np.abs(reward_values)),
            }
        
        return pd.DataFrame(report).T
    
    def plot_reward_distributions(self, scenarios: pd.DataFrame, 
                                 save_path: str = 'reward_distributions.png'):
        """Plot distributions of all reward types"""
        rewards = self.calculate_all_rewards(scenarios)
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Reward Function Distributions (1000 market scenarios)', 
                     fontsize=14, fontweight='bold')
        
        axes = axes.flatten()
        colors = plt.cm.Set2(np.linspace(0, 1, 8))
        
        for idx, (reward_type, reward_values) in enumerate(sorted(rewards.items())):
            ax = axes[idx]
            
            # Histogram
            ax.hist(reward_values, bins=50, alpha=0.7, color=colors[idx], edgecolor='black')
            
            # Statistics
            mean = np.mean(reward_values)
            std = np.std(reward_values)
            ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'μ={mean:.6f}')
            ax.axvline(mean - std, color='orange', linestyle=':', alpha=0.7, label=f'σ={std:.6f}')
            
            # Labels
            ax.set_title(f'{reward_type.replace("_", " ").title()}', fontweight='bold')
            ax.set_xlabel('Reward Value')
            ax.set_ylabel('Frequency')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_reward_correlations(self, scenarios: pd.DataFrame,
                                save_path: str = 'reward_correlations.png'):
        """Plot correlation matrix of rewards"""
        rewards = self.calculate_all_rewards(scenarios)
        rewards_df = pd.DataFrame(rewards)
        
        corr_matrix = rewards_df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                    center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'},
                    vmin=-1, vmax=1)
        ax.set_title('Reward Function Correlation Matrix', fontweight='bold', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_reward_sensitivity(self, save_path: str = 'reward_sensitivity.png'):
        """Plot reward sensitivity to key parameters"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Reward Sensitivity Analysis', fontsize=14, fontweight='bold')
        
        # 1. PnL sensitivity
        ax = axes[0, 0]
        pnls = np.linspace(-0.001, 0.002, 50)
        
        calc_basic = create_reward_calculator(RewardCalculatorType.BASIC)
        calc_sharpe = create_reward_calculator(RewardCalculatorType.WITH_SHARPE)
        calc_sortino = create_reward_calculator(RewardCalculatorType.SORTINO)
        
        rewards_basic = []
        rewards_sharpe = []
        rewards_sortino = []
        
        for pnl in pnls:
            comp = RewardComponents(
                pnl=pnl, cost=0.0001, slippage=0.00005, risk_penalty=0.0,
                volatility=0.01, position=0.5, true_reward=pnl - 0.00015
            )
            rewards_basic.append(calc_basic.calculate(comp))
            rewards_sharpe.append(calc_sharpe.calculate(comp))
            rewards_sortino.append(calc_sortino.calculate(comp))
        
        ax.plot(pnls, rewards_basic, label='BASIC', marker='o', markersize=3)
        ax.plot(pnls, rewards_sharpe, label='WITH_SHARPE', marker='s', markersize=3)
        ax.plot(pnls, rewards_sortino, label='SORTINO', marker='^', markersize=3)
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('PnL')
        ax.set_ylabel('Reward')
        ax.set_title('Sensitivity to PnL')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Volatility sensitivity
        ax = axes[0, 1]
        vols = np.linspace(0.001, 0.05, 50)
        rewards_basic = []
        rewards_sharpe = []
        rewards_sortino = []
        
        for vol in vols:
            comp = RewardComponents(
                pnl=0.0005, cost=0.0001, slippage=0.00005, risk_penalty=0.0,
                volatility=vol, position=0.5, true_reward=0.00035
            )
            rewards_basic.append(calc_basic.calculate(comp))
            rewards_sharpe.append(calc_sharpe.calculate(comp))
            rewards_sortino.append(calc_sortino.calculate(comp))
        
        ax.plot(vols, rewards_basic, label='BASIC', marker='o', markersize=3)
        ax.plot(vols, rewards_sharpe, label='WITH_SHARPE', marker='s', markersize=3)
        ax.plot(vols, rewards_sortino, label='SORTINO', marker='^', markersize=3)
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Volatility')
        ax.set_ylabel('Reward')
        ax.set_title('Sensitivity to Volatility')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Position sensitivity
        ax = axes[1, 0]
        positions = np.linspace(-1.0, 1.0, 50)
        calc_risk = create_reward_calculator(RewardCalculatorType.WITH_RISK)
        calc_calmar = create_reward_calculator(RewardCalculatorType.CALMAR)
        
        rewards_basic = []
        rewards_risk = []
        rewards_calmar = []
        
        for pos in positions:
            comp = RewardComponents(
                pnl=0.0005, cost=0.0001, slippage=0.00005, risk_penalty=0.0,
                volatility=0.01, position=pos, true_reward=0.00035
            )
            rewards_basic.append(calc_basic.calculate(comp))
            rewards_risk.append(calc_risk.calculate(comp))
            rewards_calmar.append(calc_calmar.calculate(comp))
        
        ax.plot(positions, rewards_basic, label='BASIC', marker='o', markersize=3)
        ax.plot(positions, rewards_risk, label='WITH_RISK', marker='s', markersize=3)
        ax.plot(positions, rewards_calmar, label='CALMAR', marker='^', markersize=3)
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Position')
        ax.set_ylabel('Reward')
        ax.set_title('Sensitivity to Position Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Comparison summary
        ax = axes[1, 1]
        ax.axis('off')
        
        # Create summary table
        summary_text = """
REWARD FUNCTION CHARACTERISTICS

BASIC: Greedy, no risk constraint
  ✓ Highest average reward
  ✗ Extreme position sizes
  
WITH_RISK: Balanced penalty approach  
  ✓ Smooth position constraint
  ✓ Most stable learning
  
WITH_SHARPE: Risk-normalized
  ✓ Sharpest ratios
  ✗ Division-by-vol instability
  
SORTINO: Downside-focused
  ✓ Conservative positions
  ✓ Better downside metrics
  
CALMAR: Drawdown-focused
  ✓ Lowest drawdowns
  ✗ Very conservative
  
COMPOSITE: Multi-objective
  ✓ Flexible weighting
  ✓ Fine-tuning capability
        """
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                fontfamily='monospace', fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def run_full_analysis(self, n_scenarios: int = 1000):
        """Run complete analysis and generate all plots"""
        print("\n" + "="*60)
        print("REWARD COMPARISON ANALYSIS")
        print("="*60 + "\n")
        
        # Generate scenarios
        print(f"Generating {n_scenarios} market scenarios...")
        scenarios = self.generate_test_scenarios(n_scenarios)
        print(f"✓ Generated scenarios:")
        print(f"  PnL range: [{scenarios['pnl'].min():.6f}, {scenarios['pnl'].max():.6f}]")
        print(f"  Vol range: [{scenarios['volatility'].min():.6f}, {scenarios['volatility'].max():.6f}]")
        
        # Compare rewards
        print(f"\nCalculating rewards for all 8 types...")
        report = self.create_comparison_report(scenarios)
        print("\n" + report.to_string())
        
        # Save report
        report.to_csv('reward_comparison_report.csv')
        print(f"\n✓ Saved: reward_comparison_report.csv")
        
        # Generate plots
        print(f"\nGenerating visualizations...")
        self.plot_reward_distributions(scenarios)
        self.plot_reward_correlations(scenarios)
        self.plot_reward_sensitivity()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("\nGenerated files:")
        print("  - reward_distributions.png")
        print("  - reward_correlations.png")
        print("  - reward_sensitivity.png")
        print("  - reward_comparison_report.csv")


if __name__ == "__main__":
    comparator = RewardComparator()
    comparator.run_full_analysis(n_scenarios=1000)

w
