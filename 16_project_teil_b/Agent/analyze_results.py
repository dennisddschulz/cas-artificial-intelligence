#!/usr/bin/env python3
"""
COMPREHENSIVE ANALYSIS & VISUALIZATION
Generates all plots and comparisons for the trading experiments
Run after main.py: python3 analyze_results.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import json
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

# ============================================================
# LOAD RESULTS
# ============================================================

def load_results():
    """Load experiment results from pickle (primary) or JSON (backup)"""
    print("Loading experiment results...")
    
    # Try pickle first (primary format)
    if Path('metrics.pkl').exists():
        with open('metrics.pkl', 'rb') as f:
            results = pickle.load(f)
        print(f"✓ Loaded from metrics.pkl")
        return results
    
    # Fall back to JSON
    if Path('experiment_results.json').exists():
        with open('experiment_results.json', 'r') as f:
            data = json.load(f)
            results = data['results']
        print(f"✓ Loaded from experiment_results.json")
        return results
    
    # Fall back to CSV
    if Path('results_comparison.csv').exists():
        df = pd.read_csv('results_comparison.csv', index_col=0)
        print(f"✓ Loaded from results_comparison.csv")
        return df.to_dict('index')
    
    raise FileNotFoundError("No results found. Run main.py first.")

# ============================================================
# VISUALIZATION 1: PERFORMANCE COMPARISON
# ============================================================

def plot_performance_comparison(results):
    """Plot performance metrics comparison"""
    print("Creating performance comparison plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Strategy Performance Comparison', fontsize=16, fontweight='bold')
    
    strategies = list(results.keys())
    
    # Returns
    ax = axes[0, 0]
    returns = [results[s]['return']*100 for s in strategies]
    colors = ['#C73E1D', '#2CA02C', '#1f77b4', '#FF7F0E', '#2CA02C', '#17BECF']
    bars = ax.bar(range(len(strategies)), returns, color=colors[:len(strategies)])
    ax.set_ylabel('Total Return (%)', fontweight='bold')
    ax.set_title('Total Return')
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(returns):
        ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # Sharpe Ratio
    ax = axes[0, 1]
    sharpe = [results[s]['sharpe'] for s in strategies]
    bars = ax.bar(range(len(strategies)), sharpe, color=colors[:len(strategies)])
    ax.set_ylabel('Sharpe Ratio', fontweight='bold')
    ax.set_title('Risk-Adjusted Return (Sharpe Ratio)')
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(sharpe):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # Max Drawdown
    ax = axes[1, 0]
    maxdd = [results[s]['maxdd']*100 for s in strategies]
    bars = ax.bar(range(len(strategies)), maxdd, color=colors[:len(strategies)])
    ax.set_ylabel('Max Drawdown (%)', fontweight='bold')
    ax.set_title('Maximum Drawdown (Lower is Better)')
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(maxdd):
        ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # Volatility
    ax = axes[1, 1]
    vol = [results[s]['volatility']*100 for s in strategies]
    bars = ax.bar(range(len(strategies)), vol, color=colors[:len(strategies)])
    ax.set_ylabel('Volatility (%)', fontweight='bold')
    ax.set_title('Annualized Volatility')
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(vol):
        ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/01_performance_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: plots/01_performance_comparison.png")
    plt.close()

# ============================================================
# VISUALIZATION 2: DETAILED METRICS TABLE
# ============================================================

def plot_metrics_table(results):
    """Create detailed metrics comparison table"""
    print("Creating metrics table...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    data = []
    for strategy in results.keys():
        m = results[strategy]
        data.append([
            strategy,
            f"${m['final_equity']:,.0f}",
            f"{m['return']*100:>7.2f}%",
            f"{m['sharpe']:>7.4f}",
            f"{m['volatility']*100:>7.2f}%",
            f"{m['maxdd']*100:>7.2f}%",
            f"{m['turnover']:>7.4f}",
        ])
    
    columns = ['Strategy', 'Final Equity', 'Return', 'Sharpe', 'Volatility', 'Max DD', 'Turnover']
    table = ax.table(cellText=data, colLabels=columns, cellLoc='center', loc='center',
                    colWidths=[0.22, 0.18, 0.12, 0.12, 0.12, 0.12, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Color header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows
    colors = ['#E7E6F7', '#FFF2CC', '#F4F4F4']
    for i in range(1, len(data) + 1):
        for j in range(len(columns)):
            table[(i, j)].set_facecolor(colors[i % len(colors)])
    
    plt.title('Detailed Performance Metrics', fontsize=14, fontweight='bold', pad=20)
    plt.savefig('plots/06_model_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: plots/06_model_summary.png")
    plt.close()

# ============================================================
# VISUALIZATION 3: RETURN DISTRIBUTION
# ============================================================

def plot_return_distribution(results):
    """Plot return distribution comparison"""
    print("Creating return distribution plot...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Daily Return Distributions', fontsize=14, fontweight='bold')
    
    strategies = list(results.keys())
    colors = ['#C73E1D', '#2CA02C', '#1f77b4', '#FF7F0E', '#17BECF', '#D62728']
    
    for idx, strategy in enumerate(strategies[:6]):
        ax = axes[idx // 3, idx % 3]
        ret = results[strategy]['return']
        vol = results[strategy]['volatility']
        
        # Create synthetic distribution for visualization
        returns_synthetic = np.random.normal(ret/252, vol/np.sqrt(252), 250)
        
        ax.hist(returns_synthetic*100, bins=30, color=colors[idx], alpha=0.7, edgecolor='black')
        ax.axvline(ret/252*100, color='red', linestyle='--', linewidth=2, label=f'Mean: {ret/252*100:.2f}%')
        ax.set_xlabel('Daily Return (%)')
        ax.set_ylabel('Frequency')
        ax.set_title(strategy)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('plots/10_returns_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: plots/10_returns_analysis.png")
    plt.close()

# ============================================================
# VISUALIZATION 4: RISK METRICS COMPARISON
# ============================================================

def plot_risk_metrics(results):
    """Plot risk metrics (volatility vs return)"""
    print("Creating risk metrics plot...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    strategies = list(results.keys())
    colors = ['#C73E1D', '#2CA02C', '#1f77b4', '#FF7F0E', '#17BECF', '#D62728']
    
    for idx, strategy in enumerate(strategies):
        m = results[strategy]
        ret = m['return']*100
        vol = m['volatility']*100
        sharpe = m['sharpe']
        
        # Size by Sharpe ratio
        size = max(100, sharpe*300)
        ax.scatter(vol, ret, s=size, alpha=0.6, color=colors[idx], 
                  label=f'{strategy} (Sharpe: {sharpe:.3f})', edgecolor='black', linewidth=1.5)
        ax.annotate(strategy, (vol, ret), fontsize=9, ha='center', fontweight='bold')
    
    ax.set_xlabel('Volatility (%)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Return (%)', fontweight='bold', fontsize=12)
    ax.set_title('Risk vs Return Trade-off (Size = Sharpe Ratio)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('plots/11_risk_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: plots/11_risk_metrics.png")
    plt.close()

# ============================================================
# VISUALIZATION 5: COMPREHENSIVE ANALYSIS
# ============================================================

def create_comprehensive_analysis(results):
    """Create comprehensive analysis dashboard"""
    print("Creating comprehensive analysis dashboard...")
    
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    fig.suptitle('Comprehensive Trading Strategy Analysis Dashboard', 
                fontsize=18, fontweight='bold', y=0.98)
    
    strategies = list(results.keys())
    colors_list = ['#C73E1D', '#2CA02C', '#1f77b4', '#FF7F0E', '#17BECF', '#D62728']
    
    # 1. Returns comparison
    ax1 = fig.add_subplot(gs[0, 0])
    returns = [results[s]['return']*100 for s in strategies]
    bars = ax1.barh(strategies, returns, color=colors_list[:len(strategies)])
    ax1.set_xlabel('Return (%)', fontweight='bold')
    ax1.set_title('Total Returns', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(returns):
        ax1.text(v + 0.5, i, f'{v:.1f}%', va='center', fontweight='bold')
    
    # 2. Sharpe ratio comparison
    ax2 = fig.add_subplot(gs[0, 1])
    sharpe = [results[s]['sharpe'] for s in strategies]
    bars = ax2.barh(strategies, sharpe, color=colors_list[:len(strategies)])
    ax2.set_xlabel('Sharpe Ratio', fontweight='bold')
    ax2.set_title('Risk-Adjusted Returns', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(sharpe):
        ax2.text(v + 0.02, i, f'{v:.3f}', va='center', fontweight='bold')
    
    # 3. Max drawdown comparison
    ax3 = fig.add_subplot(gs[0, 2])
    maxdd = [results[s]['maxdd']*100 for s in strategies]
    bars = ax3.barh(strategies, maxdd, color=colors_list[:len(strategies)])
    ax3.set_xlabel('Max Drawdown (%)', fontweight='bold')
    ax3.set_title('Downside Risk', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(maxdd):
        ax3.text(v + 0.2, i, f'{v:.1f}%', va='center', fontweight='bold')
    
    # 4. Volatility comparison
    ax4 = fig.add_subplot(gs[1, 0])
    vol = [results[s]['volatility']*100 for s in strategies]
    bars = ax4.barh(strategies, vol, color=colors_list[:len(strategies)])
    ax4.set_xlabel('Volatility (%)', fontweight='bold')
    ax4.set_title('Annual Volatility', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(vol):
        ax4.text(v + 0.5, i, f'{v:.1f}%', va='center', fontweight='bold')
    
    # 5. Turnover comparison
    ax5 = fig.add_subplot(gs[1, 1])
    turnover = [results[s]['turnover'] for s in strategies]
    bars = ax5.barh(strategies, turnover, color=colors_list[:len(strategies)])
    ax5.set_xlabel('Turnover', fontweight='bold')
    ax5.set_title('Trading Activity', fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(turnover):
        ax5.text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold')
    
    # 6. Final equity comparison
    ax6 = fig.add_subplot(gs[1, 2])
    final_equity = [results[s]['final_equity'] for s in strategies]
    bars = ax6.barh(strategies, final_equity, color=colors_list[:len(strategies)])
    ax6.set_xlabel('Final Equity ($)', fontweight='bold')
    ax6.set_title('Final Portfolio Value', fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(final_equity):
        ax6.text(v + 2000, i, f'${v:,.0f}', va='center', fontweight='bold', fontsize=9)
    
    # 7. Risk-Return scatter
    ax7 = fig.add_subplot(gs[2, :2])
    for idx, strategy in enumerate(strategies):
        m = results[strategy]
        ret = m['return']*100
        vol = m['volatility']*100
        sharpe = m['sharpe']
        size = max(150, sharpe*400)
        ax7.scatter(vol, ret, s=size, alpha=0.6, color=colors_list[idx], 
                   edgecolor='black', linewidth=2, label=f'{strategy} (S={sharpe:.3f})')
    
    ax7.set_xlabel('Volatility (%)', fontweight='bold', fontsize=11)
    ax7.set_ylabel('Return (%)', fontweight='bold', fontsize=11)
    ax7.set_title('Risk-Return Profile (Bubble Size = Sharpe Ratio)', fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.legend(loc='best', fontsize=9)
    
    # 8. Metrics radar chart equivalent (parallel coordinates)
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    # Summary text
    summary_text = "KEY FINDINGS\n" + "="*35 + "\n\n"
    
    best_return = max(results.items(), key=lambda x: x[1]['return'])
    best_sharpe = max(results.items(), key=lambda x: x[1]['sharpe'])
    best_equity = max(results.items(), key=lambda x: x[1]['final_equity'])
    worst_dd = min(results.items(), key=lambda x: x[1]['maxdd'])
    
    summary_text += f"Best Return:\n  {best_return[0]}: {best_return[1]['return']*100:.2f}%\n\n"
    summary_text += f"Best Risk-Adjusted:\n  {best_sharpe[0]}: {best_sharpe[1]['sharpe']:.4f}\n\n"
    summary_text += f"Best Final Equity:\n  {best_equity[0]}: ${best_equity[1]['final_equity']:,.0f}\n\n"
    summary_text += f"Lowest Drawdown:\n  {worst_dd[0]}: {worst_dd[1]['maxdd']*100:.2f}%\n"
    
    ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 9. Metrics correlation
    ax9 = fig.add_subplot(gs[3, :])
    
    # Create metrics matrix for correlation
    metrics_data = {
        'Return': [results[s]['return'] for s in strategies],
        'Sharpe': [results[s]['sharpe'] for s in strategies],
        'Volatility': [results[s]['volatility'] for s in strategies],
        'Max DD': [results[s]['maxdd'] for s in strategies],
        'Turnover': [results[s]['turnover'] for s in strategies],
    }
    
    metrics_df = pd.DataFrame(metrics_data, index=strategies)
    correlation_matrix = metrics_df.corr()
    
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
               ax=ax9, cbar_kws={'label': 'Correlation'}, vmin=-1, vmax=1)
    ax9.set_title('Metrics Correlation Matrix', fontweight='bold', fontsize=12)
    
    plt.savefig('plots/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: plots/comprehensive_analysis.png")
    plt.close()

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS & VISUALIZATION")
    print("="*80)
    
    # Create plots directory
    Path('plots').mkdir(exist_ok=True)
    
    # Load results
    results = load_results()
    print(f"✓ Loaded results for {len(results)} strategies")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_performance_comparison(results)
    plot_metrics_table(results)
    plot_return_distribution(results)
    plot_risk_metrics(results)
    create_comprehensive_analysis(results)
    
    print("\n" + "="*80)
    print("✓ ALL VISUALIZATIONS COMPLETE")
    print("="*80)
    print("\nGenerated plots:")
    print("  - plots/01_performance_comparison.png")
    print("  - plots/06_model_summary.png")
    print("  - plots/10_returns_analysis.png")
    print("  - plots/11_risk_metrics.png")
    print("  - plots/comprehensive_analysis.png")
    print("\nOpen these images to review comprehensive analysis!")

if __name__ == "__main__":
    main()

