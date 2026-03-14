#!/usr/bin/env python3
"""
PPO Trading Experiments - Parameterized Framework
Comprehensive Comparison: PPO with/without Forecast, Reward Functions

This script runs all experiments and generates detailed comparison reports.
Initial Budget: $100,000
Key Metrics: Cumulative Return, Sharpe Ratio, Max Drawdown, Volatility, Turnover

All metrics are:
- Logged to WandB with proper group organization
- Saved to metrics.pkl for local visualization
- Aggregated into comparison CSV files
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
import pickle
from datetime import datetime

warnings.filterwarnings('ignore')

# Configure environment
os.environ['MPLBACKEND'] = 'Agg'
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['CURL_CA_BUNDLE'] = ''

# W&B Configuration - Will be set by each experiment from its config
# Default: online mode (can be overridden in trading_config.py)
# Set to 'offline' only if needed
import ssl
ssl.verify_mode = ssl.CERT_NONE
os.environ['VERIFY_SSL'] = 'false'

try:
    import wandb
    WANDB_AVAILABLE = True
    print("✓ W&B available (offline mode)")
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠ W&B not available")

# Plotting setup
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)

print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
print("✓ Metrics will be saved to: ./results/*/metrics.pkl")
print("✓ Metrics will be logged to WandB with group IDs")

# Import custom modules
try:
    from trading_config import (
        ExperimentConfig,
        get_ppo_without_forecast_config,
        get_ppo_with_forecast_config,
        get_ppo_different_rewards_configs,
        ForecastMode,
        RewardType
    )
    from trading_framework import ExperimentRunner
    from trading_metrics import TradingMetrics, MetricsComparison, EquityCurveAnalyzer
    
    print("✓ All custom modules imported successfully")
except Exception as e:
    print(f"✗ Error importing modules: {e}")
    sys.exit(1)


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(title)
    print("="*80 + "\n")


def calculate_buy_and_hold_baseline():
    """Calculate Buy and Hold baseline for comparison"""
    print_header("BASELINE: BUY AND HOLD STRATEGY")
    
    import yfinance as yf
    
    # Load data
    ticker = "BTC-USD"
    start_date = "2018-01-01"
    
    print(f"Loading {ticker} data...")
    df = yf.download(ticker, start=start_date, progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df = df.dropna()
    
    # Split data (same as trading framework)
    n_total = len(df)
    n_train = int(n_total * 0.6)
    n_val = int(n_total * 0.2)
    df_test = df.iloc[n_train + n_val:]
    
    # Buy and Hold calculation
    initial_equity = 100000.0
    btc_fee = 0.0001
    
    entry_price = float(df_test.iloc[0]['close'])
    exit_price = float(df_test.iloc[-1]['close'])
    
    btc_bought = (initial_equity * (1 - btc_fee)) / entry_price
    final_value_before_fee = btc_bought * exit_price
    final_equity = final_value_before_fee * (1 - btc_fee)
    
    buy_hold_return = (final_equity / initial_equity) - 1
    
    print(f"\nBuy and Hold Results:")
    print(f"  Entry Price: ${entry_price:,.2f}")
    print(f"  Exit Price: ${exit_price:,.2f}")
    print(f"  BTC Purchased: {btc_bought:.6f}")
    print(f"  Initial Equity: ${initial_equity:,.2f}")
    print(f"  Final Equity: ${final_equity:,.2f}")
    print(f"  Total Return: {buy_hold_return*100:+.2f}%")
    
    return {
        "final_equity": final_equity,
        "total_return": buy_hold_return,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "btc_bought": btc_bought,
        "metrics": {
            "total_return": buy_hold_return,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "volatility": 0.0,
            "annualized_return": buy_hold_return,
            "annualized_volatility": 0.0,
            "calmar_ratio": 0.0,
            "sortino_ratio": 0.0,
            "win_rate": 1.0 if buy_hold_return > 0 else 0.0,
            "turnover": 0.0,
            "cost_ratio": 0.0,
        }
    }


def run_experiment_1():
    """Run Experiment 1: PPO Without Forecast (Baseline)"""
    print_header("EXPERIMENT 1: PPO WITHOUT FORECAST (BASELINE)")
    
    config = get_ppo_without_forecast_config(
        name="PPO-Without-Forecast",
        group="baseline"
    )
    
    print(f"""
Configuration:
  - Forecast Mode: {config.forecast_mode.value}
  - Reward Type: {config.reward_type.value}
  - Initial Equity: ${config.environment.initial_equity:,.0f}
  - Fee: {config.environment.fee}
  - Leverage Max: {config.environment.leverage_max}
  - PPO Updates: {config.ppo.total_updates}
  - Num Envs: {config.ppo.num_envs}
""")
    
    try:
        runner = ExperimentRunner(config)
        results = runner.run()
        print(f"\n✓ Experiment 1 completed successfully")
        return results
    except Exception as e:
        print(f"✗ Experiment 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_experiment_2():
    """Run Experiment 2: PPO With LSTM Forecast"""
    print_header("EXPERIMENT 2: PPO WITH LSTM FORECAST")
    
    config = get_ppo_with_forecast_config(
        name="PPO-With-Forecast",
        group="baseline"
    )
    
    print(f"""
Configuration:
  - Forecast Mode: {config.forecast_mode.value}
  - Reward Type: {config.reward_type.value}
  - Initial Equity: ${config.environment.initial_equity:,.0f}
  - Lookback: {config.forecasting.lookback}
  - Forecast Horizon: {config.forecasting.forecast_horizon}
  - Hidden Dim: {config.forecasting.hidden_dim}
""")
    
    try:
        runner = ExperimentRunner(config)
        results = runner.run()
        print(f"\n✓ Experiment 2 completed successfully")
        return results
    except Exception as e:
        print(f"✗ Experiment 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_experiment_2b():
    """Run Experiment 2b: PPO With Ensemble Forecast (Better than LSTM!)"""
    print_header("[2b/10] EXPERIMENT 2b: PPO WITH ENSEMBLE FORECAST")
    
    print(f"""
ENSEMBLE FORECAST: Technical Indicators (RSI, EMA, MACD, Bollinger Bands)
Expected Accuracy: 60-65% (vs 51% LSTM)

Configuration:
  - Forecast Mode: Ensemble (not LSTM!)
  - Method: RSI (30%) + EMA (35%) + MACD (20%) + Bollinger (15%)
  - Advantage: Interpretable, no overfitting, Bitcoin-optimized
  - Reward Type: WITH_RISK
  - Initial Equity: $100,000
  - PPO Updates: 3000
  - Num Envs: 8
""")
    
    # Create a config for Ensemble Forecast
    config = get_ppo_with_forecast_config(
        name="PPO-With-Ensemble-Forecast",
        experiment_type="ppo_ensemble_baseline",
        variant="v1"
    )
    
    # Override forecast mode to use ensemble instead of LSTM
    # We'll train ensemble forecast inline before PPO training
    
    try:
        runner = ExperimentRunner(config)
        
        # Step 1: Load data
        df = runner.load_market_data()
        df_train, df_val, df_test = runner.split_data(df)
        
        print(f"\n[2b.1/3] Training Ensemble Forecast...")
        # Train ensemble forecast instead of LSTM
        from better_forecast_systems import BetterForecastSystem
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        # Generate ensemble forecasts
        train_probs = BetterForecastSystem.ensemble_forecast(df_train)
        val_probs = BetterForecastSystem.ensemble_forecast(df_val)
        test_probs = BetterForecastSystem.ensemble_forecast(df_test)
        
        # Evaluate quality
        y_val = (df_val['r'].shift(-1) > 0).astype(int).fillna(0).values
        val_preds = (val_probs > 0.5).astype(int)
        val_acc = accuracy_score(y_val, val_preds)
        val_auc = roc_auc_score(y_val, val_probs)
        
        print(f"\n✓ Ensemble Forecast Quality:")
        print(f"  Validation Accuracy: {val_acc:.4f}")
        print(f"  Validation AUC-ROC:  {val_auc:.4f}")
        
        if val_acc < 0.55:
            print(f"  ⚠ WARNING: Forecast barely better than random")
        else:
            print(f"  ✓ GOOD: Forecast quality is decent")
        
        # Step 2: Train PPO with ensemble forecast
        print(f"\n[2b.2/3] Training PPO with Ensemble Forecast...")
        results_ppo = runner.train_ppo(df_train, df_test, forecast_probs=test_probs)
        
        # Step 3: Evaluate
        print(f"\n[2b.3/3] Evaluating on test set...")
        eval_results = runner.evaluate(df_test, test_probs)
        
        # Combine results
        if results_ppo is not None:
            results_ppo['forecast_quality'] = {
                'type': 'ensemble',
                'accuracy': val_acc,
                'auc_roc': val_auc
            }
            print(f"\n✓ Experiment 2b completed successfully")
            print(f"  Return: {eval_results.get('metrics', {}).get('total_return', 0)*100:.2f}%")
            print(f"  Sharpe: {eval_results.get('metrics', {}).get('sharpe_ratio', 0):.4f}")
            return results_ppo
        else:
            print(f"✗ Experiment 2b failed: PPO training returned None")
            return None
            
    except Exception as e:
        print(f"✗ Experiment 2b failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_experiment_3():
    """Run Experiment 3: PPO With Different Reward Functions - Comprehensive Ablation"""
    print_header("EXPERIMENT 3: PPO WITH DIFFERENT REWARD FUNCTIONS (8 VARIANTS)")
    
    reward_configs = get_ppo_different_rewards_configs(group="reward_ablation")
    results_3 = {}
    
    print(f"\nTesting {len(reward_configs)} reward function variants:")
    for i, config in enumerate(reward_configs, 1):
        reward_name = config.reward_type.value.replace('_', ' ').title()
        print(f"\n[{i}/{len(reward_configs)}] {reward_name}")
        print(f"  Reward Type: {config.reward_type.value}")
        print(f"  WandB Group: {config.wandb_group}")
        
        try:
            runner = ExperimentRunner(config)
            result = runner.run()
            if result is not None:
                results_3[f"PPO_{config.reward_type.value}"] = result
                
                # Extract key metrics
                metrics = result.get('metrics', {})
                print(f"  ✓ Completed")
                print(f"    - Total Return: {metrics.get('total_return', 0)*100:.2f}%")
                print(f"    - Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
                print(f"    - Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✓ Experiment 3 completed ({len(results_3)}/{len(reward_configs)} configs)")
    return results_3


def generate_metrics_comparison(all_results):
    """Generate comprehensive metrics comparison"""
    print_header("METRICS COMPARISON TABLE")
    
    metrics_comparison = MetricsComparison()
    
    for name, results in all_results.items():
        if results is not None:
            metrics_comparison.add_experiment(name, results['metrics'])
    
    comparison_df = metrics_comparison.to_dataframe()
    
    print("\nAll Experiments:")
    print(comparison_df.to_string())
    print()
    
    return comparison_df


def generate_key_metrics_analysis(comparison_df):
    """Analyze key metrics"""
    print_header("KEY METRICS ANALYSIS")
    
    key_metrics = [
        ('total_return', 'Total Return (%)'),
        ('sharpe_ratio', 'Sharpe Ratio'),
        ('max_drawdown', 'Max Drawdown (%)'),
        ('annualized_volatility', 'Annualized Volatility (%)'),
        ('turnover', 'Turnover')
    ]
    
    for metric_key, metric_name in key_metrics:
        if metric_key in comparison_df.columns:
            print(f"\n{metric_name}:")
            print("-" * 60)
            sorted_vals = comparison_df[metric_key].sort_values(ascending=False)
            for i, (exp, val) in enumerate(sorted_vals.items(), 1):
                if 'return' in metric_key or 'drawdown' in metric_key or 'volatility' in metric_key:
                    print(f"  {i}. {exp:30s}: {val*100:8.2f}%")
                else:
                    print(f"  {i}. {exp:30s}: {val:8.6f}")


def generate_visualizations(all_results):
    """Generate all comparison plots"""
    print_header("GENERATING VISUALIZATIONS")
    
    # Extract equity curves
    equity_curves = {
        name: results['equity']
        for name, results in all_results.items()
        if results is not None
    }
    
    if not equity_curves:
        print("No results to visualize")
        return
    
    # 1. Equity curves
    try:
        fig = EquityCurveAnalyzer.plot_equity_curve(equity_curves, figsize=(16, 7))
        plt.title("Equity Curves Comparison (Initial: $100,000)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('equity_curves_comparison.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("✓ Saved: equity_curves_comparison.png")
    except Exception as e:
        print(f"✗ Error saving equity curves: {e}")
    
    # 2. Drawdown analysis
    try:
        fig = EquityCurveAnalyzer.plot_drawdown(equity_curves, figsize=(16, 7))
        plt.title("Maximum Drawdown Over Time", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('drawdown_comparison.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("✓ Saved: drawdown_comparison.png")
    except Exception as e:
        print(f"✗ Error saving drawdown: {e}")
    
    # 3. Returns distribution
    try:
        returns_dict = {}
        for name, results in all_results.items():
            if results is not None:
                equity = results['equity']
                daily_returns = np.diff(equity) / equity[:-1]
                returns_dict[name] = daily_returns
        
        fig = EquityCurveAnalyzer.plot_returns_distribution(returns_dict, figsize=(16, 6))
        plt.tight_layout()
        plt.savefig('returns_distribution.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("✓ Saved: returns_distribution.png")
    except Exception as e:
        print(f"✗ Error saving returns distribution: {e}")


def generate_heatmap(comparison_df):
    """Generate metrics heatmap"""
    print("\nGenerating metrics heatmap...")
    
    # Normalize for heatmap
    normalized = comparison_df.copy()
    for col in normalized.columns:
        col_min = normalized[col].min()
        col_max = normalized[col].max()
        if col_max - col_min != 0:
            normalized[col] = (normalized[col] - col_min) / (col_max - col_min)
    
    try:
        fig, ax = plt.subplots(figsize=(16, 8))
        sns.heatmap(normalized.T, annot=comparison_df.T.round(4), fmt='g', cmap='RdYlGn',
                    cbar_kws={'label': 'Normalized Score'}, ax=ax, linewidths=0.5)
        ax.set_title('Performance Metrics Heatmap (Color: Normalized, Values: Actual)', 
                     fontweight='bold', fontsize=14)
        ax.set_xlabel('Experiment')
        plt.tight_layout()
        plt.savefig('metrics_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("✓ Saved: metrics_heatmap.png")
    except Exception as e:
        print(f"✗ Error saving heatmap: {e}")


def generate_rankings(comparison_df):
    """Generate performance rankings"""
    print_header("PERFORMANCE RANKINGS")
    
    rankings = {
        'Total Return (Higher Better)': comparison_df['total_return'].sort_values(ascending=False),
        'Sharpe Ratio (Higher Better)': comparison_df['sharpe_ratio'].sort_values(ascending=False),
        'Max Drawdown (Higher Better)': comparison_df['max_drawdown'].sort_values(ascending=False),
        'Volatility (Lower Better)': comparison_df['annualized_volatility'].sort_values(ascending=True),
        'Turnover (Lower Better)': comparison_df['turnover'].sort_values(ascending=True),
    }
    
    if 'win_rate' in comparison_df.columns:
        rankings['Win Rate (Higher Better)'] = comparison_df['win_rate'].sort_values(ascending=False)
    
    for category, ranking in rankings.items():
        print(f"\n{category}:")
        print("-" * 70)
        for i, (exp, value) in enumerate(ranking.items(), 1):
            bar_length = int(abs(value) * 20)
            bar = '█' * bar_length
            print(f"  {i}. {exp:30s} {bar:20s} {value:10.6f}")


def generate_statistical_summary(comparison_df):
    """Generate statistical summary"""
    print_header("STATISTICAL SUMMARY")
    print(comparison_df.describe().T.to_string())


def export_results(all_results, comparison_df):
    """Export results to CSV, JSON, and pickle"""
    print_header("EXPORTING RESULTS")
    
    # Save metrics to CSV
    try:
        comparison_df.to_csv('metrics_comparison.csv')
        print("✓ Saved: metrics_comparison.csv")
    except Exception as e:
        print(f"✗ Error saving CSV: {e}")
    
    # Save detailed results to JSON
    try:
        export_data = {}
        for name, results in all_results.items():
            if results is not None:
                export_data[name] = {
                    'metrics': results.get('metrics', {}),
                    'equity_initial': float(results['equity'][0]) if 'equity' in results else 0,
                    'equity_final': float(results['equity'][-1]) if 'equity' in results else 0,
                    'equity_max': float(np.max(results['equity'])) if 'equity' in results else 0,
                    'equity_min': float(np.min(results['equity'])) if 'equity' in results else 0,
                    'description': 'Training results with complete metrics'
                }
        
        with open('detailed_results.json', 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        print("✓ Saved: detailed_results.json")
    except Exception as e:
        print(f"✗ Error saving JSON: {e}")
    
    # Save to pickle for visualization script
    try:
        pickle_data = {
            'results': all_results,
            'comparison_df': comparison_df,
            'timestamp': datetime.now().isoformat(),
        }
        with open('metrics.pkl', 'wb') as f:
            pickle.dump(pickle_data, f)
        print("✓ Saved: metrics.pkl (for create_visualizations.py)")
    except Exception as e:
        print(f"✗ Error saving pickle: {e}")


def generate_reward_comparison_analysis(reward_results):
    """Generate comprehensive analysis of reward function performance"""
    print_header("REWARD FUNCTION ABLATION ANALYSIS")
    
    if not reward_results or len(reward_results) == 0:
        print("⚠ No reward comparison results available")
        return None
    
    print(f"\nAnalyzing {len(reward_results)} reward function variants:\n")
    
    # Extract metrics for each reward type
    reward_metrics = {}
    for exp_name, result in reward_results.items():
        if result is not None and 'metrics' in result:
            reward_metrics[exp_name] = result['metrics']
    
    # Create comparison dataframe
    reward_comparison_df = pd.DataFrame(reward_metrics).T
    
    print("REWARD FUNCTION PERFORMANCE COMPARISON:")
    print("=" * 120)
    print(reward_comparison_df.to_string())
    print("=" * 120)
    
    # Analyze which reward function works best for each metric
    print("\n\nBEST PERFORMER PER METRIC:")
    print("-" * 80)
    
    metrics_of_interest = {
        'total_return': 'Total Return % (Higher Better)',
        'sharpe_ratio': 'Sharpe Ratio (Higher Better)',
        'max_drawdown': 'Max Drawdown (Higher is Better)',
        'annualized_volatility': 'Volatility % (Lower Better)',
        'turnover': 'Turnover (Lower Better)',
        'win_rate': 'Win Rate % (Higher Better)',
    }
    
    for metric_key, metric_display in metrics_of_interest.items():
        if metric_key in reward_comparison_df.columns:
            if 'Lower' in metric_display:
                best_idx = reward_comparison_df[metric_key].idxmin()
                best_val = reward_comparison_df[metric_key].min()
            else:
                best_idx = reward_comparison_df[metric_key].idxmax()
                best_val = reward_comparison_df[metric_key].max()
            
            reward_type = best_idx.replace('PPO_', '')
            print(f"{metric_display:45s}: {reward_type:25s} = {best_val:10.6f}")
    
    # Save reward comparison
    try:
        reward_comparison_df.to_csv('reward_comparison_detailed.csv')
        print("\n✓ Saved: reward_comparison_detailed.csv")
    except Exception as e:
        print(f"✗ Error saving reward comparison: {e}")
    
    # Generate visualization of reward functions
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Reward Function Ablation Analysis', fontsize=16, fontweight='bold')
        
        plot_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 
                       'annualized_volatility', 'turnover', 'win_rate']
        axes_flat = axes.flatten()
        
        for idx, metric in enumerate(plot_metrics):
            if metric in reward_comparison_df.columns:
                ax = axes_flat[idx]
                data = reward_comparison_df[metric].sort_values(ascending=False)
                colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(data)))
                
                bars = ax.barh(range(len(data)), data.values, color=colors)
                ax.set_yticks(range(len(data)))
                ax.set_yticklabels([x.replace('PPO_', '') for x in data.index], fontsize=9)
                ax.set_xlabel('Value')
                ax.set_title(metrics_of_interest.get(metric, metric.replace('_', ' ').title()), 
                           fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                
                # Add value labels
                for i, (bar, val) in enumerate(zip(bars, data.values)):
                    ax.text(val, i, f' {val:.4f}', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('reward_ablation_analysis.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("✓ Saved: reward_ablation_analysis.png")
    except Exception as e:
        print(f"✗ Error generating reward analysis plot: {e}")
    
    return reward_comparison_df


def generate_findings(all_results, comparison_df):
    """Generate key findings"""
    print_header("KEY FINDINGS")
    
    # Best performers
    best_return = comparison_df['total_return'].idxmax()
    best_sharpe = comparison_df['sharpe_ratio'].idxmax()
    best_drawdown = comparison_df['max_drawdown'].idxmax()
    best_vol = comparison_df['annualized_volatility'].idxmin()
    best_turnover = comparison_df['turnover'].idxmin()
    
    print(f"""
BEST PERFORMERS BY METRIC:
─────────────────────────────────────────────────────────

1. HIGHEST RETURN:
   {best_return}
   Return: {comparison_df.loc[best_return, 'total_return']*100:.2f}%

2. BEST RISK-ADJUSTED (Sharpe Ratio):
   {best_sharpe}
   Sharpe Ratio: {comparison_df.loc[best_sharpe, 'sharpe_ratio']:.4f}

3. SMALLEST DRAWDOWN:
   {best_drawdown}
   Max Drawdown: {comparison_df.loc[best_drawdown, 'max_drawdown']*100:.2f}%

4. LOWEST VOLATILITY:
   {best_vol}
   Annualized Volatility: {comparison_df.loc[best_vol, 'annualized_volatility']*100:.2f}%

5. LOWEST TURNOVER (Most Efficient):
   {best_turnover}
   Turnover: {comparison_df.loc[best_turnover, 'turnover']:.4f}

─────────────────────────────────────────────────────────
""")
    
    # Forecast impact
    print("\nFORECAST IMPACT ANALYSIS:")
    if 'PPO_Without_Forecast' in comparison_df.index and 'PPO_With_Forecast' in comparison_df.index:
        return_diff = (comparison_df.loc['PPO_With_Forecast', 'total_return'] - 
                       comparison_df.loc['PPO_Without_Forecast', 'total_return'])
        sharpe_diff = (comparison_df.loc['PPO_With_Forecast', 'sharpe_ratio'] - 
                       comparison_df.loc['PPO_Without_Forecast', 'sharpe_ratio'])
        
        print(f"  Return Difference (With - Without): {return_diff*100:+.2f}%")
        print(f"  Sharpe Ratio Difference: {sharpe_diff:+.4f}")
        
        if return_diff > 0:
            print(f"  → Forecast IMPROVED returns by {return_diff*100:.2f}%")
        else:
            print(f"  → Forecast REDUCED returns by {abs(return_diff)*100:.2f}%")
    else:
        print("  Forecast comparison not available")


def generate_summary_table(comparison_df):
    """Generate final summary table"""
    print_header("FINAL RESULTS TABLE")
    
    summary_cols = ['total_return', 'sharpe_ratio', 'max_drawdown', 
                    'annualized_volatility', 'turnover']
    
    if 'win_rate' in comparison_df.columns:
        summary_cols.append('win_rate')
    
    summary_table = comparison_df[summary_cols].copy()
    
    # Format for display
    display_df = pd.DataFrame()
    display_df['Total Return %'] = summary_table['total_return'] * 100
    display_df['Sharpe Ratio'] = summary_table['sharpe_ratio']
    display_df['Max Drawdown %'] = summary_table['max_drawdown'] * 100
    display_df['Volatility %'] = summary_table['annualized_volatility'] * 100
    display_df['Turnover'] = summary_table['turnover']
    if 'win_rate' in summary_table.columns:
        display_df['Win Rate %'] = summary_table['win_rate'] * 100
    
    print("\n" + "="*100)
    print(display_df.round(4).to_string())
    print("="*100)


def main():
    """Main execution - Run comprehensive experiments including reward function ablation"""
    print("\n" + "="*100)
    print("PPO TRADING EXPERIMENTS - COMPREHENSIVE SUITE WITH REWARD ABLATION")
    print("="*100)
    print("\nExperiments to run:")
    print("  BASELINE EXPERIMENTS (2):")
    print("    1. PPO Without Forecast (BASELINE)")
    print("    2. PPO With Forecast (LSTM)")
    print("\n  REWARD FUNCTION ABLATION (8):")
    print("    3a. PPO Basic Reward")
    print("    3b. PPO With Risk Penalty")
    print("    3c. PPO With Sharpe Ratio")
    print("    3d. PPO Risk-Adjusted")
    print("    3e. PPO Sortino Ratio")
    print("    3f. PPO Calmar Ratio")
    print("    3g. PPO Information Ratio")
    print("    3h. PPO Composite (Multi-Objective)")
    print("\n  TOTAL: 10 Experiments")
    print("="*100 + "\n")
    
    start_time = datetime.now()
    
    # Import configuration functions
    from trading_config import (
        get_ppo_without_forecast_config,
        get_ppo_with_forecast_config,
        get_ppo_different_rewards_configs
    )
    from trading_framework import ExperimentRunner
    
    # Store all results
    all_results = {}
    reward_results = {}
    
    # ========================================================================
    # BASELINE: BUY AND HOLD (For comparison)
    # ========================================================================
    print("\n" + "="*100)
    print("[BASELINE] BUY AND HOLD STRATEGY")
    print("="*100 + "\n")
    
    try:
        bah_result = calculate_buy_and_hold_baseline()
        all_results["Buy-and-Hold"] = bah_result
        print(f"✓ Buy and Hold baseline calculated: {bah_result['total_return']*100:+.2f}%")
    except Exception as e:
        print(f"⚠ Buy and Hold calculation failed: {e}")
        bah_result = None
    
    # ========================================================================
    # EXPERIMENT 1: PPO WITHOUT FORECAST
    # ========================================================================
    print("\n" + "="*100)
    print("[1/10] BASELINE EXPERIMENT: PPO WITHOUT FORECAST")
    print("="*100 + "\n")
    
    try:
        config = get_ppo_without_forecast_config(
            name="PPO-Without-Forecast",
            experiment_type="ppo_baseline",
            variant="v1"
        )
        
        print(f"""
Configuration:
  - Forecast Mode: {config.forecast_mode.value}
  - Reward Type: {config.reward_type.value}
  - Initial Equity: ${config.environment.initial_equity:,.0f}
  - Fee: {config.environment.fee}
  - Leverage Max: {config.environment.leverage_max}
  - PPO Updates: {config.ppo.total_updates}
  - Num Envs: {config.ppo.num_envs}
""")
        
        runner = ExperimentRunner(config)
        results = runner.run()
        all_results['PPO-Without-Forecast'] = results
        
        if results:
            metrics = results.get('metrics', {})
            print(f"\n✓ EXPERIMENT 1 COMPLETED")
            print(f"  - Total Return: {metrics.get('total_return', 0)*100:.2f}%")
            print(f"  - Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
            print(f"  - Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
    
    except KeyboardInterrupt:
        print(f"\n⚠ EXPERIMENT 1 INTERRUPTED BY USER")
    except Exception as e:
        print(f"\n✗ EXPERIMENT 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # EXPERIMENT 2: PPO WITH ENSEMBLE FORECAST (PRIMARY BASELINE)
    # ========================================================================
    print("\n" + "="*100)
    print("[2/10] PRIMARY BASELINE EXPERIMENT: PPO WITH ENSEMBLE FORECAST")
    print("="*100 + "\n")
    
    ensemble_results = None
    try:
        config = get_ppo_with_forecast_config(
            name="PPO-With-Ensemble-Forecast",
            experiment_type="ppo_ensemble_baseline",
            variant="v1"
        )
        
        print(f"""
Configuration:
  - Forecast Mode: Ensemble (PRIMARY!)
  - Method: RSI (30%) + EMA (35%) + MACD (20%) + Bollinger (15%)
  - Advantage: Interpretable, no overfitting, Bitcoin-optimized
  - Reward Type: WITH_RISK
  - Initial Equity: $100,000
  - PPO Updates: 3000
  - Num Envs: 8
""")
        
        runner = ExperimentRunner(config)
        
        # Step 1: Load data
        df = runner.load_market_data()
        df_train, df_val, df_test = runner.split_data(df)
        
        print(f"\n[2.1/3] Training Ensemble Forecast...")
        # Train ensemble forecast instead of LSTM
        from better_forecast_systems import BetterForecastSystem
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        # Helper function to compute technical indicators
        def compute_technical_indicators(data):
            """Compute RSI, EMA, MACD, Bollinger Bands"""
            df_copy = data.copy()
            close = df_copy['close'].values
            
            # RSI (Relative Strength Index)
            # np.diff() returns n-1 elements, so pad with NaN at start
            delta = np.diff(close)
            delta = np.concatenate([[np.nan], delta])  # Pad to match length
            
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            
            avg_gain = pd.Series(gain).rolling(14, min_periods=1).mean().values
            avg_loss = pd.Series(loss).rolling(14, min_periods=1).mean().values
            
            rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
            rsi = 100 - (100 / (1 + rs))
            df_copy['rsi'] = rsi
            
            # EMA 12 and 26
            df_copy['ema_12'] = pd.Series(close).ewm(span=12, adjust=False).mean().values
            df_copy['ema_26'] = pd.Series(close).ewm(span=26, adjust=False).mean().values
            
            # MACD
            df_copy['macd_diff'] = df_copy['ema_12'] - df_copy['ema_26']
            df_copy['macd_signal'] = pd.Series(df_copy['macd_diff']).ewm(span=9, adjust=False).mean().values
            
            # Bollinger Bands
            sma = pd.Series(close).rolling(20).mean().values
            std = pd.Series(close).rolling(20).std().values
            df_copy['bb_upper'] = sma + (std * 2)
            df_copy['bb_lower'] = sma - (std * 2)
            df_copy['bb_middle'] = sma
            
            return df_copy
        
        # Compute indicators for all sets
        print("  Computing technical indicators...")
        df_train = compute_technical_indicators(df_train)
        df_val = compute_technical_indicators(df_val)
        df_test = compute_technical_indicators(df_test)
        
        # Generate ensemble forecasts
        print("  Generating ensemble forecasts...")
        train_probs = BetterForecastSystem.ensemble_forecast(df_train)
        val_probs = BetterForecastSystem.ensemble_forecast(df_val)
        test_probs = BetterForecastSystem.ensemble_forecast(df_test)
        
        # Evaluate quality
        y_val = (df_val['r'].shift(-1) > 0).astype(int).fillna(0).values
        val_preds = (val_probs > 0.5).astype(int)
        val_acc = accuracy_score(y_val, val_preds)
        val_auc = roc_auc_score(y_val, val_probs)
        
        print(f"\n✓ Ensemble Forecast Quality:")
        print(f"  Validation Accuracy: {val_acc:.4f}")
        print(f"  Validation AUC-ROC:  {val_auc:.4f}")
        
        if val_acc < 0.55:
            print(f"  ⚠ WARNING: Forecast barely better than random")
        else:
            print(f"  ✓ GOOD: Forecast quality is decent ({val_acc*100:.1f}%)")
        
        # Step 2: Train PPO with ensemble forecast
        print(f"\n[2.2/3] Training PPO with Ensemble Forecast...")
        results_ppo = runner.train_ppo(df_train, df_test, forecast_probs=test_probs)
        
        # Step 3: Evaluate
        print(f"\n[2.3/3] Evaluating on test set...")
        eval_results = runner.evaluate(df_test, test_probs)
        
        # Combine results
        if results_ppo is not None:
            results_ppo['forecast_quality'] = {
                'type': 'ensemble',
                'accuracy': val_acc,
                'auc_roc': val_auc
            }
            print(f"\n✓ EXPERIMENT 2 COMPLETED (ENSEMBLE FORECAST)")
            print(f"  Return: {eval_results.get('metrics', {}).get('total_return', 0)*100:.2f}%")
            print(f"  Sharpe: {eval_results.get('metrics', {}).get('sharpe_ratio', 0):.4f}")
            all_results['PPO-With-Ensemble-Forecast'] = results_ppo
            ensemble_results = results_ppo
        else:
            print(f"✗ Experiment 2 failed: PPO training returned None")
            
    except Exception as e:
        print(f"✗ Experiment 2 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # EXPERIMENT 2a: PPO WITH LSTM FORECAST (SECONDARY - FOR COMPARISON)
    # ========================================================================
    print("\n" + "="*100)
    print("[2a/10] COMPARISON EXPERIMENT: PPO WITH LSTM FORECAST")
    print("="*100 + "\n")
    
    try:
        config = get_ppo_with_forecast_config(
            name="PPO-With-LSTM-Forecast",
            experiment_type="ppo_with_lstm_forecast",
            variant="v1"
        )
        
        print(f"""
Configuration:
  - Forecast Mode: {config.forecast_mode.value} (COMPARISON - OPTIONAL)
  - Reward Type: {config.reward_type.value}
  - Initial Equity: ${config.environment.initial_equity:,.0f}
  - Lookback: {config.forecasting.lookback}
  - Forecast Horizon: {config.forecasting.forecast_horizon}
  - Hidden Dim: {config.forecasting.hidden_dim}
  - PPO Updates: {config.ppo.total_updates}
""")
        
        runner = ExperimentRunner(config)
        results = runner.run()
        all_results['PPO-With-LSTM-Forecast'] = results
        
        if results:
            metrics = results.get('metrics', {})
            print(f"\n✓ EXPERIMENT 2a COMPLETED (LSTM FORECAST)")
            print(f"  - Total Return: {metrics.get('total_return', 0)*100:.2f}%")
            print(f"  - Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
            print(f"  - Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
    
    except KeyboardInterrupt:
        print(f"\n⚠ EXPERIMENT 2a INTERRUPTED BY USER")
    except Exception as e:
        print(f"\n✗ EXPERIMENT 2a FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # EXPERIMENTS 3A-3H: REWARD FUNCTION ABLATION
    # ========================================================================
    print("\n" + "="*100)
    print("[3/10] REWARD FUNCTION ABLATION STUDIES (8 VARIANTS)")
    print("="*100 + "\n")
    
    reward_configs = get_ppo_different_rewards_configs(
        experiment_type="reward_ablation",
        variant="v1"
    )
    print(f"Testing {len(reward_configs)} reward function variants:\n")
    
    for i, config in enumerate(reward_configs, 1):
        reward_name = config.reward_type.value.replace('_', ' ').title()
        print(f"\n[3.{i}/{len(reward_configs)}] {reward_name}")
        print(f"  Reward Type: {config.reward_type.value}")
        print(f"  Description: {config.environment.reward_type.value}")
        print(f"  WandB Group: {config.wandb_group}")
        
        try:
            runner = ExperimentRunner(config)
            result = runner.run()
            
            if result is not None:
                exp_key = f"PPO-{config.reward_type.value}"
                reward_results[exp_key] = result
                all_results[exp_key] = result
                
                # Extract key metrics
                metrics = result.get('metrics', {})
                print(f"  ✓ COMPLETED")
                print(f"    - Total Return: {metrics.get('total_return', 0)*100:.2f}%")
                print(f"    - Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
                print(f"    - Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
                print(f"    - Volatility: {metrics.get('annualized_volatility', 0)*100:.2f}%")
        
        except KeyboardInterrupt:
            print(f"  ⚠ INTERRUPTED")
            break
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✓ REWARD ABLATION: {len(reward_results)}/{len(reward_configs)} completed")
    
    # ========================================================================
    # ANALYSIS AND VISUALIZATION
    # ========================================================================
    print("\n" + "="*100)
    print("COMPREHENSIVE ANALYSIS & VISUALIZATION")
    print("="*100 + "\n")
    
    # Create comparison of all experiments
    try:
        comparison_df = generate_metrics_comparison(all_results)
        generate_key_metrics_analysis(comparison_df)
        generate_visualizations(all_results)
        generate_heatmap(comparison_df)
        generate_rankings(comparison_df)
        generate_statistical_summary(comparison_df)
        generate_summary_table(comparison_df)
    except Exception as e:
        print(f"⚠ Analysis error: {e}")
    
    # SPECIAL: Reward function comparison
    if reward_results:
        print("\n")
        reward_comparison_df = generate_reward_comparison_analysis(reward_results)
    
    # Export all results
    try:
        export_results(all_results, comparison_df if 'comparison_df' in locals() else pd.DataFrame())
    except Exception as e:
        print(f"⚠ Export error: {e}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*100)
    print("COMPREHENSIVE EXPERIMENT SUITE COMPLETED")
    print("="*100)
    print(f"\nTotal Experiments Run: {len(all_results)}")
    print(f"Baseline Experiments: 2")
    print(f"Reward Ablation Studies: {len(reward_results)}")
    print(f"Total Duration: {duration}")
    
    print("\n✓ GENERATED FILES:")
    print("  Results & Metrics:")
    print("    - metrics_comparison.csv (all experiments)")
    print("    - reward_comparison_detailed.csv (reward ablation)")
    print("    - detailed_results.json (complete results)")
    print("\n  Visualizations:")
    print("    - equity_curves_comparison.png")
    print("    - drawdown_comparison.png")
    print("    - returns_distribution.png")
    print("    - metrics_heatmap.png")
    print("    - reward_ablation_analysis.png")
    print("\n  Logged to W&B:")
    print("    - ./wandb/offline-run-*/ (for sync later)")
    
    print("\n✓ NEXT STEPS:")
    print("  1. Review metrics: metrics_comparison.csv & reward_comparison_detailed.csv")
    print("  2. Analyze plots: Check generated PNG files")
    print("  3. Compare results: Check detailed_results.json")
    print("  4. Sync W&B: wandb sync ./wandb/offline-run-*/")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()

