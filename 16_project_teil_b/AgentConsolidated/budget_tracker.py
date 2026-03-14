"""
Budget and Liquidity Tracker
Tracks initial capital, transaction costs, and equity evolution over time
Logs timeseries metrics to WandB for all experiments
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Try to import WandB
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class BudgetTracker:
    """Tracks portfolio budget, equity, and transaction costs over time
    
    Logs timeseries metrics to WandB at each step if available.
    Falls back gracefully if WandB is not configured.
    """
    
    def __init__(self, initial_equity: float, dates: list = None, enable_wandb_logging: bool = True):
        self.initial_equity = initial_equity
        self.dates = dates if dates is not None else []
        self.enable_wandb_logging = enable_wandb_logging and WANDB_AVAILABLE
        
        # Time series tracking
        self.equity_values = []
        self.cash_values = []
        self.position_values = []
        self.costs = []
        self.pnl_values = []
        self.positions = []
        self.drawdowns = []
        self.cumulative_returns = []
        
        # Step counter for logging
        self.step_counter = 0
        
    def record_step(self, equity: float, position: float, cost: float, 
                   pnl: float, drawdown: float, cumulative_return: float):
        """Record one time step and log to WandB"""
        # Record locally
        self.equity_values.append(equity)
        self.positions.append(position)
        self.costs.append(cost)
        self.pnl_values.append(pnl)
        self.drawdowns.append(drawdown)
        self.cumulative_returns.append(cumulative_return)
        self.step_counter += 1
        
        # Log to WandB if enabled
        if self.enable_wandb_logging:
            self._log_step_to_wandb()
    
    def _log_step_to_wandb(self):
        """Log current step metrics to WandB
        
        Called after each step to track complete portfolio trajectory.
        Includes all equity, position, cost, and drawdown metrics.
        """
        try:
            # Calculate rolling metrics
            equity_array = np.array(self.equity_values)
            costs_array = np.array(self.costs)
            pnl_array = np.array(self.pnl_values)
            positions_array = np.array(self.positions)
            drawdowns_array = np.array(self.drawdowns)
            
            # Daily returns
            if len(equity_array) > 1:
                daily_returns = np.diff(equity_array) / equity_array[:-1]
                daily_return = daily_returns[-1] if len(daily_returns) > 0 else 0
                avg_daily_pnl = np.mean(pnl_array[-min(20, len(pnl_array)):]) if len(pnl_array) > 0 else 0
                rolling_volatility = np.std(daily_returns[-min(20, len(daily_returns)):]) if len(daily_returns) > 0 else 0
                max_equity_so_far = np.max(equity_array)
                current_drawdown_from_peak = (max_equity_so_far - equity_array[-1]) / max_equity_so_far if max_equity_so_far > 0 else 0
            else:
                daily_return = 0
                avg_daily_pnl = 0
                rolling_volatility = 0
                current_drawdown_from_peak = 0
            
            # Build timeseries metrics dict
            wandb_timeseries = {
                # Equity metrics
                "timeseries/equity": self.equity_values[-1],
                "timeseries/total_return": (self.equity_values[-1] - self.initial_equity) / self.initial_equity if len(self.equity_values) > 0 else 0,
                "timeseries/cumulative_pnl": np.sum(pnl_array) if len(pnl_array) > 0 else 0,
                
                # Cost metrics
                "timeseries/total_costs": np.sum(costs_array),
                "timeseries/daily_cost": costs_array[-1] if len(costs_array) > 0 else 0,
                "timeseries/cumulative_costs": np.sum(costs_array),
                
                # PnL metrics
                "timeseries/daily_pnl": pnl_array[-1] if len(pnl_array) > 0 else 0,
                "timeseries/avg_daily_pnl": avg_daily_pnl,
                
                # Position metrics
                "timeseries/position_size": self.positions[-1] if len(self.positions) > 0 else 0,
                "timeseries/position_absolute": abs(self.positions[-1]) if len(self.positions) > 0 else 0,
                "timeseries/avg_position": np.mean(np.abs(positions_array)) if len(positions_array) > 0 else 0,
                
                # Drawdown metrics
                "timeseries/drawdown": self.drawdowns[-1] if len(self.drawdowns) > 0 else 0,
                "timeseries/drawdown_from_peak": current_drawdown_from_peak,
                "timeseries/max_drawdown_so_far": np.min(drawdowns_array) if len(drawdowns_array) > 0 else 0,
                
                # Return metrics
                "timeseries/cumulative_return": self.cumulative_returns[-1] if len(self.cumulative_returns) > 0 else 0,
                "timeseries/daily_return": daily_return,
                
                # Risk metrics
                "timeseries/rolling_volatility_20": rolling_volatility,
                
                # Step counter
                "timeseries/step": self.step_counter,
            }
            
            # Log to WandB
            wandb.log(wandb_timeseries)
            
        except Exception as e:
            # Silent fail - don't interrupt evaluation if WandB logging fails
            pass
    
    def get_summary(self) -> dict:
        """Get summary statistics"""
        equity_array = np.array(self.equity_values)
        costs_array = np.array(self.costs)
        pnl_array = np.array(self.pnl_values)
        
        return {
            'initial_equity': self.initial_equity,
            'final_equity': equity_array[-1] if len(equity_array) > 0 else self.initial_equity,
            'total_return': (equity_array[-1] - self.initial_equity) / self.initial_equity if len(equity_array) > 0 else 0.0,
            'total_costs': np.sum(costs_array),
            'total_pnl': np.sum(pnl_array),
            'avg_daily_pnl': np.mean(pnl_array) if len(pnl_array) > 0 else 0.0,
            'max_drawdown': np.min(self.drawdowns) if len(self.drawdowns) > 0 else 0.0,
            'num_steps': len(equity_array),
            'avg_position': np.mean(np.abs(self.positions)) if len(self.positions) > 0 else 0.0,
        }
    
    def create_report_df(self) -> pd.DataFrame:
        """Create a detailed report DataFrame"""
        data = {
            'Time_Step': range(len(self.equity_values)),
            'Equity': self.equity_values,
            'Position': self.positions,
            'Daily_Cost': self.costs,
            'Daily_PnL': self.pnl_values,
            'Drawdown': self.drawdowns,
            'Cumulative_Return': self.cumulative_returns,
        }
        return pd.DataFrame(data)


def plot_equity_curve(equity_values: list, initial_equity: float, 
                     title: str = "Equity Curve Over Time", 
                     save_path: str = None):
    """Plot equity curve with initial and final values"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    steps = np.arange(len(equity_values))
    equity_array = np.array(equity_values)
    
    # Plot equity curve
    ax.plot(steps, equity_array, linewidth=2, label='Equity Value', color='#2E86AB')
    ax.axhline(y=initial_equity, color='green', linestyle='--', 
              linewidth=2, label=f'Initial Equity: ${initial_equity:,.0f}', alpha=0.7)
    
    # Fill area
    ax.fill_between(steps, initial_equity, equity_array, 
                    where=(equity_array >= initial_equity),
                    alpha=0.2, color='green', label='Profit')
    ax.fill_between(steps, initial_equity, equity_array, 
                    where=(equity_array < initial_equity),
                    alpha=0.2, color='red', label='Loss')
    
    # Format
    final_equity = equity_array[-1]
    total_return = (final_equity - initial_equity) / initial_equity
    
    ax.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Equity ($)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Add text box with final stats
    textstr = f'Initial: ${initial_equity:,.0f}\nFinal: ${final_equity:,.0f}\nReturn: {total_return*100:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_budget_breakdown(equity_values: list, positions: list, 
                         initial_equity: float,
                         title: str = "Budget Breakdown Over Time",
                         save_path: str = None):
    """Plot how equity is allocated (position value vs cash)"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    steps = np.arange(len(equity_values))
    equity_array = np.array(equity_values)
    positions_array = np.array(positions)
    
    # Position value approximation (position * price, using normalized price)
    # For simplicity, we show position exposure
    position_exposure = np.abs(positions_array)
    
    ax.plot(steps, equity_array, linewidth=2.5, label='Total Equity', 
           color='#2E86AB', zorder=3)
    ax.scatter(steps[::max(1, len(steps)//50)], equity_array[::max(1, len(steps)//50)], 
              s=30, alpha=0.5, color='#2E86AB')
    
    # Add position exposure as secondary axis
    ax2 = ax.twinx()
    ax2.bar(steps, position_exposure, alpha=0.3, label='Position Exposure',
           color='#A23B72', width=1.0)
    ax2.set_ylabel('Position Exposure (Leverage)', fontsize=11, fontweight='bold', color='#A23B72')
    ax2.tick_params(axis='y', labelcolor='#A23B72')
    
    ax.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Equity ($)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_transaction_costs_impact(costs: list, equity_values: list,
                                 title: str = "Transaction Costs Impact",
                                 save_path: str = None):
    """Plot cumulative transaction costs"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    steps = np.arange(len(costs))
    costs_array = np.array(costs)
    equity_array = np.array(equity_values)
    
    # Daily costs
    ax1.bar(steps, costs_array, alpha=0.6, color='red', width=1.0)
    ax1.set_ylabel('Daily Transaction Cost ($)', fontsize=11, fontweight='bold')
    ax1.set_title('Daily Transaction Costs', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))
    
    # Cumulative costs
    cumulative_costs = np.cumsum(costs_array)
    ax2.plot(steps, cumulative_costs, linewidth=2.5, color='darkred', label='Cumulative Cost')
    ax2.fill_between(steps, 0, cumulative_costs, alpha=0.3, color='red')
    
    # Cost as percentage of equity
    cost_pct = (cumulative_costs / equity_array[0]) * 100
    ax2_pct = ax2.twinx()
    ax2_pct.plot(steps, cost_pct, linewidth=2, color='orange', linestyle='--', 
                label='Cost (% of Initial Equity)')
    ax2_pct.set_ylabel('Cost (% of Initial Equity)', fontsize=11, fontweight='bold', color='orange')
    ax2_pct.tick_params(axis='y', labelcolor='orange')
    
    ax2.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Cost ($)', fontsize=11, fontweight='bold')
    ax2.set_title('Cumulative Transaction Costs', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))
    ax2.legend(loc='upper left', fontsize=10)
    ax2_pct.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, (ax1, ax2)


def plot_returns_and_drawdown(cumulative_returns: list, drawdowns: list,
                             title: str = "Cumulative Returns and Drawdown",
                             save_path: str = None):
    """Plot cumulative returns with drawdown overlay"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    steps = np.arange(len(cumulative_returns))
    returns_array = np.array(cumulative_returns)
    drawdowns_array = np.array(drawdowns)
    
    # Cumulative returns
    ax.plot(steps, returns_array * 100, linewidth=2.5, label='Cumulative Return',
           color='green', zorder=2)
    ax.fill_between(steps, 0, returns_array * 100, where=(returns_array >= 0),
                   alpha=0.2, color='green')
    ax.fill_between(steps, 0, returns_array * 100, where=(returns_array < 0),
                   alpha=0.2, color='red')
    
    # Drawdown
    ax2 = ax.twinx()
    ax2.fill_between(steps, 0, drawdowns_array * 100, alpha=0.3, color='red',
                    label='Drawdown')
    ax2.set_ylabel('Drawdown (%)', fontsize=11, fontweight='bold', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Return (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def create_summary_dashboard(tracker: BudgetTracker, save_dir: str = "./results"):
    """Create comprehensive dashboard with all visualizations"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    summary = tracker.get_summary()
    
    print("\n" + "="*70)
    print("BUDGET & LIQUIDITY SUMMARY")
    print("="*70)
    print(f"Initial Equity:        ${summary['initial_equity']:>15,.2f}")
    print(f"Final Equity:          ${summary['final_equity']:>15,.2f}")
    print(f"Total Return:          {summary['total_return']*100:>15.2f}%")
    print(f"Total Costs:           ${summary['total_costs']:>15,.2f}")
    print(f"Total PnL:             ${summary['total_pnl']:>15,.2f}")
    print(f"Avg Daily PnL:         ${summary['avg_daily_pnl']:>15,.2f}")
    print(f"Max Drawdown:          {summary['max_drawdown']*100:>15.2f}%")
    print(f"Avg Position Size:     {summary['avg_position']:>15.4f}")
    print(f"Number of Steps:       {summary['num_steps']:>15d}")
    print("="*70 + "\n")
    
    # Create plots
    print("Creating visualizations...")
    
    fig1, _ = plot_equity_curve(
        tracker.equity_values,
        tracker.initial_equity,
        save_path=f"{save_dir}/01_equity_curve.png"
    )
    print("  ✓ Equity curve plot")
    
    fig2, _ = plot_budget_breakdown(
        tracker.equity_values,
        tracker.positions,
        tracker.initial_equity,
        save_path=f"{save_dir}/02_budget_breakdown.png"
    )
    print("  ✓ Budget breakdown plot")
    
    fig3, _ = plot_transaction_costs_impact(
        tracker.costs,
        tracker.equity_values,
        save_path=f"{save_dir}/03_transaction_costs.png"
    )
    print("  ✓ Transaction costs plot")
    
    fig4, _ = plot_returns_and_drawdown(
        tracker.cumulative_returns,
        tracker.drawdowns,
        save_path=f"{save_dir}/04_returns_drawdown.png"
    )
    print("  ✓ Returns and drawdown plot")
    
    # Create detailed report
    df_report = tracker.create_report_df()
    df_report.to_csv(f"{save_dir}/budget_report.csv", index=False)
    print("  ✓ Detailed report CSV")
    
    # Create summary report
    with open(f"{save_dir}/budget_summary.txt", "w") as f:
        f.write("="*70 + "\n")
        f.write("BUDGET & LIQUIDITY SUMMARY\n")
        f.write("="*70 + "\n")
        f.write(f"Initial Equity:        ${summary['initial_equity']:>15,.2f}\n")
        f.write(f"Final Equity:          ${summary['final_equity']:>15,.2f}\n")
        f.write(f"Total Return:          {summary['total_return']*100:>15.2f}%\n")
        f.write(f"Total Costs:           ${summary['total_costs']:>15,.2f}\n")
        f.write(f"Total PnL:             ${summary['total_pnl']:>15,.2f}\n")
        f.write(f"Avg Daily PnL:         ${summary['avg_daily_pnl']:>15,.2f}\n")
        f.write(f"Max Drawdown:          {summary['max_drawdown']*100:>15.2f}%\n")
        f.write(f"Avg Position Size:     {summary['avg_position']:>15.4f}\n")
        f.write(f"Number of Steps:       {summary['num_steps']:>15d}\n")
        f.write("="*70 + "\n")
    print("  ✓ Summary report text file")
    
    print(f"\nAll visualizations saved to: {save_dir}/")
    
    return summary, df_report


if __name__ == "__main__":
    # Example usage
    print("Budget Tracker Module Loaded")
    print("Use: tracker = BudgetTracker(100000)")
    print("     tracker.record_step(...)")
    print("     create_summary_dashboard(tracker)")

