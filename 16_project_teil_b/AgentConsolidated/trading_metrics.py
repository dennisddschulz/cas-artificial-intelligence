"""
Trading Evaluation Metrics Module
Calculates comprehensive performance metrics for trading strategies
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class TradingMetrics:
    """Calculate trading performance metrics"""
    
    def __init__(self, initial_equity: float, risk_free_rate: float = 0.02):
        self.initial_equity = initial_equity
        self.risk_free_rate = risk_free_rate
    
    def calculate_all_metrics(
        self, equity_curve: np.ndarray, daily_returns: np.ndarray,
        positions: np.ndarray, costs: np.ndarray
    ) -> Dict[str, float]:
        """Calculate all performance metrics"""
        return {
            'total_return': self.total_return(equity_curve),
            'cumulative_return': self.cumulative_return(equity_curve),
            'annualized_return': self.annualized_return(daily_returns),
            'volatility': self.volatility(daily_returns),
            'annualized_volatility': self.annualized_volatility(daily_returns),
            'max_drawdown': self.max_drawdown(equity_curve),
            'sharpe_ratio': self.sharpe_ratio(daily_returns),
            'calmar_ratio': self.calmar_ratio(daily_returns, equity_curve),
            'sortino_ratio': self.sortino_ratio(daily_returns),
            'win_rate': self.win_rate(daily_returns),
            'profit_factor': self.profit_factor(daily_returns),
            'turnover': self.turnover(positions),
            'total_costs': self.total_costs(costs),
            'cost_ratio': self.cost_ratio(costs),
            'kurtosis': self.kurtosis(daily_returns),
            'skewness': self.skewness(daily_returns),
        }
    
    @staticmethod
    def total_return(equity_curve: np.ndarray) -> float:
        if len(equity_curve) < 2:
            return 0.0
        return (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
    
    @staticmethod
    def cumulative_return(equity_curve: np.ndarray) -> float:
        return TradingMetrics.total_return(equity_curve)
    
    @staticmethod
    def annualized_return(daily_returns: np.ndarray) -> float:
        return np.mean(daily_returns) * 252
    
    @staticmethod
    def volatility(daily_returns: np.ndarray) -> float:
        return np.std(daily_returns)
    
    @staticmethod
    def annualized_volatility(daily_returns: np.ndarray) -> float:
        return np.std(daily_returns) * np.sqrt(252)
    
    @staticmethod
    def max_drawdown(equity_curve: np.ndarray) -> float:
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return np.min(drawdown)
    
    @staticmethod
    def sharpe_ratio(daily_returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        excess = daily_returns - risk_free_rate / 252
        if np.std(excess) == 0:
            return 0.0
        return (np.mean(excess) * 252) / (np.std(excess) * np.sqrt(252))
    
    @staticmethod
    def calmar_ratio(daily_returns: np.ndarray, equity_curve: np.ndarray) -> float:
        annual_ret = np.mean(daily_returns) * 252
        max_dd = TradingMetrics.max_drawdown(equity_curve)
        if abs(max_dd) < 1e-8:
            return 0.0
        return annual_ret / abs(max_dd)
    
    @staticmethod
    def sortino_ratio(daily_returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        excess = daily_returns - risk_free_rate / 252
        downside = np.minimum(excess, 0)
        down_vol = np.std(downside)
        if down_vol == 0:
            return 0.0
        return (np.mean(excess) * 252) / (down_vol * np.sqrt(252))
    
    @staticmethod
    def win_rate(daily_returns: np.ndarray) -> float:
        if len(daily_returns) == 0:
            return 0.0
        return np.sum(daily_returns > 0) / len(daily_returns)
    
    @staticmethod
    def profit_factor(daily_returns: np.ndarray) -> float:
        gains = np.sum(daily_returns[daily_returns > 0])
        losses = np.sum(np.abs(daily_returns[daily_returns < 0]))
        if losses == 0:
            return np.inf if gains > 0 else 0.0
        return gains / losses
    
    @staticmethod
    def turnover(positions: np.ndarray) -> float:
        if len(positions) < 2:
            return 0.0
        return np.sum(np.abs(np.diff(positions)))
    
    @staticmethod
    def total_costs(costs: np.ndarray) -> float:
        return np.sum(costs)
    
    def cost_ratio(self, costs: np.ndarray) -> float:
        return np.sum(costs) / self.initial_equity
    
    @staticmethod
    def kurtosis(daily_returns: np.ndarray) -> float:
        return stats.kurtosis(daily_returns)
    
    @staticmethod
    def skewness(daily_returns: np.ndarray) -> float:
        return stats.skew(daily_returns)


class MetricsComparison:
    """Compare metrics across experiments"""
    
    def __init__(self):
        self.results: Dict[str, Dict[str, float]] = {}
    
    def add_experiment(self, name: str, metrics: Dict[str, float]):
        self.results[name] = metrics
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.results).T
    
    def plot_comparison(self, metrics=None, figsize=(15, 10)):
        if metrics is None:
            metrics = ['total_return', 'sharpe_ratio', 'max_drawdown',
                      'annualized_volatility', 'win_rate']
        
        df = self.to_dataframe()
        available = [m for m in metrics if m in df.columns]
        
        n_metrics = len(available)
        n_rows = (n_metrics + 1) // 2
        
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
        axes = axes.flatten()
        
        for idx, metric in enumerate(available):
            ax = axes[idx]
            data = df[metric]
            colors = ['green' if v > 0 else 'red' for v in data]
            data.plot(kind='bar', ax=ax, color=colors, alpha=0.7)
            ax.set_title(metric.replace('_', ' ').title())
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        for idx in range(len(available), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def export_to_csv(self, filepath: str):
        self.to_dataframe().to_csv(filepath)


class EquityCurveAnalyzer:
    """Analyze equity curves and generate plots"""
    
    @staticmethod
    def plot_equity_curve(equity_curves: Dict[str, np.ndarray], figsize=(15, 6)):
        """Plot multiple equity curves"""
        fig, ax = plt.subplots(figsize=figsize)
        
        for name, curve in equity_curves.items():
            normalized = curve / curve[0]
            ax.plot(normalized, label=name, linewidth=2, alpha=0.8)
        
        ax.set_title('Equity Curves (Normalized)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Equity (Normalized to 1.0)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_drawdown(equity_curves: Dict[str, np.ndarray], figsize=(15, 6)):
        """Plot drawdown curves"""
        fig, ax = plt.subplots(figsize=figsize)
        
        for name, curve in equity_curves.items():
            peak = np.maximum.accumulate(curve)
            drawdown = (curve - peak) / peak * 100
            ax.plot(drawdown, label=name, linewidth=2, alpha=0.8)
        
        ax.set_title('Drawdown Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Drawdown (%)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_returns_distribution(returns: Dict[str, np.ndarray], figsize=(15, 6)):
        """Plot returns distributions"""
        n_experiments = len(returns)
        fig, axes = plt.subplots(1, n_experiments, figsize=figsize)
        
        if n_experiments == 1:
            axes = [axes]
        
        for ax, (name, ret) in zip(axes, returns.items()):
            ax.hist(ret * 100, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax.axvline(np.mean(ret) * 100, color='red', linestyle='--', linewidth=2, label='Mean')
            ax.set_title(f'{name}\nDaily Returns')
            ax.set_xlabel('Daily Return (%)')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

