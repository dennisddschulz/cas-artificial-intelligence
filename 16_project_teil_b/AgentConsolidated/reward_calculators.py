"""
Modular Reward Calculators for Trading Environments
Provides pluggable reward calculation with clear separation of concerns
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
from enum import Enum


class RewardCalculatorType(Enum):
    """Types of reward calculators"""
    BASIC = "basic"
    WITH_RISK = "with_risk"
    WITH_SHARPE = "with_sharpe"
    RISK_ADJUSTED = "risk_adjusted"
    SORTINO = "sortino"
    CALMAR = "calmar"
    INFORMATION_RATIO = "information_ratio"
    COMPOSITE = "composite"


@dataclass
class RewardComponents:
    """Container for reward calculation components"""
    pnl: float  # Profit/Loss from position
    cost: float  # Transaction costs
    slippage: float  # Market impact/slippage
    risk_penalty: float  # Risk-based penalty
    volatility: float  # Current volatility/sigma
    position: float  # Current position size
    true_reward: float  # PnL - cost - slippage (actual portfolio impact)
    
    def __post_init__(self):
        """Validate that all components are finite"""
        for attr_name in ['pnl', 'cost', 'slippage', 'risk_penalty', 'volatility', 'position', 'true_reward']:
            value = getattr(self, attr_name)
            if not np.isfinite(value):
                raise ValueError(f"{attr_name}={value} is not finite!")


class RewardCalculator(ABC):
    """
    Abstract base class for reward calculation
    Each subclass implements a different reward formulation
    """
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.reward_history = []
    
    @abstractmethod
    def calculate(self, components: RewardComponents) -> float:
        """
        Calculate reward given market/portfolio components
        
        Args:
            components: RewardComponents with all relevant market/portfolio data
            
        Returns:
            float: Scalar reward signal for policy learning
        """
        pass
    
    def reset_history(self):
        """Reset reward history for new episode"""
        self.reward_history = []
    
    def record_reward(self, reward: float):
        """Record reward for analysis"""
        if np.isfinite(reward):
            self.reward_history.append(float(reward))
    
    def get_stats(self) -> Dict[str, float]:
        """Get statistics on reward history"""
        if not self.reward_history:
            return {}
        
        arr = np.array(self.reward_history)
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'median': float(np.median(arr)),
        }
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"


class BasicReward(RewardCalculator):
    """
    BASIC Reward: Pure PnL minus costs
    
    Formula:
        R = PnL - TransactionCost - Slippage
    
    Characteristics:
    - Simplest reward formulation
    - Purely greedy: maximize absolute returns
    - No explicit risk penalty
    - May lead to excessive trading and position size
    
    Good for:
    - Baseline comparisons
    - Very stable markets (low volatility)
    - Short-term horizons
    """
    
    def __init__(self):
        super().__init__(
            name="BASIC",
            description="Pure PnL minus costs: R = PnL - Cost - Slippage"
        )
    
    def calculate(self, components: RewardComponents) -> float:
        """
        Calculate basic reward: actual P&L after all costs
        """
        reward = components.true_reward
        return float(reward)


class WithRiskReward(RewardCalculator):
    """
    WITH_RISK Reward: PnL minus costs minus quadratic risk penalty
    
    Formula:
        R = (PnL - Cost - Slippage) - κ * pos^2 * σ
    
    Where:
    - κ (kappa): Risk penalty coefficient (0.01 typical)
    - pos: Current position size
    - σ: Current volatility (sigma_hat)
    
    Characteristics:
    - Discourages large positions via quadratic position penalty
    - Risk penalty scales with volatility (more aggressive in calm markets)
    - Smooth constraint without hard limits
    - Most stable learning behavior
    
    Good for:
    - General trading with moderate risk
    - Balanced return-risk tradeoff
    - Most robust learning across market conditions
    
    Parameters:
    - kappa: Risk penalty weight (default 0.01)
    """
    
    def __init__(self, kappa: float = 0.01):
        super().__init__(
            name="WITH_RISK",
            description=f"PnL - Costs - Risk Penalty: R = true_reward - κ * pos^2 * σ (κ={kappa})"
        )
        self.kappa = float(kappa)
        if not np.isfinite(self.kappa) or self.kappa < 0:
            raise ValueError(f"kappa must be non-negative and finite, got {self.kappa}")
    
    def calculate(self, components: RewardComponents) -> float:
        """
        Calculate reward with risk penalty
        Penalizes large positions more in high-volatility environments
        """
        risk_penalty = self.kappa * (components.position ** 2) * components.volatility
        reward = components.true_reward - risk_penalty
        return float(reward)


class WithSharpeReward(RewardCalculator):
    """
    WITH_SHARPE Reward: Risk-adjusted returns via Sharpe-like ratio
    
    Formula:
        R = (PnL - Cost - Slippage) / (σ + ε)
    
    Where:
    - σ: Current volatility
    - ε: Epsilon for numerical stability (default 0.001)
    
    Characteristics:
    - Explicitly optimizes risk-adjusted returns
    - Highly sensitive to volatility estimates
    - Can produce extreme values when volatility is low
    - May cause instability with poor volatility estimation
    - Rewards small positions in calm markets
    
    Good for:
    - Maximizing Sharpe ratio directly
    - Investors focused on risk-adjusted performance
    - Requires stable volatility estimation
    
    Warnings:
    - Division by small volatility can create large reward values
    - Volatility estimation quality is critical
    - May underperform in trending markets
    
    Parameters:
    - epsilon: Denominator stability term (default 0.001)
    """
    
    def __init__(self, epsilon: float = 0.001):
        super().__init__(
            name="WITH_SHARPE",
            description=f"Risk-adjusted (Sharpe-like): R = true_reward / (σ + ε) (ε={epsilon})"
        )
        self.epsilon = float(epsilon)
        if not np.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError(f"epsilon must be non-negative and finite, got {self.epsilon}")
    
    def calculate(self, components: RewardComponents) -> float:
        """
        Calculate Sharpe-like reward: normalize PnL by volatility
        Lower volatility → higher reward for same PnL
        """
        safe_sigma = max(components.volatility, self.epsilon)
        reward = components.true_reward / safe_sigma
        return float(reward)


class RiskAdjustedReward(RewardCalculator):
    """
    RISK_ADJUSTED Reward: PnL normalized by volatility minus costs
    
    Formula:
        R = (PnL / (σ + ε)) - Cost - Slippage
    
    Where:
    - σ: Current volatility
    - ε: Epsilon for stability (default 0.001)
    
    Characteristics:
    - Asymmetric: normalizes returns by risk but NOT costs
    - Costs always subtracted at face value
    - More aggressive than WITH_SHARPE in calm markets
    - Returns normalized but costs not
    
    Differences from WITH_SHARPE:
    - WITH_SHARPE: (PnL - Cost) / σ (all components normalized)
    - RISK_ADJUSTED: (PnL / σ) - Cost (only PnL normalized)
    
    Good for:
    - Positions where cost reduction is critical
    - Markets with variable transaction costs
    - More aggressive risk-taking in calm periods
    
    Parameters:
    - epsilon: Denominator stability term (default 0.001)
    """
    
    def __init__(self, epsilon: float = 0.001):
        super().__init__(
            name="RISK_ADJUSTED",
            description=f"PnL/vol - Costs: R = (PnL/(σ + ε)) - Cost - Slippage (ε={epsilon})"
        )
        self.epsilon = float(epsilon)
        if not np.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError(f"epsilon must be non-negative and finite, got {self.epsilon}")
    
    def calculate(self, components: RewardComponents) -> float:
        """
        Calculate risk-adjusted reward with asymmetric cost treatment
        """
        safe_sigma = max(components.volatility, self.epsilon)
        risk_adjusted_pnl = components.pnl / safe_sigma
        reward = risk_adjusted_pnl - components.cost - components.slippage
        return float(reward)


class SortinoReward(RewardCalculator):
    """
    SORTINO Reward: Sortino ratio-inspired reward
    
    Concept:
        Penalizes only downside volatility (loss-oriented)
        Focus on variance of negative returns only
    
    Simplified Formula:
        R = (PnL - Cost) / max(σ_downside, ε)
    
    Where:
    - σ_downside: Downside volatility (std of returns < 0)
    - ε: Epsilon for stability
    
    Characteristics:
    - More sensitive to downside risk than WITH_SHARPE
    - Ignores upside volatility
    - Better aligns with investor preferences
    - Requires tracking downside volatility separately
    
    Note:
    - Current implementation uses full volatility as proxy
    - For true Sortino, would need history of returns
    - This is a simplified version for per-step calculation
    
    Good for:
    - Risk-averse traders
    - Portfolios where downside protection is key
    - Reducing impact of positive volatility
    
    Parameters:
    - epsilon: Stability term (default 0.001)
    - downside_scale: Scale factor for downside emphasis (default 1.2)
    """
    
    def __init__(self, epsilon: float = 0.001, downside_scale: float = 1.2):
        super().__init__(
            name="SORTINO",
            description=f"Downside-focused reward (Sortino-inspired) with scale={downside_scale}"
        )
        self.epsilon = float(epsilon)
        self.downside_scale = float(downside_scale)
        if not np.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError(f"epsilon must be non-negative and finite, got {self.epsilon}")
        if not np.isfinite(self.downside_scale) or self.downside_scale <= 0:
            raise ValueError(f"downside_scale must be positive and finite, got {self.downside_scale}")
    
    def calculate(self, components: RewardComponents) -> float:
        """
        Calculate Sortino-inspired reward
        Emphasizes downside risk over upside volatility
        """
        safe_sigma = max(components.volatility, self.epsilon)
        # Amplify volatility to emphasize downside focus
        adjusted_sigma = safe_sigma * self.downside_scale
        reward = components.true_reward / adjusted_sigma
        return float(reward)


class CalmarReward(RewardCalculator):
    """
    CALMAR Reward: Calmar ratio-inspired reward
    
    Concept:
        Return per unit of maximum drawdown
        Direct optimization of return/drawdown tradeoff
    
    Simplified Formula:
        R = (PnL - Cost) / max(drawdown_estimate, ε)
    
    Where:
    - drawdown_estimate: Estimated drawdown from position
    - ε: Epsilon for stability
    
    Characteristics:
    - Directly penalizes positions that increase drawdown risk
    - More conservative than volatility-based rewards
    - Requires drawdown tracking
    
    Note:
    - True Calmar requires full portfolio history
    - This simplified version uses position-based drawdown estimate
    - Estimate = position × volatility (crude proxy)
    
    Good for:
    - Capital preservation focused trading
    - Minimizing deep underwater periods
    - Risk management strict environments
    
    Parameters:
    - epsilon: Stability term (default 0.001)
    - drawdown_multiplier: Position × vol multiplier (default 0.5)
    """
    
    def __init__(self, epsilon: float = 0.001, drawdown_multiplier: float = 0.5):
        super().__init__(
            name="CALMAR",
            description=f"Drawdown-focused reward (Calmar-inspired) with multiplier={drawdown_multiplier}"
        )
        self.epsilon = float(epsilon)
        self.drawdown_multiplier = float(drawdown_multiplier)
        if not np.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError(f"epsilon must be non-negative and finite, got {self.epsilon}")
        if not np.isfinite(self.drawdown_multiplier) or self.drawdown_multiplier <= 0:
            raise ValueError(f"drawdown_multiplier must be positive and finite, got {self.drawdown_multiplier}")
    
    def calculate(self, components: RewardComponents) -> float:
        """
        Calculate Calmar-inspired reward
        Penalizes positions based on estimated drawdown risk
        """
        drawdown_estimate = abs(components.position) * components.volatility * self.drawdown_multiplier
        safe_drawdown = max(drawdown_estimate, self.epsilon)
        reward = components.true_reward / safe_drawdown
        return float(reward)


class InformationRatioReward(RewardCalculator):
    """
    INFORMATION_RATIO Reward: Alpha generation focused reward
    
    Concept:
        Optimize alpha generation per unit of tracking error
        Focus on excess returns vs benchmark
    
    Simplified Formula:
        R = (PnL - Cost) / (σ + ε)
        But with emphasis on consistent outperformance
    
    Characteristics:
    - Similar to Sharpe but conceptually focused on alpha
    - Penalizes inconsistency
    - Good for active strategy optimization
    
    Note:
    - True Information Ratio requires benchmark returns
    - This simplified version uses Sharpe-like formulation
    - More of a conceptual variant
    
    Good for:
    - Active strategy optimization
    - Alpha-focused portfolios
    - Consistent outperformance focus
    
    Parameters:
    - epsilon: Stability term (default 0.001)
    - consistency_bonus: Bonus for consistent positive returns (default 0.1)
    """
    
    def __init__(self, epsilon: float = 0.001, consistency_bonus: float = 0.1):
        super().__init__(
            name="INFORMATION_RATIO",
            description=f"Alpha-focused reward with consistency bonus={consistency_bonus}"
        )
        self.epsilon = float(epsilon)
        self.consistency_bonus = float(consistency_bonus)
        if not np.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError(f"epsilon must be non-negative and finite, got {self.epsilon}")
    
    def calculate(self, components: RewardComponents) -> float:
        """
        Calculate Information Ratio-inspired reward
        Base Sharpe-like, plus bonus for consistency
        """
        safe_sigma = max(components.volatility, self.epsilon)
        base_reward = components.true_reward / safe_sigma
        
        # Small bonus if returns were positive (consistency)
        consistency_bonus = self.consistency_bonus if components.pnl > 0 else 0
        reward = base_reward + consistency_bonus
        return float(reward)


class CompositeReward(RewardCalculator):
    """
    COMPOSITE Reward: Weighted combination of multiple reward signals
    
    Formula:
        R = Σ(w_i * R_i)
    
    Where:
    - w_i: Weight for reward component i
    - R_i: Individual reward signal
    
    Components:
    1. Return signal: (PnL - Cost)
    2. Risk penalty: κ * pos^2 * σ
    3. Volatility adjustment: / (σ + ε)
    4. Consistency: bonus for positive returns
    
    Characteristics:
    - Highly flexible
    - Can balance multiple objectives
    - Requires careful weight tuning
    - Most complex to interpret
    
    Good for:
    - Multi-objective optimization
    - Fine-tuned strategy specific requirements
    - Research and experimentation
    
    Parameters:
    - weight_returns: Weight for raw returns (default 0.5)
    - weight_sharpe: Weight for Sharpe-like component (default 0.3)
    - weight_risk: Weight for risk penalty (default 0.2)
    - kappa: Risk penalty coefficient (default 0.01)
    - epsilon: Volatility stability (default 0.001)
    """
    
    def __init__(self, weight_returns: float = 0.5, weight_sharpe: float = 0.3, 
                 weight_risk: float = 0.2, kappa: float = 0.01, epsilon: float = 0.001):
        super().__init__(
            name="COMPOSITE",
            description=f"Weighted combination: {weight_returns:.1%} returns + {weight_sharpe:.1%} Sharpe + {weight_risk:.1%} risk"
        )
        
        # Normalize weights
        total_weight = weight_returns + weight_sharpe + weight_risk
        if total_weight == 0:
            raise ValueError("Total weight must be positive")
        
        self.weight_returns = float(weight_returns) / total_weight
        self.weight_sharpe = float(weight_sharpe) / total_weight
        self.weight_risk = float(weight_risk) / total_weight
        self.kappa = float(kappa)
        self.epsilon = float(epsilon)
        
        # Validate
        if not np.isclose(self.weight_returns + self.weight_sharpe + self.weight_risk, 1.0):
            raise ValueError("Normalized weights should sum to 1")
    
    def calculate(self, components: RewardComponents) -> float:
        """
        Calculate composite reward: weighted combination
        """
        # Component 1: Pure returns
        returns_signal = components.true_reward
        
        # Component 2: Sharpe-like (returns / volatility)
        safe_sigma = max(components.volatility, self.epsilon)
        sharpe_signal = components.true_reward / safe_sigma
        
        # Component 3: Risk penalty
        risk_penalty = self.kappa * (components.position ** 2) * components.volatility
        risk_signal = -risk_penalty  # Negative because it's a penalty
        
        # Composite
        reward = (self.weight_returns * returns_signal + 
                 self.weight_sharpe * sharpe_signal + 
                 self.weight_risk * risk_signal)
        
        return float(reward)


def create_reward_calculator(reward_type: RewardCalculatorType, 
                            **kwargs) -> RewardCalculator:
    """
    Factory function to create reward calculators
    
    Args:
        reward_type: Type of reward calculator
        **kwargs: Type-specific parameters
        
    Returns:
        RewardCalculator instance
        
    Raises:
        ValueError: If reward_type is not supported or kwargs are invalid
    """
    if reward_type == RewardCalculatorType.BASIC:
        return BasicReward()
    
    elif reward_type == RewardCalculatorType.WITH_RISK:
        return WithRiskReward(kappa=kwargs.get('kappa', 0.01))
    
    elif reward_type == RewardCalculatorType.WITH_SHARPE:
        return WithSharpeReward(epsilon=kwargs.get('epsilon', 0.001))
    
    elif reward_type == RewardCalculatorType.RISK_ADJUSTED:
        return RiskAdjustedReward(epsilon=kwargs.get('epsilon', 0.001))
    
    elif reward_type == RewardCalculatorType.SORTINO:
        return SortinoReward(
            epsilon=kwargs.get('epsilon', 0.001),
            downside_scale=kwargs.get('downside_scale', 1.2)
        )
    
    elif reward_type == RewardCalculatorType.CALMAR:
        return CalmarReward(
            epsilon=kwargs.get('epsilon', 0.001),
            drawdown_multiplier=kwargs.get('drawdown_multiplier', 0.5)
        )
    
    elif reward_type == RewardCalculatorType.INFORMATION_RATIO:
        return InformationRatioReward(
            epsilon=kwargs.get('epsilon', 0.001),
            consistency_bonus=kwargs.get('consistency_bonus', 0.1)
        )
    
    elif reward_type == RewardCalculatorType.COMPOSITE:
        return CompositeReward(
            weight_returns=kwargs.get('weight_returns', 0.5),
            weight_sharpe=kwargs.get('weight_sharpe', 0.3),
            weight_risk=kwargs.get('weight_risk', 0.2),
            kappa=kwargs.get('kappa', 0.01),
            epsilon=kwargs.get('epsilon', 0.001)
        )
    
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


if __name__ == "__main__":
    # Test all reward calculators
    print("Testing Reward Calculators\n" + "="*60)
    
    # Create test components
    test_component = RewardComponents(
        pnl=0.0005,
        cost=0.0001,
        slippage=0.00005,
        risk_penalty=0.0,
        volatility=0.01,
        position=0.5,
        true_reward=0.0005 - 0.0001 - 0.00005
    )
    
    # Test each calculator
    calculators = [
        create_reward_calculator(RewardCalculatorType.BASIC),
        create_reward_calculator(RewardCalculatorType.WITH_RISK, kappa=0.01),
        create_reward_calculator(RewardCalculatorType.WITH_SHARPE),
        create_reward_calculator(RewardCalculatorType.RISK_ADJUSTED),
        create_reward_calculator(RewardCalculatorType.SORTINO),
        create_reward_calculator(RewardCalculatorType.CALMAR),
        create_reward_calculator(RewardCalculatorType.INFORMATION_RATIO),
        create_reward_calculator(RewardCalculatorType.COMPOSITE),
    ]
    
    for calc in calculators:
        reward = calc.calculate(test_component)
        print(f"{calc.name:20s}: {reward:12.6f}  |  {calc.description}")
    
    print("\n" + "="*60)
    print("All calculators created and tested successfully!")

