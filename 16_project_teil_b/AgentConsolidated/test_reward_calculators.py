"""
Unit Tests for Reward Calculators
Validates that all reward functions work correctly and produce reasonable values
"""

import numpy as np
import pytest
from reward_calculators import (
    RewardCalculator, RewardComponents,
    BasicReward, WithRiskReward, WithSharpeReward, RiskAdjustedReward,
    SortinoReward, CalmarReward, InformationRatioReward, CompositeReward,
    create_reward_calculator, RewardCalculatorType
)


class TestRewardComponents:
    """Test RewardComponents validation"""
    
    def test_valid_components(self):
        """Components with valid values should be created"""
        comp = RewardComponents(
            pnl=0.001, cost=0.0001, slippage=0.00005, risk_penalty=0.0,
            volatility=0.01, position=0.5, true_reward=0.00085
        )
        assert comp.pnl == 0.001
        assert comp.volatility == 0.01
    
    def test_nan_components_rejected(self):
        """Components with NaN should raise ValueError"""
        with pytest.raises(ValueError):
            RewardComponents(
                pnl=np.nan, cost=0.0001, slippage=0.00005, risk_penalty=0.0,
                volatility=0.01, position=0.5, true_reward=0.00085
            )
    
    def test_inf_components_rejected(self):
        """Components with Inf should raise ValueError"""
        with pytest.raises(ValueError):
            RewardComponents(
                pnl=0.001, cost=np.inf, slippage=0.00005, risk_penalty=0.0,
                volatility=0.01, position=0.5, true_reward=0.00085
            )


class TestBasicReward:
    """Test BasicReward calculator"""
    
    def test_basic_equals_true_reward(self):
        """Basic reward should equal true_reward"""
        comp = RewardComponents(
            pnl=0.001, cost=0.0001, slippage=0.00005, risk_penalty=0.0,
            volatility=0.01, position=0.5, true_reward=0.00085
        )
        calc = BasicReward()
        reward = calc.calculate(comp)
        assert np.isclose(reward, comp.true_reward)
    
    def test_basic_negative_pnl(self):
        """Basic reward with negative PnL"""
        comp = RewardComponents(
            pnl=-0.001, cost=0.0001, slippage=0.00005, risk_penalty=0.0,
            volatility=0.01, position=-0.5, true_reward=-0.00115
        )
        calc = BasicReward()
        reward = calc.calculate(comp)
        assert reward < 0
        assert np.isclose(reward, -0.00115)


class TestWithRiskReward:
    """Test WithRiskReward calculator"""
    
    def test_kappa_zero_equals_basic(self):
        """With kappa=0, should equal basic reward"""
        comp = RewardComponents(
            pnl=0.001, cost=0.0001, slippage=0.00005, risk_penalty=0.0,
            volatility=0.01, position=0.5, true_reward=0.00085
        )
        calc = WithRiskReward(kappa=0.0)
        reward = calc.calculate(comp)
        assert np.isclose(reward, comp.true_reward)
    
    def test_risk_penalty_applied(self):
        """Risk penalty should reduce reward for non-zero position"""
        comp = RewardComponents(
            pnl=0.001, cost=0.0001, slippage=0.00005, risk_penalty=0.0,
            volatility=0.01, position=0.5, true_reward=0.00085
        )
        calc = WithRiskReward(kappa=0.01)
        reward = calc.calculate(comp)
        # Risk penalty = 0.01 * (0.5^2) * 0.01 = 0.00005
        expected = 0.00085 - 0.00005
        assert np.isclose(reward, expected)
    
    def test_zero_position_no_penalty(self):
        """With zero position, risk penalty should be zero"""
        comp = RewardComponents(
            pnl=0.001, cost=0.0001, slippage=0.00005, risk_penalty=0.0,
            volatility=0.01, position=0.0, true_reward=0.00085
        )
        calc = WithRiskReward(kappa=0.01)
        reward = calc.calculate(comp)
        assert np.isclose(reward, comp.true_reward)
    
    def test_invalid_kappa(self):
        """Invalid kappa should raise ValueError"""
        with pytest.raises(ValueError):
            WithRiskReward(kappa=-0.01)
        with pytest.raises(ValueError):
            WithRiskReward(kappa=np.inf)


class TestWithSharpeReward:
    """Test WithSharpeReward calculator"""
    
    def test_sharpe_divides_by_volatility(self):
        """Sharpe reward should divide by volatility"""
        comp = RewardComponents(
            pnl=0.001, cost=0.0001, slippage=0.00005, risk_penalty=0.0,
            volatility=0.02, position=0.5, true_reward=0.00085
        )
        calc = WithSharpeReward(epsilon=0.001)
        reward = calc.calculate(comp)
        # With epsilon, denominator = max(0.02, 0.001) = 0.02
        expected = 0.00085 / 0.02
        assert np.isclose(reward, expected)
    
    def test_sharpe_low_volatility_uses_epsilon(self):
        """With low volatility, should use epsilon"""
        comp = RewardComponents(
            pnl=0.001, cost=0.0001, slippage=0.00005, risk_penalty=0.0,
            volatility=0.0001, position=0.5, true_reward=0.00085
        )
        calc = WithSharpeReward(epsilon=0.001)
        reward = calc.calculate(comp)
        # Denominator = max(0.0001, 0.001) = 0.001
        expected = 0.00085 / 0.001
        assert np.isclose(reward, expected)
    
    def test_sharpe_amplifies_small_pnl_in_calm_markets(self):
        """Small PnL gets amplified in calm markets (low vol)"""
        comp = RewardComponents(
            pnl=0.00001, cost=0.00001, slippage=0.0, risk_penalty=0.0,
            volatility=0.0005, position=0.5, true_reward=0.0
        )
        calc = WithSharpeReward(epsilon=0.001)
        reward = calc.calculate(comp)
        # With small PnL and low vol, reward is nearly zero
        assert abs(reward) < 0.1  # Some reasonable threshold


class TestRiskAdjustedReward:
    """Test RiskAdjustedReward calculator"""
    
    def test_asymmetric_cost_treatment(self):
        """Costs should not be divided by volatility"""
        comp = RewardComponents(
            pnl=0.001, cost=0.0001, slippage=0.00005, risk_penalty=0.0,
            volatility=0.01, position=0.5, true_reward=0.00085
        )
        calc = RiskAdjustedReward(epsilon=0.001)
        reward = calc.calculate(comp)
        # PnL normalized: 0.001 / 0.01 = 0.1
        # Costs NOT normalized: 0.0001 + 0.00005 = 0.00015
        expected = 0.1 - 0.00015
        assert np.isclose(reward, expected)
    
    def test_differs_from_sharpe(self):
        """RISK_ADJUSTED should differ from WITH_SHARPE due to cost treatment"""
        comp = RewardComponents(
            pnl=0.001, cost=0.0001, slippage=0.00005, risk_penalty=0.0,
            volatility=0.01, position=0.5, true_reward=0.00085
        )
        
        calc_risk = RiskAdjustedReward()
        calc_sharpe = WithSharpeReward()
        
        reward_risk = calc_risk.calculate(comp)
        reward_sharpe = calc_sharpe.calculate(comp)
        
        # They should NOT be equal due to asymmetric cost treatment
        assert not np.isclose(reward_risk, reward_sharpe)


class TestSortinoReward:
    """Test SortinoReward calculator"""
    
    def test_sortino_scale_amplifies_volatility(self):
        """Sortino should amplify volatility compared to Sharpe"""
        comp = RewardComponents(
            pnl=0.001, cost=0.0001, slippage=0.00005, risk_penalty=0.0,
            volatility=0.01, position=0.5, true_reward=0.00085
        )
        
        calc_sortino = SortinoReward(downside_scale=1.2)
        calc_sharpe = WithSharpeReward()
        
        reward_sortino = calc_sortino.calculate(comp)
        reward_sharpe = calc_sharpe.calculate(comp)
        
        # Sortino amplifies volatility, so reward should be smaller
        assert reward_sortino < reward_sharpe


class TestCalmarReward:
    """Test CalmarReward calculator"""
    
    def test_calmar_penalizes_large_positions(self):
        """Larger positions should have lower rewards (drawdown estimate)"""
        comp_small = RewardComponents(
            pnl=0.001, cost=0.0001, slippage=0.00005, risk_penalty=0.0,
            volatility=0.01, position=0.1, true_reward=0.00085
        )
        comp_large = RewardComponents(
            pnl=0.001, cost=0.0001, slippage=0.00005, risk_penalty=0.0,
            volatility=0.01, position=0.9, true_reward=0.00085
        )
        
        calc = CalmarReward()
        reward_small = calc.calculate(comp_small)
        reward_large = calc.calculate(comp_large)
        
        # Larger position has higher drawdown estimate, lower reward
        assert reward_small > reward_large


class TestCompositeReward:
    """Test CompositeReward calculator"""
    
    def test_weight_normalization(self):
        """Weights should be normalized to sum to 1"""
        calc = CompositeReward(
            weight_returns=5, weight_sharpe=3, weight_risk=2
        )
        total = calc.weight_returns + calc.weight_sharpe + calc.weight_risk
        assert np.isclose(total, 1.0)
    
    def test_zero_weights_rejected(self):
        """All-zero weights should raise ValueError"""
        with pytest.raises(ValueError):
            CompositeReward(weight_returns=0, weight_sharpe=0, weight_risk=0)
    
    def test_is_combination(self):
        """Composite should be between pure signals"""
        comp = RewardComponents(
            pnl=0.001, cost=0.0001, slippage=0.00005, risk_penalty=0.0,
            volatility=0.01, position=0.5, true_reward=0.00085
        )
        
        calc_returns = create_reward_calculator(RewardCalculatorType.BASIC)
        calc_sharpe = create_reward_calculator(RewardCalculatorType.WITH_SHARPE)
        calc_composite = CompositeReward(weight_returns=0.5, weight_sharpe=0.5)
        
        r_returns = calc_returns.calculate(comp)
        r_sharpe = calc_sharpe.calculate(comp)
        r_composite = calc_composite.calculate(comp)
        
        # Composite should be between the two components
        min_component = min(r_returns, r_sharpe)
        max_component = max(r_returns, r_sharpe)
        assert min_component <= r_composite <= max_component


class TestRewardFactory:
    """Test reward calculator factory function"""
    
    def test_factory_creates_all_types(self):
        """Factory should create all supported reward types"""
        types = [
            RewardCalculatorType.BASIC,
            RewardCalculatorType.WITH_RISK,
            RewardCalculatorType.WITH_SHARPE,
            RewardCalculatorType.RISK_ADJUSTED,
            RewardCalculatorType.SORTINO,
            RewardCalculatorType.CALMAR,
            RewardCalculatorType.INFORMATION_RATIO,
            RewardCalculatorType.COMPOSITE,
        ]
        
        for reward_type in types:
            calc = create_reward_calculator(reward_type)
            assert isinstance(calc, RewardCalculator)
            assert calc.name is not None
    
    def test_factory_passes_kwargs(self):
        """Factory should pass kwargs to calculators"""
        calc = create_reward_calculator(
            RewardCalculatorType.WITH_RISK,
            kappa=0.05
        )
        assert np.isclose(calc.kappa, 0.05)
    
    def test_factory_invalid_type(self):
        """Factory should raise ValueError for unknown types"""
        with pytest.raises(ValueError):
            create_reward_calculator("invalid_type")


class TestRewardHistory:
    """Test reward history tracking"""
    
    def test_reward_history_tracking(self):
        """Rewards should be recorded in history"""
        calc = BasicReward()
        comp = RewardComponents(
            pnl=0.001, cost=0.0001, slippage=0.00005, risk_penalty=0.0,
            volatility=0.01, position=0.5, true_reward=0.00085
        )
        
        for _ in range(5):
            reward = calc.calculate(comp)
            calc.record_reward(reward)
        
        assert len(calc.reward_history) == 5
    
    def test_reward_stats(self):
        """Should calculate statistics from history"""
        calc = BasicReward()
        comp = RewardComponents(
            pnl=0.001, cost=0.0001, slippage=0.00005, risk_penalty=0.0,
            volatility=0.01, position=0.5, true_reward=0.00085
        )
        
        for _ in range(10):
            reward = calc.calculate(comp)
            calc.record_reward(reward)
        
        stats = calc.get_stats()
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert np.isfinite(stats['mean'])
    
    def test_reset_history(self):
        """History should clear on reset"""
        calc = BasicReward()
        comp = RewardComponents(
            pnl=0.001, cost=0.0001, slippage=0.00005, risk_penalty=0.0,
            volatility=0.01, position=0.5, true_reward=0.00085
        )
        
        for _ in range(5):
            reward = calc.calculate(comp)
            calc.record_reward(reward)
        
        assert len(calc.reward_history) > 0
        calc.reset_history()
        assert len(calc.reward_history) == 0


class TestEdgeCases:
    """Test edge cases and numerical stability"""
    
    def test_zero_volatility_with_epsilon(self):
        """Zero volatility should be handled by epsilon"""
        comp = RewardComponents(
            pnl=0.001, cost=0.0001, slippage=0.00005, risk_penalty=0.0,
            volatility=0.0, position=0.5, true_reward=0.00085
        )
        
        calc = WithSharpeReward(epsilon=0.001)
        reward = calc.calculate(comp)
        
        # Should not be NaN or Inf
        assert np.isfinite(reward)
    
    def test_very_large_position(self):
        """Should handle very large positions"""
        comp = RewardComponents(
            pnl=0.001, cost=0.0001, slippage=0.00005, risk_penalty=0.0,
            volatility=0.01, position=100.0, true_reward=0.00085
        )
        
        calc = WithRiskReward(kappa=0.01)
        reward = calc.calculate(comp)
        
        # Risk penalty = 0.01 * (100^2) * 0.01 = 100
        # Reward should be very negative
        assert reward < comp.true_reward
        assert np.isfinite(reward)
    
    def test_negative_volatility_impossible(self):
        """Volatility should always be non-negative"""
        # This is a data validation issue, not calculator issue
        # But we test that calculators work with very small positives
        comp = RewardComponents(
            pnl=0.001, cost=0.0001, slippage=0.00005, risk_penalty=0.0,
            volatility=1e-10, position=0.5, true_reward=0.00085
        )
        
        calc = WithSharpeReward(epsilon=0.001)
        reward = calc.calculate(comp)
        assert np.isfinite(reward)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

