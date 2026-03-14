"""
Trading Configuration Module
Parameterizes all trading experiments
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from enum import Enum
from datetime import datetime
import pytz


# ============================================================
# WANDB ORGANIZATION HELPERS
# ============================================================

def get_wandb_group(experiment_type: str, variant: str = "v1", seed: int = 42) -> str:
    """
    Generate hierarchical group name with date AND seed for better WandB organization
    
    Format: YYYY-MM-DD/experiment_type/variant/seed_{seed}
    Example: 2026-03-12/ppo_baseline/v1/seed_10
    
    Benefits:
    - Date-based filtering: easily find experiments from specific day
    - Seed-based separation: each seed gets its own group
    - Easy comparison: compare same experiment across seeds [10, 20, 30]
    - Hierarchical structure: date → type → version → seed
    
    This creates a tree structure in WandB UI:
    2026-03-12/
      ppo_baseline/
        v1/
          seed_10 (all seed_10 runs)
          seed_20 (all seed_20 runs)
          seed_30 (all seed_30 runs)
    """
    today = datetime.now(pytz.UTC).strftime("%Y-%m-%d")
    return f"{today}/{experiment_type}/{variant}/seed_{seed}"


def get_wandb_run_name(experiment_name: str, reward_type: str, seed: int) -> str:
    """
    Generate descriptive run name with date, seed and key configuration
    
    Format: YYYY-MM-DD_experiment_reward_seed
    Example: 2026-03-12_ppo_baseline_with_risk_10
    
    Benefits:
    - Quick identification of run configuration
    - Temporal ordering by timestamp
    - Seed number in name for quick reference
    """
    today = datetime.now(pytz.UTC).strftime("%Y-%m-%d")
    return f"{today}_{experiment_name}_{reward_type}_{seed}"


def get_wandb_tags(
    experiment_type: str,
    forecast_mode: str,
    reward_type: str,
    version: str = "v1",
    seed: int = 42
) -> list:
    """
    Generate comprehensive tags for advanced WandB filtering
    
    Tags allow multi-dimensional filtering:
    - Filter by date: tags:date_2026-03-12
    - Filter by seed: tags:seed_10
    - Filter by type: tags:ppo_baseline
    - Filter by reward: tags:reward_with_risk
    - Combine filters: tags:date_2026-03-12 AND tags:seed_10 AND tags:reward_with_risk
    
    Benefits:
    - Create charts filtered by specific criteria
    - Compare experiments across multiple dimensions
    - Track seeds, versions, and variants
    - Easy temporal filtering by date
    """
    today = datetime.now(pytz.UTC).strftime("%Y-%m-%d")
    return [
        today,                           # 2026-03-13
        experiment_type,                 # ppo_baseline
        f"forecast_{forecast_mode}",    # forecast_lstm or forecast_none
        f"reward_{reward_type}",         # reward_with_risk
        f"version_{version}",            # version_v1
        f"seed_{seed}",                  # seed_10, seed_20, seed_30
    ]


class RewardType(Enum):
    """Different reward function definitions"""
    BASIC = "basic"  # PnL - cost
    WITH_RISK = "with_risk"  # PnL - cost - risk_penalty
    WITH_SHARPE = "with_sharpe"  # Incorporates volatility
    RISK_ADJUSTED = "risk_adjusted"  # PnL / volatility
    SORTINO = "sortino"  # Downside-risk focused
    CALMAR = "calmar"  # Drawdown-risk focused
    INFORMATION_RATIO = "information_ratio"  # Alpha-focused
    COMPOSITE = "composite"  # Multi-objective weighted blend


class ForecastMode(Enum):
    """Forecast integration modes"""
    NONE = "none"  # No forecast
    LSTM = "lstm"  # LSTM-based forecast


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration"""
    ticker: str = "BTC-USD"
    start_date: str = "2018-01-01"
    end_date: Optional[str] = None
    train_frac: float = 0.6
    val_frac: float = 0.2
    # test_frac: float = 0.2  (calculated as 1 - train_frac - val_frac)


@dataclass
class ForecastingConfig:
    """LSTM Forecasting Model Configuration"""
    lookback: int = 20
    forecast_horizon: int = 5
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    early_stopping_patience: int = 20
    min_delta: float = 1e-5


@dataclass
class EnvironmentConfig:
    """Trading Environment Configuration"""
    initial_equity: float = 100000.0
    fee: float = 0.0001  # Transaction cost per unit position change
    kappa: float = 0.1   # Risk penalty weight (for WITH_RISK reward) - CHANGED from 0.01 to match notebook
    leverage_max: float = 1.0
    slippage_coef: float = 0.0
    smoothing_alpha: float = 1.0  # Position execution lag (1.0 = immediate)
    reward_scale: float = 100.0  # ← INCREASED from 1.0 to 100.0 for stronger learning signal
    include_turnover: bool = False
    min_cash_ratio: float = 0.05
    reward_type: RewardType = RewardType.WITH_RISK
    
    # ================================================================
    # Modular Reward Parameters - Configurable per Reward Type
    # ================================================================
    reward_params: Dict[str, Any] = field(default_factory=lambda: {
        # Common parameters (for normalization-based rewards)
        'epsilon': 0.001,  # Volatility stability term
        
        # WITH_RISK specific
        'kappa': 0.01,  # Risk penalty coefficient
        
        # SORTINO specific
        'downside_scale': 1.2,  # Multiplier for downside volatility emphasis
        
        # CALMAR specific
        'drawdown_multiplier': 0.5,  # Position × vol multiplier for drawdown estimate
        
        # INFORMATION_RATIO specific
        'consistency_bonus': 0.1,  # Bonus for positive returns
        
        # COMPOSITE specific (weighted blend of signals)
        'weight_returns': 0.5,      # Weight for raw returns signal
        'weight_sharpe': 0.3,       # Weight for Sharpe-like signal
        'weight_risk': 0.2,         # Weight for risk penalty signal
    })


@dataclass
class PPOConfig:
    """PPO Training Configuration"""
    num_envs: int = 8
    n_steps: int = 256
    total_updates: int = 3000
    
    # RL hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Loss coefficients
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 1.0
    
    # PPO-specific
    clip_eps: float = 0.2
    ppo_epochs: int = 20
    minibatch_size: int = 32
    target_kl: float = 0.05
    
    # Network architecture
    hidden_dim: int = 256
    num_layers: int = 2


@dataclass
class ExperimentConfig:
    """Complete experiment configuration with improved WandB organization"""
    experiment_name: str
    forecast_mode: ForecastMode
    reward_type: RewardType
    
    # Sub-configs
    data: DataConfig
    forecasting: ForecastingConfig
    environment: EnvironmentConfig
    ppo: PPOConfig
    
    # W&B logging - Improved date-based organization
    wandb_project: str = "PPO_Bitcoin_Trading_Experiments_Dennis"
    wandb_entity: str = "btcprojekt2026-bfh"
    use_wandb: bool = True
    
    # NEW: Experiment type and version for better organization
    # Examples: "ppo_baseline", "reward_ablation", "ppo_with_forecast", "multi_seed"
    wandb_experiment_type: str = "ppo_baseline"
    
    # NEW: Variant/version tracking (v1, v2, experimental, etc.)
    wandb_variant: str = "v1"
    
    # NEW: Enable date-based hierarchical grouping
    use_date_based_grouping: bool = True
    
    # Paths
    model_save_dir: str = "./models"
    results_dir: str = "./results"
    seed: int = 10
    
    # ================================================================
    # DYNAMIC WANDB PROPERTIES (Auto-generated based on date and config)
    # ================================================================
    
    @property
    def wandb_group(self) -> str:
        """
        Dynamic group name based on date, experiment type, version AND seed.
        
        Format: YYYY-MM-DD/experiment_type/variant/seed_{seed}
        Example: 2026-03-12/ppo_baseline/v1/seed_10
        
        Benefits:
        - Automatically changes with date (no manual updates needed)
        - Groups experiments by type, version AND seed
        - Each seed gets its own group for easy comparison
        - Can compare across seeds [10, 20, 30]:
          * seed_10 group contains all seed_10 runs
          * seed_20 group contains all seed_20 runs
          * seed_30 group contains all seed_30 runs
        - Easy to filter by date and seed in WandB UI
        
        This creates a tree structure:
        2026-03-12/ppo_baseline/v1/
          seed_10 (runs with seed 10)
          seed_20 (runs with seed 20)
          seed_30 (runs with seed 30)
        """
        if self.use_date_based_grouping:
            return get_wandb_group(
                self.wandb_experiment_type,
                self.wandb_variant,
                self.seed
            )
        else:
            # Fallback to simple group if date-based grouping disabled
            return f"{self.wandb_experiment_type}/seed_{self.seed}"
    
    @property
    def wandb_run_name(self) -> str:
        """
        Dynamic run name with date, experiment type, and configuration.
        
        Format: YYYY-MM-DD_experiment_name_reward_type_seed
        Example: 2026-03-12_ppo_baseline_with_risk_42
        
        Benefits:
        - Quick identification in WandB UI
        - All key config info in one string
        - Temporal ordering by date
        """
        return get_wandb_run_name(
            self.experiment_name,
            self.reward_type.value,
            self.seed
        )
    
    @property
    def wandb_tags(self) -> List[str]:
        """
        Dynamic tags for advanced filtering in WandB.
        
        Enables queries like:
        - tags:2026-03-12 (all from today)
        - tags:ppo_baseline AND tags:with_risk (baseline with risk)
        - tags:lstm (all with LSTM forecast)
        - tags:seed_42 (all with seed 42)
        
        Benefits:
        - Multi-dimensional filtering
        - Create filtered charts
        - Compare across multiple criteria
        """
        return get_wandb_tags(
            self.wandb_experiment_type,
            self.forecast_mode.value,
            self.reward_type.value,
            self.wandb_variant,
            self.seed
        )
    
    @property
    def wandb_mode(self) -> str:
        """WandB mode - online or offline"""
        return "online"  # Default to offline for corporate proxy


# ============================================================
# PRESET CONFIGURATIONS
# ============================================================

def get_ppo_without_forecast_config(
    name: str = "PPO-No-Forecast",
    experiment_type: str = "ppo_baseline",
    variant: str = "v1",
) -> ExperimentConfig:
    """PPO trading without forecast"""
    return ExperimentConfig(
        experiment_name=name,
        forecast_mode=ForecastMode.NONE,
        reward_type=RewardType.WITH_RISK,
        data=DataConfig(),
        forecasting=ForecastingConfig(),
        environment=EnvironmentConfig(),
        ppo=PPOConfig(),
        wandb_experiment_type=experiment_type,
        wandb_variant=variant,
    )


def get_ppo_with_forecast_config(
    name: str = "PPO-With-Forecast",
    experiment_type: str = "ppo_with_forecast",
    variant: str = "v1",
) -> ExperimentConfig:
    """PPO trading with LSTM forecast"""
    return ExperimentConfig(
        experiment_name=name,
        forecast_mode=ForecastMode.LSTM,
        reward_type=RewardType.WITH_RISK,
        data=DataConfig(),
        forecasting=ForecastingConfig(),
        environment=EnvironmentConfig(),
        ppo=PPOConfig(),
        wandb_experiment_type=experiment_type,
        wandb_variant=variant,
    )


def get_ppo_different_rewards_configs(
    experiment_type: str = "reward_ablation",
    variant: str = "v1",
) -> List[ExperimentConfig]:
    """PPO with different reward definitions for comprehensive ablation study
    
    Enhanced with:
    - Multiple kappa values for WITH_RISK (conservative/moderate/aggressive)
    - Reward scale variations for training dynamics
    - All parameters now configurable via reward_params
    - Proper ablation of each reward component
    - ALL use reward_scale: 100.0 for consistent learning signal
    """
    configs = []
    
    # Comprehensive reward ablation configurations
    reward_configs = [
        # ===== BASIC (Baseline - no risk penalty) =====
        (RewardType.BASIC, "basic_no_penalty", {
            "epsilon": 0.001,
            "reward_scale": 100.0,  # ← INCREASED from 1.0
        }),
        
        # ===== WITH_RISK (Test kappa sensitivity) =====
        # Baseline is now kappa=0.1, so:
        # conservative = 0.2 (2x baseline)
        # moderate = 0.1 (1x baseline - default)
        # aggressive = 0.05 (0.5x baseline)
        (RewardType.WITH_RISK, "with_risk_conservative", {
            "kappa": 0.2,  # INCREASED from 0.05 (2x baseline)
            "epsilon": 0.001,
            "reward_scale": 100.0,  # ← INCREASED from 1.0
        }),
        (RewardType.WITH_RISK, "with_risk_moderate", {
            "kappa": 0.1,  # BASELINE (matches default)
            "epsilon": 0.001,
            "reward_scale": 100.0,  # ← INCREASED from 1.0
        }),
        (RewardType.WITH_RISK, "with_risk_aggressive", {
            "kappa": 0.05,  # DECREASED from 0.001 (0.5x baseline)
            "epsilon": 0.001,
            "reward_scale": 100.0,  # ← INCREASED from 1.0
        }),
        
        # ===== WITH_SHARPE (Risk-adjusted returns) =====
        (RewardType.WITH_SHARPE, "with_sharpe_standard", {
            "epsilon": 0.001,
            "reward_scale": 100.0,  # ← INCREASED from 1.0
        }),
        (RewardType.WITH_SHARPE, "with_sharpe_scaled", {
            "epsilon": 0.001,
            "reward_scale": 50.0,  # ← INCREASED from 0.5 (scaled variant - still 100x)
        }),
        
        # ===== RISK_ADJUSTED (Direct return/volatility) =====
        (RewardType.RISK_ADJUSTED, "risk_adjusted_standard", {
            "epsilon": 0.001,
            "reward_scale": 100.0,  # ← INCREASED from 1.0
        }),
        
        # ===== SORTINO (Downside-focused) - Test scale sensitivity =====
        (RewardType.SORTINO, "sortino_moderate", {
            "epsilon": 0.001,
            "downside_scale": 1.2,  # MODERATE: 20% extra downside penalty
            "reward_scale": 100.0,  # ← INCREASED from 1.0
        }),
        (RewardType.SORTINO, "sortino_conservative", {
            "epsilon": 0.001,
            "downside_scale": 1.5,  # CONSERVATIVE: 50% extra downside penalty
            "reward_scale": 100.0,  # ← INCREASED from 1.0
        }),
        
        # ===== CALMAR (Drawdown-focused) =====
        (RewardType.CALMAR, "calmar_standard", {
            "epsilon": 0.001,
            "drawdown_multiplier": 0.5,
            "reward_scale": 100.0,  # ← INCREASED from 1.0
        }),
        (RewardType.CALMAR, "calmar_aggressive", {
            "epsilon": 0.001,
            "drawdown_multiplier": 0.3,  # Less aggressive drawdown penalty
            "reward_scale": 100.0,  # ← INCREASED from 1.0
        }),
        
        # ===== INFORMATION_RATIO (Consistency-focused) =====
        (RewardType.INFORMATION_RATIO, "info_ratio_standard", {
            "epsilon": 0.001,
            "consistency_bonus": 0.1,
            "reward_scale": 100.0,  # ← INCREASED from 1.0
        }),
        
        # ===== COMPOSITE (Multi-objective) - Test weight variations =====
        (RewardType.COMPOSITE, "composite_balanced", {
            "weight_returns": 0.5,
            "weight_sharpe": 0.3,
            "weight_risk": 0.2,
            "kappa": 0.01,
            "epsilon": 0.001,
            "reward_scale": 100.0,  # ← INCREASED from 1.0
        }),
        (RewardType.COMPOSITE, "composite_conservative", {
            "weight_returns": 0.3,   # Less return-focused
            "weight_sharpe": 0.4,    # More risk-adjusted
            "weight_risk": 0.3,      # More risk penalty
            "kappa": 0.02,           # Higher leverage penalty
            "epsilon": 0.001,
            "reward_scale": 100.0,  # ← INCREASED from 1.0
        }),
        (RewardType.COMPOSITE, "composite_aggressive", {
            "weight_returns": 0.7,   # More return-focused
            "weight_sharpe": 0.2,    # Less risk-adjusted
            "weight_risk": 0.1,      # Less risk penalty
            "kappa": 0.005,          # Lower leverage penalty
            "epsilon": 0.001,
            "reward_scale": 100.0,  # ← INCREASED from 1.0
        }),
    ]
    
    for reward_type, group_name, reward_params in reward_configs:
        env_config = EnvironmentConfig(reward_type=reward_type)
        env_config.reward_params.update(reward_params)
        
        config = ExperimentConfig(
            experiment_name=f"PPO-{reward_type.value.replace('_', ' ').title()}-{group_name}",
            forecast_mode=ForecastMode.NONE,
            reward_type=reward_type,
            data=DataConfig(),
            forecasting=ForecastingConfig(),
            environment=env_config,
            ppo=PPOConfig(),
            wandb_experiment_type=experiment_type,
            wandb_variant=variant,
        )
        configs.append(config)
    
    return configs


def get_all_experiments() -> Dict[str, ExperimentConfig]:
    """Get all experiments with improved WandB organization"""
    experiments = {
        'PPO_Without_Forecast': ExperimentConfig(
            experiment_name="PPO-Without-Forecast",
            forecast_mode=ForecastMode.NONE,
            reward_type=RewardType.WITH_RISK,
            data=DataConfig(),
            forecasting=ForecastingConfig(),
            environment=EnvironmentConfig(reward_type=RewardType.WITH_RISK),
            ppo=PPOConfig(),
            wandb_experiment_type="ppo_baseline",
            wandb_variant="v1",
        ),
        'PPO_With_Forecast': ExperimentConfig(
            experiment_name="PPO-With-Forecast",
            forecast_mode=ForecastMode.LSTM,
            reward_type=RewardType.WITH_RISK,
            data=DataConfig(),
            forecasting=ForecastingConfig(),
            environment=EnvironmentConfig(reward_type=RewardType.WITH_RISK),
            ppo=PPOConfig(),
            wandb_experiment_type="ppo_with_forecast",
            wandb_variant="v1",
        ),
        'PPO_Basic_Reward': ExperimentConfig(
            experiment_name="PPO-Basic-Reward",
            forecast_mode=ForecastMode.NONE,
            reward_type=RewardType.BASIC,
            data=DataConfig(),
            forecasting=ForecastingConfig(),
            environment=EnvironmentConfig(reward_type=RewardType.BASIC),
            ppo=PPOConfig(),
            wandb_experiment_type="reward_ablation",
            wandb_variant="v1",
        ),
        'PPO_With_Risk': ExperimentConfig(
            experiment_name="PPO-With-Risk",
            forecast_mode=ForecastMode.NONE,
            reward_type=RewardType.WITH_RISK,
            data=DataConfig(),
            forecasting=ForecastingConfig(),
            environment=EnvironmentConfig(reward_type=RewardType.WITH_RISK),
            ppo=PPOConfig(),
            wandb_experiment_type="reward_ablation",
            wandb_variant="v1",
        ),
        'PPO_With_Sharpe': ExperimentConfig(
            experiment_name="PPO-With-Sharpe",
            forecast_mode=ForecastMode.NONE,
            reward_type=RewardType.WITH_SHARPE,
            data=DataConfig(),
            forecasting=ForecastingConfig(),
            environment=EnvironmentConfig(reward_type=RewardType.WITH_SHARPE),
            ppo=PPOConfig(),
            wandb_experiment_type="reward_ablation",
            wandb_variant="v1",
        ),
        'PPO_Risk_Adjusted': ExperimentConfig(
            experiment_name="PPO-Risk-Adjusted",
            forecast_mode=ForecastMode.NONE,
            reward_type=RewardType.RISK_ADJUSTED,
            data=DataConfig(),
            forecasting=ForecastingConfig(),
            environment=EnvironmentConfig(reward_type=RewardType.RISK_ADJUSTED),
            ppo=PPOConfig(),
            wandb_experiment_type="reward_ablation",
            wandb_variant="v1",
        ),
    }
    
    return experiments


def get_forecast_only_config(
    name: str = "Forecast-Only",
    group: str = "baseline"
) -> ExperimentConfig:
    """Standalone forecasting model (no RL)"""
    return ExperimentConfig(
        experiment_name=name,
        forecast_mode=ForecastMode.LSTM,
        reward_type=RewardType.BASIC,
        data=DataConfig(),
        forecasting=ForecastingConfig(),
        environment=EnvironmentConfig(),
        ppo=PPOConfig(total_updates=0),  # No RL training
        wandb_group=group,
    )


# ============================================================
# BUILDER FOR CUSTOM CONFIGURATIONS
# ============================================================

class ConfigBuilder:
    """Builder pattern for creating custom configurations"""
    
    def __init__(self, base_name: str = "Custom-Experiment"):
        self.config = ExperimentConfig(
            experiment_name=base_name,
            forecast_mode=ForecastMode.NONE,
            reward_type=RewardType.WITH_RISK,
            data=DataConfig(),
            forecasting=ForecastingConfig(),
            environment=EnvironmentConfig(),
            ppo=PPOConfig(),
        )
    
    def with_forecast(self, mode: ForecastMode) -> "ConfigBuilder":
        self.config.forecast_mode = mode
        return self
    
    def with_reward(self, reward_type: RewardType) -> "ConfigBuilder":
        self.config.reward_type = reward_type
        self.config.environment.reward_type = reward_type
        return self
    
    def with_initial_equity(self, equity: float) -> "ConfigBuilder":
        self.config.environment.initial_equity = equity
        return self
    
    def with_fee(self, fee: float) -> "ConfigBuilder":
        self.config.environment.fee = fee
        return self
    
    def with_leverage(self, leverage: float) -> "ConfigBuilder":
        self.config.environment.leverage_max = leverage
        return self
    
    def with_ppo_updates(self, updates: int) -> "ConfigBuilder":
        self.config.ppo.total_updates = updates
        return self
    
    def with_wandb_group(self, group: str) -> "ConfigBuilder":
        self.config.wandb_group = group
        return self
    
    def build(self) -> ExperimentConfig:
        return self.config


if __name__ == "__main__":
    # Example: Create and print a configuration
    config = get_ppo_without_forecast_config()
    print(f"Experiment: {config.experiment_name}")
    print(f"Forecast Mode: {config.forecast_mode.value}")
    print(f"Reward Type: {config.reward_type.value}")
    print(f"Initial Equity: ${config.environment.initial_equity:,.0f}")
    print(f"PPO Updates: {config.ppo.total_updates}")
