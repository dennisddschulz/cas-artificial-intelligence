# WandB Organization Strategy - Improved Grouping & Filtering

## Current Issues
- Fixed group name "baseline" makes it hard to distinguish runs by date
- No temporal filtering capability
- Difficult to compare experiments from different days
- Hard to track experiment iterations and versions

## Proposed Solution: Hierarchical Organization with Date-Based Grouping

### Strategy Overview

```
wandb_project: "PPO_Bitcoin_Trading_Experiments_Dennis"
wandb_entity: "btcprojekt2026-bfh"

# WandB Organization Strategy - Improved Grouping & Filtering with Seeds

## Current Issues
- Fixed group name "baseline" makes it hard to distinguish runs by date
- No seed-based organization (can't easily compare seeds [10, 20, 30])
- No temporal filtering capability
- Difficult to compare experiments from different days
- Hard to track experiment iterations and versions

## Proposed Solution: Hierarchical Organization with Date & Seed-Based Grouping

### Strategy Overview

```
wandb_project: "PPO_Bitcoin_Trading_Experiments_Dennis"
wandb_entity: "btcprojekt2026-bfh"

wandb_group: "{date}/{experiment_type}/{variant}/seed_{seed}"
            Example: "2026-03-12/ppo_baseline/v1/seed_10"
                     "2026-03-12/ppo_baseline/v1/seed_20"
                     "2026-03-12/ppo_baseline/v1/seed_30"

tags: [date_2026-03-12, 2026-03-12, ppo_baseline, forecast_none, reward_with_risk, version_v1, seed_10, seed]
      Example with explicit date and seed tags

run_name: "{date}_{experiment_type}_{reward_type}_{seed}"
          Example: "2026-03-12_ppo_baseline_with_risk_10"
```

---

## Benefits

### 1. Seed-Based Organization
- Each seed [10, 20, 30] gets its own group
- Easy to compare same config across different seeds
- Track reproducibility across seed variations
- Identify seed-dependent behaviors

### 2. Date-Based Filtering
- Easily find all experiments from a specific date
- Compare same config across dates
- Track daily progress and improvements

### 3. Hierarchical Grouping
- `group: "2026-03-12/ppo_baseline/v1/seed_10"` creates clean structure
- Easy to expand to multiple versions/variants/seeds
- Clear hierarchy: date → experiment type → version → seed

### 4. Tag-Based Analysis
- Filter by any combination of tags
- Create charts filtered by specific criteria
- Compare experiments with same reward type across seeds

### 5. Better Run Identification
- Run name includes date, type, reward, and seed
- Easy to find specific run in WandB
- Timestamp helps track execution order

---

## WandB UI Navigation

With this structure, you can:

```
PROJECT VIEW:
├── 2026-03-12
│   ├── ppo_baseline
│   │   ├── v1
│   │   │   ├── seed_10 (all seed_10 runs for this config)
│   │   │   ├── seed_20 (all seed_20 runs for this config)
│   │   │   ├── seed_30 (all seed_30 runs for this config)
│   ├── reward_ablation
│   │   ├── v1
│   │   │   ├── seed_10
│   │   │   ├── seed_20
│   │   │   ├── seed_30
├── 2026-03-13
│   ├── ppo_baseline
│   │   ├── v1
│   │   │   ├── seed_10
│   │   │   ├── seed_20
│   │   │   ├── seed_30
│   ├── ppo_with_forecast
│   │   ├── v1
│   │   │   ├── seed_10
│   │   │   ├── seed_20
│   │   │   ├── seed_30

FILTERING OPTIONS:
✅ Filter by date: tags:2026-03-12
✅ Filter by seed: tags:seed_10, tags:seed_20, tags:seed_30
✅ Filter by experiment type: group contains "ppo_baseline"
✅ Filter by reward: tags:reward_with_risk
✅ Filter by forecast: tags:forecast_lstm
✅ Filter by version: group contains "/v1"
✅ Combine filters: tags:seed_10 AND tags:ppo_baseline
```

---

## Implementation Details

### 1. Timestamp & Seed-Based Naming Functions

```python
from datetime import datetime
import pytz

def get_wandb_group(experiment_type: str, variant: str = "v1", seed: int = 10) -> str:
    """
    Generate hierarchical group name with date AND seed

    Format: YYYY-MM-DD/experiment_type/variant/seed_{seed}
    Example: 2026-03-12/ppo_baseline/v1/seed_10

    This creates a tree structure in WandB:
    2026-03-12/ppo_baseline/v1/
      seed_10 (all seed_10 runs)
      seed_20 (all seed_20 runs)
      seed_30 (all seed_30 runs)
    """
    today = datetime.now(pytz.UTC).strftime("%Y-%m-%d")
    return f"{today}/{experiment_type}/{variant}/seed_{seed}"

def get_wandb_run_name(experiment_name: str, reward_type: str, seed: int) -> str:
    """
    Generate descriptive run name

    Format: YYYY-MM-DD_experiment_reward_seed
    Example: 2026-03-12_ppo_baseline_with_risk_10
    """
    today = datetime.now(pytz.UTC).strftime("%Y-%m-%d")
    return f"{today}_{experiment_name}_{reward_type}_{seed}"

def get_wandb_tags(
    experiment_type: str,
    forecast_mode: str,
    reward_type: str,
    version: str = "v1",
    seed: int = 10
) -> list:
    """
    Generate comprehensive tags for filtering
    """
    today = datetime.now(pytz.UTC).strftime("%Y-%m-%d")
    return [
        today,                          # Date: 2026-03-12
        experiment_type,                # ppo_baseline
        f"forecast_{forecast_mode}",   # forecast_lstm or forecast_none
        f"reward_{reward_type}",        # reward_with_risk
        f"version_{version}",           # version_v1
        f"seed_{seed}",                # seed_10, seed_20, seed_30
    ]
```

### 2. Updated ExperimentConfig

```python
@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    experiment_name: str
    forecast_mode: ForecastMode
    reward_type: RewardType

    # ... other fields ...

    # W&B logging - Now with date-based AND seed-based organization
    wandb_project: str = "PPO_Bitcoin_Trading_Experiments_Dennis"
    wandb_entity: str = "btcprojekt2026-bfh"

    # Experiment type and variant
    wandb_experiment_type: str = "ppo_baseline"  # e.g., ppo_baseline, reward_ablation
    wandb_variant: str = "v1"                    # e.g., v1, v2, experimental
    use_date_based_grouping: bool = True         # Enable date-based groups

    seed: int = 10  # Seed values [10, 20, 30]

    # Properties that generate dynamic values
    @property
    def wandb_group(self) -> str:
        """
        Dynamic group: YYYY-MM-DD/experiment_type/variant/seed_{seed}
        Example: 2026-03-12/ppo_baseline/v1/seed_10

        Each seed gets its own group for easy comparison!
        """
        if self.use_date_based_grouping:
            return get_wandb_group(
                self.wandb_experiment_type,
                self.wandb_variant,
                self.seed
            )
        else:
            return f"{self.wandb_experiment_type}/seed_{self.seed}"

    @property
    def wandb_run_name(self) -> str:
        """Dynamic run name with seed"""
        return get_wandb_run_name(
            self.experiment_name,
            self.reward_type.value,
            self.seed
        )

    @property
    def wandb_tags(self) -> list:
        """Dynamic tags including seed for filtering"""
        return get_wandb_tags(
            self.wandb_experiment_type,
            self.forecast_mode.value,
            self.reward_type.value,
            self.wandb_variant,
            self.seed
        )
```

### 3. Usage in trading_framework.py

```python
def setup_wandb(self):
    """Initialize Weights & Biases with improved organization"""
    if not (WANDB_AVAILABLE and self.config.use_wandb):
        return

    try:
        wandb.init(
            project=self.config.wandb_project,
            group=self.config.wandb_group,  # Now: "2026-03-12/ppo_baseline/v1/seed_10"
            name=self.config.wandb_run_name,  # Now: "2026-03-12_ppo_baseline_with_risk_10"
            entity=self.config.wandb_entity,
            config={...},
            tags=self.config.wandb_tags,  # Now: ["2026-03-12", "ppo_baseline", "forecast_none", "reward_with_risk", "version_v1", "seed_10"]
        )
```

---

## Comparison Strategies in WandB

### Strategy 1: Compare All Seeds for Same Config

1. Go to WandB project
2. Filter: `tags:ppo_baseline AND tags:v1`
3. Compare seed_10, seed_20, seed_30 side-by-side
4. Analyze robustness across seeds

### Strategy 2: Compare Same Experiment Across Dates

1. Filter: `tags:ppo_baseline AND tags:seed_10`
2. Compare performance over days
3. Identify trends and improvements for seed_10

### Strategy 3: Find Best Seed

1. Filter: `tags:2026-03-12 AND tags:ppo_baseline`
2. Compare all three seeds (10, 20, 30)
3. Identify which seed gives best results

### Strategy 4: Reproducibility Check

1. Filter: `tags:seed_10 AND tags:seed_20 AND tags:seed_30 AND tags:ppo_baseline`
2. Check variance across seeds
3. Low variance = robust algorithm

### Strategy 5: Debug Specific Seed Issue

1. Filter: `tags:seed_30 AND tags:2026-03-12`
2. Find all experiments with seed_30 today
3. Investigate seed-specific problems

---

## Example Configurations

### PPO Baseline with Seeds [10, 20, 30]
```python
# For seed=10:
config = ExperimentConfig(
    experiment_name="PPO-Baseline",
    forecast_mode=ForecastMode.NONE,
    reward_type=RewardType.WITH_RISK,
    wandb_experiment_type="ppo_baseline",
    wandb_variant="v1",
    seed=10,  # or 20, 30
)

# Results for seed_10:
# group: "2026-03-12/ppo_baseline/v1/seed_10"
# run_name: "2026-03-12_ppo_baseline_with_risk_10"
# tags: ["2026-03-12", "ppo_baseline", "forecast_none", "reward_with_risk", "version_v1", "seed_10"]

# Results for seed_20:
# group: "2026-03-12/ppo_baseline/v1/seed_20"
# run_name: "2026-03-12_ppo_baseline_with_risk_20"
# tags: ["2026-03-12", "ppo_baseline", "forecast_none", "reward_with_risk", "version_v1", "seed_20"]

# Results for seed_30:
# group: "2026-03-12/ppo_baseline/v1/seed_30"
# run_name: "2026-03-12_ppo_baseline_with_risk_30"
# tags: ["2026-03-12", "ppo_baseline", "forecast_none", "reward_with_risk", "version_v1", "seed_30"]
```

### Reward Ablation with Multiple Seeds
```python
# Each reward type runs with all three seeds [10, 20, 30]
config = ExperimentConfig(
    experiment_name="PPO-With-Sharpe",
    reward_type=RewardType.WITH_SHARPE,
    wandb_experiment_type="reward_ablation",
    wandb_variant="v1",
    seed=10,  # or 20, 30
)

# Results:
# seed_10: group: "2026-03-12/reward_ablation/v1/seed_10"
# seed_20: group: "2026-03-12/reward_ablation/v1/seed_20"
# seed_30: group: "2026-03-12/reward_ablation/v1/seed_30"
```

---

## WandB Filter Examples

### Compare all three seeds for baseline
```
tags:ppo_baseline AND tags:v1 AND (tags:seed_10 OR tags:seed_20 OR tags:seed_30)
```

### Find best seed for with_risk reward
```
tags:reward_with_risk AND (tags:seed_10 OR tags:seed_20 OR tags:seed_30)
Sort by: performance/sharpe_ratio
```

### Check reproducibility (low variance across seeds)
```
1. Filter: tags:ppo_baseline AND tags:v1
2. Create metric chart: sharpe_ratio
3. View lines for seed_10, seed_20, seed_30
4. Low variance = robust algorithm
```

### Debug seed_30 issues
```
tags:seed_30 AND tags:2026-03-12
```

### All experiments from today with all seeds
```
tags:2026-03-12 AND (tags:seed_10 OR tags:seed_20 OR tags:seed_30)
```

---

## WandB View Structure

```
PROJECT: PPO_Bitcoin_Trading_Experiments_Dennis

2026-03-12/
├── ppo_baseline/
│   └── v1/
│       ├── seed_10 (3 runs - one per repeat)
│       ├── seed_20 (3 runs)
│       └── seed_30 (3 runs)
├── reward_ablation/
│   └── v1/
│       ├── seed_10 (8 runs - one per reward type)
│       ├── seed_20 (8 runs)
│       └── seed_30 (8 runs)
└── ppo_with_forecast/
    └── v1/
        ├── seed_10 (2 runs)
        ├── seed_20 (2 runs)
        └── seed_30 (2 runs)

2026-03-13/
├── ppo_baseline/
│   └── v1/
│       ├── seed_10
│       ├── seed_20
│       └── seed_30
└── reward_ablation/
    └── v2/  (New version)
        ├── seed_10
        ├── seed_20
        └── seed_30
```

Much more organized with seed-based comparison! 🎯

---

## Summary

### Benefits of Seed-Based Organization

✅ **Seed comparison** - Compare results across seeds [10, 20, 30]
✅ **Reproducibility** - Track variance across seeds
✅ **Date-based filtering** - Find experiments from any day
✅ **Temporal comparison** - See trends over time
✅ **Type-based grouping** - Organize by experiment kind
✅ **Version tracking** - Track iterations (v1, v2, etc.)
✅ **Multi-dimensional filtering** - Combine tags for precise queries
✅ **Clean WandB UI** - Hierarchical seed-organized structure
✅ **Easy comparison** - All related experiments grouped together


```

---

## Benefits

### 1. Date-Based Filtering
- Easily find all experiments from a specific date
- Compare same config across dates
- Track daily progress

### 2. Hierarchical Grouping
- `group: "2026-03-12/ppo_baseline/v1"` creates logical structure in WandB UI
- Easy to expand to multiple versions/variants
- Clear hierarchy: date → experiment type → version

### 3. Tag-Based Analysis
- Filter by any combination of tags
- Create charts filtered by specific criteria
- Compare experiments with same reward type across dates

### 4. Better Run Identification
- Run name includes all key info
- Easy to find specific run in WandB
- Timestamp helps track execution order

---

## WandB UI Navigation

With this structure, you can:

```
PROJECT VIEW:
├── 2026-03-12
│   ├── ppo_baseline
│   │   ├── v1 (contains all runs from that group)
│   ├── reward_ablation
│   │   ├── v1
├── 2026-03-13
│   ├── ppo_baseline
│   │   ├── v1
│   ├── ppo_with_forecast
│   │   ├── v1

FILTERING OPTIONS:
✅ Filter by date: tag=2026-03-12
✅ Filter by experiment type: group contains "ppo_baseline"
✅ Filter by reward: tag=with_risk
✅ Filter by forecast: tag=lstm
✅ Filter by version: group contains "/v1"
✅ Filter by seed: tag=seed_42
```

---

## Implementation Details

### 1. Timestamp-Based Naming Function

```python
from datetime import datetime
import pytz

def get_wandb_group(experiment_type: str, variant: str = "v1") -> str:
    """
    Generate hierarchical group name with date

    Format: YYYY-MM-DD/experiment_type/variant
    Example: 2026-03-12/ppo_baseline/v1
    """
    # Use a fixed timezone (UTC) for consistency
    today = datetime.now(pytz.UTC).strftime("%Y-%m-%d")
    return f"{today}/{experiment_type}/{variant}"

def get_wandb_run_name(experiment_name: str, reward_type: str, seed: int) -> str:
    """
    Generate descriptive run name

    Format: YYYY-MM-DD_experiment_reward_seed
    Example: 2026-03-12_ppo_baseline_with_risk_42
    """
    today = datetime.now(pytz.UTC).strftime("%Y-%m-%d")
    return f"{today}_{experiment_name}_{reward_type}_{seed}"

def get_wandb_tags(
    experiment_type: str,
    forecast_mode: str,
    reward_type: str,
    version: str = "v1",
    seed: int = 42
) -> List[str]:
    """
    Generate comprehensive tags for filtering
    """
    today = datetime.now(pytz.UTC).strftime("%Y-%m-%d")
    return [
        today,                           # Date: 2026-03-12
        experiment_type,                 # ppo_baseline
        forecast_mode,                   # lstm or none
        reward_type,                     # with_risk
        version,                         # v1
        f"seed_{seed}",                 # seed_42
    ]
```

### 2. Updated ExperimentConfig

```python
@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    experiment_name: str
    forecast_mode: ForecastMode
    reward_type: RewardType

    # ... other fields ...

    # W&B logging - Now with date-based organization
    wandb_project: str = "PPO_Bitcoin_Trading_Experiments_Dennis"
    wandb_entity: str = "btcprojekt2026-bfh"

    # NEW: Separate components for dynamic group generation
    wandb_experiment_type: str = "ppo_baseline"  # e.g., ppo_baseline, reward_ablation, etc.
    wandb_variant: str = "v1"                    # e.g., v1, v2, experimental
    use_date_based_grouping: bool = True         # Enable date-based groups

    # Properties that generate dynamic values
    @property
    def wandb_group(self) -> str:
        """Dynamic group based on date and experiment type"""
        if self.use_date_based_grouping:
            return get_wandb_group(
                self.wandb_experiment_type,
                self.wandb_variant
            )
        else:
            return self.wandb_experiment_type

    @property
    def wandb_run_name(self) -> str:
        """Dynamic run name with date and configuration"""
        return get_wandb_run_name(
            self.experiment_name,
            self.reward_type.value,
            self.seed
        )

    @property
    def wandb_tags(self) -> List[str]:
        """Dynamic tags for filtering"""
        return get_wandb_tags(
            self.wandb_experiment_type,
            self.forecast_mode.value,
            self.reward_type.value,
            self.wandb_variant,
            self.seed
        )
```

### 3. Usage in trading_framework.py

```python
def setup_wandb(self):
    """Initialize Weights & Biases with improved organization"""
    if not (WANDB_AVAILABLE and self.config.use_wandb):
        return

    try:
        wandb.init(
            project=self.config.wandb_project,
            group=self.config.wandb_group,  # Now: "2026-03-12/ppo_baseline/v1"
            name=self.config.wandb_run_name,  # Now: "2026-03-12_ppo_baseline_with_risk_42"
            entity=self.config.wandb_entity,
            config={...},
            tags=self.config.wandb_tags,  # Now: ["2026-03-12", "ppo_baseline", "lstm", ...]
        )
```

---

## Comparison Strategies in WandB

### Strategy 1: Compare All Experiments from Today

1. Go to WandB project
2. Filter: `tags:2026-03-12`
3. View all experiments from today
4. Compare performance by reward type, forecast mode, etc.

### Strategy 2: Compare Same Experiment Across Dates

1. Filter: `tags:ppo_baseline AND tags:with_risk`
2. Compare performance over days
3. Identify trends and improvements

### Strategy 3: Compare Versions

1. Filter: `tags:ppo_baseline AND tags:v1`
2. All v1 experiments across all dates
3. Useful for tracking iteration quality

### Strategy 4: Debug Specific Seed

1. Filter: `tags:seed_42 AND tags:2026-03-12`
2. Find all experiments with seed 42 today
3. Check reproducibility

---

## Example Configurations

### Basic PPO Baseline (Today)
```python
config = ExperimentConfig(
    experiment_name="PPO-Baseline",
    forecast_mode=ForecastMode.NONE,
    reward_type=RewardType.WITH_RISK,
    wandb_experiment_type="ppo_baseline",    # NEW
    wandb_variant="v1",                      # NEW
    # ... other fields ...
)

# Results:
# group: "2026-03-12/ppo_baseline/v1"
# run_name: "2026-03-12_PPO-Baseline_with_risk_42"
# tags: ["2026-03-12", "ppo_baseline", "none", "with_risk", "v1", "seed_42"]
```

### Reward Ablation Study (Today, Version 2)
```python
config = ExperimentConfig(
    experiment_name="PPO-With-Sharpe",
    forecast_mode=ForecastMode.NONE,
    reward_type=RewardType.WITH_SHARPE,
    wandb_experiment_type="reward_ablation",  # Different type
    wandb_variant="v2",                       # Version 2
    # ... other fields ...
)

# Results:
# group: "2026-03-12/reward_ablation/v2"
# run_name: "2026-03-12_PPO-With-Sharpe_with_sharpe_42"
# tags: ["2026-03-12", "reward_ablation", "none", "with_sharpe", "v2", "seed_42"]
```

### LSTM Experiments
```python
config = ExperimentConfig(
    experiment_name="PPO-With-Forecast",
    forecast_mode=ForecastMode.LSTM,
    reward_type=RewardType.WITH_RISK,
    wandb_experiment_type="ppo_with_forecast",
    wandb_variant="v1",
    # ... other fields ...
)

# Results:
# group: "2026-03-12/ppo_with_forecast/v1"
# run_name: "2026-03-12_PPO-With-Forecast_with_risk_42"
# tags: ["2026-03-12", "ppo_with_forecast", "lstm", "with_risk", "v1", "seed_42"]
```

---

## WandB Filter Examples

### Find all with-risk experiments from today
```
tags:2026-03-12 AND tags:with_risk
```

### Find all LSTM experiments
```
tags:lstm
```

### Find baseline experiments version 1
```
tags:ppo_baseline AND tags:v1
```

### Find seed 42 results from last 3 days
```
tags:seed_42 AND (tags:2026-03-10 OR tags:2026-03-11 OR tags:2026-03-12)
```

### Find best performing experiments by date
```
1. Filter: tags:2026-03-12
2. Sort by: performance/sharpe_ratio
3. Compare top results
```

---

## Additional Metadata in Config

```python
@dataclass
class ExperimentConfig:
    # ... existing fields ...

    # Experiment metadata for organization
    wandb_experiment_type: str = "ppo_baseline"
    wandb_variant: str = "v1"
    wandb_description: str = ""  # Optional detailed description

    use_date_based_grouping: bool = True
```

---

## Implementation Steps

1. Add helper functions (get_wandb_group, get_wandb_run_name, get_wandb_tags)
2. Update ExperimentConfig with new fields and properties
3. Update trading_framework.py to use dynamic group/name/tags
4. Update all preset configurations to specify experiment_type
5. Test with a few experiments
6. Document the grouping strategy

---

## Summary

### Benefits of This Approach

✅ **Date-based filtering** - Find experiments from any day
✅ **Temporal comparison** - See trends over time
✅ **Type-based grouping** - Organize by experiment kind
✅ **Version tracking** - Track iterations (v1, v2, etc.)
✅ **Multi-dimensional filtering** - Combine tags for precise queries
✅ **Clean WandB UI** - Hierarchical group names
✅ **Reproducibility** - Seed and date in run name
✅ **Easy comparison** - All related experiments grouped together

### Example WandB View

```
PROJECT: PPO_Bitcoin_Trading_Experiments_Dennis

GROUPS:
├── 2026-03-10
│   ├── ppo_baseline / v1 (3 runs)
│   ├── reward_ablation / v1 (8 runs)
├── 2026-03-11
│   ├── ppo_baseline / v1 (3 runs)
│   ├── ppo_with_forecast / v1 (2 runs)
│   ├── reward_ablation / v2 (8 runs)
├── 2026-03-12
│   ├── ppo_baseline / v1 (3 runs)
│   ├── ppo_baseline / v2 (3 runs, experimental)
│   ├── reward_ablation / v2 (8 runs)
```

Much more organized and filterable! 🎯


