# ✅ Fixed run_all_experiments.py - Parameter Update

## Issue
The `run_all_experiments.py` script was using old `group` parameter that no longer exists in the configuration functions.

## Error Messages
```
TypeError: get_ppo_with_forecast_config() got an unexpected keyword argument 'group'
TypeError: get_ppo_different_rewards_configs() got an unexpected keyword argument 'group'
```

## Root Cause
Earlier updates changed function signatures from:
```python
def get_ppo_with_forecast_config(group: str = "baseline"):
    ...

def get_ppo_different_rewards_configs(group: str = "reward_ablation"):
    ...
```

To:
```python
def get_ppo_with_forecast_config(experiment_type: str = "ppo_with_forecast", variant: str = "v1"):
    ...

def get_ppo_different_rewards_configs(experiment_type: str = "reward_ablation", variant: str = "v1"):
    ...
```

## Changes Made

### 1. Fixed Experiment 1 Call (Line 620)
**Before:**
```python
config = get_ppo_without_forecast_config(
    name="PPO-Without-Forecast",
    group="baseline"
)
```

**After:**
```python
config = get_ppo_without_forecast_config(
    name="PPO-Without-Forecast",
    experiment_type="ppo_baseline",
    variant="v1"
)
```

### 2. Fixed Experiment 2 Call (Line 661)
**Before:**
```python
config = get_ppo_with_forecast_config(
    name="PPO-With-Forecast",
    group="baseline"
)
```

**After:**
```python
config = get_ppo_with_forecast_config(
    name="PPO-With-Forecast",
    experiment_type="ppo_with_forecast",
    variant="v1"
)
```

### 3. Fixed Experiment 3 Call (Line 701)
**Before:**
```python
reward_configs = get_ppo_different_rewards_configs(group="reward_ablation")
```

**After:**
```python
reward_configs = get_ppo_different_rewards_configs(
    experiment_type="reward_ablation",
    variant="v1"
)
```

## Benefits of New Parameters

### `experiment_type` Parameter
- Clearly identifies the type of experiment
- Used in WandB group organization
- Makes hierarchical organization possible
- Examples: `"ppo_baseline"`, `"reward_ablation"`, `"ppo_with_forecast"`

### `variant` Parameter
- Tracks versions of experiments
- Enables comparison across versions
- Default: `"v1"` (can be `"v2"`, `"experimental"`, etc.)
- Included in WandB group path: `{date}/{experiment_type}/{variant}/seed_{seed}`

## WandB Group Structure
With these changes, experiments are now organized as:

```
2026-03-12/
├── ppo_baseline/v1/
│   ├── seed_10
│   ├── seed_20
│   └── seed_30
├── ppo_with_forecast/v1/
│   ├── seed_10
│   ├── seed_20
│   └── seed_30
└── reward_ablation/v1/
    ├── seed_10
    ├── seed_20
    └── seed_30
```

## Status
✅ All three function calls have been corrected.
✅ Script should now run without parameter errors.
✅ WandB logging will use new hierarchical organization.

## Next Steps
Run the script:
```bash
python run_all_experiments.py
```

All experiments should now execute with proper WandB grouping!


