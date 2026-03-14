# WandB Seed-Based Organization - Quick Reference

## NEW Structure with Seeds [10, 20, 30]

### Group Structure
```
Format: {date}/{experiment_type}/{variant}/seed_{seed}

Examples:
✅ 2026-03-12/ppo_baseline/v1/seed_10
✅ 2026-03-12/ppo_baseline/v1/seed_20
✅ 2026-03-12/ppo_baseline/v1/seed_30
✅ 2026-03-12/reward_ablation/v1/seed_10
✅ 2026-03-12/ppo_with_forecast/v1/seed_20
```

### Run Name Structure
```
Format: {date}_{experiment_name}_{reward_type}_{seed}

Examples:
✅ 2026-03-12_ppo_baseline_with_risk_10
✅ 2026-03-12_ppo_baseline_with_risk_20
✅ 2026-03-12_ppo_baseline_with_risk_30
✅ 2026-03-12_reward_ablation_sharpe_10
```

### Tags
```
All runs tagged with explicit date and seed for filtering:
["date_2026-03-12", "2026-03-12", "ppo_baseline", "forecast_none", "reward_with_risk", "version_v1", "seed_10", "seed"]
                     ↑ implicit date     ↑ explicit date                                                    ↑ seed num  ↑ seed marker
```

---

## WandB Tree View

```
2026-03-12/
├── ppo_baseline/v1/
│   ├── seed_10 ← Group for seed 10
│   │   ├── run_1
│   │   ├── run_2
│   │   └── run_3
│   ├── seed_20 ← Group for seed 20
│   │   ├── run_1
│   │   ├── run_2
│   │   └── run_3
│   └── seed_30 ← Group for seed 30
│       ├── run_1
│       ├── run_2
│       └── run_3
├── reward_ablation/v1/
│   ├── seed_10 (8 runs for 8 reward types)
│   ├── seed_20 (8 runs)
│   └── seed_30 (8 runs)
└── ppo_with_forecast/v1/
    ├── seed_10
    ├── seed_20
    └── seed_30
```

---

## Code Usage

### Update ExperimentConfig with Different Seeds

```python
# Seed 10
config_seed_10 = ExperimentConfig(
    experiment_name="PPO-Baseline",
    forecast_mode=ForecastMode.NONE,
    reward_type=RewardType.WITH_RISK,
    wandb_experiment_type="ppo_baseline",
    wandb_variant="v1",
    seed=10,  # ← Seed 10
)

# Seed 20
config_seed_20 = ExperimentConfig(
    experiment_name="PPO-Baseline",
    forecast_mode=ForecastMode.NONE,
    reward_type=RewardType.WITH_RISK,
    wandb_experiment_type="ppo_baseline",
    wandb_variant="v1",
    seed=20,  # ← Seed 20
)

# Seed 30
config_seed_30 = ExperimentConfig(
    experiment_name="PPO-Baseline",
    forecast_mode=ForecastMode.NONE,
    reward_type=RewardType.WITH_RISK,
    wandb_experiment_type="ppo_baseline",
    wandb_variant="v1",
    seed=30,  # ← Seed 30
)

# Results in:
# config_seed_10.wandb_group → "2026-03-12/ppo_baseline/v1/seed_10"
# config_seed_20.wandb_group → "2026-03-12/ppo_baseline/v1/seed_20"
# config_seed_30.wandb_group → "2026-03-12/ppo_baseline/v1/seed_30"
```

### Run All Seeds in Loop

```python
SEEDS = [10, 20, 30]

for seed in SEEDS:
    config = ExperimentConfig(
        experiment_name="PPO-Baseline",
        forecast_mode=ForecastMode.NONE,
        reward_type=RewardType.WITH_RISK,
        wandb_experiment_type="ppo_baseline",
        wandb_variant="v1",
        seed=seed,  # ← Varies for each run
    )

    runner = ExperimentRunner(config)
    results = runner.run()  # Each run goes to different seed group
```

---

## WandB Filtering Examples

### Compare All Three Seeds

Filter in WandB:
```
tags:ppo_baseline AND tags:v1 AND (tags:seed_10 OR tags:seed_20 OR tags:seed_30)
```

Then:
1. Create metric chart: `performance/sharpe_ratio`
2. View three lines (one for each seed)
3. Compare results across seeds
4. Identify stable/robust configurations

### Find Best Seed for Reward Type

Filter:
```
tags:reward_with_risk AND (tags:seed_10 OR tags:seed_20 OR tags:seed_30)
```

Sort by: `performance/sharpe_ratio` (descending)

Result: See which seed [10, 20, 30] performs best

### Check Reproducibility

Filter:
```
tags:ppo_baseline AND tags:v1
```

Create chart showing all seeds:
- High agreement between seed_10, seed_20, seed_30 = reproducible
- Big differences = luck/sensitivity dependent
- Use this to assess algorithm robustness

### Aggregate Across Seeds

In WandB's advanced features:
1. Filter: `tags:ppo_baseline`
2. Create custom metric: `avg_sharpe_ratio = mean([seed_10, seed_20, seed_30])`
3. Plot aggregated performance
4. Compare against other experiment types

---

## Properties Generated Automatically

```python
config = ExperimentConfig(
    experiment_name="PPO-Baseline",
    seed=10,
    wandb_experiment_type="ppo_baseline",
    wandb_variant="v1",
)

# These are auto-generated from seed:
config.wandb_group
→ "2026-03-12/ppo_baseline/v1/seed_10"

config.wandb_run_name
→ "2026-03-12_ppo_baseline_with_risk_10"

config.wandb_tags
→ ["2026-03-12", "ppo_baseline", "forecast_none", "reward_with_risk", "version_v1", "seed_10"]
```

---

## Key Differences from Before

| Aspect | Before | After |
|--------|--------|-------|
| **Group** | `"baseline"` | `"2026-03-12/ppo_baseline/v1/seed_10"` |
| **Seed Tracking** | ❌ Not in group | ✅ In group path |
| **Seed Comparison** | ❌ Hard to compare | ✅ Easy - separate groups |
| **Date Filtering** | ❌ No date info | ✅ Filter by date tag |
| **Run Name** | Generic | Includes seed number |
| **Tags** | Minimal | Comprehensive with seed |

---

## Recommended Workflow

### 1. Run All Three Seeds
```python
SEEDS = [10, 20, 30]
CONFIG_TYPE = "ppo_baseline"

for seed in SEEDS:
    config = get_ppo_without_forecast_config(
        seed=seed,
        experiment_type=CONFIG_TYPE
    )
    runner = ExperimentRunner(config)
    runner.run()
```

### 2. Check WandB for Results
```
Go to: 2026-03-12/ppo_baseline/v1/
View: seed_10, seed_20, seed_30 groups
```

### 3. Compare Across Seeds
```
Filter: tags:ppo_baseline AND tags:v1
Chart: performance/sharpe_ratio
Compare: All three seed groups
```

### 4. Aggregate Results
```
Calculate: mean and std across seeds
Determine: robustness and stability
```

---

## Summary

✅ **Each seed gets its own group** for easy comparison
✅ **Date automatically included** - no manual updates needed
✅ **Seeds [10, 20, 30]** supported and organized
✅ **Hierarchical structure** - date → type → version → seed
✅ **Easy filtering** - compare seeds side-by-side
✅ **Reproducibility tracking** - see variance across seeds
✅ **Professional organization** - clean WandB UI

All seed-based grouping is **automatic** - just set `seed=10/20/30` and the rest is handled! 🎯


