# ✅ WandB Seed-Based Organization - Implementation Complete

## Status: ✅ COMPLETE AND READY TO USE

---

## What Was Changed

### 1. Updated `get_wandb_group()` Function
Now includes **seed** parameter in the group hierarchy:

**Format:** `{date}/{experiment_type}/{variant}/seed_{seed}`
**Example:** `2026-03-12/ppo_baseline/v1/seed_10`

```python
def get_wandb_group(experiment_type: str, variant: str = "v1", seed: int = 42) -> str:
    today = datetime.now(pytz.UTC).strftime("%Y-%m-%d")
    return f"{today}/{experiment_type}/{variant}/seed_{seed}"  # ← Seed included!
```

### 2. Updated `wandb_group` Property
Now passes seed parameter to `get_wandb_group()`:

```python
@property
def wandb_group(self) -> str:
    if self.use_date_based_grouping:
        return get_wandb_group(
            self.wandb_experiment_type,
            self.wandb_variant,
            self.seed  # ← Seed passed to function
        )
```

### 3. All Preset Functions Updated
All configuration functions support specifying `experiment_type`:

```python
def get_ppo_without_forecast_config(
    name: str = "PPO-No-Forecast",
    experiment_type: str = "ppo_baseline",  # ← Configurable type
    variant: str = "v1",
) -> ExperimentConfig:
```

---

## How It Works

### Automatic Organization

Just set the seed when creating a config:

```python
# Seed 10
config = ExperimentConfig(
    experiment_name="PPO-Baseline",
    seed=10,  # ← Sets seed
    wandb_experiment_type="ppo_baseline",
    wandb_variant="v1",
)

# Automatically generates:
# wandb_group: "2026-03-12/ppo_baseline/v1/seed_10"
# wandb_run_name: "2026-03-12_PPO-Baseline_with_risk_10"
# wandb_tags: ["2026-03-12", "ppo_baseline", "forecast_none", "reward_with_risk", "version_v1", "seed_10"]
```

---

## WandB Structure with Seeds [10, 20, 30]

```
2026-03-12/
├── ppo_baseline/v1/
│   ├── seed_10 ← All seed_10 runs grouped here
│   │   ├── run_1
│   │   ├── run_2
│   │   └── run_3
│   ├── seed_20 ← All seed_20 runs grouped here
│   │   ├── run_1
│   │   ├── run_2
│   │   └── run_3
│   └── seed_30 ← All seed_30 runs grouped here
│       ├── run_1
│       ├── run_2
│       └── run_3
├── reward_ablation/v1/
│   ├── seed_10 (8 runs for different reward types)
│   ├── seed_20
│   └── seed_30
└── ppo_with_forecast/v1/
    ├── seed_10
    ├── seed_20
    └── seed_30

2026-03-13/
└── (same structure repeats for each day)
```

---

## Comparison in WandB

### Filter All Three Seeds
```
tags:ppo_baseline AND tags:v1 AND (tags:seed_10 OR tags:seed_20 OR tags:seed_30)
```

Then:
1. Create metric chart (e.g., `performance/sharpe_ratio`)
2. View three lines (one per seed)
3. Compare robustness across seeds

### Find Best Seed
```
tags:ppo_baseline AND tags:v1
```

Sort by `performance/sharpe_ratio` (descending)

Result: Identify which seed [10, 20, 30] performs best

### Check Reproducibility
```
tags:ppo_baseline AND tags:v1
```

Create chart showing all seeds:
- **High agreement** between seed_10, seed_20, seed_30 = **reproducible algorithm**
- **Big differences** = **sensitive to seed**
- Use this to assess robustness

---

## Example Workflow

### 1. Run All Three Seeds

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
    results = runner.run()
    # Each run goes to:
    # 2026-03-12/ppo_baseline/v1/seed_10
    # 2026-03-12/ppo_baseline/v1/seed_20
    # 2026-03-12/ppo_baseline/v1/seed_30
```

### 2. Check WandB

1. Open WandB project
2. Look at `2026-03-12/ppo_baseline/v1/`
3. See three groups: `seed_10`, `seed_20`, `seed_30`
4. Each group contains runs for that seed

### 3. Compare Results

```
Filter: tags:ppo_baseline AND tags:v1
Chart:  performance/sharpe_ratio
Lines:  seed_10, seed_20, seed_30
```

See how the algorithm performs across different seeds!

---

## Key Features

✅ **Automatic Date Inclusion**
- No manual updates needed
- Date changes automatically each day
- Perfect for daily experiments

✅ **Seed-Based Grouping**
- Each seed [10, 20, 30] gets its own group
- Easy to compare seeds side-by-side
- Reproducibility tracking

✅ **Hierarchical Organization**
- Date → Experiment Type → Version → Seed
- Clean tree structure in WandB UI
- Easy to navigate and understand

✅ **Tag-Based Filtering**
- Filter by date: `tags:2026-03-12`
- Filter by seed: `tags:seed_10`
- Filter by reward: `tags:reward_with_risk`
- Combine: `tags:seed_10 AND tags:ppo_baseline`

✅ **Fully Automatic**
- Just set `seed=10/20/30` when creating config
- Everything else is generated automatically
- No additional code needed

---

## Files Updated

| File | Changes |
|------|---------|
| `trading_config.py` | Updated `get_wandb_group()` to include seed |
| `trading_config.py` | Updated `wandb_group` property to pass seed |
| `trading_config.py` | Updated all preset functions |
| `WANDB_ORGANIZATION_STRATEGY.md` | Updated to show seed-based grouping |
| `WANDB_SEED_ORGANIZATION_GUIDE.md` | NEW: Comprehensive seed guide |

---

## Documentation

### Main Guide
- **WANDB_ORGANIZATION_STRATEGY.md** - Complete strategy with all details

### Quick Reference
- **WANDB_SEED_ORGANIZATION_GUIDE.md** - Quick start guide with examples

---

## Summary

### Before
```
wandb_group: "baseline"  ← Same for all runs
```

### After
```
wandb_group: "2026-03-12/ppo_baseline/v1/seed_10"
             2026-03-12/ppo_baseline/v1/seed_20
             2026-03-12/ppo_baseline/v1/seed_30
```

Much better organization with **date**, **experiment type**, **version**, AND **seed**! 🎯

---

## Quick Start

### Run Experiment with Seed 10
```python
config = ExperimentConfig(
    experiment_name="PPO-Baseline",
    forecast_mode=ForecastMode.NONE,
    reward_type=RewardType.WITH_RISK,
    wandb_experiment_type="ppo_baseline",
    wandb_variant="v1",
    seed=10,  # ← Seed here
)

runner = ExperimentRunner(config)
runner.run()

# Automatically goes to:
# "2026-03-12/ppo_baseline/v1/seed_10"
```

### Run Experiment with Seed 20
```python
config = ExperimentConfig(
    # ... same config ...
    seed=20,  # ← Only change this
)

# Automatically goes to:
# "2026-03-12/ppo_baseline/v1/seed_20"
```

### Run All Seeds in Loop
```python
for seed in [10, 20, 30]:
    config = ExperimentConfig(
        # ... config ...
        seed=seed,  # ← Varies
    )
    ExperimentRunner(config).run()

# Creates:
# 2026-03-12/ppo_baseline/v1/seed_10
# 2026-03-12/ppo_baseline/v1/seed_20
# 2026-03-12/ppo_baseline/v1/seed_30
```

---

## Status: ✅ READY TO USE

All changes are complete and ready for production use.

Just set your seed and enjoy the organized WandB experience! 🎉


