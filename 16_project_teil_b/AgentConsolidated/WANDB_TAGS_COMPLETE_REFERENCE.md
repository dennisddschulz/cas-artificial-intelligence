# ✅ WandB Tags - Complete Implementation Reference

## Status: ✅ IMPLEMENTATION COMPLETE

All explicit date and seed tags have been successfully added to the WandB tagging system.

---

## Implementation Summary

### What Was Added

**Explicit Date Tag:**
```python
f"date_{today}"  # Format: date_2026-03-12
```

**Generic Seed Tag:**
```python
"seed"  # Marks all seed-based runs
```

### Location
**File:** `trading_config.py`
**Function:** `get_wandb_tags()`
**Lines:** 57-105

---

## Tags Generated

### Example for Seed 10

```python
config = ExperimentConfig(
    experiment_name="PPO-Baseline",
    seed=10,
    wandb_experiment_type="ppo_baseline",
    wandb_variant="v1",
)

# Generates these tags:
[
    "date_2026-03-12",           # ✅ NEW: Explicit date
    "2026-03-12",                # Backward compatible
    "ppo_baseline",              # Experiment type
    "forecast_none",             # Forecast mode
    "reward_with_risk",          # Reward type
    "version_v1",                # Version
    "seed_10",                   # Specific seed
    "seed"                       # ✅ NEW: Generic seed marker
]
```

---

## Filtering Capabilities

### Date-Based Filtering

```
Find all experiments from today:
  tags:date_2026-03-12

Find all experiments from specific date:
  tags:date_2026-03-11

Find all experiments from multiple dates:
  tags:date_2026-03-10 OR tags:date_2026-03-11 OR tags:date_2026-03-12
```

### Seed-Based Filtering

```
Find all seed 10 experiments:
  tags:seed_10

Find all seed 20 experiments:
  tags:seed_20

Find ALL seed-based (reproducibility) experiments:
  tags:seed

Find all seed-based baseline experiments:
  tags:seed AND tags:ppo_baseline
```

### Combined Filtering

```
Find all seed-based baseline experiments from today:
  tags:date_2026-03-12 AND tags:ppo_baseline AND tags:seed

Find seed 10 baseline experiments from today:
  tags:date_2026-03-12 AND tags:seed_10 AND tags:ppo_baseline

Find best seed from today:
  tags:date_2026-03-12 AND tags:ppo_baseline AND tags:seed
  Sort by: performance/sharpe_ratio (descending)

Find seed 10 performance progression:
  tags:seed_10 AND tags:ppo_baseline
  Compare across dates

Check reproducibility across seeds:
  tags:date_2026-03-12 AND tags:ppo_baseline AND tags:seed
  Create metric chart showing seed_10, seed_20, seed_30
```

---

## Tag Categories

### Date Tags
| Tag Format | Example | Purpose |
|-----------|---------|---------|
| `date_YYYY-MM-DD` | `date_2026-03-12` | Explicit date filtering (recommended) |
| `YYYY-MM-DD` | `2026-03-12` | Simple date (backward compatible) |

### Experiment Type Tags
| Tag | Example | Purpose |
|-----|---------|---------|
| Experiment Type | `ppo_baseline` | Find by experiment type |
| | `reward_ablation` | |
| | `ppo_with_forecast` | |

### Configuration Tags
| Tag Format | Example | Purpose |
|-----------|---------|---------|
| `forecast_{mode}` | `forecast_lstm` | Find by forecast mode |
| | `forecast_none` | |
| `reward_{type}` | `reward_with_risk` | Find by reward type |
| `version_{v}` | `version_v1` | Find by version |

### Seed Tags
| Tag Format | Example | Purpose |
|-----------|---------|---------|
| `seed_{n}` | `seed_10` | Find specific seed (10, 20, 30) |
| `seed` | `seed` | Mark all seed-based runs |

---

## Use Cases and Queries

### Use Case 1: Compare Three Seeds from Today
```
Query: tags:date_2026-03-12 AND tags:ppo_baseline AND (tags:seed_10 OR tags:seed_20 OR tags:seed_30)
Chart: performance/sharpe_ratio
View: Three lines (one per seed)
Purpose: Compare algorithm robustness across seeds
```

### Use Case 2: Find Best Seed
```
Query: tags:date_2026-03-12 AND tags:ppo_baseline AND tags:seed
Sort by: performance/sharpe_ratio (descending)
Purpose: Identify which seed performs best
```

### Use Case 3: Check Reproducibility
```
Query: tags:ppo_baseline AND tags:seed
Chart: performance/sharpe_ratio
View: All dates showing seed_10, seed_20, seed_30
Purpose: Check if results consistent across seeds
```

### Use Case 4: Track Seed 10 Progress
```
Query: tags:seed_10 AND tags:ppo_baseline
Filter: Group by date
Purpose: See how seed 10 performs over time
```

### Use Case 5: Find All Today's Experiments
```
Query: tags:date_2026-03-12
Purpose: Quick overview of all today's runs
```

### Use Case 6: Find All Seed-Based Experiments
```
Query: tags:seed
Purpose: Find all reproducibility runs
```

### Use Case 7: Debug Specific Seed Issue
```
Query: tags:seed_30 AND tags:date_2026-03-12
Purpose: Investigate problems with seed 30 today
```

---

## Tag Generation Automatic

All tags are **automatically generated** based on:

| Parameter | Tag Generated | Example |
|-----------|---------------|---------|
| `seed=10` | `seed_10`, `seed` | Both always included |
| Today's date | `date_2026-03-12`, `2026-03-12` | Both always included |
| `experiment_type="ppo_baseline"` | `ppo_baseline` | Automatically included |
| `forecast_mode=ForecastMode.NONE` | `forecast_none` | Automatically included |
| `reward_type=RewardType.WITH_RISK` | `reward_with_risk` | Automatically included |
| `variant="v1"` | `version_v1` | Automatically included |

---

## Backward Compatibility

### Old Filters Still Work
```
Old: tags:2026-03-12          ✅ Still works
New: tags:date_2026-03-12     ✅ Also works (recommended)

Old: tags:seed_10             ✅ Still works
New: tags:seed_10 AND tags:seed   ✅ Also possible
```

### Migration Path
1. New experiments use new explicit tags
2. Old experiments keep their original tags
3. Can use both old and new filters
4. Gradual adoption is possible

---

## Tag Structure for Today (2026-03-12)

### PPO Baseline with Seeds
```
Seed 10:
  ["date_2026-03-12", "2026-03-12", "ppo_baseline", "forecast_none", "reward_with_risk", "version_v1", "seed_10", "seed"]

Seed 20:
  ["date_2026-03-12", "2026-03-12", "ppo_baseline", "forecast_none", "reward_with_risk", "version_v1", "seed_20", "seed"]

Seed 30:
  ["date_2026-03-12", "2026-03-12", "ppo_baseline", "forecast_none", "reward_with_risk", "version_v1", "seed_30", "seed"]
```

### Reward Ablation with Seeds
```
For each reward type (basic, with_risk, with_sharpe, etc.):
  Seed 10: ["date_2026-03-12", "2026-03-12", "reward_ablation", "forecast_none", "reward_{type}", "version_v1", "seed_10", "seed"]
  Seed 20: ["date_2026-03-12", "2026-03-12", "reward_ablation", "forecast_none", "reward_{type}", "version_v1", "seed_20", "seed"]
  Seed 30: ["date_2026-03-12", "2026-03-12", "reward_ablation", "forecast_none", "reward_{type}", "version_v1", "seed_30", "seed"]
```

---

## Quick Reference

### Finding Experiments

| Task | Filter Query |
|------|--------------|
| All from today | `tags:date_2026-03-12` |
| All seed 10 | `tags:seed_10` |
| All seed-based | `tags:seed` |
| Today's seed-based | `tags:date_2026-03-12 AND tags:seed` |
| Today's baseline | `tags:date_2026-03-12 AND tags:ppo_baseline` |
| Today's baseline seeds | `tags:date_2026-03-12 AND tags:ppo_baseline AND tags:seed` |
| Seed 10 baseline | `tags:seed_10 AND tags:ppo_baseline` |
| Reward ablation seeds | `tags:reward_ablation AND tags:seed` |
| Today's LSTM forecast | `tags:date_2026-03-12 AND tags:forecast_lstm` |

---

## Documentation

### New Guide
📄 **WANDB_EXPLICIT_TAGS_UPDATE.md**
- Complete guide to new explicit tags
- Examples and use cases
- Filtering cheat sheet

### Related Guides
📄 **WANDB_SEED_ORGANIZATION_GUIDE.md**
- Seed-based organization details

📄 **WANDB_ORGANIZATION_STRATEGY.md**
- Complete strategy (updated with new tags)

---

## Summary

### What Changed
✅ Added explicit date tags: `date_YYYY-MM-DD`
✅ Added generic seed tag: `seed`
✅ Maintained backward compatibility

### Benefits
✅ Clearer tag naming and intent
✅ Better filtering capabilities
✅ Easier to find seed-based experiments
✅ More professional organization

### Ready to Use
✅ Automatic tag generation
✅ No additional code needed
✅ Just set your seed - tags handle themselves!

---

## Example: Complete Workflow

```python
# 1. Create configs for all three seeds
seeds = [10, 20, 30]
configs = []

for seed in seeds:
    config = ExperimentConfig(
        experiment_name="PPO-Baseline",
        seed=seed,
        wandb_experiment_type="ppo_baseline",
        wandb_variant="v1",
    )
    configs.append(config)

# 2. Run all experiments
for config in configs:
    runner = ExperimentRunner(config)
    runner.run()

# 3. In WandB, filter to compare
# Filter: tags:date_2026-03-12 AND tags:ppo_baseline AND tags:seed
# Result: All three seeds grouped and easy to compare!

# 4. Create comparison chart
# Metric: performance/sharpe_ratio
# View: Three lines (seed_10, seed_20, seed_30)
# Analyze: Which seed is most robust?
```

---

## Status: ✅ COMPLETE

All explicit date and seed tags are fully implemented and ready to use.

Just set your seed and enjoy organized, filterable WandB experiments! 🎯


