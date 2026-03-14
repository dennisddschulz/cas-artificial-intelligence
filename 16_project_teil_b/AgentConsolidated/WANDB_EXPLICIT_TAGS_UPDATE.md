# ✅ WandB Tags Update - Explicit Date and Seed Tags

## What Changed

Added **explicit date and seed tags** to improve searchability and filtering in WandB.

---

## New Tags Structure

### Before
```python
tags = [
    "2026-03-12",                    # Date (implicit)
    "ppo_baseline",
    "forecast_none",
    "reward_with_risk",
    "version_v1",
    "seed_10",                       # Seed (one tag)
]
```

### After
```python
tags = [
    "date_2026-03-12",              # ✅ NEW: Explicit date tag
    "2026-03-12",                    # Still included for backward compatibility
    "ppo_baseline",
    "forecast_none",
    "reward_with_risk",
    "version_v1",
    "seed_10",                       # Still included
    "seed",                          # ✅ NEW: Generic seed tag (all seed-based runs)
]
```

---

## Tag Details

### Date Tags
- `date_2026-03-12` - Explicit date tag (recommended for filtering)
- `2026-03-12` - Simple date tag (kept for backward compatibility)

**Use cases:**
```
Filter by date: tags:date_2026-03-12
Find all today's experiments: tags:date_2026-03-12
Compare across dates: tags:date_2026-03-10 OR tags:date_2026-03-11 OR tags:date_2026-03-12
```

### Seed Tags
- `seed_10` - Specific seed (e.g., seed 10)
- `seed_20` - Specific seed (e.g., seed 20)
- `seed_30` - Specific seed (e.g., seed 30)
- `seed` - Generic tag (ALL seed-based runs)

**Use cases:**
```
Find seed 10 runs: tags:seed_10
Find seed 20 runs: tags:seed_20
Find seed 30 runs: tags:seed_30
Find ANY seed-based run: tags:seed
Find all seed-based ppo_baseline: tags:seed AND tags:ppo_baseline
```

---

## Enhanced Filtering Examples

### Example 1: Find All Experiments from Today
```
Filter: tags:date_2026-03-12
Result: All experiments from 2026-03-12
```

### Example 2: Compare All Seeds for Today's Baseline
```
Filter: tags:date_2026-03-12 AND tags:ppo_baseline AND tags:seed
Result: All seed-based baseline experiments from today
Breakdown: seed_10, seed_20, seed_30 all shown
```

### Example 3: Find Specific Seed Across All Days
```
Filter: tags:seed_10 AND tags:ppo_baseline
Result: All ppo_baseline experiments with seed 10 (across all dates)
Use case: Track seed 10 performance over time
```

### Example 4: Find Best Seed Today
```
Filter: tags:date_2026-03-12 AND tags:ppo_baseline AND tags:seed
Sort by: performance/sharpe_ratio (descending)
Result: Best seed [10, 20, 30] for today's baseline
```

### Example 5: Check Seed Reproducibility
```
Filter: tags:ppo_baseline AND tags:version_v1 AND tags:seed
Create chart: performance/sharpe_ratio
View: All three seed groups (seed_10, seed_20, seed_30)
Analysis: Are results consistent across seeds?
```

---

## Code Examples

### Run with Explicit Tags
```python
config = ExperimentConfig(
    experiment_name="PPO-Baseline",
    seed=10,
    wandb_experiment_type="ppo_baseline",
    wandb_variant="v1",
)

# Tags automatically generated:
# ["date_2026-03-12", "2026-03-12", "ppo_baseline",
#  "forecast_none", "reward_with_risk", "version_v1", "seed_10", "seed"]
```

### Run All Seeds
```python
for seed in [10, 20, 30]:
    config = ExperimentConfig(
        experiment_name="PPO-Baseline",
        seed=seed,
        wandb_experiment_type="ppo_baseline",
        wandb_variant="v1",
    )
    runner = ExperimentRunner(config)
    runner.run()

# Each gets appropriate tags:
# seed_10: [..., "seed_10", "seed"]
# seed_20: [..., "seed_20", "seed"]
# seed_30: [..., "seed_30", "seed"]
```

---

## Benefits of New Tags

✅ **Clearer Intent**
- `date_2026-03-12` is obviously a date tag
- `seed_10` is obviously a seed tag

✅ **Generic Seed Tag**
- `seed` tag marks all seed-based experiments
- Easy to find all reproducibility runs

✅ **Better Filtering**
- Can filter by explicit date: `tags:date_2026-03-12`
- Can filter all seed experiments: `tags:seed`

✅ **Backward Compatible**
- Simple date `2026-03-12` still included
- All existing filters still work

✅ **Professional Organization**
- More explicit and searchable
- Clearer tag hierarchy

---

## Tag Summary

| Tag | Format | Example | Purpose |
|-----|--------|---------|---------|
| Explicit Date | `date_YYYY-MM-DD` | `date_2026-03-12` | Find experiments from specific date |
| Simple Date | `YYYY-MM-DD` | `2026-03-12` | Backward compatibility |
| Experiment Type | `{type}` | `ppo_baseline` | Identify experiment type |
| Forecast Mode | `forecast_{mode}` | `forecast_lstm` | Filter by forecast type |
| Reward Type | `reward_{type}` | `reward_with_risk` | Filter by reward function |
| Version | `version_{v}` | `version_v1` | Track versions |
| Specific Seed | `seed_{n}` | `seed_10` | Find specific seed runs |
| Seed Marker | `seed` | `seed` | Mark all seed-based experiments |

---

## Complete Example

### Configuration
```python
config = ExperimentConfig(
    experiment_name="PPO-Baseline",
    forecast_mode=ForecastMode.NONE,
    reward_type=RewardType.WITH_RISK,
    wandb_experiment_type="ppo_baseline",
    wandb_variant="v1",
    seed=10,
)
```

### Auto-Generated WandB Info
```
group: "2026-03-12/ppo_baseline/v1/seed_10"
run_name: "2026-03-12_ppo_baseline_with_risk_10"
tags: [
    "date_2026-03-12",      ← Find by explicit date
    "2026-03-12",           ← Backward compatible
    "ppo_baseline",         ← Find by experiment type
    "forecast_none",        ← Find by forecast mode
    "reward_with_risk",     ← Find by reward type
    "version_v1",           ← Find by version
    "seed_10",              ← Find by specific seed
    "seed",                 ← Mark as seed-based
]
```

---

## Filtering Cheat Sheet

```
Find all today's runs:
  tags:date_2026-03-12

Find all seed_10 runs today:
  tags:date_2026-03-12 AND tags:seed_10

Find all baseline runs today:
  tags:date_2026-03-12 AND tags:ppo_baseline

Find all seed-based baseline runs today:
  tags:date_2026-03-12 AND tags:ppo_baseline AND tags:seed

Compare three seeds today:
  tags:date_2026-03-12 AND tags:ppo_baseline AND (tags:seed_10 OR tags:seed_20 OR tags:seed_30)

Find best seed:
  tags:ppo_baseline AND tags:seed
  Sort by: performance/sharpe_ratio

Track seed_10 progress:
  tags:seed_10 AND tags:ppo_baseline
  Compare across dates

Check reproducibility:
  tags:ppo_baseline AND tags:seed
  Create metric chart
  View all seed groups
```

---

## Summary

### What Was Added
✅ Explicit date tags: `date_2026-03-12`
✅ Generic seed tag: `seed`

### Benefits
✅ Clearer tag naming
✅ Better filtering options
✅ Easier to find seed-based experiments
✅ Improved searchability

### Backward Compatibility
✅ Simple date tag `2026-03-12` still included
✅ All existing filters still work
✅ No breaking changes

All tags are **automatically generated** - just set your seed and date is handled automatically! 🎯


