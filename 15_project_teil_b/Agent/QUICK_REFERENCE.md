# Quick Reference - Optimierte Parameter

## 🔧 Hyperparameter

| Parameter | Alt | Neu | Grund |
|-----------|-----|-----|-------|
| **FEE** | 0.0005 | 0.0001 | ↓ Transaktionskosten für mehr Rewards |
| **KAPPA** | 0.1 | 0.01 | ↓ Risk Penalty für größere Positions |
| **LEVERAGE_MAX** | 1.0 | 1.0 | — (unverändert) |
| **lr** | 3e-4 | 1e-4 | ↓ Stabilere Gradient Updates |
| **ent_coef** | 0.001 | 0.01 | ↑ Mehr Exploration |
| **max_grad_norm** | 0.5 | 1.0 | ↑ Größere Gradienten erlaubt |
| **num_envs** | 16 | 8 | ↓ Stabilere Sampler |
| **n_steps** | 128 | 256 | ↑ Längere Trajektorien |
| **ppo_epochs** | 10 | 20 | ↑ Mehr Policy Updates |
| **minibatch_size** | 64 | 32 | ↓ Kleinere, häufigere Updates |
| **target_kl** | 0.1 | 0.05 | ↓ Engeres KL-Control |

---

## 🧠 Network-Architektur

### ALT:
```
Input (obs_dim)
  ↓
Linear(obs_dim, 128)
  ↓
Tanh
  ↓
Linear(128, 128)
  ↓
Tanh
  ↓
[Actor Head] [Critic Head]
```

### NEU:
```
Input (obs_dim)
  ↓
Linear(obs_dim, 256) → LayerNorm → Tanh
  ↓
Linear(256, 256) → LayerNorm → Tanh
  ↓
Linear(256, 128) → Tanh
  ↓
[Actor Head] [Critic Head]

✓ LayerNorm für numerische Stabilität
✓ Größere Layers für komplexere Muster
```

---

## 🎁 Reward-Shaping

### Vorher:
```python
reward = pnl - transaction_cost - risk_penalty
```

### Nachher:
```python
base_reward = pnl - transaction_cost - risk_penalty
position_alignment = clip(a * mu_hat, -0.01, 0.01)
reward = base_reward + position_alignment * 0.1
reward = clip(reward, -0.1, 0.1)  # ← CRITICAL
```

**Auswirkungen:**
- ✓ Rewards sind bounded und stabil
- ✓ Alignment mit Marktausblick wird incentiviert
- ✓ Weniger Training Noise

---

## 📊 State Space (14 Features)

### Market (8):
```
[r, r_lag1, mu_hat, sigma_hat, rsi, macd_diff, bb_width, ema_ratio]
                  ↓
         Clipped zu [-3, 3]
```

### Portfolio (6):
```
[
  tanh(pos/0.5),                    # current_position
  clip(cash/equity, 0, 1),          # cash_ratio
  min(|pos|/1.0, 1.0),              # leverage
  clip(drawdown, 0, 1),             # drawdown
  tanh((equity-init)/init),         # cumulative_pnl
  tanh((equity-last_equity)/last)   # recent_return
]
```

**Alle Features sind bounded!** ✓

---

## 🎯 Action Space

```
a_t ∈ [-1.0, 1.0]

Interpretation:
├─ a = +0.5  →  50% Long (invest 50% of capital)
├─ a = -0.5  →  50% Short (short 50% of capital)
├─ a = +1.0  →  100% Long (full leverage)
└─ a = -1.0  →  100% Short (full leverage short)
```

---

## 💰 Reward Komponenten (Diagnostics)

Das Notebook misst automatisch:

1. **PnL**: `pos_{t-1} * r_t` (Return von Position)
2. **Cost**: `fee * |a_t - pos_{t-1}|` (Transaktionskosten)
3. **Risk Penalty**: `kappa * a_t^2 * sigma_t` (Volatilität Strafe)
4. **Alignment**: `a * mu_hat` (Positionen mit Momentum)
5. **Final Reward**: `clip(base + alignment_bonus, -0.1, 0.1)`

Beispielwerte während Training:
```
PnL:          [0.000001, 0.000050]  ← sehr klein!
Cost:         [0.000001, 0.000005]  ← sehr klein!
Penalty:      [0.000001, 0.000010]  ← sehr klein!
Alignment:    [-0.010, +0.010]       ← bonus range
Final Reward: [-0.1, +0.1]           ← normalized!
```

---

## 📈 Training Progression

### Phase 1: Exploration (Updates 0-500)
- Policy ist zufällig
- Returns sind hochvolatil: [-2, +2]
- Agent exploriert Action Space
- **Erwartung**: Sehr variable Episode Returns

### Phase 2: Early Learning (Updates 500-1500)
- Policy beginnt zu lernen
- Returns sollten stabilisieren: [-0.5, +0.5]
- Agent findet erste profitable Muster
- **Erwartung**: Reducing volatility, trending positive

### Phase 3: Convergence (Updates 1500-3000)
- Policy ist gelernt
- Returns sollten positiv sein: [0.0, +0.5]
- Konsistente Handelsregel
- **Erwartung**: Stable positive returns

---

## 🔍 Monitoring Checklist

Während Training beobachten:
- [ ] Returns trending upward (nicht fallend!)
- [ ] log_std zwischen -2.0 und -0.5
- [ ] Avg|Adv| > 0 (positive learning signals)
- [ ] Keine KL divergence blow-ups

Nach Training prüfen:
- [ ] Mean Episode Return > 0
- [ ] Equity Curve aufwärts trending
- [ ] Position Stats sane (nicht extreme swings)
- [ ] Win Rate > 50%

---

## ⚡ Performance Tuning

### Wenn Returns zu negativ sind:
```python
KAPPA = 0.001      # vs 0.01
FEE = 0.00005      # vs 0.0001
ent_coef = 0.02    # vs 0.01 (mehr exploration)
```

### Wenn Training zu instabil ist:
```python
lr = 5e-5          # vs 1e-4 (slower learning)
minibatch_size = 64 # vs 32 (larger batches)
ent_coef = 0.005    # vs 0.01 (less exploration)
```

### Wenn Agent nicht handelt:
```python
self.log_std = torch.ones(act_dim) * -0.3  # vs -0.5
ent_coef = 0.05    # vs 0.01 (force exploration)
```

---

## 📊 Key Metrics nach Training

```python
# Should see:
Mean return (last 100):     > 0.0  ✓
Total episodes:              > 200  ✓
Win rate:                    > 50%  ✓
Sharpe ratio:                > 0.5  ✓
Max drawdown:                < 10%  ✓
```

---

## 🎬 Execution Checklist

1. ✅ Notebook öffnen
2. ✅ Alle Cells von oben nach unten ausführen
3. ✅ Nach Update 100: ersten diagnostics Output überprüfen
4. ✅ Nach Update 1000: Trends überprüfen
5. ✅ Nach Training fertig: finalen Report überprüfen
6. ✅ Grafiken speichern/analysieren
7. ✅ Ergebnisse dokumentieren

---

## 💡 Debugging Tips

```python
# Problem: Zu viele NaN/Inf Werte
→ Überprüfen Sie market features in _get_obs()
→ Check: market_features = np.clip(..., -3.0, 3.0)

# Problem: Policy zu konservativ
→ Reduzieren Sie log_std initiales Negativ-Level
→ Oder erhöhen Sie entropy coefficient

# Problem: Rewards immer ähnlich
→ Überprüfen Sie Reward Clipping in step()
→ Alignment Bonus könnte zu klein sein

# Problem: Value Function schlechte Predictions
→ Erhöhen Sie vf_coef
→ Oder verringern Sie policy_loss durch weniger clipping
```

---

**Viel Erfolg beim Training! 🚀**

Bei Fragen oder Problemen → `OPTIMIZATION_GUIDE.md` konsultieren

