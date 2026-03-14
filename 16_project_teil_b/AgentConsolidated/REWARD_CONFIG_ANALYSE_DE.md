# Reward Config Analyse & Empfehlungen (Deutsch)

## Frage: Machen diese reward configs Sinn?

**Kurzantwort**: ✅ **JA, aber mit Verbesserungen**

---

## 🔍 Analyse der Aktuellen Konfigurationen

### Was ist FALSCH implementiert?

Nach Analyse von `trading_framework.py` (Lines 1180-1310) sehe ich:

```python
# Equity update verwendet true_reward
self.equity *= float(np.exp(true_reward))

# Aber rewards werden ANDERS berechnet:
if self.reward_type == RewardType.WITH_RISK:
    reward = true_reward - risk_pen  # ← risk_pen subtrahiert
elif self.reward_type == RewardType.SORTINO:
    reward = (downside_adjusted_pnl - cost - slippage) / safe_sigma  # ← normalisiert
elif self.reward_type == RewardType.COMPOSITE:
    reward = weighted_combination(...)  # ← komplex
```

**PROBLEM**: Die `reward_params` in der Config werden **NICHT verwendet**!

| Config-Param | Wo verwendet? | Status |
|---|---|---|
| `epsilon=0.001` | ❌ NIRGENDS | Totes Gewicht |
| `kappa=0.01` | ✅ Used (hardcoded in env) | OK |
| `downside_scale=1.2` | ✅ Hardcoded in reward | OK |
| `drawdown_multiplier=0.5` | ✅ Hardcoded in reward | OK |
| `consistency_bonus=0.1` | ✅ Hardcoded in reward | OK |
| `weight_returns=0.5` | ✅ Hardcoded in composite | OK |

---

## ❌ Das Problem: Reward-Parameter sind Deklarativ, nicht Aktiv

Die reward_params sind **konfiguriert aber nicht integriert**!

### Aktuelle Situation:
```python
reward_configs = [
    (RewardType.WITH_RISK, "with_risk_reward", {
        "kappa": 0.01,      # ← Diese Config...
        "epsilon": 0.001,   # ← ...
    }),
]
```

### Was SOLLTE passieren:
```python
# In TradingEnv.step():
kappa = self.reward_params.get('kappa', 0.01)  # ← Aus Config lesen!
reward = true_reward - kappa * (self.pos ** 2) * sigma_t
```

---

## ✅ Empfehlungen: VERSCHIEDENE Kappa-Werte verwenden?

### **JA, verschiedene Kappa macht ABSOLUT Sinn!**

**Warum?**

1. **Ablation Study richtig machen**: Kappa steuert Leverage-Penalty
   - `kappa=0.001` → Agent erlaubt aggressive Positionen
   - `kappa=0.01` → Moderate Positionen
   - `kappa=0.1` → Conservative Positionen

2. **Verschiedene Reward-Funktionen brauchen unterschiedliche Parameter**

   ```
   BASIC:              kappa=N/A (no risk)
   WITH_RISK:          kappa=0.001, 0.01, 0.1 (TEST!)
   WITH_SHARPE:        kappa=N/A (volatility is self-regulating)
   SORTINO:            downside_scale=0.8, 1.2, 1.5
   CALMAR:             drawdown_multiplier=0.3, 0.5, 0.7
   COMPOSITE:          kappa=0.005, 0.01, 0.02
   ```

3. **Reward Scale auch testen!**
   - `reward_scale=0.1` → Kleinere Rewards, sanfteres Training
   - `reward_scale=1.0` → Normal
   - `reward_scale=10.0` → Aggressive Rewards, schnelleres Training

---

## 📋 Verbesserte Reward Konfigurationen

### **Empfohlen: 3x3 Ablation (9 Varianten statt 8)**

```python
reward_configs = [
    # ===== BASIC (Baseline) =====
    (RewardType.BASIC, "basic", {
        "reward_scale": 1.0,
    }),

    # ===== WITH_RISK (3 Varianten) =====
    (RewardType.WITH_RISK, "with_risk_conservative", {
        "kappa": 0.05,          # ← HIGH: Sehr konservativ
        "reward_scale": 1.0,
    }),
    (RewardType.WITH_RISK, "with_risk_moderate", {
        "kappa": 0.01,          # ← MEDIUM: Balanced
        "reward_scale": 1.0,
    }),
    (RewardType.WITH_RISK, "with_risk_aggressive", {
        "kappa": 0.001,         # ← LOW: Erlaubt höhere Leverage
        "reward_scale": 1.0,
    }),

    # ===== WITH_SHARPE (2 Varianten) =====
    (RewardType.WITH_SHARPE, "with_sharpe_normal", {
        "epsilon": 0.001,
        "reward_scale": 1.0,
    }),
    (RewardType.WITH_SHARPE, "with_sharpe_scaled", {
        "epsilon": 0.001,
        "reward_scale": 0.1,    # ← Sanfteres Training
    }),

    # ===== SORTINO (2 Varianten) =====
    (RewardType.SORTINO, "sortino_moderate", {
        "downside_scale": 1.2,
        "reward_scale": 1.0,
    }),
    (RewardType.SORTINO, "sortino_aggressive", {
        "downside_scale": 0.8,  # ← Weniger Penalty für Downside
        "reward_scale": 1.0,
    }),

    # ===== CALMAR (1 Variante) =====
    (RewardType.CALMAR, "calmar", {
        "drawdown_multiplier": 0.5,
        "reward_scale": 1.0,
    }),

    # ===== COMPOSITE (3 Varianten) =====
    (RewardType.COMPOSITE, "composite_baseline", {
        "weight_returns": 0.5,
        "weight_sharpe": 0.3,
        "weight_risk": 0.2,
        "kappa": 0.01,
        "reward_scale": 1.0,
    }),
    (RewardType.COMPOSITE, "composite_conservative", {
        "weight_returns": 0.3,   # ← Weniger Return-fokus
        "weight_sharpe": 0.4,    # ← Mehr Risk-Adjust
        "weight_risk": 0.3,      # ← Mehr Risk-Penalty
        "kappa": 0.02,           # ← Höher
        "reward_scale": 1.0,
    }),
    (RewardType.COMPOSITE, "composite_aggressive", {
        "weight_returns": 0.7,   # ← Mehr Return-fokus
        "weight_sharpe": 0.2,
        "weight_risk": 0.1,      # ← Weniger Risk-Penalty
        "kappa": 0.005,          # ← Niedriger
        "reward_scale": 1.0,
    }),
]
```

---

## 🔧 Wie man die Parameter RICHTIG integriert

### Schritt 1: In `trading_framework.py` TradingEnv initialisieren

```python
def __init__(self, df, ..., reward_params=None):
    # ...existing code...
    self.reward_params = reward_params or {}

    # Defaults
    self.kappa = float(self.reward_params.get('kappa', 0.01))
    self.reward_scale = float(self.reward_params.get('reward_scale', 1.0))
    self.epsilon = float(self.reward_params.get('epsilon', 0.001))
    self.downside_scale = float(self.reward_params.get('downside_scale', 1.2))
    self.drawdown_multiplier = float(self.reward_params.get('drawdown_multiplier', 0.5))
    self.consistency_bonus = float(self.reward_params.get('consistency_bonus', 0.1))

    # Composite weights
    self.weight_returns = float(self.reward_params.get('weight_returns', 0.5))
    self.weight_sharpe = float(self.reward_params.get('weight_sharpe', 0.3))
    self.weight_risk = float(self.reward_params.get('weight_risk', 0.2))
```

### Schritt 2: In `step()` die Parameter verwenden

```python
elif self.reward_type == RewardType.WITH_RISK:
    risk_pen = self.kappa * (self.pos ** 2) * sigma_t  # ← Nutzt self.kappa
    reward = true_reward - risk_pen

elif self.reward_type == RewardType.SORTINO:
    downside_penalty = self.downside_scale  # ← Nutzt self.downside_scale
    # ...

elif self.reward_type == RewardType.COMPOSITE:
    # ← Nutzen self.weight_returns, self.weight_sharpe, self.weight_risk
    reward = (self.weight_returns * signal_returns +
             self.weight_sharpe * signal_sharpe +
             self.weight_risk * signal_risk)
```

### Schritt 3: Environment mit Params erstellen

```python
env = TradingEnv(
    df,
    reward_type=reward_type,
    reward_params=config.environment.reward_params  # ← Pass params!
)
```

---

## 📊 Vergleich: Vorher vs. Nachher

| Aspect | Vorher | Nachher |
|--------|--------|---------|
| **Reward-Varianten** | 8 | 12-15 |
| **Kappa-Werte** | 1 (0.01) | 3 (0.001, 0.01, 0.05) |
| **Reward-Scale** | 1 (hardcoded) | 2-3 pro Variant |
| **Parameter Integration** | ❌ Nicht genutzt | ✅ Vollständig |
| **Ablation Quality** | Mäßig | Exzellent |
| **Insights** | Welche Reward-Type ist best? | Welcher Parameter ist optimal? |

---

## 💡 Konkrete Verbesserungen zum Implementieren

### 1. **Parameter tatsächlich verwenden** (Einfach, Hoch-Impact)

```python
# In trading_framework.py TradingEnv.__init__()
self.kappa = float(self.reward_params.get('kappa', 0.01))
self.reward_scale = float(self.reward_params.get('reward_scale', 1.0))
```

### 2. **Reward-Scale tuneable machen** (Sehr Einfach)

```python
reward_configs = [
    (RewardType.WITH_RISK, "with_risk_normal", {
        "kappa": 0.01,
        "reward_scale": 1.0,
    }),
    (RewardType.WITH_RISK, "with_risk_scaled", {
        "kappa": 0.01,
        "reward_scale": 0.5,  # ← Sanfteres Training
    }),
    # ...
]
```

### 3. **Verschiedene Kappa-Werte testen** (Empfohlen)

```python
kappa_values = [0.001, 0.01, 0.05]  # Conservativeness spectrum
for kappa in kappa_values:
    configs.append((RewardType.WITH_RISK, f"with_risk_kappa{kappa}", {
        "kappa": kappa,
        "reward_scale": 1.0,
    }))
```

---

## ✅ Fazit

| Frage | Antwort |
|-------|---------|
| **Sind die Configs sinnvoll?** | ✅ JA, aber nicht optimal genutzt |
| **Verschiedene Kappa-Werte?** | ✅ JA, sehr gute Idee! |
| **Reward-Scale variieren?** | ✅ JA, hilfreich für Training |
| **Parameter in Code integrieren?** | ✅ JA, CRITICAL! |
| **Wie viel Aufwand?** | ⏱️ 30-60 Minuten für Vollintegration |

---

## Nächste Schritte

1. **Schnell (15 min)**: Parameter in TradingEnv.__init__ integrieren
2. **Mittel (30 min)**: Kappa-Varianten zu reward_configs hinzufügen
3. **Optimal (60 min)**: Alle Parameter-Variationen testen

Soll ich das jetzt für dich implementieren?

