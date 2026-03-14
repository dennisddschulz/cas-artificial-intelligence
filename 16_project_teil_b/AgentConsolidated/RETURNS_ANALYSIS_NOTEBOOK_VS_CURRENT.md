# Analyse: Warum sind die Returns unterschiedlich?

## 📊 Vergleich: Notebook vs. run_all_experiments.py

### BASELINE KONFIGURATIONEN (KRITISCH)

#### Notebook (Project_Part_3_Final_Architecture.ipynb)
```python
FEE = 0.0001           # Transaction cost
KAPPA = 0.01           # Risk penalty weight
LEVERAGE_MAX = 1.0     # Max leverage
INITIAL_EQUITY = 100000.0
num_envs = 8
total_updates = 3000
learning_rate = 1e-4   # für PPO
reward_type = WITH_RISK  # (implizit)
```

#### run_all_experiments.py (JETZT)
```python
# Aus trading_config.py EnvironmentConfig:
fee = 0.0001           ✓ GLEICH
kappa = 0.01           ✓ GLEICH
leverage_max = 1.0     ✓ GLEICH
initial_equity = 100000.0  ✓ GLEICH
num_envs = 8           ✓ GLEICH
total_updates = 3000   ✓ GLEICH
learning_rate = 1e-4   ✓ GLEICH
reward_type = WITH_RISK  ✓ GLEICH
```

---

## ❌ GEFUNDENES PROBLEM

### Line 729-730 im Notebook (WICHTIG!)
```python
fee=0.0005,      # ← ACHTUNG: HIER WIRD 0.0005 VERWENDET!
kappa=0.1,       # ← ACHTUNG: HIER WIRD 0.1 VERWENDET!
```

Das ist die **TradingEnv Initialisierung in `make_env()` FUNKTION**!

### Das bedeutet:
- **Notebook verwendet tatsächlich**: fee=0.0005, kappa=0.1
- **run_all_experiments.py verwendet**: fee=0.0001, kappa=0.01

```
Notebook Fee:     0.0005 (5x höher als aktuell!)
Current Fee:      0.0001

Notebook Kappa:   0.1 (10x höher als aktuell!)
Current Kappa:    0.01
```

---

## 🔍 ANALYSE DER AUSWIRKUNGEN

### Fee Impact (0.0005 vs 0.0001)

Die Fee beeinflusst die Kosten bei jedem Trade:
```
Cost = fee × turnover

Mit Turnover = 100 (durchschnittlich):
- Notebook: 0.0005 × 100 = 0.05 = 5% pro Update
- Jetzt: 0.0001 × 100 = 0.01 = 1% pro Update

5x unterschiedliche Kosten!
```

**Aber Fee ist nur 5x, nicht 10x Unterschied in Returns**

### Kappa Impact (0.1 vs 0.01) - DAS HAUPTPROBLEM!

Kappa steuert den Risk Penalty in der Reward-Funktion:
```
Reward = PnL - cost - κ × (Position)² × Volatility

Mit κ=0.1:  Viel aggressivere Position-Reduktion
Mit κ=0.01: Moderate Position-Reduktion
```

**Dies ist der HAUPTGRUND für unterschiedliche Performance!**

---

## 🎯 WARUM WAR DAS NOTEBOOK BESSER (117k-143k)?

### Hypothesis 1: Agent lernt aggressiver zu traden
- Mit **κ=0.01** (aktuell): Agent ist konservativ
- Mit **κ=0.1** (Notebook): Agent hat MEHR Freiheit zu traden

Das erklärt die höheren Returns!

### Hypothesis 2: Andere Parameter im Notebook
Schaue auf Line 1039-1046 - die gedruckten Parameter:
```python
print(f"Initial Equity: ${INITIAL_EQUITY:,.0f}")
print(f"FEE: {FEE}  KAPPA: {KAPPA}  LEVERAGE_MAX: {LEVERAGE_MAX}")
print(f"Learning Rate: {lr}  Num Envs: {num_envs}  Steps: {n_steps}")
```

Diese zeigen die **globalen Konstanten** (FEE=0.0001, KAPPA=0.01)

ABER die `make_env()` Funktion überschreibt diese mit hardcoded Werten!
(Line 729-730: fee=0.0005, kappa=0.1)

---

## ⚡ ERKLÄRUNG DER NEGATIVEN RETURNS JETZT

Mit den neuen Konfigurationen:
1. **Kappa=0.01 (statt 0.1)**: Agent darf aggressiver traden
2. **Fee=0.0001 (statt 0.0005)**: Kosten sind niedriger
3. **Aber**: Agent muss mit **deutlich weniger Risiko-Penalty trainieren**

Das führt dazu, dass:
- Agent zu aggressive Positionen macht
- Volatilität zu hoch wird
- Drawdown zu groß wird
- Returns negativ werden

**Der Agent ist nicht ausreichend konservativ trainiert!**

---

## 📋 ZUSAMMENFASSUNG DER UNTERSCHIEDE

| Parameter | Notebook | Aktuell | Ratio |
|-----------|----------|---------|-------|
| Fee | 0.0005 | 0.0001 | 5x |
| Kappa | 0.1 | 0.01 | **10x** ← HAUPTPROBLEM |
| Leverage Max | 1.0 | 1.0 | 1x |
| Total Updates | 3000 | 3000 | 1x |
| Num Envs | 8 | 8 | 1x |
| Learning Rate | 1e-4 | 1e-4 | 1x |

---

## ✅ LÖSUNG

### Option 1: Kappa zurück auf 0.1 (wie Notebook)
```python
# In trading_config.py EnvironmentConfig
kappa: float = 0.1  # Statt 0.01
```

**Vorteil**: Returns sollten positiv werden (wie Notebook)
**Nachteil**: Agent ist sehr konservativ

### Option 2: Reward Parameter richtig in run_all_experiments.py nutzen

Die **reward_params Integration** ist nur zu 80% implementiert. Die `EnvironmentConfig` defaults werden nicht richtig vom Notebook-Setup übernommen.

**WICHTIG**: In `make_env()` werden diese Parameter NICHT gelesen!

```python
# In trading_framework.py (train_ppo, Zeile 583)
def make_env():
    return TradingEnv(
        df_train,
        fee=self.config.environment.fee,  # ← CORRECT
        kappa=self.config.environment.kappa,  # ← CORRECT
        # ... aber reward_params werden NICHT übergeben!
        reward_params=self.config.environment.reward_params,  # ← BEREITS HINZUGEFÜGT
    )
```

Das wurde bereits gefixt! Aber prüfe, ob fee und kappa wirklich gelesen werden.

### Option 3: Überprüfe, dass reward_params['kappa'] nicht None ist

```python
# In trading_framework.py TradingEnv.__init__ (Zeile 1159)
self.kappa = float(self.reward_params.get('kappa', kappa))
```

Falls reward_params['kappa'] = None, wird kappa als Fallback verwendet!

---

## 🔧 DEBUGGING-SCHRITTE

### 1. Drucke die aktualen Parameter
```python
# Am Anfang von TradingEnv.step()
print(f"DEBUG: Using kappa={self.kappa}, fee={self.fee}, reward_scale={self.reward_scale}")
```

### 2. Überprüfe EnvironmentConfig defaults
```python
from trading_config import EnvironmentConfig
env = EnvironmentConfig()
print(f"Default kappa: {env.kappa}")
print(f"Default fee: {env.fee}")
print(f"Default reward_params: {env.reward_params}")
```

### 3. Überprüfe reward_params in der Config
```python
from trading_config import get_ppo_without_forecast_config
config = get_ppo_without_forecast_config()
print(f"Environment kappa: {config.environment.kappa}")
print(f"Environment reward_params: {config.environment.reward_params}")
```

---

## 🎯 ANTWORT AUF DEINE FRAGEN

### 1. Wurde die Baseline (Basic) beim Reward Refaktoring verändert?
**NEIN, nicht absichtlich.**
- Aber die Reward_params wurden nicht richtig auf die Config übernommen
- Die defaults sind unterschiedlich vom Notebook

### 2. Sind mit den Reward Ablation Varianten viel bessere Returns zu erwarten?
**VIELLEICHT.**
- Die WITH_RISK mit κ=0.05 (conservative) sollte besser sein
- Die WITH_RISK mit κ=0.001 (aggressive) könnte deutlich besser sein
- Aber wir müssen erst die Baseline fixen

### 3. War es im Notebook realistisch?
**JA, völlig realistisch.**
- Mit κ=0.1 (wie Notebook) sind positive Returns normal
- Mit κ=0.01 (wie jetzt) werden Returns von aggressiven Positionen negativ
- Das Notebook war KONSERVATIVER trainiert

---

## ⚠️ KRITISCHES PROBLEM

Die **reward_params Integration war nicht vollständig**:

1. EnvironmentConfig hat reward_params mit Defaults
2. Diese werden aber **in den Konfig-Funktionen nicht gemerkt**!
3. get_ppo_without_forecast_config() erstellt eine neue EnvironmentConfig()
4. Diese hat die DEFAULT reward_params, nicht die angepassten!

**Lösung**: Explizit die reward_params in der Config setzen, wenn sie unterschiedlich sein sollen!

```python
def get_ppo_without_forecast_config(...):
    env_config = EnvironmentConfig()
    # Wenn ihr kappa=0.1 wollt (wie Notebook):
    env_config.reward_params['kappa'] = 0.1
    # oder direkt:
    env_config.kappa = 0.1
```

---

## FAZIT

**Hauptgrund für negative Returns**:
- Kappa=0.01 ist zu aggressiv (zu wenig Risiko-Penalty)
- Agent über-traded und riskiert zu viel
- Mit κ=0.1 (Notebook) wären Returns wahrscheinlich positiv

**Empfehlung**:
1. Setze kappa=0.1 in EnvironmentConfig (wie Notebook)
2. Oder nutze die reward_ablation κ=0.05 (conservative) für bessere Results
3. Teste mit κ=0.001 (aggressive) um zu sehen, wie viel schlimmer es werden kann


