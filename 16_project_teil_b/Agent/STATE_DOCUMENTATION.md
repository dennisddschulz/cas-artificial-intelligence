# State Space Dokumentation - Forecast-Augmented RL Trading

## Überblick

Der **State** ist ein 14-dimensionaler Vektor, der den aktuellen Zustand des Trading-Systems beschreibt. Er wird dem Agenten bei jedem Schritt übergeben und soll alle relevanten Informationen enthalten, um optimale Trading-Entscheidungen zu treffen.

```
State = [Market Features (8) | Portfolio Features (6)]
      = [r, r_lag1, mu_hat, sigma_hat, rsi, macd_diff, bb_width, ema_ratio |
         current_position, cash_ratio, current_leverage, drawdown, cumulative_pnl, recent_return]
```

**Dimension**: 14 (oder 15 mit optionalem Forecast)
**Datentyp**: `np.float32`
**Range**: Alle Features sind normalisiert auf [-3, 3] oder ähnliche Bounds

---

## PART 1: MARKET FEATURES (8 Felder)

Diese Felder beschreiben die **Marktbedingungen** und technischen Signale.

### 1. `r` - Log Return (Aktuelle Tagesrendite)

**Index**: 0
**Berechnung**:
```
r_t = log(price_t / price_{t-1})
```

**Interpretation**:
- Zeigt die prozentuale Veränderung des Preises
- Positiv (z.B. +0.02): Preis ist um 2% gestiegen
- Negativ (z.B. -0.03): Preis ist um 3% gefallen
- Null (0.0): Keine Preisveränderung

**Range**: [-0.1, 0.1] (typisch für tägliche Returns)
**Praktisches Beispiel**:
- Bitcoin steigt von $50,000 auf $51,000 → r ≈ +0.0198 ≈ +1.98%
- Bitcoin fällt von $50,000 auf $48,000 → r ≈ -0.0408 ≈ -4.08%

**Warum wichtig für Agent?**
- Agent sieht unmittelbare Marktreaktion
- Hilft beim Verständnis von Markttrends
- Wird für PnL-Berechnung verwendet

---

### 2. `r_lag1` - Verzögerter Return (Vorheriger Tag)

**Index**: 1
**Berechnung**:
```
r_lag1_t = r_{t-1}
```

**Interpretation**:
- Return vom **vorherigen Tag** (t-1)
- Hilft dem Agent, Trend-Kontinuität zu erkennen

**Range**: [-0.1, 0.1]
**Praktisches Beispiel**:
- Tag 1: Bitcoin +2% (r = +0.02)
- Tag 2: Bitcoin +1.5% (r = +0.015)
  - Der Agent sieht: r=+0.015, r_lag1=+0.02
  - Signal: Momentum schwächt sich ab

**Warum wichtig für Agent?**
- Erkennt Trends (zwei positive Returns = aufwärts Trend)
- Hilft bei Momentum-Strategien
- Reduziert Rauschen durch Verzögerung

---

### 3. `mu_hat` - Forecasted Return (Prognostizierter Return)

**Index**: 2
**Berechnung**:
```
mu_hat_t = EWMA(r, span=20)
```
EWMA = Exponentially Weighted Moving Average

**Interpretation**:
- **Durchschnittlicher erwarteter Return** basierend auf den letzten 20 Tagen
- Gewichtet neuere Tage stärker als ältere
- Positive Werte: Erwartet Preisanstieg
- Negative Werte: Erwartet Preisfall

**Range**: [-0.02, 0.02] (normalerweise kleiner als r)
**Praktisches Beispiel**:
- Letzte 5 Tage: +1%, +1.5%, +1.2%, +0.8%, -0.1%
- mu_hat ≈ +0.008 = +0.8% erwarteter Return
- Signal: Schwacher Aufwärtstrend

**Warum wichtig für Agent?**
- Zeigt durchschnittliche Marktrichtung
- Wird verwendet für Alignment Bonus in der Reward-Funktion
- Stabiler als einzelner Return (weniger Rauschen)

---

### 4. `sigma_hat` - Volatilitätsschätzung

**Index**: 3
**Berechnung**:
```
sigma_hat_t = std(r, window=20)
```

**Interpretation**:
- Misst die **Preisschwankungen** (Volatilität)
- Hohe Werte: Markt ist volatil/unsicher
- Niedrige Werte: Markt ist stabil/ruhig

**Range**: [0.005, 0.05] (normalerweise positiv)
**Praktisches Beispiel**:
- Ruhiger Markt: sigma_hat = 0.008 = 0.8% tägliche Volatilität
- Volatiler Markt: sigma_hat = 0.035 = 3.5% tägliche Volatilität

**Warum wichtig für Agent?**
- Risk-Penalty hängt davon ab: `risk_penalty = kappa * position^2 * sigma_hat`
- Agent wird angehalten, in volatilen Zeiten kleinere Positionen zu nehmen
- Schützt vor Überhebung in unsicheren Märkten

---

### 5. `rsi` - Relative Strength Index

**Index**: 4
**Berechnung**:
```
RS = average_gain / average_loss (über 14 Tage)
RSI = 100 - (100 / (1 + RS))
RSI_normalized = (RSI - 50) / 50  → [-1, 1]
```

**Interpretation**:
- Misst **Momentum** und **Überverkauftheit/Überverkäuftheit**
- **+1.0**: Stark überverkauft (viele Käufe, könnte Korrektur folgen)
- **-1.0**: Stark überverkauft (viele Verkäufe, könnte Anstieg folgen)
- **0.0**: Neutrale Momentum

**Range**: [-1.0, 1.0]
**Praktisches Beispiel**:
- RSI = 80 (überverkauft) → normalized = (80-50)/50 = +0.6
- RSI = 20 (überverkauft) → normalized = (20-50)/50 = -0.6

**Warum wichtig für Agent?**
- Mean-Reversion Signal: Extremwerte deuten auf Umkehrungen hin
- Momentum-Indikator
- Weitverbreiteter Indikator in technischer Analyse

---

### 6. `macd_diff` - MACD Differenz (Trend-Signal)

**Index**: 5
**Berechnung**:
```
MACD = EMA(close, 12) - EMA(close, 26)
MACD_Signal = EMA(MACD, 9)
MACD_Diff = MACD - MACD_Signal
MACD_Diff_normalized = MACD_Diff / std(MACD_Diff)
```

**Interpretation**:
- Zeigt die **Differenz zwischen schnellem und langsamem Trend**
- Positiv: Schneller Trend liegt über langsamem → Aufwärtstrend
- Negativ: Schneller Trend liegt unter langsamem → Abwärtstrend
- Kreuzungen sind Kauf-/Verkaufssignale

**Range**: [-3, 3] (normalisiert)
**Praktisches Beispiel**:
- MACD_Diff = +0.5 (positiv): Aufwärtstrend
- MACD_Diff = -0.3 (negativ): Abwärtstrend
- MACD_Diff kreuzzt 0: Trendwechsel

**Warum wichtig für Agent?**
- Klassischer Trend-Indikator
- Hilft bei Trend-Following-Strategien
- Identifiziert Momentum-Shift

---

### 7. `bb_width` - Bollinger Bands Breite

**Index**: 6
**Berechnung**:
```
BB_Width = 2 * std(close, 20) / SMA(close, 20)
BB_Width_normalized = BB_Width / mean(BB_Width)
```

**Interpretation**:
- Misst die **Spannweite der Preisschwankungen**
- Hohe Werte: Preis schwankt stark (volatil)
- Niedrige Werte: Preis ist stabil (kann vor Ausbruch deuten)

**Range**: [0.5, 3.0] (typisch)
**Praktisches Beispiel**:
- BB_Width = 0.05 (eng): Markt ist ruhig, könnte Ausbruch folgen
- BB_Width = 0.15 (weit): Markt ist volatil, Preisbewegungen groß

**Warum wichtig für Agent?**
- Alternative Volatilitätsmessung (wie sigma_hat)
- Identifiziert Squeeze (schmale Bänder = Ausbruch wahrscheinlich)
- Hilft bei Volatilität-basiertem Position-Sizing

---

### 8. `ema_ratio` - EMA Momentum Ratio

**Index**: 7
**Berechnung**:
```
EMA12 = EMA(close, 12)
EMA26 = EMA(close, 26)
Ratio = EMA12 / EMA26
Momentum = (Ratio - mean(Ratio)) / std(Ratio)
```

**Interpretation**:
- Zeigt die **Relation zwischen kurzfristigen und langfristigen Trends**
- Positiv: Kurzfristiger Trend stärker (Aufwärtstrend)
- Negativ: Langfristiger Trend stärker (Abwärtstrend)
- Ähnlich wie MACD, aber etwas anders berechnet

**Range**: [-3, 3] (normalisiert)
**Praktisches Beispiel**:
- EMA12/EMA26 = 1.05 (12er über 26er) → Aufwärtstrend
- EMA12/EMA26 = 0.95 (12er unter 26er) → Abwärtstrend

**Warum wichtig für Agent?**
- Bestätigung von MACD-Signal
- Momentum-Indikator
- Hilft bei Trend-Klassifikation

---

## PART 2: PORTFOLIO FEATURES (6 Felder)

Diese Felder beschreiben den **Zustand des Trading-Portfolios** und der **Risikoposition**.

### 9. `current_position` - Aktuelle Position (Normalized)

**Index**: 8
**Berechnung**:
```
current_position = tanh(pos / 0.5)
```
wobei `pos ∈ [-1, 1]` die tatsächliche Position ist.

**Interpretation**:
- Zeigt die **aktuelle Expositionsgröße und -richtung**
- **+0.9**: Fast vollständig long (95% Kapital investiert long)
- **-0.9**: Fast vollständig short (95% Kapital investiert short)
- **0.0**: Neutral (keine Position, 100% Cash)
- **+0.5**: 50% long
- **-0.3**: 30% short

**Range**: [-0.9999, 0.9999] (tanh output)
**Praktisches Beispiel**:
- pos = 0.5 → current_position = tanh(0.5/0.5) = tanh(1.0) ≈ 0.762
- pos = -0.3 → current_position = tanh(-0.3/0.5) = tanh(-0.6) ≈ -0.537

**Warum wichtig für Agent?**
- Agent muss wissen, welche Position er gerade hat
- Essentiell für Reward-Berechnung (PnL hängt davon ab)
- Hilft bei Position-Management Entscheidungen
- Verhindert, dass Agent vergisst, was er trägt

---

### 10. `cash_ratio` - Cash-Anteil (Liquidität)

**Index**: 9
**Berechnung**:
```
cash_ratio = cash / equity = (equity - |position| * equity) / equity
           = 1 - |position|
```

**Interpretation**:
- Zeigt den **Anteil der liquiden Mittel** (nicht investiert)
- **1.0**: 100% Cash (vollständig liquid, keine Position)
- **0.5**: 50% Cash, 50% investiert
- **0.0**: 0% Cash (vollständig investiert oder leveraged)
- **0.05**: Minimum 5% Cash (Liquidity-Constraint)

**Range**: [0.0, 1.0]
**Praktisches Beispiel**:
- Equity = $100,000
- Position = 0.6 (60% long)
- Position Value = 0.6 * $100,000 = $60,000
- Cash = $100,000 - $60,000 = $40,000
- cash_ratio = $40,000 / $100,000 = 0.4

**Warum wichtig für Agent?**
- Agent muss wissen, wie viel Liquidität verfügbar ist
- Verhindert zu aggressive Positionen
- Enforces Margin-Requirements (mind. 5%)
- Hilft bei Risk-Management

---

### 11. `current_leverage` - Aktuelle Leveragenutzung

**Index**: 10
**Berechnung**:
```
current_leverage = min(|position| / LEVERAGE_MAX, 1.0)
```
wobei `LEVERAGE_MAX = 1.0`

**Interpretation**:
- Zeigt, wie **viel des verfügbaren Hebels** verwendet wird
- **0.0**: Kein Hebel (keine Position)
- **0.5**: 50% des maximalen Hebels (50% Position)
- **1.0**: Voll ausgelastet (100% long oder short)

**Range**: [0.0, 1.0]
**Praktisches Beispiel**:
- Position = 0.7, LEVERAGE_MAX = 1.0 → leverage = 0.7
- Position = -0.3, LEVERAGE_MAX = 1.0 → leverage = 0.3
- Position = 0.0, LEVERAGE_MAX = 1.0 → leverage = 0.0

**Warum wichtig für Agent?**
- Agent muss wissen, wie exponiert er bereits ist
- Hilft beim Risk-Scaling
- Verhindert Über-Leveraging
- Input für Risk-Management Logik

---

### 12. `drawdown` - Maximaler Rückgang vom Peak

**Index**: 11
**Berechnung**:
```
drawdown = (peak_equity - current_equity) / peak_equity
         = 1 - (current_equity / peak_equity)
```

**Interpretation**:
- Zeigt, wie **weit unter dem Höchststand** das Portfolio ist
- **0.0**: Auf neuem Allzeithoch
- **0.1**: 10% unter dem Peak
- **0.3**: 30% unter dem Peak (großer Drawdown)

**Range**: [0.0, 1.0]
**Praktisches Beispiel**:
- Peak Equity = $100,000
- Current Equity = $85,000
- Drawdown = (100,000 - 85,000) / 100,000 = 0.15 = 15%

**Warum wichtig für Agent?**
- Wichtige Risk-Metrik
- Zeigt, wie schlecht es dem Portfolio gerade geht
- Agent kann konservativ reagieren bei großem Drawdown
- Hilft bei Mean-Reversion Strategien

---

### 13. `cumulative_pnl` - Kumulativer Profit/Loss

**Index**: 12
**Berechnung**:
```
cumulative_pnl = tanh((equity - initial_equity) / initial_equity)
```

**Interpretation**:
- Zeigt den **Gesamtgewinn/Verlust** seit Anfang
- **+0.7**: Große Gewinne (tanh ≈ 70% Gewinn)
- **-0.5**: Verluste (tanh ≈ -50% Gewinn)
- **0.0**: Break-Even (tanh ≈ 0% Gewinn)

**Range**: [-0.9999, 0.9999] (tanh output)
**Praktisches Beispiel**:
- Initial Equity = $100,000
- Current Equity = $110,000
- cumulative_pnl = tanh((110,000 - 100,000) / 100,000) = tanh(0.1) ≈ 0.0997 ≈ +10%

**Warum wichtig für Agent?**
- Agent sieht seine Gesamtperformance
- Hilft bei Momentum-Entscheidungen
- Positive Werte ermutigen zu größeren Positionen
- Negative Werte können zu konservativerem Verhalten führen

---

### 14. `recent_return` - Tagesrendite (Letzter Schritt)

**Index**: 13
**Berechnung**:
```
recent_return = tanh((equity - last_equity) / last_equity)
```

**Interpretation**:
- Zeigt die **Veränderung vom letzten Schritt**
- **+0.5**: +50% Rendite vom letzten Tag
- **-0.3**: -30% Rendite vom letzten Tag
- **0.0**: Keine Veränderung

**Range**: [-0.9999, 0.9999] (tanh output)
**Praktisches Beispiel**:
- Last Equity = $100,000
- Current Equity = $102,000
- recent_return = tanh((102,000 - 100,000) / 100,000) = tanh(0.02) ≈ 0.02 = +2%

**Warum wichtig für Agent?**
- Zeigt die **unmittelbare Reaktion** auf die Position
- Kurzfristiger Feedback-Loop
- Hilft bei Live-Adjustments
- Ähnlich wie `r`, aber auf Portfolio-Level

---

## Zusätzliches Feature (Optional)

### 15. `forecast_signal` - Forecast-Wahrscheinlichkeit (Optional)

**Index**: 14 (nur wenn `include_forecast=True`)
**Berechnung**:
```
forecast_signal = forecast_probability * 2 - 1
```
wobei `forecast_probability ∈ [0, 1]` die LSTM-Vorhersage ist.

**Interpretation**:
- Zeigt die **LSTM-Vorhersage** für nächste Tagesrichtung
- **+1.0**: LSTM ist sehr sicher, dass Preis steigt
- **-1.0**: LSTM ist sehr sicher, dass Preis fällt
- **0.0**: LSTM ist unsicher (50% Wahrscheinlichkeit)

**Range**: [-1.0, 1.0]
**Praktisches Beispiel**:
- LSTM probability = 0.75 (75% up) → signal = 0.75 * 2 - 1 = +0.5
- LSTM probability = 0.30 (30% up) → signal = 0.30 * 2 - 1 = -0.4

**Warum wichtig für Agent?**
- Externe Signal für den Agent
- Lenkt Exploration in vielversprechende Richtung
- Reward-Bonus wenn Position mit Signal aligned ist
- Hilft beim schnelleren Training

---

## Zusammenfassung: State-Struktur

```python
State Vector (14-dimensional):
┌─────────────────────────────────────────────────────┐
│ MARKET FEATURES (8):                                │
│ [0] r             : Heutiger Log-Return             │
│ [1] r_lag1        : Gestrigen Log-Return            │
│ [2] mu_hat        : Erwarteter Return (EWMA)        │
│ [3] sigma_hat     : Volatilität                     │
│ [4] rsi           : Momentum-Indikator              │
│ [5] macd_diff     : Trend-Signal                    │
│ [6] bb_width      : Volatilitäts-Signal             │
│ [7] ema_ratio     : Momentum-Ratio                  │
├─────────────────────────────────────────────────────┤
│ PORTFOLIO FEATURES (6):                             │
│ [8]  current_position : Aktuelle Positionsgröße    │
│ [9]  cash_ratio       : Liquidität (Cash%)         │
│ [10] current_leverage : Hebel-Auslastung           │
│ [11] drawdown         : Rückgang vom Peak          │
│ [12] cumulative_pnl   : Gesamtgewinn/-verlust      │
│ [13] recent_return    : Tagesrendite               │
└─────────────────────────────────────────────────────┘
```

---

## Normalisierungen & Bounds

| Feature | Type | Range | Normalisierung |
|---------|------|-------|---|
| r | float32 | [-0.1, 0.1] | Log-Returns |
| r_lag1 | float32 | [-0.1, 0.1] | Log-Returns |
| mu_hat | float32 | [-0.02, 0.02] | EWMA |
| sigma_hat | float32 | [0.005, 0.05] | Rolling Std |
| rsi | float32 | [-1, 1] | (RSI - 50) / 50 |
| macd_diff | float32 | [-3, 3] | Normalized |
| bb_width | float32 | [0.5, 3.0] | Normalized |
| ema_ratio | float32 | [-3, 3] | Z-Score |
| current_position | float32 | [-0.9999, 0.9999] | tanh(pos/0.5) |
| cash_ratio | float32 | [0, 1] | Direct Ratio |
| current_leverage | float32 | [0, 1] | Direct Ratio |
| drawdown | float32 | [0, 1] | Direct Ratio |
| cumulative_pnl | float32 | [-0.9999, 0.9999] | tanh() |
| recent_return | float32 | [-0.9999, 0.9999] | tanh() |
| forecast_signal | float32 | [-1, 1] | prob * 2 - 1 |

---

## Praktisches Beispiel: Ein vollständiger State

```
Annahme: Bitcoin, 10:00 Uhr MEZ

Markt-Situation:
- Bitcoin: $50,000 → $51,000 (+2%)
- Volatilität: Moderat
- Trend: Aufwärts

State Vector:
[
  # Market Features
  0.0198,      # [0] r: +1.98% today
  0.0102,      # [1] r_lag1: +1.02% yesterday
  0.0085,      # [2] mu_hat: +0.85% erwarteter Return
  0.0152,      # [3] sigma_hat: 1.52% Volatilität
  0.65,        # [4] rsi: +0.65 (strong momentum)
  0.32,        # [5] macd_diff: +0.32 (uptrend)
  1.2,         # [6] bb_width: 1.2 (moderate volatility)
  0.58,        # [7] ema_ratio: +0.58 (uptrend)

  # Portfolio Features
  0.612,       # [8] current_position: 50% long
  0.5,         # [9] cash_ratio: 50% cash
  0.5,         # [10] current_leverage: 50% ausgelastet
  0.02,        # [11] drawdown: 2% unter Peak
  0.087,       # [12] cumulative_pnl: +8.7% Gesamtgewinn
  0.0199,      # [13] recent_return: +2.0% heute
]

Interpretation:
→ Markt ist im Aufwärtstrend (+2% today)
→ Volatilität ist moderat
→ Agent hat 50% long Position
→ Noch 50% Liquidität verfügbar
→ Portfolio ist +8.7% im Gewinn
```

---

Diese Dokumentation sollte Dir einen vollständigen Überblick über den State-Space geben! 🎯

