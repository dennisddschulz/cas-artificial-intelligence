# State Space - Quick Reference & Visualisierung

## 🎯 State Vector im Überblick

```
┌─────────────────────────────────────────────────────────────────┐
│                      STATE VECTOR (14-D)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  MARKET FEATURES (Was passiert am Markt?)                      │
│  ────────────────────────────────────────                      │
│  [0] r           │ Current Return      │  Preisbewegung heute  │
│  [1] r_lag1      │ Yesterday Return    │  Preisbewegung gestern│
│  [2] mu_hat      │ Expected Return     │  Trend-Richtung      │
│  [3] sigma_hat   │ Volatility          │  Risiko/Unsicherheit │
│  [4] rsi         │ Momentum            │  Überkauft/Verkauft  │
│  [5] macd_diff   │ Trend Signal        │  Trend-Stärke        │
│  [6] bb_width    │ Band Width          │  Schwankungsbreite   │
│  [7] ema_ratio   │ Moving Avg Ratio    │  Short vs Long Term  │
│                                                                 │
│  PORTFOLIO FEATURES (Wie ist meine Position?)                 │
│  ──────────────────────────────────────────                   │
│  [8]  current_pos │ Position Size      │  Wie viel bin ich long?│
│  [9]  cash_ratio  │ Cash Available     │  Wie viel Cash noch?  │
│  [10] leverage    │ Leverage Used      │  Wie viel Hebel?      │
│  [11] drawdown    │ Distance from Peak │  Wie weit unten?      │
│  [12] cum_pnl     │ Total Profit/Loss  │  Wie profitabel?      │
│  [13] recent_ret  │ Today's Change     │  Wie gut war heute?   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Feature-Kategorien

### 1. RETURNS (Features 0-1)
```
Was: Preisveränderungen
Warum: Direktes Feedback vom Markt

r & r_lag1
├─ Beide positiv: Aufwärtstrend
├─ Beide negativ: Abwärtstrend
└─ Unterschiedlich: Trend-Wechsel
```

### 2. TREND INDICATORS (Features 2, 5, 7)
```
Was: Langfristige Richtung
Warum: Hilft bei Position-Alignment

mu_hat, macd_diff, ema_ratio
├─ Alle positiv: Starker Aufwärtstrend
├─ Alle negativ: Starker Abwärtstrend
└─ Gemischt: Schwacher/Konfus Trend
```

### 3. VOLATILITY (Features 3, 6)
```
Was: Preisschwankungen & Unsicherheit
Warum: Risk-Management

sigma_hat & bb_width
├─ Hoch: Großes Risiko → kleinere Positionen
└─ Niedrig: Stabiler Markt → größere Positionen
```

### 4. MOMENTUM (Features 4)
```
Was: Überverkauft/Überverkauft
Warum: Mean-Reversion Signale

rsi
├─ Positiv: Überverkauft → Anstieg wahrscheinlich
└─ Negativ: Überverkauft → Fall wahrscheinlich
```

### 5. POSITION STATUS (Features 8-10)
```
Was: Wo bin ich aktuell exponiert?
Warum: Muss wissen, welche Position ich halte

current_position & cash_ratio & leverage
├─ High Position: Viel exponiert
├─ High Cash: Viel Liquidität
└─ Zusammen: Risikomanagenent-Info
```

### 6. RISK METRICS (Features 11-13)
```
Was: Wie gut/schlecht läuft es?
Warum: Performance-Feedback

drawdown & cumulative_pnl & recent_return
├─ Hoch drawdown: Conservative traden
├─ Positiv pnl: Mehr Risiko eingehen
└─ Negativer return: Vorsicht nächste Moves
```

---

## 🎨 Visuelle Darstellung

### Zeitliche Ebene
```
         PAST    |    CURRENT    |    FUTURE
                 │               │
r_lag1 ────────────r────────────→ (Agent muss handeln)
                 │               │
        Gestern  │  Heute 10:00  │  Morgen?
                 ↓
         Agent sieht: [r_lag1, r, mu_hat, ...]
         Agent entscheidet: action
```

### Markt vs Portfolio
```
                          STATE VECTOR
              ┌─────────────────────────────────┐
              │                                 │
    ┌─────────┴──────────┐          ┌──────────┴─────┐
    │                    │          │                │
MARKET FEATURES      PORTFOLIO FEATURES
(Extern)            (Intern)
    │                    │          │                │
Preis              Position      Cash           PnL
Trend              Leverage      Drawdown       Return
Volatility         Exposure      Risk
    │                    │          │                │
    └─────────┬──────────┘          └──────────┬─────┘
              │                                 │
         Marktdaten              Portfoliozustand
         (öffentlich)            (Agentenwissen)
```

### Numerische Beispiele

#### Example 1: BULL MARKET
```
State:
[
  +0.03,   # r: +3% heute (stark ansteigend!)
  +0.02,   # r_lag1: +2% gestern
  +0.015,  # mu_hat: +1.5% erwartet
  +0.01,   # sigma_hat: 1% Volatilität (stabil)
  +0.8,    # rsi: überverkauft (Aufwärts-Potential)
  +0.6,    # macd_diff: stark positiv (Uptrend)
  +1.5,    # bb_width: normal
  +0.7,    # ema_ratio: stark positiv

  +0.6,    # current_position: 60% long (profitable!)
  +0.4,    # cash_ratio: noch 40% cash verfügbar
  +0.6,    # leverage: 60% ausgelastet
  +0.01,   # drawdown: sehr nah am Peak
  +0.15,   # cum_pnl: +15% Gewinn!
  +0.03,   # recent_return: +3% heute
]

Agent sollte: AGGRESSIVE LONG gehen
Grund: Alles positiv → Bull Market Signal
```

#### Example 2: BEAR MARKET
```
State:
[
  -0.04,   # r: -4% heute (stark fallend)
  -0.02,   # r_lag1: -2% gestern
  -0.012,  # mu_hat: -1.2% erwartet
  +0.025,  # sigma_hat: 2.5% Volatilität (unsicher)
  -0.7,    # rsi: überverkauft (Anstieg-Potential?)
  -0.5,    # macd_diff: negativ (Downtrend)
  +2.0,    # bb_width: hoch (volatil)
  -0.6,    # ema_ratio: negativ

  -0.3,    # current_position: 30% short (gut!)
  +0.7,    # cash_ratio: 70% cash (sicher)
  +0.3,    # leverage: nur 30% ausgelastet
  +0.25,   # drawdown: 25% unter Peak (!)
  -0.08,   # cum_pnl: -8% Verlust
  -0.04,   # recent_return: -4% heute (Achtung!)
]

Agent sollte: DEFENSIVE SHORT / ALL CASH
Grund: Alles negativ → Bear Market → Schutz gefragt
```

#### Example 3: CONFLICTED MARKET
```
State:
[
  +0.01,   # r: +1% heute
  -0.02,   # r_lag1: -2% gestern (Konflikt!)
  +0.001,  # mu_hat: ~0% erwartet (unsicher)
  +0.02,   # sigma_hat: 2% Volatilität (erhöht)
  +0.1,    # rsi: schwach positiv
  -0.2,    # macd_diff: negativ (Downtrend)
  +1.2,    # bb_width: normal
  +0.05,   # ema_ratio: schwach positiv

  +0.1,    # current_position: 10% long (klein)
  +0.9,    # cash_ratio: 90% cash (defensiv)
  +0.1,    # leverage: kaum genutzt
  +0.08,   # drawdown: 8% unter Peak
  +0.02,   # cum_pnl: +2% (bescheiden)
  +0.01,   # recent_return: +1% (ok)
]

Agent sollte: WAIT AND SEE / SMALL POSITIONS
Grund: Unsicheres Signal → Reduziertes Risiko
```

---

## 🎯 Feature-Relationen

### Positive Korrelationen (Agent sollte beachten)
```
r ↑  +  mu_hat ↑  =  Starker Aufwärtstrend
r ↓  +  mu_hat ↓  =  Starker Abwärtstrend
sigma ↑  +  bb_width ↑  =  Markt wird volatiler
rsi ↑  +  macd ↑  =  Konfirmation des Trends
```

### Inverse Relationen
```
drawdown ↑  →  Risiko ↑  →  Position sollte ↓
cash_ratio ↓  →  Liquidität ↓  →  Vorsicht!
recent_return ↑  →  Momentum ↑  →  Trend-Following
```

---

## 💡 Praktische Interpretationen

| Wenn... | Dann ist | Agent sollte |
|---------|---------|--------------|
| r > 0, r_lag1 > 0 | Uptrend | Long nehmen |
| r < 0, r_lag1 < 0 | Downtrend | Short nehmen |
| sigma ↑ | Volatilität | Positionen reduzieren |
| drawdown > 0.2 | Großes Minus | Defensiv agieren |
| cash_ratio ↓ 0.05 | Margin Low | Vorsicht! |
| rsi extreme | Umkehr? | Kontra-Trend traden |

---

## 📏 Dimensionen & Shape

```
State Shape: (14,)
Data Type: np.float32
Valid Range: [-3.0, +3.0] (mit clipping)

Memory: 14 float32 = 56 bytes
Sequence: 8 parallel envs × 256 steps = 28,672 states pro Update
```

---

Diese Dokumentation gibt Dir den kompletten Überblick über jeden einzelnen Feature im State-Space! 🎯


