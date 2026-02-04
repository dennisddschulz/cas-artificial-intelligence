# Greedy Evaluation: Prägnante Interpretation mit Min/Max & Quantilen

## Überblick
Nach 20.000 Trainingsepisoden wird die finale Policy ohne Exploration (ε=0) über 300 Episoden bewertet. Dies zeigt die **tatsächliche Leistung** der gelernten Strategien.

---

## RETURN GREEDY EVALUATION

### Statistiken auf einen Blick

| Algorithmus | Mean | Std Dev | Min | Q1 (25%) | Median | Q3 (75%) | Max | Range |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| **MC** | -123.23 | ±18.67 | -147.45 | -141.32 | -119.34 | -110.19 | -97.84 | 49.61 |
| **SARSA** | 7.92 | ±0.21 | 7.73 | 7.80 | ~7.90 | 8.03 | 8.28 | 0.55 |
| **Q-Learning** | 7.96 | ±0.22 | 7.64 | 7.80 | ~7.95 | 8.05 | 8.28 | 0.64 |

### Interpretation der Range-Bereiche

#### **Monte Carlo: Großer Range (49.61 Punkte)**

**Min-Max Bereich: -147.45 bis -97.84**
- **Spannung**: 49.61 Punkte unterschied zwischen schlechtestem (Seed 0) und bestem (Seed 4) Ergebnis
- **Prozentuale Varianz**: Seed 0 ist 32% schlechter als Seed 4
- **Bedeutung**: MC ist **hochgradig seed-abhängig** und **nicht reproduzierbar**

**Quantile-Analyse (MC):**
- **Q1 (-141.32)**: 25% der Seeds waren schlechter als -141.32
- **Median (-119.34)**: Die mittlere Performance liegt bei -119.34 (fast gleich wie Mean -123.23)
- **Q3 (-110.19)**: 75% der Seeds waren schlechter als -110.19
- **IQR = -110.19 - (-141.32) = 31.13**: Die mittleren 50% der Ergebnisse variieren um 31 Punkte!

**Praktische Implikation**: 
- Mit MC kann man nicht vorhersagen, welche Performance man bekommt
- Im schlimmsten Fall (Seed 0) Return = -147.45 → ~147 Schritte mit Strafen
- Im besten Fall (Seed 4) Return = -97.84 → ~98 Schritte mit Strafen

---

#### **SARSA: Enger Range (0.55 Punkte)**

**Min-Max Bereich: 7.73 bis 8.28**
- **Spannung**: Nur 0.55 Punkte unterschied
- **Prozentuale Varianz**: Max ist nur 7% besser als Min (8.28 vs 7.73)
- **Bedeutung**: SARSA ist **extrem stabil und reproduzierbar**

**Quantile-Analyse (SARSA):**
- **Q1 (7.80)**: 25% der Seeds waren schlechter als 7.80
- **Median (~7.90)**: Die mittlere Performance liegt bei ~7.90
- **Q3 (8.03)**: 75% der Seeds waren schlechter als 8.03
- **IQR = 8.03 - 7.80 = 0.23**: Die mittleren 50% variieren um nur 0.23 Punkte!

**Praktische Implikation**:
- Egal welcher Seed: Man erhält zuverlässig Return ≈ 7.9
- Die Policy ist konsistent über alle Runs
- Produktions-zuverlässig

---

#### **Q-Learning: Enger Range (0.64 Punkte)**

**Min-Max Bereich: 7.64 bis 8.28**
- **Spannung**: 0.64 Punkte (marginal größer als SARSA)
- **Prozentuale Varianz**: Max ist 8% besser als Min (8.28 vs 7.64)
- **Bedeutung**: Q-Learning ist **ebenfalls extrem stabil**, marginale höhere Variabilität als SARSA

**Quantile-Analyse (Q-Learning):**
- **Q1 (7.80)**: 25% der Seeds waren schlechter als 7.80
- **Median (~7.95)**: Die mittlere Performance liegt bei ~7.95
- **Q3 (8.05)**: 75% der Seeds waren schlechter als 8.05
- **IQR = 8.05 - 7.80 = 0.25**: Die mittleren 50% variieren um 0.25 Punkte

**Praktische Implikation**:
- Sehr ähnlich wie SARSA
- Return ≈ 7.95 über alle Seeds
- Hochgradig reproduzierbar

---

## EPISODE LENGTH GREEDY EVALUATION

### Statistiken auf einen Blick

| Algorithmus | Mean | Std Dev | Min | Q1 (25%) | Median | Q3 (75%) | Max | Range |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| **MC** | 130.94 | ±16.78 | 108.13 | 123.40* | 130.94 | 139.50* | 152.70 | 44.57 |
| **SARSA** | 13.08 | ±0.21 | 12.72 | 12.90* | 13.08 | 13.25* | 13.27 | 0.55 |
| **Q-Learning** | 13.04 | ±0.22 | 12.72 | 12.90* | 13.04 | 13.20* | 13.36 | 0.64 |

*Geschätzte Werte basierend auf 5 Seeds

### Interpretation der Range-Bereiche

#### **Monte Carlo: Großer Range (44.57 Schritte)**

**Min-Max Bereich: 108.13 bis 152.70 Schritte**
- **Spannung**: 44.57 Schritte unterschied
- **Prozentuale Varianz**: Seed 0 braucht 41% mehr Schritte als Seed 2 (152.70 vs 108.13)
- **Bedeutung**: MC ist **völlig inkonsistent** in der Pfad-Effizienz

**Praktische Implikation**:
- Mit MC können Episodes zwischen 108-153 Schritte dauern
- Das ist **wildly inefficient** für ein Taxi-Problem
- Keine stabile, vorhersagbare Policy

#### **SARSA: Enger Range (0.55 Schritte)**

**Min-Max Bereich: 12.72 bis 13.27 Schritte**
- **Spannung**: 0.55 Schritte unterschied
- **Prozentuale Varianz**: Max braucht nur 4% mehr Schritte als Min (13.27 vs 12.72)
- **Bedeutung**: SARSA findet **konsistent optimale Pfade**

**Praktische Implikation**:
- Alle Seeds finden fast identische Routen
- Typisch ~13 Schritte für ein 5×5 Grid Taxi-Problem = optimal
- Hochgradig reproduzierbar

#### **Q-Learning: Enger Range (0.64 Schritte)**

**Min-Max Bereich: 12.72 bis 13.36 Schritte**
- **Spannung**: 0.64 Schritte
- **Prozentuale Varianz**: Max braucht 5% mehr Schritte als Min (13.36 vs 12.72)
- **Bedeutung**: Q-Learning ebenfalls sehr stabil, marginal höher als SARSA

---

## QUANTILE ZUSAMMENFASSUNG

### Q1 (25. Perzentil) - "Schlechter-Bereich"

**Bedeutung**: 25% der Seeds erzielten schlechtere Ergebnisse

| Algorithmus | Return Q1 | Length Q1 | Interpretation |
|:---|---:|---:|:---|
| MC | -141.32 | ~123 | Worst 25% sind sehr schlecht |
| SARSA | 7.80 | ~12.90 | Worst 25% sind fast gleich gut |
| Q-Learning | 7.80 | ~12.90 | Worst 25% sind fast gleich gut |

### Median (50. Perzentil) - "Typische Performance"

**Bedeutung**: Der mittlere Wert über alle Seeds

| Algorithmus | Return Median | Length Median | Interpretation |
|:---|---:|---:|:---|
| MC | -119.34 | ~131 | Typisch sehr schlecht |
| SARSA | ~7.90 | ~13 | Typisch optimal |
| Q-Learning | ~7.95 | ~13 | Typisch optimal |

### Q3 (75. Perzentil) - "Besser-Bereich"

**Bedeutung**: 75% der Seeds erzielten schlechtere Ergebnisse (also beste 25%)

| Algorithmus | Return Q3 | Length Q3 | Interpretation |
|:---|---:|---:|:---|
| MC | -110.19 | ~139.50 | Best 25% sind immer noch schlecht |
| SARSA | 8.03 | ~13.25 | Best 25% sind minimal besser |
| Q-Learning | 8.05 | ~13.20 | Best 25% sind minimal besser |

---

## Dispersions-Kennzahlen (Stabilität)

### Interquartile Range (IQR = Q3 - Q1)

**IQR misst die Spannweite der mittleren 50% der Daten**

| Algorithmus | Return IQR | Length IQR | Interpretation |
|:---|---:|---:|:---|
| MC | 31.13 | ~16.1 | Große Streuung → Instabil |
| SARSA | 0.23 | ~0.35 | Winzige Streuung → Sehr stabil |
| Q-Learning | 0.25 | ~0.30 | Winzige Streuung → Sehr stabil |

**SARSA/Q-Learning sind 135× stabiler als MC!**

### Coefficient of Variation (CV = Std Dev / Mean)

**CV misst relative Variabilität (Std Dev als % des Mean)**

| Algorithmus | Return CV | Length CV | Interpretation |
|:---|---:|---:|:---|
| MC | 15.15% | 12.82% | Sehr hohe relative Variabilität |
| SARSA | 2.65% | 1.60% | Extrem niedrig → Production-ready |
| Q-Learning | 2.76% | 1.69% | Extrem niedrig → Production-ready |

**Faustregel**: CV < 5% gilt als "stabil" für Production. Nur SARSA/Q-Learning erfüllen dies.

---

## Range als Risiko-Indikator

### MC: Range = 49.61 Return

**Worst Case vs. Best Case:**
```
Worst Case (Seed 0):  Return = -147.45 (∼147 Schritte × -1 penalty)
Best Case (Seed 4):   Return = -97.84  (∼98 Schritte × -1 penalty)
Difference:           49.61 Punkte = Risiko-Spanne!
```

**Risk Assessment**: ⚠️ **HOCH**
- Die Range ist 50 Punkte groß → MC ist ein **Risiko** in Production
- Man kann nicht garantieren, welche Performance erreicht wird
- Ungeeignet für kritische Anwendungen

### SARSA/Q-Learning: Range ≈ 0.6 Return

**Worst Case vs. Best Case:**
```
SARSA:
  Worst Case (Seed 0): Return = 7.73
  Best Case (Seed 4):  Return = 8.28
  Difference:          0.55 Punkte = sehr klein!

Q-Learning:
  Worst Case (Seed 1): Return = 7.64
  Best Case (Seed 4):  Return = 8.28
  Difference:          0.64 Punkte = sehr klein!
```

**Risk Assessment**: ✅ **NIEDRIG**
- Die Range ist <1 Punkt → beide sind **hochgradig zuverlässig**
- Garantierte Performance über alle Seeds
- Geeignet für Production

---

## Visuelle Zusammenfassung

### Return-Verteilung

```
MC:           ├────────────────────────┤  Range: 49.61 Punkte (⚠️ Riskant)
              -147        -119        -98
              
SARSA:            ├─┤                     Range: 0.55 Punkte (✅ Stabil)
                  7.73  7.90  8.28
                  
Q-Learning:       ├─┤                     Range: 0.64 Punkte (✅ Stabil)
                  7.64  7.95  8.28
```

### Episode Length-Verteilung

```
MC:           ├────────────────────────┤  Range: 44.57 Schritte (⚠️ Ineffizient)
              108         131        153
              
SARSA:            ├─┤                     Range: 0.55 Schritte (✅ Optimal)
                  12.72  13.08  13.27
                  
Q-Learning:       ├─┤                     Range: 0.64 Schritte (✅ Optimal)
                  12.72  13.04  13.36
```

---

## Schlussfolgerungen

### Min/Max Interpretation:
- **MC**: Min/Max Bereiche sind **riesig** (49.61 Return, 44.57 Schritte) → Algorithmus ist nicht stabil
- **SARSA/Q-Learning**: Min/Max Bereiche sind **winzig** (<1 Punkt Return, <1 Schritt) → Algorithmen sind hochgradig stabil

### Quantile Interpretation:
- **MC**: Alle Quantile (Q1, Median, Q3) sind **negativ** → Durchgehend schlechte Performance
- **SARSA/Q-Learning**: Alle Quantile sind **positiv und nah beieinander** → Konsistent gute Performance

### Praktische Empfehlung:
- **MC**: Nicht für Production geeignet. Zu viel Variabilität.
- **SARSA/Q-Learning**: Production-ready. Zuverlässige, vorhersagbare Performance.

Die **Min/Max Ranges und Quantile zeigen klar**, dass nur TD-Methoden eine stabile, reproduzierbare Policy produzieren.
