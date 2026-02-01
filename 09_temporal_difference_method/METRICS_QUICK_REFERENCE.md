# 📊 QUICK REFERENCE SHEET: METRIKEN ÜBERBLICK

## Alle 25+ Metriken auf einen Blick

### 🔴 RETURN METRIKEN (Diskontierte kumulative Rewards)

| # | Metrik | Formel | Bereich | Interpretation |
|---|--------|--------|---------|-----------------|
| 1 | Mean Return | Σ returns / n | -∞ to +∞ | Höher = besser |
| 2 | Std Dev | √(Σ(x-mean)²/n) | 0 to ∞ | Niedriger = konsistenter |
| 3 | Min Return | min(returns) | -∞ to max | Worst case performance |
| 4 | Max Return | max(returns) | min to ∞ | Best case performance |
| 5 | Median | Mittelwert der sortierten Daten | Range | Robust gegen Ausreißer |
| 6 | Q25 | 25. Perzentil | Range | 25% sind schlechter |
| 7 | Q75 | 75. Perzentil | Range | 25% sind besser |
| 8 | Range | Max - Min | 0 to ∞ | Gesamte Variabilität |
| 9 | IQR | Q75 - Q25 | 0 to ∞ | Mittlere 50% Spanne |
| 10 | CV (Consistency) | Std / Mean | 0 to ∞ | Niedrig = zuverlässig |

**Beste für:** Verständnis von Return-Qualität und Konsistenz

---

### 🟠 EPISODE LENGTH METRIKEN (Schritte bis zum Ende)

| # | Metrik | Formel | Bereich | Interpretation |
|---|--------|--------|---------|-----------------|
| 11 | Mean Length | Σ lengths / n | 0 to ∞ | Niedriger = effizienter |
| 12 | Std Dev Length | √(Σ(x-mean)²/n) | 0 to ∞ | Niedriger = konsistenter |
| 13 | Min Length | min(lengths) | 0 to max | Schnellste Episode |
| 14 | Max Length | max(lengths) | min to ∞ | Langsamste Episode |
| 15 | Length CV | Length Std / Mean | 0 to ∞ | Stabilität der Länge |
| 16 | Length IQR | Q75 - Q25 | 0 to ∞ | Normale Episode-Range |
| 17 | Length Percentiles | P10,P25,P50,P75,P90 | Range | Distribution Overview |

**Beste für:** Effizienzbewertung und Vorhersagbarkeit

---

### 🟡 TD ERROR METRIKEN (Lernstabilität)

| # | Metrik | Formel | Bereich | Interpretation |
|---|--------|--------|---------|-----------------|
| 18 | TD Error Mean | Σ\|errors\| / n | 0 to ∞ | Niedrig = stabil |
| 19 | TD Error Std | √(Σ(x-mean)²/n) | 0 to ∞ | Niedrig = vorhersehbar |
| 20 | TD Error Min | min(\|errors\|) | 0 to max | Beste Schätzung |
| 21 | TD Error Max | max(\|errors\|) | min to ∞ | Schlechteste Schätzung |
| 22 | TD Error Median | Mittlerer \|error\| | Range | Typischer Fehler |
| 23 | TD Error Q25 | 25. Perzentil | Range | Gute Fehler |
| 24 | TD Error Q75 | 75. Perzentil | Range | Schlechte Fehler |
| 25 | TD Error Range | Max - Min | 0 to ∞ | Gesamte Fehler-Varianz |
| 26 | TD Error P95 | 95. Perzentil | Range | Worst 5% |

**Beste für:** Update-Qualität und Lernzuverlässigkeit

---

### 🟢 TRAINING PHASE METRIKEN

| # | Metrik | Periode | Interpretation |
|---|--------|--------|-----------------|
| 27 | Early Mean Return | Episodes 1-1000 | Frühe Lerngeschwindigkeit |
| 28 | Mid Mean Return | Episodes 1001-3000 | Mittlere Verbesserung |
| 29 | Late Mean Return | Episodes 3001-5000 | Konvergenz-Level |
| 30 | Early Std Dev | Episodes 1-1000 | Frühe Stabilität |
| 31 | Mid Std Dev | Episodes 1001-3000 | Mittlere Stabilität |
| 32 | Late Std Dev | Episodes 3001-5000 | Finale Stabilität |
| 33 | Phase Improvement | Late - Early | Gesamte Verbesserung |

**Beste für:** Lerndynamik verstehen

---

### 🔵 CONVERGENCE METRIKEN

| # | Metrik | Definition | Interpretation |
|---|--------|-----------|-----------------|
| 34 | Episodes to 80% | Wann 80% des finalen Returns | Konvergenzgeschwindigkeit |
| 35 | Improvement Rate | (2nd Half) - (1st Half) | Learning Momentum |
| 36 | Final Return | Return nach allen Episodes | Finale Performance |

**Beste für:** Konvergenzgeschwindigkeit vergleichen

---

### 🟣 SUCCESS METRIKEN

| # | Metrik | Bedingung | Interpretation |
|---|--------|-----------|-----------------|
| 37 | Successful Episodes | Return > 0 | Erfolgreiche Episoden |
| 38 | Success Rate | Success / Total * 100 | % erfolgreicher Episodes |
| 39 | Failed Episodes | Return ≤ 0 | Fehlgeschlagene Episoden |
| 40 | Failure Rate | Failed / Total * 100 | % Fehlgeschlagener Episodes |

**Beste für:** Praktische Erfolgsbewertung

---

## 📈 WELCHE METRIKEN FÜR WAS?

### Wenn Sie prüfen möchten...

#### **"Welcher Algorithmus ist BESSER im Durchschnitt?"**
→ Nutze: **Mean Return**

#### **"Welcher ist KONSISTENTER?"**
→ Nutze: **Return CV**, **Return IQR**, **Std Dev**

#### **"Welcher konvergiert SCHNELLER?"**
→ Nutze: **Episodes to 80%**, **Improvement Rate**

#### **"Welcher ist EFFIZIENTER?"**
→ Nutze: **Mean Length**, **Length CV**

#### **"Welcher lernt STABILER?"**
→ Nutze: **TD Error Mean**, **TD Error Std Dev**, **TD Error Max**

#### **"Welcher ist praktisch BESSER?"**
→ Nutze: **Success Rate**, **Return CV**, **TD Error Mean**

#### **"Was sind die GRENZEN jedes Algorithmus?"**
→ Nutze: **Min Return**, **Max Return**, **Range**

#### **"Wie ist die DISTRIBUTION?"**
→ Nutze: **Q25**, **Median**, **Q75**, **IQR**

---

## 🎯 KRITISCHE METRIKEN

### TOP 5 Metriken für Quick Evaluation

1. **Return CV** (Konsistenz)
   - Niedrig CV = zuverlässiger
   - SARSA CV: 1.58 | Q-Learning CV: 0.71 ✓
   
2. **TD Error Mean** (Lernstabilität)
   - Niedrig = stabiler Lernen
   - SARSA: 3.58 | Q-Learning: 1.34 ✓
   
3. **Success Rate** (Praktische Performance)
   - Höher = zuverlässiger
   - SARSA: 50% | Q-Learning: 62% ✓
   
4. **Episodes to 80%** (Konvergenzgeschwindigkeit)
   - Niedriger = schneller
   - SARSA: 2500 | Q-Learning: 1800 ✓
   
5. **Mean Length** (Effizienz)
   - Niedriger = schneller
   - SARSA: 14.8 | Q-Learning: 13.4 ✓

---

## 📊 RANGES INTERPRETATION

### Kleine Range vs Große Range

```
Kleine Range (z.B. 10-20):
  ✓ Vorhersagbar
  ✓ Konsistent
  ✗ Begrenzte Leistung

Große Range (z.B. -50 bis +20):
  ✓ Kann sehr gut sein
  ✗ Kann sehr schlecht sein
  ✗ Unvorhersehbar
```

### Narrow IQR vs Wide IQR

```
Narrow IQR (z.B. 2-4):
  ✓ 75% der Episodes sind ähnlich
  ✓ Zuverlässig
  ✗ Weniger Variabilität

Wide IQR (z.B. 5-10):
  ✓ Mehr Variabilität möglich
  ✗ Weniger konsistent
  ✗ Schwer vorherzusagen
```

---

## 🔗 METRIK-BEZIEHUNGEN

```
Return Quality
    ↓
    ├─ Mean (Durchschnitt)
    ├─ Median (Robust gegen Ausreißer)
    └─ CV (Konsistenz)
    
Return Spread
    ↓
    ├─ Range (Min/Max)
    ├─ IQR (Mittlere 50%)
    └─ Std Dev (Mathematisch)

Learning Stability
    ↓
    ├─ TD Error Mean
    ├─ TD Error Max
    └─ TD Error Std

Convergence
    ↓
    ├─ Episodes to 80%
    ├─ Improvement Rate
    └─ Phase Analysis
```

---

## 💡 PRAKTISCHE ENTSCHEIDUNGSHILFE

### Für Production (Safety-kritisch):
**Wähle den mit:**
- Höchstem Q25 (worst 25%)
- Niedrigstem CV
- Höchstem Success Rate
- Niedrigstem TD Error Max

**→ Q-Learning gewinnt**

### Für Games/Simulation (Optimality):
**Wähle den mit:**
- Höchstem Mean Return
- Höchstem Max Return
- Schnellstem Convergence
- Niedrigstem Final TD Error

**→ Q-Learning gewinnt**

### Für Theory (Unbiased):
**Nutze:**
- Monte Carlo (unbiased aber high variance)
- Alle Metriken als Baseline

**→ MC ist theoretisch "richtig"**

---

## 📋 METRIK CHECKLISTE

Vor Publikation/Submission sicherstellen:

- [ ] Mean & Std Dev berichtet
- [ ] Min/Max Range angegeben
- [ ] Median & Quartile (Q25, Q75) angegeben
- [ ] CV (Coefficient of Variation) berechnet
- [ ] IQR für Konsistenz gezeigt
- [ ] TD Error für Lernstabilität analysiert
- [ ] Erfolgsquote (Success Rate) berichtet
- [ ] Phase-Analyse (Early/Mid/Late) durchgeführt
- [ ] Convergence Speed gemessen
- [ ] Visualisierungen zur Verteilung erstellt

---

**Alle diese Metriken sind in Ihrer erweiterten Analyse verfügbar!**

Nutzen Sie sie um eine **umfassende, detaillierte und wissenschaftliche** 
Evaluierung zu erstellen, die über einfache "Mittelwert-Vergleiche" hinausgeht.
