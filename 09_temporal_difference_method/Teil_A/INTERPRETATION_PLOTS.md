# Interpretation: Monte Carlo vs. SARSA vs. Q-Learning

## Übersicht
Vergleich dreier RL-Algorithmen im Taxi-v3 über **5 Seeds × 20.000 Episoden**. Die Ergebnisse zeigen fundamentale Unterschiede zwischen episodischem Lernen (MC) und Online TD-Lernen (SARSA/Q-Learning).

---

## PLOT 1: Learning Curves

### Return Lernkurve

| Algorithmus | Mean | Std Dev | Min | Max | Range |
|:---|---:|---:|---:|---:|---:|
| **MC** | -123.23 | ±18.67 | -147.45 | -97.84 | 49.61 |
| **SARSA** | 7.92 | ±0.21 | 7.73 | 8.28 | 0.55 |
| **Q-Learning** | 7.96 | ±0.22 | 7.64 | 8.28 | 0.64 |

**Interpretationen:**
- **MC**: Bleibt durchgehend negativ (-123.23). Kein Lernfortschritt erkennbar. Hohe Variabilität (Range: 49.61).
- **SARSA/Q-Learning**: Springen nach ~1.000-2.000 Episoden in positive Bereiche. Konvergieren schnell und stabil zu ähnlichen Werten (~7.9).

**Warum MC so schlecht?** MC lernt nur episodisch (am Episode-Ende), nicht online. Mit zu schnellem ε-Decay wird Exploration reduziert, bevor gute Strategien gelernt werden.

### Episode Length Lernkurve

| Algorithmus | Mean | Std Dev | Min | Max | Range |
|:---|---:|---:|---:|---:|---:|
| **MC** | 130.94 | ±16.78 | 108.13 | 152.70 | 44.57 |
| **SARSA** | 13.08 | ±0.21 | 12.72 | 13.27 | 0.55 |
| **Q-Learning** | 13.04 | ±0.22 | 12.72 | 13.36 | 0.64 |

**Interpretationen:**
- **MC**: ~130 Schritte pro Episode = ineffiziente, lange Pfade. Keine Verbesserung.
- **SARSA/Q-Learning**: ~13 Schritte = quasi-optimale Pfade im Taxi-Problem.
- **Korrelation**: Längere Episodes → mehr negative Rewards → schlechterer Return.

---

## PLOT 2: Greedy Evaluation Bar Charts

### Return Stabilitäts-Analyse

**Coefficient of Variation (CV):**
- **MC**: 15.15% (hoch, seed-abhängig)
- **SARSA**: 2.65% (sehr stabil)
- **Q-Learning**: 2.76% (sehr stabil)

**→ SARSA ist 88× stabiler als MC!**

**Range-Analyse:**
- **MC**: Range = 49.61 Punkte → Seed 0 ist 32% schlechter als Seed 4
- **SARSA/Q-Learning**: Range ≈ 0.6 Punkte → Alle Seeds liefern ähnliche Ergebnisse

**Fazit**: TD-Methoden sind hochgradig reproduzierbar. MC ist nicht produktionsgeeignet.

### Episode Length Stabilitäts-Analyse

- **MC**: Range 44.57 Schritte, CV 12.82% → Hohe Variabilität
- **SARSA/Q-Learning**: Range ~0.6 Schritte, CV ~1.6% → Extrem stabil

---

## On-Policy (SARSA) vs. Off-Policy (Q-Learning)

### TD-Zielwert-Unterschied

```
SARSA:      Target = r + γ * Q(s', a_aktuell)   [aktuell gewählte Aktion]
Q-Learning: Target = r + γ * max(Q(s', :))     [beste Aktion]
```

### Empirische Beobachtung

| Metrik | SARSA | Q-Learning |
|:---|---:|---:|
| Return | 7.92 | 7.96 |
| Std Dev | ±0.21 | ±0.22 |
| Lerngeschwindigkeit | ~2000 Ep | ~2000 Ep |

**Unterschied ist minimal!** Beide konvergieren zu ähnlichen optimalen Policies.

**Warum?** Der Aktionsraum (6 Aktionen) ist klein genug, dass beide Algorithmen schnell genug explorieren und die optimale Policy finden. Q-Learning's theoretischer Vorteil wird erst bei größeren Aktionsräumen relevant.

---

## Monte Carlo vs. Temporal Difference (Kritischer Vergleich)

### Fundamentale Unterschiede

| Aspekt | Monte Carlo | TD (SARSA/Q-Learning) |
|:---|:---|:---|
| **Lernzeitpunkt** | Episode-Ende | Nach jedem Schritt |
| **Lernquelle** | Komplette Trajectories | Ein-Schritt Fehler + Bootstrap |
| **Konvergenz** | Langsam | Schnell |

### Warum MC so dramatisch schlechter performt

1. **Episodisches Lernen ist zu langsam**: MC braucht die komplette Episode → tardives Feedback
2. **Zu schneller ε-Decay**: Nach 15.000 Episoden ε=0.05 → Exploration stoppt, bevor gute Policy gelernt
3. **"Reward vergiftung"**: Eine schlechte Episode (-100) "vergiftet" viele Q-Werte lange Zeit
4. **TD-Bootstrap ist robust**: TD-Methoden korrigieren kontinuierlich → schnelle Fehlerwiederherstellung

**Resultat**: MC konvergiert zu suboptimaler Policy und bleibt dort stecken.

---

## Stabilitäts-Kennzahlen (alle 5 Seeds)

### Return

| Algoritmus | Mean | Std | Min | Max | CV | 
|:---|---:|---:|---:|---:|---:|
| MC | -123.23 | 18.67 | -147.45 | -97.84 | 15.15% |
| SARSA | 7.92 | 0.21 | 7.73 | 8.28 | 2.65% |
| Q-Learn | 7.96 | 0.22 | 7.64 | 8.28 | 2.76% |

### Episode Length

| Algoritmus | Mean | Std | Min | Max | CV | 
|:---|---:|---:|---:|---:|---:|
| MC | 130.94 | 16.78 | 108.13 | 152.70 | 12.82% |
| SARSA | 13.08 | 0.21 | 12.72 | 13.27 | 1.60% |
| Q-Learn | 13.04 | 0.22 | 12.72 | 13.36 | 1.69% |

---

## Antworten zu Aufgabe A3

**F1: Warum bleibt MC lange negativ?**
- MC lernt episodisch (nicht online). Zu schneller ε-Decay → Exploration stoppt zu früh.
- Konvergiert zu suboptimaler Policy. 130 Schritte pro Episode = ineffiziente Pfade → negative Returns.

**F2: Warum lernen SARSA/Q-Learning schneller online?**
- TD-Updates nach jedem Schritt, nicht Episode-Ende. Sofortiges Feedback → kontinuierliche Verbesserung.
- Nach ~2.000 Episoden stabile, gute Policy erreicht.

**F3: Warum ist Q-Learning aggressiver?**
- Q-Learning nutzt max(Q) vs. SARSA nutzt aktuelle Aktion → optimistischere TD-Targets.
- **Im Taxi-Problem minimal** (beide ~8 Return). Vorteil wächst mit Aktionsraum-Größe.

**F4: Warum sind SARSA/Q-Learning am Ende ähnlich?**
- Nach 20.000 Episoden haben beide konvergiert. Kleine Umgebung (6 Aktionen) → beide finden optimale Policy.
- Beide erreichen Return ≈ +8 und Length ≈ 13 Schritte.

---

## Zusammenfassung

**TD-Methoden dominieren MC:**
- 131 Punkte besserer Return
- 10× kürzere, effizientere Episodes  
- 88× stabiler (CV: 2.7% vs 15%)
- Seed-unabhängig und reproduzierbar

**SARSA vs. Q-Learning:** Praktisch identisch. Unterschied wächst mit Aktionsraum-Größe.

**Kernempfehlung:** TD-Methoden für Taxi-v3 klar überlegen. MC nicht produktionsgeeignet.
