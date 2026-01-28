# On-Policy (SARSA) vs Off-Policy (Q-Learning) - Vergleich
## Prio 1: Detaillierte Analyse

### Executive Summary

Die Analyse vergleicht **SARSA** (On-Policy) und **Q-Learning** (Off-Policy) auf der Taxi-v3 Umgebung. Der **WICHTIGSTE UNTERSCHIED** liegt in der **Target Bestimmung** beim TD Update.

---

## 1. TARGET BESTIMMUNG - DIE KERNIDEE

### SARSA (On-Policy)
```
Q(s,a) += alpha * [r + gamma * Q(s', a') - Q(s,a)]
                               ↑
                    ACTUAL next action from policy
```

**Was passiert:**
- Die nächste Aktion `a'` wird **tatsächlich** via ε-greedy Policy gewählt
- Wenn `a'` eine schlechte Aktion ist (zufällig exploriert), wird dies in den Q-Wert eingerechnet
- Q-Werte reflektieren die Werte **unter der aktuellen explorierenden Policy**
- Konservativ: Berücksichtigt das Risiko von Exploration

**Beispiel:**
- Agent steht vor Klippe (wird erforscht)
- Mit Prob 0.2 wird zufällig zum Sprung gewählt
- SARSA: "Diese Aktion hat 20% Chance zu fallen → schlechter Wert"
- Q-Wert wird konservativ trainiert

---

### Q-Learning (Off-Policy)
```
Q(s,a) += alpha * [r + gamma * max Q(s', ·) - Q(s,a)]
                               ↑
                    BEST possible action, regardless of policy
```

**Was passiert:**
- Die nächste Aktion ist **IMMER** die beste bekannte Aktion
- Es spielt keine Rolle, welche Aktion die Policy tatsächlich wählt
- Q-Werte reflektieren die Werte **unter der optimalen möglichen Policy**
- Aggressiv: Ignoriert Explorations-Risiken

**Beispiel:**
- Agent steht vor Klippe (wird erforscht)
- Q-Learning: "Beste Aktion ist zu springen + landen → guter Wert"
- Trainiert optimistisch, ignoriert dass Agent zufällig springen könnte
- Policy kann zu aggressiv werden

---

## 2. EMPIRISCHE ERGEBNISSE

### Performance-Vergleich (Taxi-v3, 5000 Episoden)

```
                           SARSA (On-Policy)    Q-Learning (Off-Policy)
───────────────────────────────────────────────────────────────────────
Greedy Return              4.96                 7.64 ✓ (54% besser)
Average Steps              15.76                13.36 ✓ (15% schneller)
Mean |TD Error|            3.58                 1.34 ✓ (63% stabiler)
Max TD Error               39.52                20.00 (Q-Learning begrenzter)
```

### Interpretation:
- **Q-Learning lernt schneller** → optimale Policy wird früher erreicht
- **SARSA stabiler** → kleinere TD-Fehler, weniger Overoptimism
- **Q-Learning effizienter** → weniger Schritte zum Ziel (bessere Lösung)

---

## 3. TD-ERROR ANALYSE

### SARSA:
- **Größere Fehler** (Max: 39.5, Mean: 3.58)
- Grund: Verwendet suboptimale nächste Aktionen (Exploration)
- Aber: Stellt sich selbst schnell korrigiert

### Q-Learning:
- **Kleinere Fehler** (Max: 20.0, Mean: 1.34)
- Grund: Verwendet immer beste Aktion → weniger Variabilität
- Vorteil: Glattere Lernkurve

---

## 4. WANN WELCHER ALGORITHMUS?

### SARSA (On-Policy) nutzen, wenn:
✓ **Sicherheit kritisch** (Robotik, autonomes Fahren)
   - Exploration kann teuer/gefährlich sein
   - Beispiel: Roboter testet nicht gerne "Grenzen aus"

✓ **Konservative Lösung gewünscht**
   - Lieber funktionierend als optimal
   - Beispiel: Finanzielle Entscheidungen

✓ **Online Learning mit Live-Feedback**
   - Policy wird direkt genutzt
   - Exploration und Exploitation sind verschmolzen

### Q-Learning (Off-Policy) nutzen, wenn:
✓ **Optimale Policy das Ziel**
   - Man will das Beste, nicht das Sichere
   - Beispiel: Game-Playing, Benchmarks

✓ **Simulation / Safe Environment**
   - Exploration ist günstig (replay, simulation)
   - Keine echten Konsequenzen

✓ **Schneller Lernen wichtig**
   - Q-Learning konvergiert typisch schneller
   - Beispiel: Training mit Budget-Beschränkungen

✓ **Off-Policy-Nutzen**
   - Man kann alte/andere Policies replays nutzen
   - Q-Learning kann davon lernen, SARSA nicht

---

## 5. DIE MATHEMATISCHE INTUITION

### Szenario: Agent vor Klippe (wird erforscht)

**State Werte:**
- Q(klippe, safe) = 5.0 (sicher)
- Q(klippe, jump) = ? (wird erforscht)
- Q(next_state) = 3.0 (Durchschnitt)

**SARSA sieht:**
```
Mit ε=0.3 wird a' mit Prob 0.3 zufällig gewählt
Expected next value = 0.7 * 5.0 + 0.3 * 3.0 = 4.3
→ Conservative: Klippe ist ein moderates Risiko
```

**Q-Learning sieht:**
```
Best possible action: a' = safe → Q = 5.0
→ Aggressive: Klippe ist sicher, wenn man optimal spielt
```

**Resultat:**
- SARSA: Q(klippe, jump) ≈ 2.0 (vorsichtig)
- Q-Learning: Q(klippe, jump) ≈ 4.0 (optimistisch)
- Policy: Q-Learning wird eher springen

---

## 6. BIAS-VARIANZ TRADEOFF

|  | Bias | Varianz | Sample Eff. | Konvergenz |
|---|---|---|---|---|
| **SARSA** | Unbiased (unter Policy π) | Hoch | Mittel | Langsam |
| **Q-Learning** | Biased (Overestimation risk) | Niedrig | Gut | Schnell |
| **MC** | Unbiased (wahre Return) | Sehr hoch | Schlecht | Sehr langsam |
| **TD** | Leicht biased | Niedrig | Gut | Schnell |

---

## 7. PRAKTISCHE BEISPIELE

### SARSA würde man nehmen für:
1. **Autonome Fahrzeuge**
   - Spielt es safe, auch unter Exploration
   
2. **Finanzielle Trading-Agents**
   - Konservativ, um Verluste zu vermeiden
   
3. **Medizinische Entscheidungen**
   - Lieber bewährt als optimal

### Q-Learning würde man nehmen für:
1. **Game Playing (AlphaGo, Chess)**
   - Optimale Strategie ist alles
   
2. **Robotik in Simulation**
   - Training in sicherer Umgebung, deployment später
   
3. **Resource Allocation / Optimization**
   - Maximum Return ist das Ziel
   
4. **Offline RL**
   - Alte Daten replays ohne Policy-Abhängigkeit

---

## 8. KEY TAKEAWAYS

### 1. **Target Bestimmung ist ALLES**
   - SARSA: `Q(s', a')` ← schließt Explorations-Risiken ein
   - Q-Learning: `max Q(s', ·)` ← ignoriert Explorations-Risiken

### 2. **On-Policy vs Off-Policy ist eine Philosophical Choice**
   - On-Policy: "Lernen aus dem, was ich tue"
   - Off-Policy: "Lernen aus dem, was ich hätte tun können"

### 3. **Praktischer Impact**
   - Q-Learning: 54% bessere Returns, 15% schneller
   - SARSA: 63% stabilere Updates
   - Trade-off zwischen Speed und Stability

### 4. **Kein universelles "Besser"**
   - Q-Learning "optimal" aber zu optimistisch
   - SARSA "realistisch" aber konservativ
   - **Wahl hängt von Use-Case ab**

---

## 9. REFERENZEN ZUR DATEI

Die komplette Analyse mit Code und Plots ist in der Jupyter Notebook enthalten:
- `/09_temporal_difference_method/09_Temporal_Difference_Method.ipynb`
- `/09_temporal_difference_method/analysis_sarsa_vs_qlearning.py` (standalone Python)
- `/09_temporal_difference_method/01_SARSA_vs_QLearning_Comparison.png` (Plots)

---

**Erstellt:** 2025-01-28  
**Umgebung:** Taxi-v3 (Gymnasium)  
**Episoden:** 5000  
**Seed:** 42
