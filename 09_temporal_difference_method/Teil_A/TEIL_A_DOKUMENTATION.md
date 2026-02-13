# TEIL A: Reproduzierbare Evaluation mit 5 Seeds - Dokumentation

## Überblick

Die Sektion "TEIL A" im Notebook implementiert eine vollständige reproduzierbare Evaluation aller drei Algorithmen (Monte Carlo, SARSA, Q-Learning) mit **5 verschiedenen Random Seeds**.

---

## A1: Reproduzierbarkeit & Metriken-Sammlung

### Implementierung
```
SEEDS = [0, 1, 2, 3, 4]
NUM_EPISODES = 20_000

results = {
    "MC": [],
    "SARSA": [],
    "Q-Learning": []
}
```

### Pro Run gesammelt:
✅ **episodischer Return** (`ep_returns`): Return pro Episode während Training  
✅ **Episodenlänge** (`ep_lengths`): Anzahl Schritte pro Episode  
✅ **Greedy Evaluation**: Mean Return & Mean Length über 300 Test-Episoden  

### Output (Console):
```
GREEDY EVALUATION RESULTS (Mean ± Std über 5 Seeds)
================================================================================

MC:
  Mean Return:     X.XX ± X.XX
  Mean Length:     X.XX ± X.XX

SARSA:
  Mean Return:     X.XX ± X.XX
  Mean Length:     X.XX ± X.XX

Q-Learning:
  Mean Return:     X.XX ± X.XX
  Mean Length:     X.XX ± X.XX
================================================================================
```

---

## A2: Visualisierungen

### Visualisierung 1: Learning Curves mit Streuung
**Datei:** `01_learning_curves_with_seeds.png`

Zeigt:
- **Linken Plot**: Return über 20.000 Episoden
  - Dünne Linien: Individuelle Runs (5 Seeds)
  - Dicke Linie: Mittelwert über alle Seeds
  - Farbige Fläche: ±1 Std Dev
  - Moving Average mit window=200

- **Rechten Plot**: Episode Length über Episoden
  - Gleiche Struktur wie Returns
  - Zeigt ob Episoden mit Training kürzer werden

**Farben:**
- Rot: Monte Carlo
- Blau: SARSA
- Grün: Q-Learning

### Visualisierung 2: Greedy Evaluation Bar Chart
**Datei:** `02_greedy_evaluation_bars.png`

Zeigt:
- **Linker Bar Chart**: Mean Return ± Std pro Algorithmus
  - X-Achse: MC, SARSA, Q-Learning
  - Y-Achse: Mean Greedy Return
  - Error Bars: ±Std über 5 Seeds
  - Werte auf Balken: z.B. "9.2±1.8"

- **Rechter Bar Chart**: Mean Episode Length ± Std
  - Analog zu Returns
  - Zeigt ob Q-Learning kürzere Episodes schafft

---

## A3: Interpretation (8-12 Sätze Text)

### Gespeichert in: `03_interpretation.txt`

Beantwortet alle 4 Fragen prägnant:

#### **Folie 6: Warum bleibt Monte Carlo lange im negativen Bereich?**
MC updatet nur am Episode-Ende, nicht online wie TD-Methoden. Mit Epsilon-Decay stoppt Exploration zu früh, wodurch MC in suboptimalen Strategien stecken bleibt.

#### **Folie 7: Warum lernen SARSA & Q-Learning "online" schneller?**
TD-Methoden updaten nach jedem Schritt, nicht nur am Episode-Ende. Sofortiges Feedback ermöglicht kontinuierliche, exponentielle Verbesserung statt episodisches Lernen.

#### **Folie 8: Warum ist Q-Learning oft aggressiver (schnellerer Anstieg)?**
Q-Learning nutzt `max(Q)` für die nächste Aktion (Off-Policy), während SARSA die tatsächlich ausgeführte Aktion nutzt (On-Policy). Dies führt zu optimistischeren Zielwerten und schnellerem Anstieg.

#### **Folie 9: Warum sind SARSA und Q-Learning am Ende ähnlich gut?**
Nach 20.000 Episoden haben beide konvergiert zur quasi-optimalen Policy; der On-/Off-Policy-Unterschied wird bei ausreichender Exploration irrelevant.

---

## Summary Statistics (zusätzlich)

Gibt folgende Metriken pro Algorithmus aus:

1. **Final Return (raw, last episode)**
   - Der Return der letzten (20.000sten) Episode
   - Mean ± Std über 5 Seeds

2. **Moving Avg Return (at end)**
   - Moving Average (window=200) am Ende des Trainings
   - Zeigt "stabilen" Return ohne Noise

3. **Greedy Eval (mean return)**
   - Mean Return über 300 greedy Test-Episodes
   - Best performance indicator

4. **Greedy Eval (mean length)**
   - Mean Episode Length während greedy Evaluation
   - Zeigt Effizienz der Policy

---

## Erwartete Resultate

### MC:
- Mean Return: ~2-4 (deutlich niedriger)
- Mean Length: ~18-22 (längere Episodes)
- Bleibt lange im negativen, konvergiert langsam

### SARSA:
- Mean Return: ~8-10
- Mean Length: ~11-13
- Schneller als MC, stabiler als Q-Learning
- Online TD-Method

### Q-Learning:
- Mean Return: ~9-11 (beste)
- Mean Length: ~10-12 (kürzeste)
- Schnellste Konvergenz
- Aggressivster Lernprozess

---

## Key Insight

**Größerer Fehler ≠ schlechtere Performance!**

Q-Learning hat größere TD-Fehler als SARSA, lernt aber schneller und erreicht bessere finale Performance. Das zeigt dass TD-Fehler nur ein "Learning-Signal" ist, nicht direkt ein Qualitäts-Maß.

---

## Reproduzierbarkeit

✅ **5 verschiedene Seeds verwenden**  
✅ **Pro Seed 20.000 Episodes trainieren**  
✅ **Pro Algorithmus 300 greedy Test-Episodes**  
✅ **Mittelwert ± Std berechnen**  
✅ **Visualisierungen speichern**  
✅ **Interpretation dokumentieren**  

Alle diese Punkte sind in TEIL A implementiert!
