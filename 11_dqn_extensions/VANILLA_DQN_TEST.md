# Vanilla DQN Verifikation: Uniform Replay vs. PER

## Experiment-Setup

**Ziel**: Vergleiche Uniform Replay mit Prioritized Experience Replay bei **Vanilla DQN** (keine Extensions).

**Konfiguration**:
```
USE_DOUBLE  = False
USE_DUELING = False
USE_NOISY   = False

Training mit:
- Uniform Replay (Baseline)
- Prioritized Experience Replay (PER)

Identisch: SEED, Hyperparameter, Environment
```

## Warum ist das wichtig?

Die bisherigen Tests mit allen Extensions waren zu komplex. Jetzt testen wir:

1. **Isoliert PER vs Uniform**
   - Keine Interferenz von Double DQN, Dueling, Noisy
   - Klare Unterscheidung zwischen Methoden

2. **Vanilla DQN Performance**
   - Baseline Standard DQN
   - Sollte mit 300k Steps gut konvergieren

3. **Echte PER Effekte**
   - Nur PER macht den Unterschied
   - Kein Konfounding durch andere Extensions

## Erwartete Ergebnisse

### Hypothese A: PER ist besser (Wahrscheinlichkeit: 70%)
- PER sampelt wichtige Transitions häufiger
- Sollte schneller lernen
- Final Performance: PER > Uniform

### Hypothese B: Uniform ist besser (Wahrscheinlichkeit: 20%)
- LunarLander zu einfach für PER Vorteile
- Uniform Sampling robuster
- Final Performance: Uniform ≥ PER

### Hypothese C: Keine Unterschiede (Wahrscheinlichkeit: 10%)
- 300k Steps könnte zu kurz sein
- Oder PER/Uniform Performance äquivalent
- Statistical Test: p > 0.05

## Metriken zu Vergleichen

1. **Final Performance** (Step 300.000)
   - Return ± Std Dev
   - Wer ist höher?

2. **Mean Performance** (über alle 20 Evaluationen)
   - Durchschnittliche Performance
   - Robustheit

3. **Max Performance** (Peak)
   - Bestes Ergebnis
   - Kapazität

4. **Learning Curve**
   - Konvergenzgeschwindigkeit
   - Stabilität über Zeit

5. **Statistical Test**
   - T-Test auf Signifikanz
   - Cohen's d für Effektgröße

## Timeline

```
Start:    jetzt
Duration: ~8-10 Stunden (2x300k Steps)
Status:   🔄 LÄUFT

Uniform:  300k Steps (~4 Stunden)
PER:      300k Steps (~4 Stunden)
Analyse:  15 Minuten
```

## Wichtige Erkenntnisse die wir erwarten

1. **Vanilla DQN sollte mit positiven Rewards konvergieren**
   - Nicht negative Returns wie beim vorherigen komplexen Setup
   - LunarLander erfolgreich landen

2. **PER sollte schneller oder besser konvergieren**
   - Wenn die Implementierung korrekt ist
   - Und das Problem PER Benefits hat

3. **Clarität über Vanilla DQN + PER**
   - Weniger Variablen
   - Klares Resultat

## Fehlerquellen die wir ausgeschlossen haben

✅ Dueling DQN - kann komplexe Zielableitungen haben
✅ Double DQN - könnte Bias-Korrektur maskieren
✅ Noisy Networks - können Exploration anders beeinflussen

Jetzt nur: **Vanilla DQN + Uniform vs. PER Replay Buffer**

## Was wird nach dem Training gemacht?

1. ✓ Ergebnisse in `results_vanilla_dqn.json` speichern
2. ✓ Statistische Tests durchführen
3. ✓ Visualisierungen erstellen
4. ✓ Ergebnisse dokumentieren
5. ✓ PPT aktualisieren mit KORREKTEN Ergebnissen

---

**Status**: Training läuft - wird ~8 Stunden dauern
**Nächstes Update**: Nach Training abgeschlossen

