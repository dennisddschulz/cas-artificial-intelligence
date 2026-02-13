# 🚀 PRIORITIZED EXPERIENCE REPLAY (PER) ASSIGNMENT - EXECUTION GUIDE

## 📌 Überblick

Sie haben erhalten:

```
✅ PER_Assignment.ipynb              - Hauptnotebook (Code + Experimente)
✅ PER_Presentation.pptx             - PowerPoint (22 Slides)  
✅ PER_ASSIGNMENT_README.md          - Detaillierte Dokumentation
✅ PER_CHEAT_SHEET.md               - Quick Reference Guide
✅ EXECUTION_GUIDE.md               - Diese Datei
```

**Status:** Bereit zum Ausführen ✓

---

## ⚡ SCHNELLSTART (5 Minuten)

### Schritt 1: Notebook öffnen
```bash
cd /home/isc-den/cas-artificial-intelligence/11_dqn_extensions
jupyter notebook PER_Assignment.ipynb
```

### Schritt 2: Alle Zellen ausführen
```
Menü → Cell → Run All
oder: Keyboard Shortcut Ctrl+A dann Shift+Enter
```

### Schritt 3: Warten
⏱️ Training dauert ~2-4 Minuten (zwei 300k-step Trainings)

### Schritt 4: Ergebnisse ansehen
- Plots werden automatisch angezeigt
- Analyse wird automatisch gedruckt
- PPT ist bereits erstellt

### Schritt 5: Präsentation vorbereiten
```bash
Öffne: PER_Presentation.pptx
```

---

## 📊 WAS PASSIERT WÄHREND TRAINING?

### Experiment 1: Uniform Replay
```
Training für 300,000 Schritte...
Evaluation alle 15,000 Schritte (20 Evaluationen)
Ergebnis: Stabile aber langsamere Konvergenz
```

### Experiment 2: Prioritized Experience Replay  
```
Training für 300,000 Schritte...
Mit Prioritäten basierend auf TD-Error
Ergebnis: Schnellere Konvergenz + bessere Performance
```

### Generierte Visualisierungen
Nach Abschluss werden automatisch erzeugt:
1. `plot_01_eval_return.png` - Hauptvergleich
2. `plot_02_training_loss.png` - Loss-Verlauf
3. `plot_03_td_error.png` - TD-Error Dynamik
4. `plot_04_episode_returns.png` - Episode Returns
5. `plot_05_convergence_comparison.png` - Konvergenz-Speed
6. `plot_06_distribution.png` - Return Verteilung
7. `plot_07_metrics_table.png` - Metrik-Tabelle
8. `plot_08_learning_efficiency.png` - Learning Efficiency

---

## 🎯 NOTEBOOK-NAVIGATION

### TEIL 1: Konzeptionelle Erklärung
**Lesen:** Theoretischer Überblick
**Action:** Keine (nur Markdown)

### TEIL 2-4: Setup & Imports
**Lesen:** Notwendige Imports
**Action:** Zelle ausführen (automatisch)

### TEIL 5-6: Implementierung
**Lesen:** Buffer-Implementierungen & Code
**Action:** Zelle ausführen (automatisch)

### TEIL 7-9: Training ⏱️ (Dauert ~2-4 Min)
**Wichtig:** 
- Experiment 1 startet mit Uniform Replay
- Experiment 2 startet mit PER
- Progress wird angezeigt via `tqdm`

**Was passiert:**
```
[Experiment Name]: [====>                    ] X%
```

### TEIL 10: Visualisierungen 🎨
**Action:** Automatisch generiert
**Ergebnis:** 8 hochwertige Plots

### TEIL 11: Analyse & Statistik 📊
**Lesen:** Automatisch berechnete Metriken
**Ergebnis:** Quantitative Vergleiche

### TEIL 12: Reflexion 🧠
**Lesen:** Detaillierte Diskussion
**Nutzen:** Für Präsentation verwenden

---

## 📝 WÄHREND DER AUSFÜHRUNG

### Progress-Anzeige
```
[Uniform Replay (Baseline)]: 78%|████████▊  | 234567/300000 [01:23<00:34, 1923 episodes/s]

[step  15000] eval return:  150.5 ± 45.2 | buffer:  15000 | last_loss: 0.0234
[step  30000] eval return:  175.3 ± 38.1 | buffer:  30000 | last_loss: 0.0145
...
```

### Was bedeuten die Ausgaben?
- `78%`: Prozentuale Vollendung
- `234567/300000`: Aktuelle Steps / Total Steps
- `01:23`: Verstrichene Zeit
- `34s`: Geschätzte verbleibende Zeit
- `eval return: 150.5 ± 45.2`: Durchschnitt ± Standardabweichung
- `buffer: 15000`: Aktuelle Buffer-Größe

### Normal Verhalten
```
✓ Loss sinkt über Zeit
✓ Eval Return steigt
✓ TD-Error sinkt
✓ Training wird langsamer (Buffer wächst)
```

### Warnsignale ⚠️
```
✗ Loss wächst → Problem mit Hyperparametern
✗ Eval Return fällt → Zu aggressives Training
✗ Out of Memory → Buffer zu groß
✗ NaN Werte → Numerische Instabilität
```

---

## 🎨 NACH DEM TRAINING

### Automatisch generierte Plots
Alle Plots sind hochauflösend (300 dpi) und publikationsreif

**Interaktive Anzeige:**
- Hovern für Werte
- Zoom mit Scroll
- Speichern via Rechtsklick

### Interpretation der Plots

#### Plot 1: Evaluation Return
```
Y-Achse: Durchschnittliche Rewards
X-Achse: Trainingsschritte

✓ Kurve sollte ansteigen
✓ Mit Fehlerbalken (±Std)
✓ PER typischerweise steiler anfangs
```

#### Plot 2: Training Loss
```
Y-Achse: Loss (log-Skala!)
X-Achse: Update Steps

✓ Loss sollte fallen
✓ Geglättet mit Moving Average (window=50)
✓ Kleine Oszillationen normal
```

#### Plot 3: TD-Error
```
Y-Achse: Mittlerer absoluter TD-Error
X-Achse: Update Steps

✓ TD-Error sollte fallen
✓ PER typischerweise schneller fallend
✓ Zeigt "Agent lernt"
```

---

## 📊 ANALYSE & INTERPRETATION

### Automatisch berechnete Statistiken
```
UNIFORM REPLAY (BASELINE):
  Final Return:        XXX.XX ± YY.YY
  Max Return:          ZZZ.ZZ
  Avg Episode Return:  XXX.XX ± YY.YY
  Training Time:       XXX.X s
  Avg Loss:            X.XXXXXX
  Avg TD-Error:        X.XXXXXX

PRIORITIZED EXPERIENCE REPLAY (PER):
  Final Return:        YYY.YY ± ZZ.ZZ
  Max Return:          WWW.WW
  Avg Episode Return:  YYY.YY ± ZZ.ZZ
  Training Time:       YYY.Y s
  Avg Loss:            Y.YYYYYY
  Avg TD-Error:        Y.YYYYYY

PER IMPROVEMENT: +X.X% (oder -X.X% wenn Uniform besser)
```

### Statistische Tests
```
T-Test: p-value = X.XXX
→ Unterschied ist ['signifikant' wenn p<0.05 else 'nicht signifikant']

Cohen's d = X.XXX
→ Effektgröße: ['negligible', 'small', 'medium', 'large']
```

### Was bedeutet das?
- **p < 0.05**: Unterschied ist statistisch signifikant ✓
- **Cohen's d > 0.5**: Großer praktischer Unterschied ✓
- **Beides positiv**: PER ist klar besser ✓

---

## 💾 OUTPUT-DATEIEN

Nach Ausführung befinden sich folgende Dateien im Verzeichnis:

```
/home/isc-den/cas-artificial-intelligence/11_dqn_extensions/

📊 Visualisierungen (PNG):
   plot_01_eval_return.png              (1000x600px)
   plot_02_training_loss.png            (1000x600px)
   plot_03_td_error.png                 (1000x600px)
   plot_04_episode_returns.png          (1000x600px)
   plot_05_convergence_comparison.png   (1400x600px)
   plot_06_distribution.png             (1600x600px)
   plot_07_metrics_table.png            (1400x600px)
   plot_08_learning_efficiency.png      (1000x600px)

📄 Daten:
   results_summary.json                 (Alle Metriken)
   
📑 Dokumentation:
   PER_Assignment.ipynb                 (Hauptnotebook)
   PER_Presentation.pptx                (Präsentation)
   PER_ASSIGNMENT_README.md             (Detaillierte Docs)
   PER_CHEAT_SHEET.md                   (Quick Reference)
```

---

## 🔧 ANPASSUNGEN & DEBUGGING

### Wenn Training zu lange dauert

**Option 1: Weniger Schritte**
```python
TOTAL_STEPS = 150_000  # statt 300_000 (2x schneller)
```

**Option 2: Weniger häufige Evaluation**
```python
EVAL_EVERY = 30_000  # statt 15_000 (2x schneller Eval)
```

**Option 3: Kleinerer Batch**
```python
BATCH_SIZE = 128  # statt 256 (etwas schneller)
```

### Wenn Training zu schnell trainiert (evtl. instabil)

**Reduziere Learning Rate:**
```python
LR = 1e-3  # statt 2e-3
```

**Oder erhöhe Gradient Clipping:**
```python
GRAD_CLIP_NORM = 15.0  # statt 10.0
```

### Wenn PER instabil wird

**Reduziere Alpha:**
```python
PER_ALPHA = 0.4  # statt 0.6 (weniger aggressiv)
```

**Oder starte Beta höher:**
```python
PER_BETA_START = 0.6  # statt 0.4
```

---

## 📋 CHECKLISTE VOR PRÄSENTATION

- [ ] Notebook wurde komplett ausgeführt
- [ ] Keine Error-Meldungen
- [ ] 8 Plots wurden generiert
- [ ] results_summary.json existiert
- [ ] PowerPoint existiert und hat 22 Slides
- [ ] Alle Dateien sind lesbar
- [ ] Ich verstehe die Ergebnisse
- [ ] Ich kann die Plots erklären
- [ ] Ich habe die Cheat Sheet gelesen
- [ ] Ich bin bereit zur Präsentation

---

## 🎓 LERNZIELE ÜBERPRÜFUNG

Nach dem Durcharbeiten können Sie erklären:

- [ ] Was Experience Replay ist
- [ ] Warum Uniform Replay suboptimal ist
- [ ] Wie PER Prioritäten berechnet
- [ ] Was α und β Parameter bedeuten
- [ ] Wie SumTree funktioniert (O(log n))
- [ ] Was Importance Sampling Weights sind
- [ ] Empirische Vorteile von PER (schneller, besser)
- [ ] Nachteile von PER (komplex, tuning-intensiv)
- [ ] Wann PER hilft und wann nicht
- [ ] Praktische Hyperparameter

Wenn ≥8/10: Sie sind ready! 🚀

---

## 📞 HÄUFIGE PROBLEME & LÖSUNGEN

### Problem 1: "ModuleNotFoundError: gymnasium"
**Lösung:**
```bash
pip install gymnasium torch numpy matplotlib seaborn scipy pandas
```

### Problem 2: "CUDA out of memory"
**Lösung:**
```python
BATCH_SIZE = 128  # Reduzieren
BUFFER_SIZE = 100_000  # Reduzieren
device = torch.device('cpu')  # CPU statt GPU
```

### Problem 3: "Plots werden nicht angezeigt"
**Lösung:**
```python
# Am Anfang hinzufügen:
import matplotlib
matplotlib.use('Agg')  # Alternative Backend
import matplotlib.pyplot as plt
```

### Problem 4: "Training divergiert (Loss = NaN)"
**Lösung:**
```python
PER_ALPHA = 0.4  # Weniger aggressiv
GRAD_CLIP_NORM = 20.0  # Stärker clipping
LR = 1e-3  # Niedriger
```

### Problem 5: "Kein Unterschied zwischen Uniform und PER"
**Lösung:**
- LunarLander könnte zu einfach sein
- Versuche schwierigere Umgebung (z.B. Atari)
- Oder: Es bedeutet Uniform ist schon gut genug! ✓

---

## ⏱️ ZEITSCHÄTZUNG

| Aktivität | Zeit |
|-----------|------|
| **Notebook ausführen** | 2-4 Min |
| **Training Experiment 1** | 1-2 Min |
| **Training Experiment 2** | 1-2 Min |
| **Plots generieren** | 30 Sec |
| **Alle Zellen insgesamt** | 3-5 Min |
| **Ergebnisse interpretieren** | 5-10 Min |
| **Präsentation vorbereiten** | 10-20 Min |
| **GESAMT** | ~20-30 Min |

---

## 🎯 NÄCHSTE SCHRITTE

### Nach erfolgreichem Training:

1. **Verstehen Sie die Ergebnisse**
   - Lesen Sie die automatisch generierten Analysen
   - Schauen Sie die Plots an
   - Vergleichen Sie Uniform vs. PER

2. **Vorbereitung zur Präsentation**
   - Öffnen Sie PER_Presentation.pptx
   - Passen Sie an wenn nötig
   - Merken Sie sich die Key Points

3. **Vertiefte Analyse (optional)**
   - Modifizieren Sie Hyperparameter
   - Führen Sie zusätzliche Experimente durch
   - Erstellen Sie eigene Visualisierungen

4. **Dokumentation**
   - Notizen machen
   - Screenshots machen
   - Ergebnisse dokumentieren

---

## 📚 WEITERE RESSOURCEN

**Im Verzeichnis:**
- `PER_ASSIGNMENT_README.md` - Detaillierte Dokumentation
- `PER_CHEAT_SHEET.md` - Schnelle Referenz
- `DQN_Extensions.ipynb` - Original Notebook

**Online:**
- Schaul et al., "Prioritized Experience Replay" (2015)
- Sutton & Barto, "Reinforcement Learning" (2018)
- OpenAI Spinning Up in Deep RL

---

## ✅ FERTIG?

Wenn alles lief:
1. ✅ Notebook wurde ausgeführt
2. ✅ Plots wurden generiert
3. ✅ Ergebnisse wurden analysiert
4. ✅ Sie verstehen die Unterschiede
5. ✅ PowerPoint ist vorbereitet

**Dann: Sie sind ready für die Präsentation!** 🎉

---

**Viel Erfolg!** 🚀

Bei Fragen: Schauen Sie in:
- `PER_CHEAT_SHEET.md` für schnelle Antworten
- `PER_ASSIGNMENT_README.md` für detaillierte Erklärungen
- Notebook-Zellen für den Code

