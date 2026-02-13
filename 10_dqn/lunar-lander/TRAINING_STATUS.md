# LunarLander DQN Training - Status & Anleitung

## ✅ Was wurde erledigt:

### 1. **Angepasster DQN Code für LunarLander-v3**
   - ✅ `train_lunar_lander.py` erstellt (Original mit 500k Steps)
   - ✅ `train_lunar_lander_fast.py` erstellt (Schnelle Version mit 200k Steps)
   - ✅ Environment aktualisiert von v2 zu v3
   - ✅ Netzwerk-Größe erhöht (128 → 256 hidden units für komplexere Aufgabe)
   - ✅ Hyperparameter angepasst für LunarLander

### 2. **Automatische Visualisierungen**
   Das Skript generiert automatisch:
   - ✅ `01_episode_returns.png` - Episode Returns Vergleich
   - ✅ `02_mean_max_q.png` - Mean Max Q-Values über Training
   - ✅ `03_gap.png` - Overestimation Gap Analyse
   - ✅ `04_episode_lengths.png` - Episode Length Trending
   - ✅ `results_summary.txt` - Textausgabe mit Statistiken

### 3. **Evaluation**
   Nach dem Training werden beide Modelle mit 20 Episoden im Greedy Mode evaluiert
   und die Ergebnisse verglichen.

---

## 🚀 Trainings-Status:

### Schnelles Training (`train_lunar_lander_fast.py`)
- **Steps**: 200,000 
- **Status**: ✅ **LÄUFT JETZT IM HINTERGRUND!**
- **Start Zeit**: ~16:00 Uhr
- **Erwartete Fertigstellung**: ~18:00 Uhr (2 Stunden später)
- **Output-Verzeichnis**: `/home/isc-den/cas-artificial-intelligence/10_dqn/lunar-lander/plots/`
- **Log-Datei**: `/home/isc-den/cas-artificial-intelligence/10_dqn/lunar-lander/training.log`

**Um Status zu überprüfen:**
```bash
tail -50 /home/isc-den/cas-artificial-intelligence/10_dqn/lunar-lander/training.log
ps aux | grep train_lunar_fast


---

## 📊 Erwartete Ergebnisse:

Im Vergleich zu CartPole sollten wir sehen:

### **DQN vs Double DQN in LunarLander:**
- **DQN**: Aggressives Q-Value Learning, schnellerer anfänglicher Lernfortschritt
- **Double DQN**: Konservativere Q-Schätzungen, stabilere langfristige Performance

### **Wichtiger Unterschied zu CartPole:**
- **CartPole**: Einfache 2-Aktion Umgebung → DQN war hier besser
- **LunarLander**: Komplexe 4-Aktion Umgebung mit kontinuierlichen Zuständen → **Double DQN sollte hier besser sein!**

**Grund**: In komplexeren Umgebungen wird die Overestimation-Reduktion von Double DQN wichtig.

---

## 🎯 Konfiguration des Fast-Trainings:

```python
Total Steps:              200,000
Batch Size:              64
Learning Rate:           0.0005
Gamma:                   0.99
Target Update Frequency: 5,000 steps
Epsilon Decay:           100,000 steps
Start Learning:          5,000 steps
```

---

## 📈 Was die Plots zeigen werden:

### **Plot 1: Episode Returns**
- Y-Achse: Return pro Episode (Moving Average window=50)
- X-Achse: Episode Nummer
- Erwartet: Beide Algorithmen sollten von ca. -200 zu ca. +100-200 gehen
- Success Threshold: +200 (offizielle LunarLander-v3 Erfolgs-Schwelle)

### **Plot 2: Mean Max Q-Values**
- Zeigt durchschnittliche maximale Q-Werte über Trainingsverlauf
- DQN: Höhere Werte (optimistischer)
- Double DQN: Niedrigere Werte (realistischer)

### **Plot 3: Overestimation Gap**
- Gap = max(Q(s', :)) - Q(s', a_genommen)
- DQN: Größerer Gap (mehr Overestimation)
- Double DQN: Kleinerer Gap (weniger Overestimation)

### **Plot 4: Episode Lengths**
- Zeigt wie lange Episodes sind über Zeit
- Kurze Episodes = gutes Landen = höhere Returns
- Erwartet: Beide sollten von 500-1000 Schritten zu 100-300 Schritten gehen

---

## 📁 Output-Verzeichnis:

```
/home/isc-den/cas-artificial-intelligence/10_dqn/lunar-lander/plots/
├── 01_episode_returns.png      (4 plot lines: 2 algos × thin/mean)
├── 02_mean_max_q.png            (2 curves mit markers)
├── 03_gap.png                   (2 curves mit markers)
├── 04_episode_lengths.png       (2 curves)
└── results_summary.txt          (Statistiken & Evaluationsergebnisse)
```

---

## 🔍 Wie man die Plots überprüft:

```bash
# Nach fertigstellung:
ls -lh /home/isc-den/cas-artificial-intelligence/10_dqn/lunar-lander/plots/
cat /home/isc-den/cas-artificial-intelligence/10_dqn/lunar-lander/plots/results_summary.txt
```

---

## ⚙️ Technische Details:

### **Umgebung:**
- **LunarLander-v3**: 
  - Observation Space: 8 Dimensionen (x, y, vx, vy, angle, angular_velocity, left_leg_contact, right_leg_contact)
  - Action Space: 4 diskrete Aktionen (noop, fire left, fire main, fire right)
  - Rewards: -0.3 für jeden Schritt, -1 für crash, +0 bis +10 für sanfte Landung

### **Q-Network Architektur:**
```
Input (8) → Linear → ReLU → Linear → ReLU → Linear → Output (4)
           ↓         ↓      ↓       ↓      ↓       ↓
          256       ReLU   256     ReLU   256      4 Q-values
```

### **Training Loop:**
1. Epsilon-greedy Action Selection
2. Buffer Storage (Replay Memory)
3. Batch Sampling & Training
4. Target Network Update (alle 5k Steps)
5. Bias Metrics Berechnung

---

## ✨ Nächste Schritte:

1. **Warten auf fertigstellung** (ca. 10-15 Min)
2. **Plots überprüfen** - sollten zeigen, dass Double DQN besser ist
3. **Ergebnisse analysieren** - warum ist es anders als CartPole?
4. **Optional**: Längeres Training mit 500k Steps für bessere Konvergenz

---

**Status**: 🟢 Training läuft - Plots werden bald verfügbar!

