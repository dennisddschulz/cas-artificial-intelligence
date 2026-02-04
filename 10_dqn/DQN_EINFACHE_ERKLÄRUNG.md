# DQN Code - Einfache Erklärung

## Was ist DQN?
DQN = "Deep Q-Network" - ein Algorithmus, der einem Computer beibringt, Spiele zu spielen, indem er lernt, welche Aktionen gut sind.

---

## Was macht der Code? (Schritt für Schritt)

### 1. **CartPole-Spiel** (das Ziel)
```
Ein Stab balanciert auf einem Wagen.
Der Computer muss entscheiden:
- Links fahren
- Rechts fahren
Ziel: Den Stab so lange wie möglich aufrecht halten
```

### 2. **Das Gehirn: QNet (das neuronale Netz)**
```
Input:  Position des Wagens, Winkel des Stabs, Geschwindigkeiten (4 Zahlen)
  ↓
Versteckte Schicht (128 Neuronen)
  ↓
Versteckte Schicht (128 Neuronen)
  ↓
Output: Wie gut ist "Links fahren"? Wie gut ist "Rechts fahren"? (2 Zahlen)
```

**Einfach gesagt**: Das Netz gibt Bewertungen für jede Aktion aus. Je höher die Bewertung, desto besser die Aktion.

---

### 3. **Zwei Netze: Online vs. Target**
```
q_online:  Das Netz, das ständig trainiert wird (lernt schnell)
q_target:  Eine alte Kopie von q_online (bleibt stabiler)
```

**Warum zwei?** Das Target-Netz ist wie ein "Ruherufer". Es stabilisiert das Lernen, damit das Online-Netz nicht zu wild herumspringt.

---

### 4. **Replay Buffer (das Gedächtnis)**
```
Der Computer speichert seine Erfahrungen:
- Was sah ich? (obs)
- Was tat ich? (action)
- Bekam ich Belohnung? (reward)
- Was sah ich danach? (obs2)
- War das Spiel vorbei? (done)
```

**Warum?** Später lernt der Computer aus diesen gespeicherten Erfahrungen in Batches, nicht nur aus aktuellen.

---

### 5. **Die Hauptschleife: Training**

#### **Schritt 1: Aktion wählen (Epsilon-Greedy)**
```python
if random.random() < eps:
    return random.randrange(n_actions)  # Zufall (Exploration)
else:
    return torch.argmax(Q(obs))         # Beste bekannte Aktion (Exploitation)
```
**Einfach**: Am Anfang wählt der Computer zufällig (lernt neue Dinge). Mit der Zeit nutzt er sein Wissen (beste Aktionen).

#### **Schritt 2: Speichern der Erfahrung**
```python
buffer.push(obs, a, r, obs2, done)
```
**Der Computer merkt sich: "Ich war in Zustand X, machte Aktion Y, bekam Belohnung Z"**

#### **Schritt 3: Aus der Vergangenheit lernen**
```
Nimm eine zufällige Batch von Erfahrungen aus dem Buffer
Berechne: "Was hätte ich lernen sollen?" (target)
Berechne: "Was dachte ich damals?" (q_sa)
Fehler = (target - q_sa)²
Lerne: Reduziere diesen Fehler!
```

---

### 6. **DQN vs. Double DQN (Unterschied)**

#### **DQN (Standard)**
```
Best-Aktion wählen:  a* = argmax Q_target(s')
Target berechnen:     target = r + γ * Q_target(s', a*)
```
**Problem**: Überschätzt oft die Qualität von Aktionen!

#### **Double DQN (Besser)**
```
Best-Aktion wählen:    a* = argmax Q_online(s')      ← nutze Online-Netz
Target berechnen:      target = r + γ * Q_target(s', a*) ← evaluiere mit Target-Netz
```
**Vorteil**: Weniger Überschätzung, stabiler!

---

### 7. **Bias-Messung (Diagnostik)**
```python
mean_maxQ = durchschnittliche beste Q-Werte
gap = (max Q_target) - (Q_target bei Online-Aktion)
```
**Was bedeutet das?**
- Wenn `gap` klein ist → DQN schätzt realistisch
- Wenn `gap` groß ist → DQN überschätzt

---

## Das Gesamtbild: Was passiert?

```
1. Computer spielt CartPole zufällig
   ↓
2. Speichert Erfahrungen (Buffer)
   ↓
3. Lernt aus Batch von Erfahrungen
   ↓
4. Q-Netz wird besser (predicts bessere Aktionen)
   ↓
5. Weniger zufällig, mehr gezielt spielen
   ↓
6. Nach 200.000 Schritte: Kann Stab lange aufrecht halten!
```

---

## Wichtige Parameter erklärt

| Parameter | Bedeutung | Beispiel |
|:---|:---|:---|
| `batch_size` | Wieviele Erfahrungen auf einmal lernen? | 64 |
| `lr` | Wie schnell lernen? | 0.001 (langsam) |
| `gamma` | Wie sehr zählen zukünftige Belohnungen? | 0.99 (sehr viel) |
| `eps_start` | Anfangs zufälligkeit | 1.0 (100%) |
| `eps_end` | Endzufälligkeit | 0.05 (5%) |
| `target_update_every` | Wie oft Kopie aktualisieren? | 1000 Schritte |

---

## Output des Codes

```
1. Episode Returns Plot
   → Zeigt: Wird der Computer besser?
   
2. Mean Max-Q Plot
   → Zeigt: Wie hoch schätzt der Computer Aktionen?
   
3. Gap Plot
   → Zeigt: Überschätzt DQN oder Double DQN mehr?
   
4. Evaluation
   → Misst: Wie gut ist die finale Policy?
   
5. GIFs
   → Zeigt: Der Computer beim Spielen!
```

---

## Zusammenfassung in 3 Sätzen

1. **DQN trainiert ein Netz, das sagt "wie gut ist diese Aktion"**
2. **Der Computer lernt, indem er gespeicherte Erfahrungen analysiert (Replay Buffer)**
3. **Double DQN ist besser, weil es weniger überschätzt**

