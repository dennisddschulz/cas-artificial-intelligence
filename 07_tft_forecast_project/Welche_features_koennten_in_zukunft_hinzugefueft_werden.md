Gerne — hier ist die **präzise deutsche Übersetzung**, sauber strukturiert und fachlich korrekt formuliert.

---

## ### Plan: Erweiterte Feature-Engineering-Strategie für einen stabilen Index (SMI)

#### **Ziel**

Verbesserung des prädiktiven Signals für einen relativ stabilen, niedrig-volatilen Aktienindex durch die Priorisierung glatter, rauschärmerer und „leakage-sicherer“ Kovariaten, die Trend, langsame Saisonalität, Regimewechsel sowie Momentum/Volatilität über lange Horizonte abbilden — und gleichzeitig übermäßig reaktive, laute Indikatoren vermeiden.

---

### **1) Audit der aktuellen Features ✓**

* Bereits vorhanden: `log_return`, `rolling_vol` (kurz), `sma_fast`, `sma_slow`, `bb_mid/upper/lower`
* Basis: Diese als Kern beibehalten; Fensterlängen für Indizes eher länger wählen.

---

### **2) Niedrig-Varianz-Kalender-/Saisonalitäts-Features (empfohlen als erstes)**

* Dummy- oder zyklische Kodierung:

    * Wochentag: `dow_sin`, `dow_cos`
    * Monat: `moy_sin`, `moy_cos`
    * Monatsende-/Quartalsende-Flags: `is_tom`, `is_eoq`, `is_eoy`
    * Handelstag im Monat (1..N) als zyklisch: `tdim_sin`, `tdim_cos`
    * Feiertagsnähe (Schweiz): `pre_holiday`, `post_holiday` (benötigt CH-Feiertagskalender)
* Begründung: Fängt leichte Saisonalitätseffekte auf breiten Indizes ein, ohne Rauschen zu erhöhen.

---

### **3) Trend- und Mean-Reversion-Features (lange Horizonte)**

* Längere SMAs/EMAs: `sma_100`, `sma_200`, `ema_63`, `ema_126`
* Abstand zum Durchschnitt: `dist_sma_200 = close/sma_200 - 1` und Z-Score
* Rollierende lineare Trendsteigung: `slope_63` via OLS auf Preis oder Log-Preis
* Preis-Kanal/Donchian: `donchian_mid_55`, `donchian_width_55`
* Bollinger-Ableitungen: `%b` (Position im Band), `bb_bw` (Bandbreite)
* Begründung: Indizes zeigen langsame Trends; Distanz und Steigung zeigen Über-/Untertreibung sowie Mean-Reversion-Potenzial.

---

### **4) Momentum (längere Fenster bevorzugen, kurze vermeiden)**

* Kumulative Renditen: `mom_21`, `mom_63`, `mom_126` (auf Log-Returns)
* RSI mit längeren Perioden: `rsi_21`, optional `rsi_63`
* MACD mit de-noisierten Parametern: `macd = ema_12-ema_26`, `macd_signal = ema_9(macd)` (oder langsamere 24/52/18-Variante für Indizes)
* „Streak“-Länge: Anzahl aufeinanderfolgender Up/Down-Tage (gekappt), über 10–20 Tage
* Begründung: Glatteres Momentum ist bei Indizes robuster.

---

### **5) Volatilitäts- und Risiko-Features (lange Fenster betonen)**

* Realisierte Volatilität lang/kurz: `rv_21`, `rv_63`, `rv_ratio = rv_21/rv_63`
* Vol of Vol: Std-Abw. von `rolling_vol` über 63 Tage
* Drawdown-Metriken: `drawdown`, `max_dd_126`, `time_since_high`
* Ulcer Index: `ulcer_63`
* Falls OHLC verfügbar: ATR `atr_14`, Parkinson-Volatilität `parkinson_20`
* Begründung: Regime (ruhig vs. moderat volatil) beeinflussen Vorhersagbarkeit und Fehlermuster.

---

### **6) Autokorrelations-/Statistik-Features**

* Verzögerte Ziel-/Renditewerte: `lag_ret_1`, `lag_ret_5`, `lag_ret_10`
* Rollierende Autokorrelation von Returns bei Lag 1 und 5 über 63 Tage: `acf1_63`, `acf5_63`
* Rollierende Z-Scores: `z_close_63`, `z_ret_63`
* Optional: Grober Hurst-Exponent über 126 Tage (falls Rechenbudget reicht)
* Begründung: Erfasst Persistenz vs. Mean-Reversion.

---

### **7) Volumen/Flow-Features (falls `volume` vorhanden)**

* Volumen-Z-Score: `z_vol_63`
* OBV und seine EMA: `obv`, `ema_obv_21`
* Price-Volume Trend (PVT)
* Begründung: Volumen spiegelt Marktteilnahme; lange Fenster reduzieren Rauschen.

---

### **8) Externe Kovariaten (optional, hoher Nutzen falls verfügbar)**

* Verwandte Märkte: DAX, EUROSTOXX50, S&P500 als vergangene Kovariaten (ausgerichtet und skaliert)
* FX: `USDCHF`, `EURCHF` (CHF-Stärke vs. globales Risikosentiment)
* Zinsen: Schweizer 10-Jahres-Rendite; Credit-Spread-Proxy (z. B. EUR IG Spread)
* Begründung: Indizes bewegen sich mit globalen Faktoren; diese stabilisieren oft das Modell.

---

### **9) Implementierungsplan im aktuellen Repo**

* Erweiterungen in `FeatureConfig` (als Booleans + Fenstergrößen):

    * `use_calendar`, `use_long_sma`, `use_ema`, `use_distance_to_sma`, `use_trend_slope`,
      `use_momentum`, `use_rsi`, `use_macd`, `use_vol_long`,
      `use_vol_ratio`, `use_drawdown`, `use_autocorr`, `use_volume_feats`
    * Fenster: `sma_long_window=200`, `ema_windows=(21,63)`,
      `mom_windows=(21,63,126)`, `rsi_window=21`, `trend_window=63`,
      `vol_long_window=63`, etc.
* Anpassungen in `feature_engineering.py`:

    * Kalenderfeatures sauber aus dem Index berechnen; Leakage vermeiden
    * Lange Fenster **vor** `dropna()` berechnen
    * OHLC-/Volumen-abhängige Features mit Spalten-Checks absichern
* Alle Features als past_covariates implementieren; sicherstellen, dass nur Vergangenheitswerte genutzt werden.

---

### **10) Validierung und Feature-Selektion**

* Ablation-Tests: Start mit Kalender + SMA-Distanz (lang) + lange RV; Gruppen schrittweise hinzufügen
* RMSE und sMAPE beobachten; einfachere Sätze bevorzugen, die beide verbessern
* TFT-Variable-Importances nutzen, um redundante Indikatoren zu entfernen (z. B. SMA vs. EMA-Distanz)
* Multikollinearität prüfen: zu viele überlappende Fenster vermeiden

---

### **11) Parametrische Leitlinien für einen stabilen Index**

* Längere Fenster bevorzugen: 63/126/200 statt 5/10/14
* Normalisierung via Z-Scores und Verhältnisse, kompatibel mit Scaler
* Möglichst binäre/kategorische Regime-Flags einsetzen (z. B. Hochvola-Regime)

---

### **12) Qualitäts- und Leakage-Kontrollen**

* Alle Features mit `.shift()` oder Rolling-Fenstern ohne Vorholausblick berechnen
* Vorhandene `check_timeseries()` erneut ausführen, um NaN/Inf nach allen Transformationsschritten zu verhindern
* Feature-Anzahl moderat halten (6–12 starke Features genügen am Anfang)

---

## ### Konkrete Feature-Liste (empfohlene erste Auswahl)

* **Kalender:** `dow_sin`, `dow_cos`, `moy_sin`, `moy_cos`, `is_tom`
* **Trend:** `sma_200`, `dist_sma_200`, `slope_63`, `z_close_63`
* **Momentum:** `mom_63`
* **Volatilität:** `rv_63`, `rv_ratio = rv_21/rv_63`
* **Risiko:** `drawdown`, `time_since_high`
* **Falls Volumen vorhanden:** `z_vol_63`

Dieses Set ist kompakt, rauscharm und gut geeignet für die Stabilität des SMI. Weitere Features wie RSI(21), %b oder MACD können später ergänzt werden.

---

Wenn du möchtest, kann ich dir auch:

* die exakten Feature-Formeln,
* oder einen fertigen Code-Patch für `feature_engineering.py`
  formulieren.
