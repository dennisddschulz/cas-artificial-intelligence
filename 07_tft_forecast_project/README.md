
# 📘 TFT Forecast Starter Kit – Komplettes README

Dieses Repository enthält ein vollständiges, didaktisch strukturiertes Starter-Kit für **moderne Zeitreihenprognosen** mit **PyTorch**, **Darts**, **Temporal Fusion Transformer (TFT)**, **Chronos 2**, **TSMixer**, **TiDE**, und weiteren State-of-the-Art-Modellen.

Es wurde speziell entwickelt für **Studierende des CAS Artificial Intelligence (BFH)**, um ein praktisches, real einsatzfähiges Framework zu erhalten – inklusive Feature Engineering, Backtesting, Trading-Simulation und Explainability.

---

# 1. Projektübersicht

Das Starter-Kit ermöglicht:

- Laden von Finanzdaten (oder beliebigen Zeitreihen)
- Feature-Engineering (technische Indikatoren)
- Erstellung und Skalierung von Darts-TimeSeries
- Training eines Temporal Fusion Transformers (TFT)
- Evaluation mit Backtesting
- Durchführung einer einfachen Trading-Simulation
- Feature Importances (Variable Selection Network)
- Vergleich mit modernen Foundation Models wie **Chronos 2**
- Test zusätzlicher Modelle wie **TSMixer**, **TiDE**, **DLinear**, **NLinear**

Alle Komponenten sind modular und erweiterbar.

---

# 📂 2. Projektstruktur

```
forecast_project/
│
├── data/
│   ├── raw/                     # Originale CSV-Dateien
│   └── processed/               # Normierte/gefüllte CSVs
│
├── src/
│   ├── main.py                  # Haupteinstiegspunkt
│   ├── config.py                # Konfigurationsklassen
│   ├── yahoo_data.py            # Laden und Speichern von Yahoo Finance Daten
│   ├── feature_engineering.py   # Finanz-Features (Returns, SMA, Volatilität, BB)
│   ├── timeseries_prep.py       # Konvertierung zu Darts-TimeSeries, Splits, Scaling
│   ├── train_tft.py             # TFT-Training & Evaluation
│   ├── backtesting_and_explain.py # Backtesting, Trading Simulation, Feature Importances
│   ├── models_foundation.py     # Ladepunkt für Chronos2, TSMixer, etc.
│   └── model_tft.py             # TFT-Model
│
├── environment.yml
│
└── README.md                    # Dieses Dokument
```

---

# 🛠️ 3. Installation

# 3.1. Voraussetzungen

- Python **3.10**
- **conda** (Miniconda empfohlen)
- Internetzugang für Yahoo Finance und HuggingFace (Chronos-2)
- CPU reicht völlig

Wichtige Libraries:

- `darts==0.39.0`
- `torch` (CPU)
- `numpy<2.0` (wichtig!)
- `pandas`, `matplotlib`, `yfinance`
---

Wir liefern eine gemeinsames `environment.yml` für alle Plattforme.

---
spi vom 01.01.2025

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh

# conda config --set proxy_servers.https https://isc-den:Sommer\!500@proxy.infet.ejpd.admin.ch:8080
conda config --set ssl_verify false

## 3.2 Installation
```bash
conda env create -f environment.yml
conda activate tft_env
python -m ipykernel install --user --name=tft_env
```

---

# 4. Starten des Projekts

## 4.1 Daten laden (z. B. SPY)
```bash
# python src/main.py --download SPY
python -m src.main --download SPY
```

## 4.2 Vollen TFT-Experimentlauf starten
```bash
python src/main.py
```

Output enthält:

- RMSE & sMAPE
- Historical Forecasts
- Trading-Simulation
- Feature Importances

---

# 5. Feature Engineering (Übersicht)

Die Datei `feature_engineering.py` erzeugt u. a.:

- log_return
- rolling_vol
- sma_10, sma_30
- Bollinger Bands (mid, upper, lower)

Beispiel:
```python
df["log_return"] = np.log(df[target]/df[target].shift(1))
df["sma_10"] = df[target].rolling(10).mean()
```

Alles ist sauber dokumentiert und lässt sich beliebig erweitern.

---

# 6. TimeSeries-Erstellung & Scaling

`timeseries_prep.py` übernimmt:

✔ Sicherstellen von sauberem DatetimeIndex  
✔ Füllen fehlender Werte  
✔ Umwandlung in Darts-TimeSeries  
✔ Splitting in train/val/test  
✔ Scaling via `Scaler()`

---

# 7. Temporal Fusion Transformer (TFT)

Der TFT ist ein hybrides Modell:

- LSTM Encoder & Decoder
- Multi-Head Attention
- Variable Selection Netzwerk
- Gated Residual Networks
- Quantile-Loss oder MSE

Wir trainieren ihn so:

```python
model.fit(
    series=train,
    past_covariates=cov_train,
    val_series=val,
    val_past_covariates=cov_val,
)
```

---

# 8. Backtesting

Das Modul `backtesting_and_explain.py` enthält:

- Historical Forecasts
- Rolling Window Evaluation
- Error-Metriken

Beispiel:

```bash
RMSE (historical): 0.1827
sMAPE (historical): 14.92%
```

---

# 9. Trading-Simulation (einfach)

In `trading_simulation_long_short()`:

- long, wenn Modell ↑ erwartet
- short, wenn ↓ erwartet
- kumulative Equity-Kurve
- Sharpe-ähnlicher Wert
- Max Drawdown

Plot erzeugt direkten visuellen Vergleich:

- Strategie vs. Buy&Hold

---

# 10. Feature Importances (TFT Explainability)

Darts bietet Zugriff auf:

- Encoder Importances
- Decoder Importances
- Static Covariates Importances

Beispiel:

```
sma_10  → 0.70
rolling_vol → 16.7
bb_mid → 18.3
```

---

# 11. Moderne Modelle: Foundation Models & SOTA

Im Ordner `models_foundation.py befindet sich ein vorbereiteter, aber leerer Einstiegspunkt,
in dem die Studierenden mindestens ein modernes Modell integrieren und evaluieren sollen.

Beispiele für aktuelle, in Darts verfügbare SOTA-Modelle::

### ✔ Chronos 2 (Amazon)
- Foundation Model  
- Kein Fine-Tuning notwendig  
- Extrem stark auf generischer Zeitreihenprognose  

### ✔ TiDE (Meta)
- Deep Encoder for Time-Series  

### ✔ TSMixer  
- Token-Mixing Architektur  

### ✔ NLinear / DLinear  
- Neue lineare SOTA-Modelle  

### ✔ N-BEATS / N-HiTS  
- Sehr starke univariate Architekturen  

---

>Hinweis für Studierende:
Mindestens ein dieser Modelle muss im Rahmen der Projektarbeit in `models_foundation.py` implementiert, trainiert und mit TFT verglichen werden.

# 12. Evaluationsfluss für Studierende

(Details stehen im separaten **Assignment**)

Studierende sollen:

1. TFT testen (deterministisch)  
2. TFT erneut testen (mit Quantile Loss)  
3. Moderne Modelle integrieren:  
   - TiDE  
   - TSMixer  
   - NLinear / DLinear  
   - Chronos2  
4. Alle Modelle gegeneinander evaluieren:  
   - RMSE  
   - sMAPE  
   - Backtesting Qualität  
   - Trading-Ergebnis  
5. Alles visuell dokumentieren und präsentieren  

---

# 13. Fazit

Dieses Starter-Kit ist:

- Vollständig modular  
- Extrem flexibel  
- Ideal für Forschung, Lehre und Prototyping  
- Bereit für den direkten Einsatz in Finance, IoT, Energie, Demand-Forecasting usw.

Wenn du Feedback oder Erweiterungswünsche hast – einfach melden 😊
