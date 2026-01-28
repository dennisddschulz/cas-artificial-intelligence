# MLflow Integration für TFT Forecasting mit Darts & PyTorch Lightning

## Übersicht

Diese Dokumentation beschreibt die Integration von MLflow in die TFT (Temporal Fusion Transformer) Forecasting Pipeline. Die Integration ermöglicht:

- **Experiment Tracking**: Alle Trainingsläufe werden protokolliert
- **Parameter Logging**: Reproduzierbare Experimente
- **Metriken Vergleich**: Verschiedene Modelle vergleichen
- **Artifact Storage**: Visualisierungen und Modelle speichern
- **Offline-Betrieb**: Kein Server erforderlich

---

## Geänderte Dateien

| Datei | Zweck |
|-------|-------|
| `src/config.py` | MLflow Konfiguration (Experiment-Name, Tracking-Pfad) |
| `src/logger/mlflow_util.py` | Setup-Funktionen für MLflow |
| `src/train_tft.py` | Logging von Params, Metrics, Datasets, Models, Artifacts |
| `src/visualizer/candlestick.py` | Visualisierung für Artifacts |

---

## 1. Konfiguration (`src/config.py`)

```python
@dataclass
class MlFlowConfig:
    experiment_name: str = "TFT_Forecasting"
    tracking_dir: Path = Path("logs/mlruns")  # Lokaler Speicher
```

**Wichtig:** Kein HTTP-Server nötig! Logs werden direkt ins Dateisystem geschrieben.

---

## 2. MLflow Setup (`src/logger/mlflow_util.py`)

```python
def setup_mlflow(cfg: MlFlowConfig) -> str:
    cfg.tracking_dir.mkdir(parents=True, exist_ok=True)
    
    # Lokalen Pfad als file:// URI setzen für Dateisystem-Speicherung
    local_uri = cfg.tracking_dir.resolve().as_uri()  # z.B. file:///C:/git/.../logs/mlruns
    mlflow.set_tracking_uri(local_uri)
    mlflow.set_experiment(cfg.experiment_name)
    
    return local_uri
```

**Key Point:** `Path.as_uri()` erzeugt `file:///...` URI → funktioniert komplett offline!

---

## 3. PyTorch Lightning Logger (`src/logger/mlflow_util.py`)

```python
def create_mlflow_logger(cfg: MlFlowConfig, run_id: str) -> MLFlowLogger:
    local_uri = cfg.tracking_dir.resolve().as_uri()
    
    return MLFlowLogger(
        experiment_name=cfg.experiment_name,
        tracking_uri=local_uri,
        run_id=run_id,  # WICHTIG: Gleiche Run-ID wie Hauptrun!
        log_model=True,
    )
```

**Integration mit Darts TFT:**

```python
from pytorch_lightning.loggers import MLFlowLogger

model = TFTModel(
    ...,
    pl_trainer_kwargs={"logger": pl_logger}  # ← Lightning Logger
)
```

---

## 4. Training Pipeline Ablauf (`src/train_tft.py`)

```
┌─────────────────────────────────────────────────────────────┐
│  1. mlflow.start_run()                                      │
│     └── Startet einen neuen Run                             │
├─────────────────────────────────────────────────────────────┤
│  2. mlflow.log_params({...})                                │
│     └── Modell-Parameter, Feature-Flags, Datenset-Infos     │
├─────────────────────────────────────────────────────────────┤
│  3. mlflow.log_input(dataset)                               │
│     └── Trainings-Datensatz tracken                         │
├─────────────────────────────────────────────────────────────┤
│  4. model.fit(...) ← mit PLLogger                           │
│     └── PyTorch Lightning loggt automatisch:                │
│         • train_loss (pro Epoch)                            │
│         • val_loss (pro Epoch)                              │
├─────────────────────────────────────────────────────────────┤
│  5. mlflow.log_metrics({...})                               │
│     └── Test-Metriken: RMSE, MAE, R², Direction Accuracy    │
├─────────────────────────────────────────────────────────────┤
│  6. mlflow.log_artifact("candlestick.html")                 │
│     └── Visualisierungen speichern                          │
├─────────────────────────────────────────────────────────────┤
│  7. mlflow.pytorch.log_model(model)                         │
│     └── Trainiertes Modell speichern                        │
├─────────────────────────────────────────────────────────────┤
│  8. mlflow.end_run()                                        │
│     └── Run abschließen                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Was wird geloggt?

### Parameter (für Reproduzierbarkeit)

```python
mlflow.log_params({
    # Modell-Parameter
    "ticker": "BTCUSDT",
    "hidden_size": 32,
    "lstm_layers": 2,
    "n_epochs": 20,
    "lr": 0.001,
    "random_state": 42,
    
    # Feature-Engineering Flags
    "use_log_return": True,
    "use_sma_fast": True,
    "use_bollinger_bands": True,
    
    # Datenset-Infos
    "train_size": 2501,
    "val_size": 100,
    "test_size": 200,
    "date_start": "2018-01-01",
    "date_end": "2025-12-10",
})
```

### Metriken (für Vergleich)

```python
mlflow.log_metrics({
    "test_rmse": 1234.5,
    "test_mae": 987.6,
    "test_smape": 0.05,
    "test_mape": 0.08,
    "test_r2": 0.85,
    "direction_accuracy": 0.52,  # Wichtig für Trading!
    "training_time_seconds": 120,
})
```

### Artifacts (für Analyse)

| Artifact | Beschreibung |
|----------|--------------|
| `candlestick_forecast.html` | Interaktiver Plotly Chart |
| `tft_model/` | Gespeichertes PyTorch Modell |

### Dataset Tracking

```python
dataset = mlflow.data.from_pandas(
    df_raw,
    source=str(cfg.data.csv_path),
    name="BTCUSDT_BINANCE",
)
mlflow.log_input(dataset, context="training")
```

---

## 6. Architektur: Darts ↔ PyTorch Lightning ↔ MLflow

```
┌──────────────┐     ┌────────────────────┐     ┌─────────────┐
│   Darts      │────▶│  PyTorch Lightning │────▶│   MLflow    │
│   TFTModel   │     │  Trainer           │     │   Logger    │
└──────────────┘     └────────────────────┘     └─────────────┘
       │                      │                        │
       │                      │                        │
   model.fit()          Callbacks &              Automatisches
                        Logging Hooks            Loss-Tracking
```

**Der Trick:** Darts nutzt intern PyTorch Lightning. Durch `pl_trainer_kwargs={"logger": MLFlowLogger}` werden alle Training-Metriken (Loss pro Epoch) automatisch geloggt!

---

## 7. MLflow UI starten

```bash
mlflow ui --backend-store-uri logs/mlruns
```

Dann im Browser öffnen: [http://localhost:5000](http://localhost:5000)

### Features der UI:

- **Runs vergleichen**: Mehrere Experimente nebeneinander
- **Metriken sortieren**: z.B. nach `direction_accuracy` für Trading
- **Artifacts anschauen**: HTML Charts direkt im Browser
- **Modelle laden**: Für spätere Inferenz

---

## 8. Vorteile dieser Integration

| Feature | Vorteil |
|---------|---------|
| **Offline-fähig** | Kein Server nötig, alles lokal in `logs/mlruns/` |
| **Reproduzierbar** | Alle Parameter und Random Seeds gespeichert |
| **Vergleichbar** | Runs nach Metriken sortieren und filtern |
| **Visualisierbar** | Charts als interaktive HTML Artifacts |
| **Model Registry** | Modelle versioniert speichern und laden |

---

## 9. Beispiel: Modell später laden

```python
import mlflow.pytorch

# Modell aus einem Run laden
model = mlflow.pytorch.load_model("runs:/<RUN_ID>/tft_model")

# Oder aus dem Model Registry
model = mlflow.pytorch.load_model("models:/TFT_BTCUSDT/latest")
```

---

## 10. Ordnerstruktur nach dem Training

```
logs/mlruns/
├── 0/                          # Default Experiment
├── <experiment_id>/            # TFT_Forecasting Experiment
│   ├── <run_id_1>/
│   │   ├── artifacts/
│   │   │   ├── candlestick_forecast.html
│   │   │   └── tft_model/
│   │   ├── metrics/
│   │   │   ├── test_rmse
│   │   │   ├── test_r2
│   │   │   └── ...
│   │   ├── params/
│   │   │   ├── hidden_size
│   │   │   ├── n_epochs
│   │   │   └── ...
│   │   └── meta.yaml
│   └── <run_id_2>/
│       └── ...
└── models/                     # Model Registry (optional)
```

MLFlow Aufruf:
mlflow ui --backend-store-uri file:///home/isc-den/cas-artificial-intelligence/0_mlflow_logs
