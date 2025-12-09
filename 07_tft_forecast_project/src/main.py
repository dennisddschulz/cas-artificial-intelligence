# main.py

from __future__ import annotations

from src.config import ExperimentConfig
from src.yahoo_data import download_yahoo_to_csv
from src.train_tft import run_experiment
from src.evaluate import plot_forecast


def main():
    # 1. Konfiguration laden
    cfg = ExperimentConfig()

    # Für einen schnellen Test: weniger Epochen
    cfg.model.n_epochs = 25

    # 2. Yahoo-Daten herunterladen (z.B. SPY ETF als Proxy für S&P500)
    # Du kannst symbol auch auf "^GSPC" ändern, wenn du direkt den Index willst.
    symbol = "^SSMI"
    print(f"=== Starte Yahoo-Download für Symbol: {symbol} ===")
    download_yahoo_to_csv(
        symbol=symbol,
        cfg=cfg.data,
        start="2024-01-01",
        end=None,          # bis heute
        interval="1d",
    )

    # 3. Experiment/Pipeline starten
    print("=== Starte Experiment (TFT Forecasting Pipeline) ===")
    results = run_experiment(cfg)

    print("=== Fertig! ===")
    print("Verfügbare Keys in results:", list(results.keys()))

    # Plot forecast vs. actual each time main is executed
    try:
        actual = results.get("y_test")
        forecast = results.get("forecast")
        if actual is not None and forecast is not None:
            plot_forecast(actual=actual, forecast=forecast, title="Forecast vs Actual")
        else:
            print("[WARN] Konnte Plot nicht erstellen: 'y_test' oder 'forecast' fehlt in results.")
    except Exception as e:
        print(f"[WARN] Plotten fehlgeschlagen: {e}")


if __name__ == "__main__":
    main()
