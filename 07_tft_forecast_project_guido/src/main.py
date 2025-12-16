# main.py

from __future__ import annotations

import os
import pickle
from src.config import ExperimentConfig
from src.yahoo_data import download_yahoo_to_csv
from src.train_tft import run_experiment
from src.backtesting_and_explain import (
    reconstruct_price_from_log_return,
    trading_simulation_long_short,
)
import requests

def main():
    # 1. Konfiguration laden
    cfg = ExperimentConfig()
    cfg.model.n_epochs = 20 # Schnelltest

    # 2. Daten herunterladen
    symbol = "^SSMI"
    print(f"=== Starte Yahoo-Download für Symbol: {symbol} ===")
    download_yahoo_to_csv(
        symbol=symbol,
        cfg=cfg.data,
        start="2015-01-01",
        end=None,
        interval="1d",
    )

    # 3. Experiment starten
    print("=== Starte Experiment (TFT Forecasting Pipeline) ===")
    results = run_experiment(cfg)

    # 4. Log-Returns → Preis rekonstruieren
    forecast_price = reconstruct_price_from_log_return(results["forecast"], base_price=100.0)
    forecast_price_quantile = reconstruct_price_from_log_return(results["forecast"], base_price=100.0)


    actual_price = reconstruct_price_from_log_return(results["y_test"], base_price=100.0)

    # 5. Saubere Trading-Simulation mit Preisreihen
    trade = trading_simulation_long_short(actual=actual_price, forecast=forecast_price)

    # 5. Saubere Trading-Simulation mit Preisreihen
    trading_quantile = trading_simulation_long_short(actual=actual_price, forecast=forecast_price)



    # 6. Ergebnisse sichern
    results["forecast_price"] = forecast_price
    results["actual_price"] = actual_price
    results["trading"] = trade
    results["trading_quantaile"] = trading_quantile

    print("=== Fertig! ===")
    print("Verfügbare Keys in results:", list(results.keys()))

    output_path = os.path.join("notebooks", "results.pkl")
    os.makedirs("notebooks", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    print("Saved to:", os.path.abspath(output_path))


if __name__ == "__main__":
    main()
