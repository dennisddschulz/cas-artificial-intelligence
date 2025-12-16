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
from src.evaluate import (
    plot_forecast,
    plot_compare_deterministic_vs_quantile,
    plot_compare_models,
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
    #forecast_price = reconstruct_price_from_log_return(results["deterministic"]["forecast"], base_price=100.0)
    #forecast_q_median_price = reconstruct_price_from_log_return(results["forecast_q_median"], base_price=100.0)
    #actual_price = reconstruct_price_from_log_return(results["y_test"], base_price=100.0)

    # 5. Saubere Trading-Simulation mit Preisreihen
    #trade = trading_simulation_long_short(actual=actual_price, forecast=forecast_price)
    #trade_quantile = trading_simulation_long_short(actual=actual_price, forecast=forecast_q_median_price)

    # 6. Ergebnisse sichern
    #results["forecast_price"] = forecast_price
    #results["actual_price"] = actual_price
    #results["trading"] = trade
    #results["trading_quantile"] = trade_quantile

    # 3b. Optional: Foundation Model (DLinear/NLinear)
    # foundation_results = {}
    # if getattr(cfg, "foundation", None) and cfg.foundation.enabled:
    #     print("=== Starte Foundation Model Pipeline ({} ) ===".format(cfg.foundation.model_type))
    #     try:
    #         foundation_results = run_foundation_experiment(cfg)
    #     except Exception as e:
    #         print(f"[WARN] Foundation Pipeline fehlgeschlagen: {e}")


    print("=== Fertig! ===")
    print("Verfügbare Keys in results:", list(results.keys()))


    # Plot comparison if multiple forecasts exist; otherwise fall back to single plot
    try:
        # Prefer y_test from foundation if TFT did not set it
        actual = results.get("actual_price") #or foundation_results.get("y_test")
        det_fc = results["deterministic"]["forecast"]
        qmed_fc = results["quantile"]["price_forecasts"]["q05"]
        # fnd_fc = foundation_results.get("forecast_foundation")

        if actual is None:
            print("[WARN] 'y_test' fehlt in results; kann nicht plotten.")
        else:
            if det_fc is not None or qmed_fc is not None:
                plot_compare_deterministic_vs_quantile(
                    actual=actual,
                    det_forecast=det_fc,
                    q_median_forecast=qmed_fc,
                    title="Deterministic vs Quantile (median) — Forecast vs Actual",
                )

            else:
                # Fallback to any default forecast
                forecast = results.get("forecast")
                if forecast is None:
                    forecast = None #fnd_fc
                if forecast is not None:
                    plot_forecast(actual=actual, forecast=forecast, title="Forecast vs Actual")
                else:
                    print("[WARN] Konnte Plot nicht erstellen: keine Forecasts in results.")
    except Exception as e:
        print(f"[WARN] Plotten fehlgeschlagen: {e}")


    output_path = os.path.join("src/notebooks", "results.pkl")
    # os.makedirs("notebooks", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    print("Saved to:", os.path.abspath(output_path))


if __name__ == "__main__":
    main()
