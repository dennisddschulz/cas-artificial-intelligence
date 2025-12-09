# main.py

from __future__ import annotations

from src.config import ExperimentConfig
from src.yahoo_data import download_yahoo_to_csv
from src.train_tft import run_experiment
from src.train_foundation import run_foundation_experiment
from src.evaluate import (
    plot_forecast,
    plot_compare_deterministic_vs_quantile,
    plot_compare_models,
)


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

    # 3. Experiment/Pipeline starten (TFT)
    print("=== Starte Experiment (TFT Forecasting Pipeline) ===")
    results = run_experiment(cfg)

    # 3b. Optional: Foundation Model (DLinear/NLinear)
    foundation_results = {}
    if getattr(cfg, "foundation", None) and cfg.foundation.enabled:
        print("=== Starte Foundation Model Pipeline ({} ) ===".format(cfg.foundation.model_type))
        try:
            foundation_results = run_foundation_experiment(cfg)
        except Exception as e:
            print(f"[WARN] Foundation Pipeline fehlgeschlagen: {e}")

    print("=== Fertig! ===")
    print("Verfügbare Keys in results:", list(results.keys()))

    # Plot comparison if multiple forecasts exist; otherwise fall back to single plot
    try:
        # Prefer y_test from foundation if TFT did not set it
        actual = results.get("y_test") or foundation_results.get("y_test")
        det_fc = results.get("forecast_det")
        qmed_fc = results.get("forecast_q_median")
        fnd_fc = foundation_results.get("forecast_foundation")

        if actual is None:
            print("[WARN] 'y_test' fehlt in results; kann nicht plotten.")
        else:
            if (det_fc is not None or qmed_fc is not None) and fnd_fc is None:
                plot_compare_deterministic_vs_quantile(
                    actual=actual,
                    det_forecast=det_fc,
                    q_median_forecast=qmed_fc,
                    title="Deterministic vs Quantile (median) — Forecast vs Actual",
                )
            elif fnd_fc is not None:
                plot_compare_models(
                    actual=actual,
                    tft_det=det_fc,
                    tft_qmed=qmed_fc,
                    foundation=fnd_fc,
                    title="Actual vs TFT vs Foundation",
                )
            else:
                # Fallback to any default forecast
                forecast = results.get("forecast")
                if forecast is None:
                    forecast = fnd_fc
                if forecast is not None:
                    plot_forecast(actual=actual, forecast=forecast, title="Forecast vs Actual")
                else:
                    print("[WARN] Konnte Plot nicht erstellen: keine Forecasts in results.")
    except Exception as e:
        print(f"[WARN] Plotten fehlgeschlagen: {e}")

    # 4. Metriken kurz ausgeben
    try:
        det_metrics = results.get("metrics_det")
        qmed_metrics = results.get("metrics_q_median")
        fnd_metrics = foundation_results.get("metrics_foundation")

        if det_metrics or qmed_metrics or fnd_metrics:
            print("=== Metriken Vergleich ===")
            if det_metrics:
                print(f"TFT (det):    RMSE={det_metrics.get('rmse'):.4f}  sMAPE={det_metrics.get('smape'):.2f}%")
            if qmed_metrics:
                print(f"TFT (q50):    RMSE={qmed_metrics.get('rmse'):.4f}  sMAPE={qmed_metrics.get('smape'):.2f}%")
            if fnd_metrics:
                print(
                    f"Foundation ({cfg.foundation.model_type}): RMSE={fnd_metrics.get('rmse'):.4f}  sMAPE={fnd_metrics.get('smape'):.2f}%"
                )
    except Exception as e:
        print(f"[WARN] Ausgabe der Metriken fehlgeschlagen: {e}")


if __name__ == "__main__":
    main()
