# main.py

from __future__ import annotations

from src.config import ExperimentConfig
from src.yahoo_data import download_yahoo_to_csv
from src.train_tft import run_experiment


def main():
    # 1. Konfiguration laden
    cfg = ExperimentConfig()

    # Für einen schnellen Test: weniger Epochen
    # cfg.model.n_epochs = 5 # ATTENTION: wenn kommentiert nimmt er aus config.py

    # 2. Yahoo-Daten herunterladen (z.B. SPY ETF als Proxy für S&P500)
    # Du kannst symbol auch auf "^GSPC" ändern, wenn du direkt den Index willst.
    # symbol = "SPY" # ATTENTION: hier unsere symbol nehmen
    # print(f"=== Starte Yahoo-Download für Symbol: {symbol} ===")
    # download_yahoo_to_csv(
    #     symbol=symbol,
    #     cfg=cfg.data,
    #     start="2018-01-01",
    #     end=None,          # bis heute
    #     interval="1d",
    # )

    # 3. Experiment/Pipeline starten
    print("=== Starte Experiment (TFT Forecasting Pipeline) ===")
    results = run_experiment(cfg)

    print("=== Fertig! ===")
    print("Verfügbare Keys in results:", list(results.keys()))


if __name__ == "__main__":
    main()
