from pathlib import Path

from darts.metrics import rmse, smape
import numpy as np

from src.timeseries_prep import split_series
from src.data_loading import load_timeseries_csv
from src.timeseries_prep import prepare_series_with_features, scale_series
from src.model_tft import create_tft_model
from src.backtesting_and_explain import (
        BacktestConfig,
        run_darts_backtest,
        trading_simulation_long_short,
        explain_tft_variables,
    )
from src.config import ExperimentConfig

def check_timeseries(ts, name="series"):
    """
    Проверяет Darts TimeSeries на NaN, Inf, -Inf и печатает статистики.
    Работает на Darts 0.39+ (использует .to_dataframe()).
    """
    if ts is None:
        print(f"[WARN] {name}: None (нет данных)")
        return

    df = ts.to_dataframe()   # <-- правильный метод в Darts 0.39+

    print(f"\n=== Sanity Check: {name} ===")
    print("Shape:", df.shape)
    print("NaN:", df.isna().any().any())
    print("Inf:", np.isinf(df.values).any())
    print("-Inf:", np.isneginf(df.values).any())
    print("All finite:", np.isfinite(df.values).all())
    print(df.describe())
    print("=== End Check ===\n")



def run_experiment(cfg: ExperimentConfig):
    # 1. Load raw data
    df_raw = load_timeseries_csv(cfg.data)

    # 2. Prepare series and engineered features
    y_train_val, y_test, cov_train_val, cov_test = prepare_series_with_features(
        df_raw,
        cfg.data,
        cfg.features,
    )

    # 3. Scale
    (
        y_train_val_sc,
        y_test_sc,
        cov_train_val_sc,
        cov_test_sc,
        target_scaler,
        cov_scaler,
    ) = scale_series(y_train_val, y_test, cov_train_val, cov_test)

    # 4. Further split train/val (optional, here simple)
    train, val = split_series(y_train_val_sc, n=cfg.data.val_size)
    cov_train = cov_val = None
    if cov_train_val_sc is not None:
        cov_train, cov_val = split_series(cov_train_val_sc, n=cfg.data.val_size)


    # === SANITY CHECK nach Skalierung ===
    check_timeseries(y_train_val_sc, "y_train_val_sc")
    check_timeseries(y_test_sc, "y_test_sc")
    check_timeseries(cov_train_val_sc, "cov_train_val_sc")
    check_timeseries(cov_test_sc, "cov_test_sc")


    # 5. Create TFT model
    model = create_tft_model(cfg.model)

    # 6. Fit model
    model.fit(
        series=train,
        past_covariates=cov_train,
        val_series=val,
        val_past_covariates=cov_val,
        verbose=True,
    )

    # 7. Forecast on test window (scaled)

    # (train+val+test)
    full_cov_sc = None
    if cov_train_val_sc is not None and cov_test_sc is not None:
        full_cov_sc = cov_train_val_sc.concatenate(cov_test_sc)

    forecast = model.predict(
        n=len(y_test_sc),
        series=y_train_val_sc,       # model starts from the end of train_val
        past_covariates=full_cov_sc, # Länge: train + val + test
    )


    # 8. Invert scaling for metrics & trading simulation
    forecast_inv = target_scaler.inverse_transform(forecast)
    y_test_inv = target_scaler.inverse_transform(y_test_sc)

    # 9. Metrics (auf echter Skala)
    print("RMSE:", rmse(y_test_inv, forecast_inv))
    print("sMAPE:", smape(y_test_inv, forecast_inv))


    # 10. Backtesting auf der gesamten Trainings+Test-Serie (scaled)
    print("=== Darts Backtest ===")

    bt_cfg = BacktestConfig(
        start=0.7,
        forecast_horizon=cfg.model.output_chunk_length,
        stride=cfg.model.output_chunk_length,
    )

    full_series_sc = y_train_val_sc.concatenate(y_test_sc)

    bt_results = run_darts_backtest(
        model=model,
        series=full_series_sc,
        past_covariates=full_cov_sc,
        cfg=bt_cfg,
    )

    # 11. Trading-Simulation (auf Testfenster, auf echter Skala - unscaled)
    print("=== Trading-Simulation auf Testfenster ===")
    trading_simulation_long_short(
        actual=y_test_inv,
        forecast=forecast_inv,  # Testfenster
    )

    # 12. TFT Feature-Importances (scaled, wie trainiert)
    print("=== TFT Feature Importances ===")
    explain_tft_variables(
        model=model,
        background_series=y_train_val_sc,
        background_past_covariates=cov_train_val_sc,
    )

    return {
        "model": model,
        "forecast_scaled": forecast,
        "forecast": forecast_inv,
        "y_test": y_test_inv,
        "backtest": bt_results,
    }



