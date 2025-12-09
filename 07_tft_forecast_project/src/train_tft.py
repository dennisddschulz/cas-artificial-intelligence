
from darts import TimeSeries
from darts.metrics import rmse, smape
import numpy as np
import pandas as pd

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


    results = {}

    # (train+val+test) covariates concatenated for prediction
    full_cov_sc = None
    if cov_train_val_sc is not None and cov_test_sc is not None:
        full_cov_sc = cov_train_val_sc.concatenate(cov_test_sc)

    # Inverse-scaled y_test for metrics
    y_test_inv = target_scaler.inverse_transform(y_test_sc)

    # 5a. Deterministic (MSE) model
    if cfg.model.train_deterministic:
        model_det = create_tft_model(cfg.model, use_quantile=False)
        model_det.fit(
            series=train,
            past_covariates=cov_train,
            val_series=val,
            val_past_covariates=cov_val,
            verbose=True,
        )

        fc_det_sc = model_det.predict(
            n=len(y_test_sc),
            series=y_train_val_sc,
            past_covariates=full_cov_sc,
        )

        fc_det = target_scaler.inverse_transform(fc_det_sc)
        det_rmse = rmse(y_test_inv, fc_det)
        det_smape = smape(y_test_inv, fc_det)
        print("[Deterministic] RMSE:", det_rmse)
        print("[Deterministic] sMAPE:", det_smape)

        # Backtesting and explainability on deterministic model
        print("=== Darts Backtest (Deterministic) ===")
        bt_cfg = BacktestConfig(
            start=0.7,
            forecast_horizon=cfg.model.output_chunk_length,
            stride=cfg.model.output_chunk_length,
        )
        full_series_sc = y_train_val_sc.concatenate(y_test_sc)
        bt_results = run_darts_backtest(
            model=model_det,
            series=full_series_sc,
            past_covariates=full_cov_sc,
            cfg=bt_cfg,
        )

        print("=== TFT Feature Importances (Deterministic) ===")
        explain_tft_variables(
            model=model_det,
            background_series=y_train_val_sc,
            background_past_covariates=cov_train_val_sc,
        )

        print("=== Trading-Simulation auf Testfenster (Deterministic) ===")
        trading_simulation_long_short(actual=y_test_inv, forecast=fc_det)

        results.update({
            "model_deterministic": model_det,
            "forecast_det_scaled": fc_det_sc,
            "forecast_det": fc_det,
            "metrics_det": {"rmse": det_rmse, "smape": det_smape},
            "backtest_det": bt_results,
            "y_test": y_test_inv,
        })
        # default plot fallback
        results.setdefault("forecast", fc_det)

    # 5b. Quantile model
    if cfg.model.train_quantile:
        model_q = create_tft_model(cfg.model, use_quantile=True)
        model_q.fit(
            series=train,
            past_covariates=cov_train,
            val_series=val,
            val_past_covariates=cov_val,
            verbose=True,
        )

        # Use median quantile for point forecast
        median_q = 0.5 if 0.5 in cfg.model.quantiles else sorted(cfg.model.quantiles)[len(cfg.model.quantiles)//2]
        if hasattr(model_q, "predict_quantiles"):
            print("=== predict_quantiles ===")
            fc_q_median_sc = model_q.predict_quantiles(
                n=len(y_test_sc),
                series=y_train_val_sc,
                past_covariates=full_cov_sc,
                quantiles=[median_q],
            )[0]
        else:
            # Sampling-based quantiles when predict_quantiles() is unavailable
            fc_samples_sc = model_q.predict(
                n=len(y_test_sc),
                series=y_train_val_sc,
                past_covariates=full_cov_sc,
                num_samples=500,
            )
            # Darts 0.39: TimeSeries no longer has quantile_timeseries; use quantiles_df
            try:
                qdf = fc_samples_sc.quantiles_df([median_q])
                # qdf may have MultiIndex columns; take the first column
                if isinstance(qdf, pd.DataFrame):
                    col = qdf.columns[0]
                    q_series = qdf[col]
                else:
                    q_series = qdf
                fc_q_median_sc = TimeSeries.from_series(q_series)
            except Exception:
                # Fallback of last resort: use the mean across samples (approx median)
                fc_q_median_sc = fc_samples_sc

        fc_q_median = target_scaler.inverse_transform(fc_q_median_sc)
        q_rmse = rmse(y_test_inv, fc_q_median)
        q_smape = smape(y_test_inv, fc_q_median)
        print("[Quantile/Median] RMSE:", q_rmse)
        print("[Quantile/Median] sMAPE:", q_smape)

        results.update({
            "model_quantile": model_q,
            "forecast_q_median_scaled": fc_q_median_sc,
            "forecast_q_median": fc_q_median,
            "metrics_q_median": {"rmse": q_rmse, "smape": q_smape},
        })
        # if no deterministic model, use quantile median as default forecast
        results.setdefault("forecast", fc_q_median)

    return results



