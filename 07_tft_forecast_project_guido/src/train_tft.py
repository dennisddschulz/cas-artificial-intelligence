from pathlib import Path

from darts.metrics import rmse, smape
import numpy as np
from src.quantile_utils import predict_median_quantile
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
        reconstruct_price_from_log_return,
    )
from src.config import ExperimentConfig

def check_timeseries(ts, name="series"):
    if ts is None:
        print(f"[WARN] {name}: None (kein Daten)")
        return

    df = ts.to_dataframe()

    print(f"\n=== Sanity Check: {name} ===")
    print("Shape:", df.shape)
    print("NaN:", df.isna().any().any())
    print("Inf:", np.isinf(df.values).any())
    print("-Inf:", np.isneginf(df.values).any())
    print("All finite:", np.isfinite(df.values).all())
    print(df.describe())
    print("=== End Check ===\n")


def run_experiment(cfg: ExperimentConfig):
    global model_q, q_rmse, q_smape, fc_q_median, fc_q_median_sc, quantile_results, trade_quantile, bt_fc_inv_quantile, bt_actual_inv_quantile, forecast_price_bt_quantile, forecast_inv_quantile, median_forecast_log_ret_quantile_01, median_forecast_log_ret_quantile_05, median_forecast_log_ret_quantile_09, forecast_price_bt_quantile_09, forecast_price_bt_quantile_05, forecast_price_bt_quantile_01
    df_raw = load_timeseries_csv(cfg.data)

    y_train_val, y_test, cov_train_val, cov_test, df_feat = prepare_series_with_features(
        df_raw,
        cfg.data,
        cfg.features,
    )

    (
        y_train_val_sc,
        y_test_sc,
        cov_train_val_sc,
        cov_test_sc,
        target_scaler,
        cov_scaler,
    ) = scale_series(y_train_val, y_test, cov_train_val, cov_test)

    train, val = split_series(y_train_val_sc, n=cfg.data.val_size)
    cov_train = cov_val = None
    if cov_train_val_sc is not None:
        cov_train, cov_val = split_series(cov_train_val_sc, n=cfg.data.val_size)

    check_timeseries(y_train_val_sc, "y_train_val_sc")
    check_timeseries(y_test_sc, "y_test_sc")

    model = create_tft_model(cfg.model)

    model.fit(
        series=train,
        past_covariates=cov_train,
        val_series=val,
        val_past_covariates=cov_val,
        verbose=True,
    )

    full_cov_sc = None
    if cov_train_val_sc is not None and cov_test_sc is not None:
        full_cov_sc = cov_train_val_sc.concatenate(cov_test_sc)

    forecast_scaled = model.predict(
        n=len(y_test_sc),
        series=y_train_val_sc,
        past_covariates=full_cov_sc,
    )

    forecast_inv = target_scaler.inverse_transform(forecast_scaled)
    y_test_inv = target_scaler.inverse_transform(y_test_sc)

    test_rmse = rmse(y_test_inv, forecast_inv)
    test_smape = smape(y_test_inv, forecast_inv)
    print("RMSE:", test_rmse)
    print("sMAPE:", test_smape)

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

    # ------------------------------------------------------------
    # ⬇️ KORREKTUR DER PREISREKONSTRUKTION
    # ------------------------------------------------------------

    # 1. Basispreis für die Rekonstruktion bestimmen
    # Wir nehmen den Original-Preis EINEN TAG VOR dem Startpunkt des Backtests.
    start_index = int(len(df_raw) * bt_cfg.start)
    # df_raw enthält die Original-Preise, die für die Basis benötigt werden.
    # Wichtig: Der erste Log-Return im Backtest beginnt BEI start_index. Die Basis
    # muss daher der Preis VOR start_index sein.
    BASE_PRICE = df_raw.iloc[start_index - 1][cfg.data.target_col]

    print(f"Basispreis für Preisrekonstruktion (Tag vor Backtest-Start): {BASE_PRICE:.2f}")

    # Log-Returns der Test-Forecasts (für direkten Forecast-Return)
    forecast_inv = target_scaler.inverse_transform(forecast_scaled)
    actual_log_ret_test = target_scaler.inverse_transform(y_test_sc)

    # Log-Returns der Backtest-Ergebnisse (invertiert)
    bt_fc_inv = target_scaler.inverse_transform(bt_results["historical_forecasts"])
    bt_actual_inv = target_scaler.inverse_transform(bt_results["historical_actual"])

    # Wenn TFT Quantile verwendet, müssen wir für die Trading-Simulation
    # und Preisrekonstruktion nur den Median ('q0.5') auswählen,
    # um die kumulative Divergenz zu vermeiden.
    # Der Fehler liegt fast immer hier: die Rekonstruktion wird auf alle Quantile gleichzeitig angewendet.
    try:
        # Versuche, das 0.5-Quantil zu extrahieren (falls vorhanden)
        median_forecast_log_ret = bt_fc_inv['q0.5']
    except Exception:
        # Falls das Modell nur eine univariate Serie (z.B. Mean-Prediction) ausgibt
        median_forecast_log_ret = bt_fc_inv

    # Preise rekonstruieren (mit korrektem BASE_PRICE und Median-Quantil)
    forecast_price_bt = reconstruct_price_from_log_return(median_forecast_log_ret, base_price=BASE_PRICE)
    actual_price_bt = reconstruct_price_from_log_return(bt_actual_inv, base_price=BASE_PRICE)

    # ------------------------------------------------------------
    # ⬇️ 2. Trading-Simulation mit den BACKTEST-PREISEN durchführen
    # ------------------------------------------------------------
    trade = trading_simulation_long_short(
        actual=actual_price_bt,  # <--- WICHTIG: Verwende actual_price_bt
        forecast=forecast_price_bt,  # <--- WICHTIG: Verwende forecast_price_bt
    )

    expl = explain_tft_variables(
        model=model,
        background_series=y_train_val_sc,
        background_past_covariates=cov_train_val_sc,
    )



    if cfg.model.train_quantile:

        model_q = create_tft_model(cfg.model, use_quantile=True)

        model_q.fit(
            series=train,
            past_covariates=cov_train,
            val_series=val,
            val_past_covariates=cov_val,
            verbose=True,
        )

        median_sc = predict_median_quantile(
            model=model_q,
            n=len(y_test_sc),
            series=y_train_val_sc,
            past_covariates=full_cov_sc,
            quantiles=cfg.model.quantiles,
        )

        median = target_scaler.inverse_transform(median_sc)

        quantile_results = {
            "model": model_q,
            "forecast_median": median,
            "rmse": rmse(y_test_inv, median),
            "smape": smape(y_test_inv, median),
        }




        forecast_scaled_quantile = model.predict(
            n=len(y_test_sc),
            series=y_train_val_sc,
            past_covariates=full_cov_sc,
        )

        forecast_inv_quantile = target_scaler.inverse_transform(forecast_scaled_quantile)
        y_test_inv = target_scaler.inverse_transform(y_test_sc)

        test_rmse_quantile = rmse(y_test_inv, forecast_inv_quantile)
        test_smape_quantile = smape(y_test_inv, forecast_inv_quantile)
        print("RMSE:", test_rmse_quantile)
        print("sMAPE:", test_smape_quantile)

        bt_cfg = BacktestConfig(
            start=0.7,
            forecast_horizon=cfg.model.output_chunk_length,
            stride=cfg.model.output_chunk_length,
        )

        full_series_sc = y_train_val_sc.concatenate(y_test_sc)

        bt_results_quantile = run_darts_backtest(
            model=model,
            series=full_series_sc,
            past_covariates=full_cov_sc,
            cfg=bt_cfg,
        )

        # ------------------------------------------------------------
        # ⬇️ KORREKTUR DER PREISREKONSTRUKTION
        # ------------------------------------------------------------

        # 1. Basispreis für die Rekonstruktion bestimmen
        # Wir nehmen den Original-Preis EINEN TAG VOR dem Startpunkt des Backtests.
        start_index = int(len(df_raw) * bt_cfg.start)
        # df_raw enthält die Original-Preise, die für die Basis benötigt werden.
        # Wichtig: Der erste Log-Return im Backtest beginnt BEI start_index. Die Basis
        # muss daher der Preis VOR start_index sein.
        BASE_PRICE = df_raw.iloc[start_index - 1][cfg.data.target_col]

        print(f"Basispreis für Preisrekonstruktion (Tag vor Backtest-Start): {BASE_PRICE:.2f}")

        # Log-Returns der Test-Forecasts (für direkten Forecast-Return)
        forecast_inv_quantile = target_scaler.inverse_transform(forecast_scaled_quantile)
        actual_log_ret_test = target_scaler.inverse_transform(y_test_sc)

        # Log-Returns der Backtest-Ergebnisse (invertiert)
        bt_fc_inv_quantile = target_scaler.inverse_transform(bt_results_quantile["historical_forecasts"])
        bt_actual_inv_quantile = target_scaler.inverse_transform(bt_results_quantile["historical_actual"])

        # Wenn TFT Quantile verwendet, müssen wir für die Trading-Simulation
        # und Preisrekonstruktion nur den Median ('q0.5') auswählen,
        # um die kumulative Divergenz zu vermeiden.
        # Der Fehler liegt fast immer hier: die Rekonstruktion wird auf alle Quantile gleichzeitig angewendet.
        try:
            # Versuche, das 0.5-Quantil zu extrahieren (falls vorhanden)
            median_forecast_log_ret_quantile_01 = bt_fc_inv_quantile['q0.1']
            median_forecast_log_ret_quantile_05 = bt_fc_inv_quantile['q0.5']
            median_forecast_log_ret_quantile_09 = bt_fc_inv_quantile['q0.9']
        except Exception:
            # Falls das Modell nur eine univariate Serie (z.B. Mean-Prediction) ausgibt
            median_forecast_log_ret_quantile_01 = bt_fc_inv_quantile
            median_forecast_log_ret_quantile_05 = bt_fc_inv_quantile
            median_forecast_log_ret_quantile_09 = bt_fc_inv_quantile


       # forecast_price_bt_quantile = bt_fc_inv_quantile['q0.5']
        # Preise rekonstruieren (mit korrektem BASE_PRICE und Median-Quantil)
        forecast_price_bt_quantile_01 = reconstruct_price_from_log_return(median_forecast_log_ret_quantile_01, base_price=BASE_PRICE)
        forecast_price_bt_quantile_05 = reconstruct_price_from_log_return(median_forecast_log_ret_quantile_05, base_price=BASE_PRICE)
        forecast_price_bt_quantile_09 = reconstruct_price_from_log_return(median_forecast_log_ret_quantile_09, base_price=BASE_PRICE)

        #actual_price_bt = reconstruct_price_from_log_return(bt_actual_inv, base_price=BASE_PRICE)


        # Preise rekonstruieren (mit korrektem BASE_PRICE und Median-Quantil)
        forecast_price_bt_quantile = reconstruct_price_from_log_return(quantile_results["forecast_median"], base_price=BASE_PRICE)
        #actual_price_bt = reconstruct_price_from_log_return(bt_actual_inv, base_price=BASE_PRICE)

        trade_quantile = trading_simulation_long_short(
            actual=actual_price_bt,  # <--- WICHTIG: Verwende actual_price_bt
            forecast=forecast_price_bt_quantile,  # <--- WICHTIG: Verwende forecast_price_bt
        )

    return {
        "model": model,
        "model_quantile": quantile_results["model"],
        "forecast_q_median": forecast_inv_quantile, #quantile_results["forecast_median"],
        "forecast_q_median_price": forecast_price_bt_quantile,
        "forecast_q_median_price_01": forecast_price_bt_quantile_01,
        "forecast_q_median_price_05": forecast_price_bt_quantile_05,
        "forecast_q_median_price_09": forecast_price_bt_quantile_09,
        "metrics_q_median": {"rmse": quantile_results["rmse"], "smape": quantile_results["smape"]},
        # "model_quantile": model_q,
        # "forecast_q_median_scaled": fc_q_median_sc,
        # "forecast_q_median": fc_q_median,
        # "forecast_q_median_price": reconstruct_price_from_log_return(fc_q_median, base_price=BASE_PRICE),
        "target_scaler": target_scaler,
        "cov_scaler": cov_scaler,
        "y_train_val": y_train_val,
        "y_test": actual_log_ret_test,  # y_test ist Log-Return
        "forecast": forecast_inv,
        "forecast_price": reconstruct_price_from_log_return(forecast_inv, base_price=BASE_PRICE),
        # Für den direkten Test-Split
        "actual_price": reconstruct_price_from_log_return(actual_log_ret_test, base_price=BASE_PRICE),
        # Für den direkten Test-Split
        "rmse": test_rmse,
        "smape": test_smape,
        "backtest": bt_results,
        "historical_forecasts": bt_fc_inv,
        "historical_forecasts_quantile": bt_fc_inv_quantile,

        "historical_actual_log_ret": bt_actual_inv,
        "historical_actual_log_ret_quantile": bt_actual_inv_quantile,

        "historical_forecast_price": forecast_price_bt,
        "historical_forecast_price_quantile": forecast_price_bt_quantile,

        "historical_actual_price": actual_price_bt,
        "trading": trade,
        "trading_quantile": trade_quantile,
        "feature_importances": expl["feature_importances"],
        "df_raw": df_raw,
        "df_feat": df_feat,
        "cov_train_val": cov_train_val,
        "cov_test_sc": cov_test_sc,
        "y_train_val_sc": y_train_val_sc,
        "cov_train_val_sc" : cov_train_val_sc
    }
