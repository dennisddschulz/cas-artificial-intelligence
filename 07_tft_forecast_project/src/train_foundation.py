from __future__ import annotations

from typing import Dict, Any

from darts.metrics import rmse, smape

from src.config import ExperimentConfig
from src.data_loading import load_timeseries_csv
from src.timeseries_prep import (
    prepare_series_with_features,
    scale_series,
)
from src.models_foundation import create_dlinear, create_nlinear


def run_foundation_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    """
    Train and evaluate a simple foundation model (DLinear or NLinear) using the
    same data preparation and scaling as the TFT pipeline.

    Returns a results dict with keys:
      - model_foundation
      - forecast_foundation (inverse-transformed)
      - metrics_foundation {rmse, smape}
      - y_test (inverse-transformed target for test span)
    """
    # 1) Load raw data
    df_raw = load_timeseries_csv(cfg.data)

    # 2) Prepare target and features (we will ignore covariates for D/NLinear)
    y_train_val, y_test, cov_train_val, cov_test = prepare_series_with_features(
        df_raw, cfg.data, cfg.features
    )

    # 3) Scale target (and covariates for consistency, though not used)
    (
        y_train_val_sc,
        y_test_sc,
        cov_train_val_sc,
        cov_test_sc,
        target_scaler,
        cov_scaler,
    ) = scale_series(y_train_val, y_test, cov_train_val, cov_test)

    # 4) Select and create model
    fcfg = cfg.foundation
    model_type = (fcfg.model_type or "dlinear").lower()
    if model_type == "dlinear":
        model = create_dlinear(fcfg)
    elif model_type == "nlinear":
        model = create_nlinear(fcfg)
    else:
        raise ValueError("Unknown foundation model_type: %s" % fcfg.model_type)

    # 5) Fit on scaled train+val target only (no covariates)
    model.fit(series=y_train_val_sc, verbose=True)

    # 6) Predict length of test span from the end of train_val
    fc_sc = model.predict(n=len(y_test_sc), series=y_train_val_sc)

    # 7) Inverse transform forecast and test target
    fc = target_scaler.inverse_transform(fc_sc)
    y_test_inv = target_scaler.inverse_transform(y_test_sc)

    # 8) Metrics
    m_rmse = rmse(y_test_inv, fc)
    m_smape = smape(y_test_inv, fc)

    results = {
        "model_foundation": model,
        "forecast_foundation": fc,
        "metrics_foundation": {"rmse": m_rmse, "smape": m_smape},
        "y_test": y_test_inv,
    }

    return results
