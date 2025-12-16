from darts import TimeSeries


import numpy as np
from darts import TimeSeries


def predict_quantiles_range(
        model,
        n,
        series,
        past_covariates,
        quantiles,
        num_samples: int = 1000,
):
    """
    Darts 0.39 + TFT + QuantileRegression
    Quantiles are extracted from the probabilistic forecast,
    NOT from components.
    """

    # get probabilistic forecast
    pred = model.predict(
        n=n,
        series=series,
        past_covariates=past_covariates,
        num_samples=num_samples,
    )

    # Preferred path (exists in Darts 0.39)
    if hasattr(pred, "quantile_timeseries"):
        return {q: pred.quantile_timeseries(q) for q in quantiles}

    # Fallback: compute manually from samples
    vals = pred.all_values()  # (time, component, sample)

    if vals.ndim != 3:
        raise RuntimeError(f"Expected 3D all_values(), got {vals.shape}")

    if vals.shape[1] != 1:
        raise RuntimeError(f"Expected univariate target, got {vals.shape[1]} components")

    res = {}
    for q in quantiles:
        qv = np.quantile(vals[:, 0, :], q, axis=1)
        res[q] = TimeSeries.from_times_and_values(
            pred.time_index,
            qv,
            columns=[f"q{q}"],
        )

    return res


def predict_median_quantile(
        model,
        n,
        series,
        past_covariates,
        quantiles,
):
    median_q = 0.5 if 0.5 in quantiles else sorted(quantiles)[len(quantiles)//2]

    pred = model.predict(
        n=n,
        series=series,
        past_covariates=past_covariates,
        num_samples=500,
    )

    return pred.quantile_timeseries(median_q)
