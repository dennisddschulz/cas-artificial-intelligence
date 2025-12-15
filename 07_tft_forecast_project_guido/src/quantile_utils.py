# quantile_utils.py
import numpy as np
import pandas as pd
from darts import TimeSeries

def predict_median_quantile(
        model,
        n,
        series,
        past_covariates,
        quantiles,
):
    """
    Always returns a TimeSeries corresponding to the median forecast.
    Works across Darts versions and prediction modes.
    """

    median_q = 0.5 if 0.5 in quantiles else sorted(quantiles)[len(quantiles) // 2]

    # --------------------------------------------------
    # 1. Native quantile prediction (BEST case)
    # --------------------------------------------------
    if hasattr(model, "predict_quantiles"):
        return model.predict_quantiles(
            n=n,
            series=series,
            past_covariates=past_covariates,
            quantiles=[median_q],
        )[0]

    # --------------------------------------------------
    # 2. Sampling-based fallback
    # --------------------------------------------------
    samples = model.predict(
        n=n,
        series=series,
        past_covariates=past_covariates,
        num_samples=500,
    )

    # Case A: Darts returns a samples container
    if hasattr(samples, "quantiles_df"):
        qdf = samples.quantiles_df([median_q])
        col = qdf.columns[0]
        return TimeSeries.from_series(qdf[col])

    # Case B: Darts returns stacked TimeSeries (most common)
    # Shape: (time, components=samples)
    values = samples.values()  # shape (T, S, 1) or (T, S)

    if values.ndim == 3:
        values = values[:, :, 0]

    median_values = np.quantile(values, median_q, axis=1)

    return TimeSeries.from_times_and_values(
        times=samples.time_index,
        values=median_values,
        columns=["median"],
    )
