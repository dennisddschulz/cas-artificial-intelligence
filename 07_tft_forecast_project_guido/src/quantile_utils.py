# src/quantile_utils.py

import numpy as np
from darts import TimeSeries

# src/quantile_utils.py
import numpy as np
from darts import TimeSeries


def _select_quantile_component(ts: TimeSeries, q: float) -> TimeSeries:
    """
    Darts QuantileRegression usually names components like: "q0.1", "q0.5", "q0.9".
    This ensures we select the right component, not accidentally the same one.
    """
    if ts.n_components == 1:
        return ts

    name = f"q{q}"
    if name in ts.components:
        return ts[name]

    raise ValueError(f"Quantile component '{name}' not found. Available: {list(ts.components)}")


import numpy as np
from darts import TimeSeries

def predict_quantiles_range(
        model,
        n,
        series,
        past_covariates,
        quantiles,
        num_samples=1000,
):
    """
    Robust quantile prediction for Darts TFT.
    Returns {q: TimeSeries} where each TimeSeries is UNIVARIATE.
    """

    quantiles = list(quantiles)

    # ---- 1) Preferred path: predict_quantiles() ----
    if hasattr(model, "predict_quantiles"):
        out = model.predict_quantiles(
            n=n,
            series=series,
            past_covariates=past_covariates,
            quantiles=quantiles,
        )

        # Case A: dict {q: ts}
        if isinstance(out, dict):
            return {q: out[q] for q in quantiles}

        # Case B: list/tuple aligned to quantiles
        if isinstance(out, (list, tuple)):
            return {q: out[i] for i, q in enumerate(quantiles)}

        # Case C: single multi-component TimeSeries: components like "q0.1", "q0.5", "q0.9"
        if isinstance(out, TimeSeries):
            res = {}
            comps = list(out.components)
            for q in quantiles:
                name = f"q{q}"
                if name not in comps:
                    raise ValueError(f"Missing component '{name}'. Available components: {comps}")
                res[q] = out[name]  # correct component selection
            return res

        raise TypeError(f"Unsupported output type from predict_quantiles(): {type(out)}")

    # ---- 2) Older Darts: predict(..., num_samples) fallback ----
    samples = model.predict(
        n=n,
        series=series,
        past_covariates=past_covariates,
        num_samples=num_samples,
    )

    vals = samples.values()
    if vals.ndim == 3:
        vals = vals[:, :, 0]

    res = {}
    for q in quantiles:
        qv = np.quantile(vals, q, axis=1)
        res[q] = TimeSeries.from_times_and_values(samples.time_index, qv, columns=[f"q{q}"])
    return res


# --------------------------------------------------
# Median-only quantile prediction
# --------------------------------------------------
def predict_median_quantile(
        model,
        n,
        series,
        past_covariates,
        quantiles,
):
    """
    Always returns the median (0.5) quantile as a TimeSeries.
    Version-agnostic across Darts releases.
    """

    median_q = 0.5 if 0.5 in quantiles else sorted(quantiles)[len(quantiles) // 2]

    # --------------------------------------------------
    # 1) Native quantile prediction (best case)
    # --------------------------------------------------
    if hasattr(model, "predict_quantiles"):
        q_ts = model.predict_quantiles(
            n=n,
            series=series,
            past_covariates=past_covariates,
            quantiles=[median_q],
        )

        # dict-like
        if isinstance(q_ts, dict):
            return q_ts[median_q]

        # list-like
        if isinstance(q_ts, (list, tuple)):
            return q_ts[0]

        # multi-component TimeSeries
        if isinstance(q_ts, TimeSeries):
            return _select_quantile_component(q_ts, median_q)

        raise TypeError(f"Unsupported output type: {type(q_ts)}")

    # --------------------------------------------------
    # 2) Sampling-based fallback (always works)
    # --------------------------------------------------
    samples = model.predict(
        n=n,
        series=series,
        past_covariates=past_covariates,
        num_samples=500,
    )

    values = samples.values()
    if values.ndim == 3:
        values = values[:, :, 0]

    median_vals = np.quantile(values, median_q, axis=1)

    return TimeSeries.from_times_and_values(
        samples.time_index,
        median_vals,
        columns=["q0.5"],
    )

