import matplotlib.pyplot as plt
from darts import TimeSeries

from src.config import FoundationConfig


def plot_forecast(
        actual: TimeSeries,
        forecast: TimeSeries,
        title: str = "Forecast vs Actual",
):
    plt.figure(figsize=(10, 5))
    actual.plot(label="Actual")
    forecast.plot(label="Forecast")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


from typing import Optional
import matplotlib.pyplot as plt
from darts import TimeSeries

def plot_compare_deterministic_vs_quantile(
        actual: TimeSeries,
        det_forecast: Optional[TimeSeries] = None,
        q_median_forecast: Optional[TimeSeries] = None,
        title: str = "Deterministic vs Quantile (median)",
):
    """
    Plot Actual vs Deterministic forecast vs Quantile-median forecast on one chart.

    Any of the forecasts can be None; the function will plot whatever is provided.
    Includes simple diagnostics to help debug empty / non-overlapping forecasts.
    """
    def diag(ts: TimeSeries, name: str):
        try:
            length = len(ts)
            start = ts.start_time()
            end = ts.end_time()
            has_nans = ts.pd_series().isna().any()
            print(f"{name}: len={length}, start={start}, end={end}, has_nans={has_nans}")
        except Exception as e:
            print(f"{name}: failed diagnostics: {e}")

    plt.figure(figsize=(11, 6))
    fig, ax = plt.subplots(figsize=(11, 6))
    # plot actual on explicit axis
    actual.plot(label="Actual", linewidth=2, color="#333333", ax=ax)

    if det_forecast is not None:
        diag(det_forecast, "Deterministic forecast")
        # only plot if it has points
        if len(det_forecast) > 0 and not det_forecast.pd_series().isna().all():
            det_forecast.plot(label="Deterministic", color="#1f77b4", ax=ax)
        else:
            print("Deterministic forecast is empty or all-NaN; not plotted.")

    if q_median_forecast is not None:
        diag(q_median_forecast, "Quantile-median forecast")
        if len(q_median_forecast) > 0 and not q_median_forecast.pd_series().isna().all():
            q_median_forecast.plot(label="Quantile (median)", color="#ff7f0e", ax=ax)
        else:
            print("Quantile-median forecast is empty or all-NaN; not plotted.")

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_compare_models(
        actual: TimeSeries,
        tft_det: TimeSeries | None = None,
        tft_qmed: TimeSeries | None = None,
        foundation: TimeSeries | None = None,
        title: str = "Actual vs TFT vs Foundation",
):
    """
    Plot Actual vs. any combination of:
      - TFT deterministic
      - TFT quantile median
      - Foundation (DLinear/NLinear)
    """

    cfg = FoundationConfig()
    plt.figure(figsize=(11, 6))
    actual.plot(label="Actual", linewidth=2, color="#222222")
    if tft_det is not None:
        tft_det.plot(label="TFT (det)", color="#1f77b4")
    if tft_qmed is not None:
        tft_qmed.plot(label="TFT (q50)", color="#ff7f0e")
    if foundation is not None:
        foundation.plot(label="Foundation " + cfg.model_type, color="#2ca02c")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
