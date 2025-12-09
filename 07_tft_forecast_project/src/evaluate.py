import matplotlib.pyplot as plt
from darts import TimeSeries


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


def plot_compare_deterministic_vs_quantile(
    actual: TimeSeries,
    det_forecast: TimeSeries | None,
    q_median_forecast: TimeSeries | None,
    title: str = "Deterministic vs Quantile (median)",
):
    """
    Plot Actual vs Deterministic forecast vs Quantile-median forecast on one chart.

    Any of the forecasts can be None; the function will plot whatever is provided.
    """
    plt.figure(figsize=(11, 6))
    actual.plot(label="Actual", linewidth=2, color="#333333")
    if det_forecast is not None:
        det_forecast.plot(label="Deterministic", color="#1f77b4")
    if q_median_forecast is not None:
        q_median_forecast.plot(label="Quantile (median)", color="#ff7f0e")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
