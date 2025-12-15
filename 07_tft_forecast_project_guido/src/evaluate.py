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
