from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    # Path to CSV with time series
    csv_path: Path = Path("data/processed/gold_silver_abs.csv")
    datetime_col: str = "timestamp"
    target_col: str = "GOLD_Close"   # TARGET: main series to forecast
    target_value: str = "USD"
    freq: str | None = "B"

    use_exclude_features: bool = False
    features_to_exclude = lambda c: c not in {
        "GOLD_MACD",
        "GOLD_MACD_SIGNAL",
        "GOLD_MACD_HIST",
        "SILVER_MACD",
        "SILVER_MACD_SIGNAL",
        "SILVER_MACD_HIST"
    }

    # Optional columns for finance (can be missing in non-finance data)
    gold_close_col: str | None = "GOLD_Close"
    gold_open_col: str | None = "GOLD_Open"
    gold_high_col: str | None = "GOLD_High"
    gold_low_col: str | None = "GOLD_Low"
    gold_volume_col: str | None = "GOLD_Volume"

    # How many last points to reserve for test
    test_size: int = 215
    val_size: int = 215
    train_size: int = 979


@dataclass
class FeatureConfig: # event. brauchen wir das nicht mehr, wenn die features schon mitkommen
    use_log_return: bool = False
    use_rolling_volatility: bool = False
    use_sma_fast: bool = False
    use_sma_slow: bool = False
    use_bollinger_bands: bool = False

    # window lengths
    sma_fast_window: int = 10
    sma_slow_window: int = 30
    vol_window: int = 20
    bb_window: int = 20
    bb_std: float = 2.0



@dataclass
class ModelConfig: # Config entweder clonen oder parametrisierbar machen
    input_chunk_length: int = 30   # encoder length
    output_chunk_length: int = 14  # forecast horizon
    hidden_size: int = 32
    lstm_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 32
    n_epochs: int = 24
    lr: float = 1e-3
    random_state: int = 42
    use_gpu: bool = True


@dataclass
class ExperimentConfig:
    data: DataConfig = DataConfig
    features: FeatureConfig = FeatureConfig
    model: ModelConfig = ModelConfig
