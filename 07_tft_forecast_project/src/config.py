from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    # Path to CSV with time series
    csv_path: Path = Path("data/raw/example_prices.csv")
    datetime_col: str = "date"
    target_col: str = "close"   # main series to forecast

    # Optional columns for finance (can be missing in non-finance data)
    open_col: str | None = "open"
    high_col: str | None = "high"
    low_col: str | None = "low"
    volume_col: str | None = "volume"

    # How many last points to reserve for test
    test_size: int = 200
    val_size: int = 100


@dataclass
class FeatureConfig:
    use_log_return: bool = True
    use_rolling_volatility: bool = True
    use_sma_fast: bool = True
    use_sma_slow: bool = True
    use_bollinger_bands: bool = True

    # window lengths
    sma_fast_window: int = 10
    sma_slow_window: int = 30
    vol_window: int = 20
    bb_window: int = 20
    bb_std: float = 2.0


@dataclass
class ModelConfig:
    # 60 Tage zuruck
    input_chunk_length: int = 60   # encoder length
    # 10 Tage voraus
    output_chunk_length: int = 10  # forecast horizon
    hidden_size: int = 32
    lstm_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 32
    n_epochs: int = 20
    lr: float = 1e-3
    random_state: int = 42
    use_gpu: bool = False

    # Training variants
    train_deterministic: bool = True
    train_quantile: bool = True
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)


@dataclass
class FoundationConfig:
    # Enable/disable foundation model pipeline
    enabled: bool = True
    # 'dlinear' or 'nlinear'
    model_type: str = "nlinear"

    # Align sensible defaults with current TFT setup
    input_chunk_length: int = 60
    output_chunk_length: int = 10

    # Training hyperparameters
    n_epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    random_state: int = 42
    use_gpu: bool = False


@dataclass
class ExperimentConfig:
    data: DataConfig = DataConfig()
    features: FeatureConfig = FeatureConfig()
    model: ModelConfig = ModelConfig()
    foundation: FoundationConfig = FoundationConfig()
