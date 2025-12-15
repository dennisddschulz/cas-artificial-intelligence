from dataclasses import dataclass
from pathlib import Path
from dataclasses import dataclass, field
from dataclasses import field
import torch
from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression



@dataclass
class ModelConfig:
    ...
    optimizer_class: any = torch.optim.Adam
    optimizer_kwargs: dict = field(default_factory=lambda: {"lr": 1e-3})


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
    log_return_morgen: bool = True
    use_log_return: bool = True
    use_rolling_volatility: bool = True
    use_sma_fast: bool = True
    use_sma_slow: bool = True
    use_bollinger_bands: bool = True
    use_rate_of_change:bool = True
    use_rsi: bool = True
    use_sma: bool = True
    use_usd_chf:bool = True
    use_eur_chf:bool = True
    use_market_context:bool = True
    use_macd:bool = True
    use_hedonometer:bool = False


    # window lengths
    sma_fast_window: int = 10
    sma_slow_window: int = 30
    vol_window: int = 20
    bb_window: int = 20
    bb_std: float = 2.0
    roc_window: float = 10
    rsi_window: float = 10



@dataclass
class ModelConfig:
    input_chunk_length: int = 60  # encoder length
    output_chunk_length: int = 1  # forecast horizon
    hidden_size: int = 42
    lstm_layers: int = 1
    dropout: float = 0.17
    batch_size: int = 64
    n_epochs: int = 20
    lr: float = 0.00047
    random_state: int = 42
    use_gpu: bool = True

    # ⬇️ Jetzt korrekt: Likelihood mit default_factory
    likelihood: any = field(default_factory=lambda: QuantileRegression(quantiles=[0.1, 0.5, 0.9]))

    add_relative_index: bool = True
    add_encoders: dict | None = field(default_factory=lambda: {
        "datetime_attribute": {"past": ["dayofweek", "month"]}
    })

    pl_trainer_kwargs: dict = field(default_factory=dict)
    optimizer_class: any = torch.optim.AdamW
    optimizer_kwargs: dict = field(default_factory=lambda: {"lr": 1e-3})
    lr_scheduler_class: any = None
    lr_scheduler_kwargs: dict = field(default_factory=dict)

    # Training variants
    train_deterministic: bool = True
    train_quantile: bool = True
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)


@dataclass
class FoundationConfig:
    # Enable/disable foundation model pipeline
    enabled: bool = True
    # 'dlinear' or 'nlinear'
    model_type: str = "dlinear"

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

