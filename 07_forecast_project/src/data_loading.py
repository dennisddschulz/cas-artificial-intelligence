import pandas as pd
from pathlib import Path
from src.config import DataConfig


def load_timeseries_csv(cfg: DataConfig) -> pd.DataFrame:
    """
    Loads a univariate or multivariate time series from CSV.
    Expects at least:
        - datetime column (cfg.datetime_col)
        - target column   (cfg.target_col)

    Additional columns (open, high, low, volume) are optional.
    """
    path: Path = cfg.csv_path
    df = pd.read_csv(path)

    # Basic checks
    if cfg.datetime_col not in df.columns:
        raise ValueError(f"datetime_col '{cfg.datetime_col}' not in CSV columns {df.columns.tolist()}")

    if cfg.target_col not in df.columns:
        raise ValueError(f"target_col '{cfg.target_col}' not in CSV columns {df.columns.tolist()}")

    # Parse datetime and sort
    df[cfg.datetime_col] = pd.to_datetime(df[cfg.datetime_col])
    df = df.sort_values(cfg.datetime_col).reset_index(drop=True)
    df = df.set_index(cfg.datetime_col)

    return df
