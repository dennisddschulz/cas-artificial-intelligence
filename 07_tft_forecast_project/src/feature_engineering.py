# feature_engineering.py

import pandas as pd
import numpy as np
from src.config import FeatureConfig


def add_finance_features(df: pd.DataFrame, cfg: FeatureConfig, target_col: str) -> pd.DataFrame:
    """
    Fügt dem Roh-DataFrame einfache Finanz-Features hinzu:
      - logarithmische Renditen
      - rollierende Volatilität
      - einfache gleitende Durchschnitte (SMA)
      - Bollinger Bänder

    Wichtig:
      - Entfernt NaN und Inf-Werte am Ende, damit das Modell
        keine ungültigen Zahlen sieht.
    """
    df_feat = df.copy()

    # Sicherstellen, dass das Target numerisch ist
    df_feat[target_col] = pd.to_numeric(df_feat[target_col], errors="coerce")

    # 1) Logarithmische Rendite
    if cfg.use_log_return:
        df_feat["log_return"] = np.log(
            df_feat[target_col] / df_feat[target_col].shift(1)
        )

    # 2) Rollierende Volatilität (Std der Renditen)
    if cfg.use_rolling_volatility:
        if "log_return" not in df_feat.columns:
            df_feat["log_return"] = np.log(
                df_feat[target_col] / df_feat[target_col].shift(1)
            )
        df_feat["rolling_vol"] = df_feat["log_return"].rolling(cfg.vol_window).std()

    # 3) Gleitende Durchschnitte (SMA)
    if cfg.use_sma_fast:
        df_feat[f"sma_{cfg.sma_fast_window}"] = (
            df_feat[target_col].rolling(cfg.sma_fast_window).mean()
        )

    if cfg.use_sma_slow:
        df_feat[f"sma_{cfg.sma_slow_window}"] = (
            df_feat[target_col].rolling(cfg.sma_slow_window).mean()
        )

    # 4) Bollinger Bänder
    if cfg.use_bollinger_bands:
        mid = df_feat[target_col].rolling(cfg.bb_window).mean()
        std = df_feat[target_col].rolling(cfg.bb_window).std()
        df_feat["bb_mid"] = mid
        df_feat["bb_upper"] = mid + cfg.bb_std * std
        df_feat["bb_lower"] = mid - cfg.bb_std * std

    # 5) Ungültige Werte bereinigen:
    #    - Inf / -Inf → NaN
    #    - alle NaN-Zeilen entfernen
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    df_feat = df_feat.dropna()


    return df_feat
