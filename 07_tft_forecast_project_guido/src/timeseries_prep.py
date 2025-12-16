from typing import Tuple

import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

import pandas as pd

from src.config import DataConfig, FeatureConfig
from src.feature_engineering import add_finance_features


def dataframe_to_timeseries(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
) -> Tuple[TimeSeries, TimeSeries | None]:
    """
    Konvertiert einen pandas DataFrame in Darts TimeSeries (robust für Finanzdaten):
      - ergänzt fehlende Business Days
      - füllt Werte stabil mit ffill/bfill (kein fillna_value=0!)
      - erzeugt garantiert NaN-freie TimeSeries

    WICHTIG:
    - Wir füllen die Lücken zuerst im DataFrame, NICHT in Darts.
    - Dadurch entstehen keine künstlichen 0-Werte.
    - Die Frequenz ist sauber 'B' (Business Days).
    """

    # -----------------------------------------
    # 1) Fehlende Business Days einfügen
    # -----------------------------------------
    df = df.asfreq("B")

    # -----------------------------------------
    # 2) Lücken füllen (Feiertage, fehlende Daten)
    # -----------------------------------------
    df = df.ffill().bfill()

    # -----------------------------------------
    # 3) TARGET TimeSeries (ohne fill_missing_dates=True!)
    # -----------------------------------------
    target_series = TimeSeries.from_dataframe(
        df,
        value_cols="log_return_morgen",
        fill_missing_dates=False,
        freq="B",
    )

    # -----------------------------------------
    # 4) FEATURES / COVARIATES
    # -----------------------------------------
    covariates_series = None
    if feature_cols:
        covariates_series = TimeSeries.from_dataframe(
            df,
            value_cols=feature_cols,
            fill_missing_dates=False,
            freq="B",
        )

    return target_series, covariates_series



def split_series(series: TimeSeries, n: int):
    """
    Teilt eine TimeSeries in:
      - linken Teil (alles außer den letzten n Punkten)
      - rechten Teil (die letzten n Punkte)
    """
    return series[:-n], series[-n:]


def prepare_series_with_features(
    df_raw: pd.DataFrame,
    data_cfg: DataConfig,
    feat_cfg: FeatureConfig,
) -> Tuple[TimeSeries, TimeSeries, TimeSeries | None, TimeSeries | None]:
    """
    1. Fügt dem Roh-DataFrame technische Features hinzu.
    2. Konvertiert in Darts TimeSeries (Target + optionale Kovariaten).
    3. Splittet die Zeitreihen in:
         - y_train_val
         - y_test
         - cov_train_val
         - cov_test

    Rückgabe:
        y_train_val, y_test, cov_train_val, cov_test
    """
    # 1) Feature-Engineering

    df_feat = add_finance_features(df_raw, feat_cfg, data_cfg.target_col)
    # 2) Feature-Spalten bestimmen (alle außer Target)
    exclude =[data_cfg.target_col, "log_return","log_return_morgen"]
    feature_cols = [c for c in df_feat.columns if c not in exclude]
    #df_feat = df_feat.drop(columns=exclude, axis=1)

    # 3) In TimeSeries konvertieren (mit konsistenter Frequenz "B")
    y_all, cov_all = dataframe_to_timeseries(df_feat, data_cfg.target_col, feature_cols)

    # 4) Train/Val/Test-Split nach Zeit
    n_total = len(y_all)
    test_size = data_cfg.test_size
    val_size = data_cfg.val_size  # aktuell nicht separat verwendet, aber in Config vorhanden

    # Letzte test_size Punkte → Test
    y_train_val, y_test = y_all[:-test_size], y_all[-test_size:]
    cov_train_val = cov_test = None
    if cov_all is not None:
        cov_train_val, cov_test = cov_all[:-test_size], cov_all[-test_size:]

    # Hier könnten wir y_train/y_val weiter splitten, wenn du explizit
    # zwischen Train und Val auf Zeitreihenebene unterscheiden willst.
    # Für TFT nutzen wir aktuell val_series direkt im model.fit().

    return y_train_val, y_test, cov_train_val, cov_test, df_feat


def scale_series(
    y_train_val: TimeSeries,
    y_test: TimeSeries,
    cov_train_val: TimeSeries | None,
    cov_test: TimeSeries | None,
):
    """
    Skaliert Target und Kovariaten mit dem Darts Scaler.

    Rückgabe:
        y_train_val_scaled, y_test_scaled,
        cov_train_val_scaled, cov_test_scaled,
        target_scaler, cov_scaler
    """
    # Target-Skalierung
    target_scaler = Scaler()
    y_train_val_scaled = target_scaler.fit_transform(y_train_val)
    y_test_scaled = target_scaler.transform(y_test)

    # Kovariaten-Skalierung (falls vorhanden)
    cov_scaler = None
    cov_train_val_scaled = None
    cov_test_scaled = None

    if cov_train_val is not None:
        cov_scaler = Scaler()
        cov_train_val_scaled = cov_scaler.fit_transform(cov_train_val)
        cov_test_scaled = cov_scaler.transform(cov_test)

    return (
        y_train_val_scaled,
        y_test_scaled,
        cov_train_val_scaled,
        cov_test_scaled,
        target_scaler,
        cov_scaler,
    )
