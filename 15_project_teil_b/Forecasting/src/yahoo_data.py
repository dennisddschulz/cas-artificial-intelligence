# src/yahoo_data.py

from __future__ import annotations
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from src.config import DataConfig


def download_yahoo_ohlcv(
    symbol: str,
    start: str = "2015-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Lädt OHLCV-Daten von Yahoo Finance über yfinance.

    Parameter:
    ----------
    symbol : str
        Börsenticker, z.B. "SPY", "^GSPC", "AAPL".
    start : str
        Startdatum im Format "YYYY-MM-DD".
    end : Optional[str]
        Enddatum. Wenn None → bis heute.
    interval : str
        Zeitintervall ("1d", "1h", etc.).

    Rückgabe:
    ---------
    df : pd.DataFrame
        OHLCV-Daten mit DatetimeIndex.
    """
    df = yf.download(
        tickers=symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,      # echte OHLC-Daten behalten
        progress=False,
    )

    if df.empty:
        raise ValueError(f"Keine Daten für {symbol} erhalten.")

    # Sicherstellen, dass der Index ein DatetimeIndex ist
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    return df


def normalize_ohlcv_dataframe_for_project(
    df: pd.DataFrame,
    cfg: DataConfig,
) -> pd.DataFrame:
    """
    Normalisiert das von yfinance geladene OHLCV-Format für unser Projekt.

    Hintergrundproblem:
    -------------------
    yfinance liefert seit 2023 häufig MultiIndex-Spalten:
        Level 0 = Ticker (z.B. "SPY")
        Level 1 = Felder ("Open", "High", ...)

    Beispiel:
        ('SPY', 'Open'), ('SPY', 'Close'), ...

    Beim Speichern als CSV wird daraus:
        close,open,high,low,volume
        SPY, SPY, SPY, SPY, SPY     ← zweite Header-Zeile

    Das muss entfernt werden, sonst liest pandas Strings statt Zahlen.

    Schritte:
    ---------
    1. MultiIndex-Spalten “abflachen”
    2. Spalten auf Projektnamen mappen (close, open, high, low, volume)
    3. Datetime-Spalte herstellen
    4. Nur relevante Spalten behalten
    """

    df = df.copy()

    # ---------------------------------------------------------
    # 1) MultiIndex-Spalten erkennen und abflachen
    # ---------------------------------------------------------
    if isinstance(df.columns, pd.MultiIndex):
        if df.columns.nlevels == 2:
            # Wir übernehmen nur Level 1, also "Open", "High", ...
            df.columns = df.columns.get_level_values(0)
            print(df.columns)
        else:
            # Falls noch andere MultiIndex-Strukturen auftreten:
            df.columns = [
                "_".join(
                    str(part) for part in col
                    if part is not None and part != ""
                )
                for col in df.columns
            ]

    # ---------------------------------------------------------
    # 2) Spaltennamen in Projektformat überführen
    # ---------------------------------------------------------
    rename_map = {}

    if "Open" in df.columns:
        rename_map["Open"] = cfg.open_col
    if "High" in df.columns:
        rename_map["High"] = cfg.high_col
    if "Low" in df.columns:
        rename_map["Low"] = cfg.low_col
    if "Close" in df.columns:
        rename_map["Close"] = cfg.target_col
    if "Volume" in df.columns and cfg.volume_col is not None:
        rename_map["Volume"] = cfg.volume_col

    df = df.rename(columns=rename_map)

    # ---------------------------------------------------------
    # 3) Datetime-Spalte hinzufügen (z.B. "date")
    # ---------------------------------------------------------
    df[cfg.datetime_col] = df.index

    # ---------------------------------------------------------
    # 4) Auf gewünschte Spalten reduzieren
    # ---------------------------------------------------------
    keep_cols = [cfg.datetime_col, cfg.target_col]

    for col in [cfg.open_col, cfg.high_col, cfg.low_col, cfg.volume_col]:
        if col is not None and col in df.columns:
            keep_cols.append(col)

    df = df[keep_cols].reset_index(drop=True)


    return df


def download_yahoo_to_csv(
    symbol: str,
    cfg: DataConfig,
    start: str = "2015-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
) -> Path:
    """
    Vollständige Pipeline:
    1. Download der Yahoo-Finance-Daten
    2. Normalisierung
    3. Speichern unter cfg.csv_path

    Rückgabe:
    ---------
    Pfad zur erzeugten CSV-Datei.
    """

    print(f"▶ Lade Yahoo Finance Daten für: {symbol}")
    df_raw = download_yahoo_ohlcv(symbol, start=start, end=end, interval=interval)

    print("▶ Normalisiere Daten ...")
    df_norm = normalize_ohlcv_dataframe_for_project(df_raw, cfg)

    csv_path: Path = cfg.csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df_norm.to_csv(csv_path, index=False)
    print(f"✔ Daten gespeichert unter: {csv_path}")

    return csv_path
