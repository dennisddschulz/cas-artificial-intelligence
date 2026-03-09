# feature_engineering.py
import requests
import pandas as pd
import numpy as np
from src.config import FeatureConfig
import yfinance as yf








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
        # 1) Logarithmische Rendite morgen
    if cfg.log_return_morgen:
        df_feat["log_return_morgen"] = np.log(
            df_feat[target_col] / df_feat[target_col].shift(1)).shift(-1)

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


    # 5) rate of change
    if cfg.use_rate_of_change:
        df_feat["roc"] = df_feat[target_col].pct_change(cfg.roc_window)

    # 6 RSI ( relative-strength-index/)
    if cfg.use_rsi:
        window = cfg.rsi_window
        delta = df_feat[target_col].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()

        rs = avg_gain / avg_loss
        df_feat[f"rsi_{window}"] = 100 - (100 / (1 + rs))

        df_feat[f"rsi_{window}"] = df_feat[f"rsi_{window}"].ffill()

    # 6) Markt / Einzeltitel als Kontext-Features
    if cfg.use_market_context:
        start = df_feat.index.min()
        end = df_feat.index.max()

        tickers = [
            "NESN.SW", "NOVN.SW", "RO.SW", "UBSG.SW", "ZURN.SW",
            "SRENH.SW", "CSGN.SW", "GIVN.SW", "CFR.SW", "ROG.SW",
            "ABBN.SW", "SGSN.SW", "SLHN.SW", "ADEN.SW", "LONN.SW",
            "SCMN.SW", "BAER.SW", "ALC.SW", "PGHN.SW", "GEBN.SW",
            "^VIX", "^GSPC"
        ]

        # Download
        market_data = yf.download(
            tickers,
            start=start,
            end=end,
            interval="1d",
            progress=False,
            auto_adjust=False
        )

        # Nur Close-Preise
        close_prices = market_data["Close"]

        # Log-Returns für ALLE Ticker
        log_returns = np.log(close_prices / close_prices.shift(1))

        # Saubere Feature-Namen (TFT-freundlich)
        log_returns.columns = [
            f"{c.replace('^', '').replace('.', '_')}_logret"
            for c in log_returns.columns
        ]

        # Leere Spalten entfernen (nur NaNs)
        log_returns = log_returns.dropna(axis=1, how="all")

        # Merge mit Feature-DF
        df_feat = df_feat.join(log_returns, how="left")

        # Fehlende Werte behandeln
        df_feat = df_feat.ffill()

    # 7) usd / chf
    if cfg.use_usd_chf:
        start_date = df_feat.index.min()
        end_date = df_feat.index.max()

        # Download
        usd_chf = yf.download(
            "CHF=X",
            start=start_date,
            end=end_date,
            interval="1d",
            progress=False
        )

        if isinstance(usd_chf.columns, pd.MultiIndex):
            usd_chf.columns = usd_chf.columns.get_level_values(0)

        usd_chf = usd_chf[['Close']].rename(columns={'Close': 'USD_CHF'})

        df_feat = df_feat.join(usd_chf, how="left")
        df_feat['USD_CHF'] = df_feat['USD_CHF'].ffill()

    #8 chf / CHF
    if cfg.use_eur_chf:
        start_date = df_feat.index.min()
        end_date = df_feat.index.max()

        eur_chf = yf.download(
            "EURCHF=X",
            start=start_date,
            end=end_date,
            interval="1d",
            progress=False
        )

        if isinstance(eur_chf.columns, pd.MultiIndex):
            eur_chf.columns = eur_chf.columns.get_level_values(0)

        eur_chf = eur_chf[['Close']].rename(
            columns={'Close': 'EUR_CHF'}
        )

        df_feat = df_feat.join(eur_chf, how="left")
        df_feat['EUR_CHF'] = df_feat['EUR_CHF'].ffill()


    if cfg.use_macd:

        ema12 = df_feat[target_col].ewm(span=12, adjust=False).mean() # 12 days
        ema26 = df_feat[target_col].ewm(span=26, adjust=False).mean() # 26 days
        df_feat["EMA12"] = ema12
        df_feat["EMA26"] = ema26

        df_feat["macd"] = ema12 - ema26
        df_feat["macd_signal"] = df_feat["macd"].ewm(span=9, adjust=False).mean()
        df_feat["macd_hist"] = df_feat["macd"] - df_feat["macd_signal"]



    if cfg.use_hedonometer:

        url = "https://hedonometer.org/api/v1/happiness/?format=json&timeseries__title=de_all"

        r = requests.get(url, verify=False)
        data = r.json()

        hedonometer_df = pd.DataFrame(data['objects'])

        hedonometer_df['date'] = pd.to_datetime(hedonometer_df['date'])
        hedonometer_df = hedonometer_df.set_index('date')

        hedonometer_df = hedonometer_df[['happiness']].astype(float).rename(columns={'happiness': 'Hedonometer'})

        df_feat = df_feat.join(hedonometer_df, how='left')
        #df_feat['Hedonometer'] = df_feat['Hedonometer'].ffill()
        df_feat['Hedonometer'] = (df_feat["Hedonometer"].diff() > 0).fillna(-1).astype(dtype=int).shift(-1)

    # ) Ungültige Werte bereinigen:
    #    - Inf / -Inf → NaN
    #    - alle NaN-Zeilen entfernen
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    df_feat = df_feat.dropna()

    return df_feat