"""
asset_loader.py

Full implementation of the Asset/Indicator builder & fetcher.

Features:
- Enums for Indicator, Asset, Timeframe, TimePeriod, HLOC
- Fluent builder API to select assets, indicators and per-indicator configuration
- Fetch OHLCV via yfinance
- Compute indicators on-demand: MA, MACD, RSI, ATR, Bollinger Bands, ZigZag
- Compute divergence indicators (d_price, d_macd, d_rsi) when two assets are present
- Export merged CSV with configurable columns

Dependencies: pandas, numpy, yfinance
Install: pip install pandas numpy yfinance

Usage example at the bottom of the file.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# ENUMS
# -----------------------------
class Indicator(Enum):
    MA = "ma"
    MACD = "macd"
    RSI = "rsi"
    ATR = "atr"
    BB = "bb"
    ZIGZAG = "zigzag"
    D_PRICE = "d_price"
    D_MACD = "d_macd"
    D_RSI = "d_rsi"

class Asset(Enum):
    GOLD = "GC=F"
    SILVER = "SI=F"

class Timeframe(Enum):
    H1 = "60m"
    H4 = "4h"
    D = "1d"
    W = "1wk"
    M = "1mo"

class TimePeriod(Enum):
    D1 = 1
    W1 = 7
    M1 = 30
    M6 = 180
    Y1 = 365
    Y2 = 730
    Y3 = 1095
    Y5 = 1825
    Y10 = 3650

class HLOC(Enum):
    H = "High"
    L = "Low"
    O = "Open"
    C = "Close"

# -----------------------------
# Config dataclass
# -----------------------------
@dataclass
class IndicatorConfig:
    indicator: Indicator
    hloc: Optional[Tuple[HLOC, ...]] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def on(self, *hlocs: HLOC) -> "IndicatorConfig":
        self.hloc = hlocs
        return self

    def __getattr__(self, name):
        def setter(value):
            self.params[name] = value
            return self
        return setter

# -----------------------------
# Column normalizer (FIX)
# -----------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize yfinance columns to single-level Capitalized names:
    - tuple columns: take first level
    - split on '_' and take first token (handles open_gold, open_gc=f)
    - Capitalize token -> 'Open','High','Low','Close','Volume'
    """
    new_cols = []
    for col in df.columns:
        c = col
        if isinstance(c, tuple):
            c = c[0]
        c = str(c)
        c = c.split("_")[0]
        c = c.capitalize()
        new_cols.append(c)
    df.columns = new_cols
    return df

# -----------------------------
# Indicator calculations
# -----------------------------
def calc_ma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=1).mean()

def calc_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()

def calc_bb(series: pd.Series, period: int = 20, dev: float = 2.0):
    basis = series.rolling(period, min_periods=1).mean()
    std = series.rolling(period, min_periods=1).std()
    upper = basis + dev * std
    lower = basis - dev * std
    return upper, basis, lower

def calc_zigzag(df: pd.DataFrame, depth: int = 12, deviation: float = 5.0, backstep: int = 3):
    high = df["High"]
    low = df["Low"]
    n = len(df)
    zz_high = pd.Series(index=df.index, dtype=float)
    zz_low = pd.Series(index=df.index, dtype=float)

    last_pivot_index = None
    last_pivot_value = None

    for i in range(depth, n - depth):
        window_high = high.iloc[i - depth:i + depth + 1]
        window_low = low.iloc[i - depth:i + depth + 1]
        is_high = high.iloc[i] == window_high.max()
        is_low = low.iloc[i] == window_low.min()

        if is_high:
            val = float(high.iloc[i])
            if last_pivot_value is None or (abs(val - last_pivot_value) / last_pivot_value * 100 >= deviation):
                if last_pivot_index is None or (i - last_pivot_index) >= backstep:
                    zz_high.iloc[i] = val
                    last_pivot_index = i
                    last_pivot_value = val
        if is_low:
            val = float(low.iloc[i])
            if last_pivot_value is None or (abs(val - last_pivot_value) / last_pivot_value * 100 >= deviation):
                if last_pivot_index is None or (i - last_pivot_index) >= backstep:
                    zz_low.iloc[i] = val
                    last_pivot_index = i
                    last_pivot_value = val

    return zz_high, zz_low

# -----------------------------
# Builder
# -----------------------------
class AssetBuilder:
    def __init__(self, *assets: Asset):
        if len(assets) == 0:
            raise ValueError("At least one asset must be provided")
        self.assets: Tuple[Asset, ...] = assets
        self.timeframe: Timeframe = Timeframe.D
        self.timeperiod: TimePeriod = TimePeriod.Y1
        self.requested_indicators: Dict[Asset, List[Indicator]] = {a: [] for a in assets}
        self.configs: Dict[Asset, List[IndicatorConfig]] = {a: [] for a in assets}
        self._current_asset: Optional[Asset] = None

    def timeframeToFetch(self, timeframe: Timeframe) -> "AssetBuilder":
        self.timeframe = timeframe
        return self

    def timePeriodToApply(self, timeperiod: TimePeriod) -> "AssetBuilder":
        self.timeperiod = timeperiod
        return self

    def on(self, asset: Asset) -> "AssetBuilder":
        if asset not in self.assets:
            raise ValueError(f"Asset {asset} not in builder assets: {self.assets}")
        self._current_asset = asset
        return self

    def apply(self, *inds: Indicator) -> "AssetBuilder":
        if self._current_asset is None:
            raise ValueError("Call .on(asset) before .apply(...)")
        for ind in inds:
            if ind not in self.requested_indicators[self._current_asset]:
                self.requested_indicators[self._current_asset].append(ind)
        return self

    def configureIndicator(self, ind: Indicator) -> IndicatorConfig:
        if self._current_asset is None:
            raise ValueError("Call .on(asset) before .configureIndicator(ind)")
        cfg = IndicatorConfig(indicator=ind)
        self.configs[self._current_asset].append(cfg)
        return cfg

    def build(self) -> "DataFetcher":
        divergence_inds = {Indicator.D_PRICE, Indicator.D_MACD, Indicator.D_RSI}
        all_requested = set()
        for a, inds in self.requested_indicators.items():
            all_requested.update(inds)
        if any(d in all_requested for d in divergence_inds) and len(self.assets) < 2:
            raise ValueError("Divergence indicators require at least 2 assets configured in the builder")
        return DataFetcher(self)

# -----------------------------
# DataFetcher
# -----------------------------
class DataFetcher:
    def __init__(self, builder: AssetBuilder):
        self.builder = builder

    def fetch(self) -> "APICsvResult":
        results: Dict[Any, pd.DataFrame] = {}

        # Fetch data per asset
        for asset in self.builder.assets:
            ticker = asset.value
            days = self.builder.timeperiod.value
            interval = self.builder.timeframe.value
            logger.info(f"Downloading {ticker} period={days}d interval={interval}")

            df = yf.download(tickers=ticker, period=f"{days}d", interval=interval, progress=False)
            if df is None or df.empty:
                logger.warning(f"No data for {ticker}")
                results[asset] = pd.DataFrame()  # keep as empty df
                continue

            # normalize columns
            df = normalize_columns(df)

            # ensure OHLCV exist with exact names
            expected = ["Open", "High", "Low", "Close", "Volume"]
            for col in expected:
                if col not in df.columns:
                    df[col] = np.nan

            # compute per-asset indicators based on configs
            cfgs = self.builder.configs.get(asset, [])

            def find_configs(ind: Indicator) -> List[IndicatorConfig]:
                return [c for c in cfgs if c.indicator == ind]

            # MA
            for c in find_configs(Indicator.MA):
                period = int(c.params.get("period", 20))
                src = c.hloc[0].value if c.hloc else "Close"
                colname = f"MA_{period}"
                df[colname] = calc_ma(df[src], period)

            # MACD
            for c in find_configs(Indicator.MACD):
                fast = int(c.params.get("fast", 12))
                low = int(c.params.get("low", 26))
                signal = int(c.params.get("signal", 9))
                src = c.hloc[0].value if c.hloc else "Close"
                macd, macd_signal, macd_hist = calc_macd(df[src], fast=fast, slow=low, signal=signal)
                df["MACD"] = macd
                df["MACD_SIGNAL"] = macd_signal
                df["MACD_HIST"] = macd_hist

            # RSI
            for c in find_configs(Indicator.RSI):
                period = int(c.params.get("period", 14))
                src = c.hloc[0].value if c.hloc else "Close"
                df["RSI"] = calc_rsi(df[src], period=period)

            # ATR
            for c in find_configs(Indicator.ATR):
                period = int(c.params.get("period", 14))
                df["ATR"] = calc_atr(df, period=period)

            # BB
            for c in find_configs(Indicator.BB):
                period = int(c.params.get("period", 20))
                dev = float(c.params.get("dev", 2.0))
                src = c.hloc[0].value if c.hloc else "Close"
                up, mid, lo = calc_bb(df[src], period=period, dev=dev)
                df["BB_UPPER"] = up
                df["BB_BASIS"] = mid
                df["BB_LOWER"] = lo

            # ZIGZAG
            for c in find_configs(Indicator.ZIGZAG):
                depth = int(c.params.get("depth", 12))
                deviation = float(c.params.get("deviation", 5.0))
                backstep = int(c.params.get("backstep", 3))
                zh, zl = calc_zigzag(df, depth=depth, deviation=deviation, backstep=backstep)
                df[f"ZIGZAG_HIGH_{depth}"] = zh
                df[f"ZIGZAG_LOW_{depth}"] = zl

            results[asset] = df

        # compute divergence indicators if requested
        all_requested = set()
        for inds in self.builder.requested_indicators.values():
            all_requested.update(inds)

        if any(ind in all_requested for ind in (Indicator.D_PRICE, Indicator.D_MACD, Indicator.D_RSI)):
            if len(self.builder.assets) < 2:
                raise ValueError("Divergence indicators require two assets configured in the builder")
            a1, a2 = self.builder.assets[0], self.builder.assets[1]
            df1 = results.get(a1, pd.DataFrame()).copy()
            df2 = results.get(a2, pd.DataFrame()).copy()

            # align on intersection index
            common_index = df1.index.intersection(df2.index)
            df1 = df1.loc[common_index]
            df2 = df2.loc[common_index]

            if Indicator.D_PRICE in all_requested:
                results["D_PRICE"] = pd.DataFrame(index=common_index)
                results["D_PRICE"]["D_PRICE"] = df1["Close"] / df2["Close"]

            if Indicator.D_MACD in all_requested:
                if not ("MACD" in df1.columns and "MACD_SIGNAL" in df1.columns and "MACD" in df2.columns and "MACD_SIGNAL" in df2.columns):
                    raise ValueError("To compute D_MACD both assets must have MACD computed")
                results["D_MACD"] = pd.DataFrame(index=common_index)
                results["D_MACD"]["D_MACD"] = (df1["MACD"] - df1["MACD_SIGNAL"]) - (df2["MACD"] - df2["MACD_SIGNAL"])

            if Indicator.D_RSI in all_requested:
                if not ("RSI" in df1.columns and "RSI" in df2.columns):
                    raise ValueError("To compute D_RSI both assets must have RSI computed")
                results["D_RSI"] = pd.DataFrame(index=common_index)
                results["D_RSI"]["D_RSI"] = df1["RSI"] - df2["RSI"]

            # replace original with aligned frames
            results[a1] = df1
            results[a2] = df2

        return APICsvResult(results, self.builder)

# -----------------------------
# APICsvResult
# -----------------------------
class APICsvResult:
    def __init__(self, dfs: Dict[Any, pd.DataFrame], builder: AssetBuilder):
        self.dfs = dfs
        self.builder = builder

    def to_csv(self, path: Optional[str] = None) -> pd.DataFrame:
        parts: List[pd.DataFrame] = []

        for asset in self.builder.assets:
            df = self.dfs.get(asset)
            if df is None or df.empty:
                continue
            cols: List[str] = []
            for c in ["Open", "High", "Low", "Close", "Volume"]:
                if c in df.columns:
                    cols.append(c)
                else:
                    df[c] = np.nan
                    cols.append(c)
            ma_cols = [c for c in df.columns if c.startswith("MA_")]
            cols.extend(sorted(ma_cols, key=lambda x: int(x.split("_")[1]) if "_" in x and x.split("_")[1].isdigit() else x))
            for c in ["MACD", "MACD_SIGNAL", "MACD_HIST"]:
                if c in df.columns:
                    cols.append(c)
            for c in ["ATR", "RSI"]:
                if c in df.columns:
                    cols.append(c)
            for c in ["BB_UPPER", "BB_BASIS", "BB_LOWER"]:
                if c in df.columns:
                    cols.append(c)
            zz_cols = [c for c in df.columns if c.startswith("ZIGZAG_")]
            cols.extend(sorted(zz_cols))

            prefixed = {col: f"{asset.name}_{col}" for col in cols}
            part = df[cols].rename(columns=prefixed)
            parts.append(part)

        # divergence parts uppercase
        for key in ["D_PRICE", "D_MACD", "D_RSI"]:
            if key in self.dfs:
                part = self.dfs[key].copy()
                parts.append(part)

        if not parts:
            return pd.DataFrame()

        combined = pd.concat(parts, axis=1, join="inner")
        combined.index.name = "timestamp"
        if path:
            combined.to_csv(path)
            logger.info(f"Wrote CSV to {path}")
        return combined


def create_df_log(df):
    # convert zigzag columns to events first
    df_events = df.copy()

    zigzag_cols = [c for c in df.columns if "ZIGZAG" in c]
    numeric_cols = [c for c in df.columns if c not in zigzag_cols]

    # 1. ZigZag event transform
    df_events[zigzag_cols] = df_events[zigzag_cols].applymap(lambda v: 1 if v != 0 else 0)

    # 2. Log returns on numeric columns
    df_log_num = df_events[numeric_cols].apply(lambda col: np.log(col / col.shift(1)))

    # 3. Diff returns on ZigZag
    df_log_zig = df_events[zigzag_cols].diff()

    # 4. Combine
    df_log = pd.concat([df_log_num, df_log_zig], axis=1)

    # 5. Remove first row (NaN caused by shift & diff)
    df_log = df_log.iloc[1:]

    df_log.to_csv("../data/processed/gold_silver_log.csv")
    print(df_log.head())

def getScaleDataframe(df):
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(
        scaler.fit_transform(df.values),
        index=df.index,
        columns=df.columns
    )
    return scaled_df

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Example: build gold+silver, daily for 5 year
    builder = AssetBuilder(Asset.GOLD, Asset.SILVER).timeframeToFetch(Timeframe.D).timePeriodToApply(TimePeriod.Y10)

    # Request per-asset indicators (must pair with configureIndicator calls to compute them)
    builder.on(Asset.GOLD).apply(
        Indicator.MA, Indicator.MACD, Indicator.RSI, Indicator.ATR, Indicator.BB, Indicator.ZIGZAG, Indicator.D_PRICE, Indicator.D_MACD, Indicator.D_RSI
    )

    # gold settings
    builder.on(Asset.GOLD).configureIndicator(Indicator.MA).on(HLOC.C).period(20)
    builder.on(Asset.GOLD).configureIndicator(Indicator.MA).on(HLOC.C).period(50)
    builder.on(Asset.GOLD).configureIndicator(Indicator.MA).on(HLOC.C).period(100)
    builder.on(Asset.GOLD).configureIndicator(Indicator.MACD).on(HLOC.C).fast(12).low(26).signal(9)
    builder.on(Asset.GOLD).configureIndicator(Indicator.RSI).on(HLOC.C).period(14)
    builder.on(Asset.GOLD).configureIndicator(Indicator.ATR).period(14)
    builder.on(Asset.GOLD).configureIndicator(Indicator.BB).on(HLOC.C).period(20).dev(2.0)
    builder.on(Asset.GOLD).configureIndicator(Indicator.ZIGZAG).on(HLOC.H, HLOC.L).depth(12).deviation(5).backstep(3)
    builder.on(Asset.GOLD).configureIndicator(Indicator.ZIGZAG).on(HLOC.H, HLOC.L).depth(24).deviation(5).backstep(3)
    builder.on(Asset.GOLD).configureIndicator(Indicator.ZIGZAG).on(HLOC.H, HLOC.L).depth(48).deviation(5).backstep(3)

    # divergence indicators needs to be added only once
    builder.on(Asset.GOLD).configureIndicator(Indicator.D_PRICE)
    builder.on(Asset.GOLD).configureIndicator(Indicator.D_RSI)
    builder.on(Asset.GOLD).configureIndicator(Indicator.D_MACD)

    # mirror silver
    builder.on(Asset.SILVER).apply(
        Indicator.MA, Indicator.MACD, Indicator.RSI, Indicator.ATR, Indicator.BB, Indicator.ZIGZAG, Indicator.D_PRICE, Indicator.D_MACD, Indicator.D_RSI
    )

    builder.on(Asset.SILVER).configureIndicator(Indicator.MA).on(HLOC.C).period(20)
    builder.on(Asset.SILVER).configureIndicator(Indicator.MA).on(HLOC.C).period(50)
    builder.on(Asset.SILVER).configureIndicator(Indicator.MA).on(HLOC.C).period(100)
    builder.on(Asset.SILVER).configureIndicator(Indicator.MACD).on(HLOC.C).fast(12).low(26).signal(9)
    builder.on(Asset.SILVER).configureIndicator(Indicator.RSI).on(HLOC.C).period(14)
    builder.on(Asset.SILVER).configureIndicator(Indicator.ATR).period(14)
    builder.on(Asset.SILVER).configureIndicator(Indicator.BB).on(HLOC.C).period(20).dev(2.0)
    builder.on(Asset.SILVER).configureIndicator(Indicator.ZIGZAG).on(HLOC.H, HLOC.L).depth(12).deviation(5).backstep(3)
    builder.on(Asset.SILVER).configureIndicator(Indicator.ZIGZAG).on(HLOC.H, HLOC.L).depth(24).deviation(5).backstep(3)
    builder.on(Asset.SILVER).configureIndicator(Indicator.ZIGZAG).on(HLOC.H, HLOC.L).depth(48).deviation(5).backstep(3)

    # divergence indicators are computed automatically when at least 2 assets are added to the builder

    fetcher = builder.build()
    result = fetcher.fetch()

    df = result.to_csv("../data/raw/raw_gold_silver.csv")

    # 1. Replace ZIGZAG NaN values with 0 - whenever zigzag produces null, replace it with 0
    zigzag_cols = [c for c in df.columns if "ZIGZAG" in c.upper()]
    df[zigzag_cols] = df[zigzag_cols].fillna(0)

    # data clean up - warm up phase because of the indicator's periods
    # ma, rsi, macd, atr, zigzag: these indicators need to process at least their period once to start producing values - remove first entries (null values)
    # If you know the max warm-up period (e.g., 100 periods because of MA_100):
    warmup = 100
    df = df.iloc[warmup:].copy()

    df.to_csv("../data/processed/gold_silver_abs.csv")
    print(df.head())

    # scaled version
    df_scaled = getScaleDataframe(df)
    df_scaled.to_csv("../data/processed/gold_silver_abs_scaled.csv")

    # --------------------------
    # Create log-return version - In case we want to try it out with these values as well
    # --------------------------
    # create_df_log(df) # >>> Work in progress <<<

    # Done

