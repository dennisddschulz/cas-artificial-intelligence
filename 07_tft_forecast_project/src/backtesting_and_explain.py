# src/backtesting_and_explain.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.metrics import mape, rmse, smape
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.explainability import TFTExplainer  # nur für TFTModel
from darts import TimeSeries, concatenate


@dataclass
class BacktestConfig:
    """
    Konfiguration für das Backtesting mit Darts.

    - start: ab welchem Anteil der Zeitreihe das Backtesting beginnen soll.
      Beispiel: start=0.6 → die ersten 60% der Daten werden nur zum Training genutzt,
      danach wird so getan, als würden wir in "Echtzeit" vorwärts rollen.
    - forecast_horizon: wie viele Schritte in die Zukunft wir pro Schritt vorhersagen.
    - stride: wie oft wir neu vorhersagen (z.B. alle 5 Schritte).
    - retrain: ob das Modell bei jedem Schritt neu trainiert werden soll.
      (retrain=True ist realistischer, aber teurer)
    """
    start: float = 0.7
    forecast_horizon: int = 10
    stride: int = 5
    retrain: bool = False

# -----------------------------------------------------------
# Hilfsfunktion: historical_forecasts → EIN TimeSeries
# -----------------------------------------------------------
def _to_single_series_from_hfc(hfc) -> TimeSeries:
    """
    Wandelt die Ausgabe von `model.historical_forecasts()` in genau EIN TimeSeries um.
    Warum?

    Darts kann je nach Parametern Folgendes zurückgeben:
        - TimeSeries
        - list[TimeSeries]
        - list[list[TimeSeries]]

    Für Backtesting und Plotten benötigen wir jedoch EINEN konsistenten TimeSeries.
    Diese Funktion normalisiert also alle Varianten.
    """

    # Fall 1: Es ist bereits ein TimeSeries
    if isinstance(hfc, TimeSeries):
        return hfc

    # Fall 2: Liste aller Forecasts
    if isinstance(hfc, list):
        if len(hfc) == 0:
            raise ValueError("historical_forecasts hat eine leere Liste zurückgegeben.")

        first = hfc[0]

        # Fall 2a: list[TimeSeries]
        if isinstance(first, TimeSeries):
            return concatenate(hfc)

        # Fall 2b: list[list[TimeSeries]]
        if isinstance(first, list):
            if len(first) == 0:
                raise ValueError("historical_forecasts hat list[list[]] zurückgegeben, aber inneres ist leer.")

            inner_first = first[0]
            if not isinstance(inner_first, TimeSeries):
                raise TypeError("Unerwarteter geschachtelter Typ in historical_forecasts.")

            return concatenate(first)

    # Alles andere ist ein Fehler
    raise TypeError(f"Nicht unterstützter Typ von historical_forecasts(): {type(hfc)}")


def run_darts_backtest(
    model: ForecastingModel,
    series: TimeSeries,
    past_covariates: Optional[TimeSeries] = None,
    cfg: BacktestConfig = BacktestConfig(),
):
    """
    Führt ein "historical_forecasts"-Backtesting mit Darts durch
    und gibt sowohl die historische Vorhersage als auch Kennzahlen aus.

    Idee:
    - Darts simuliert, als würden wir in der Vergangenheit an verschiedenen Zeitpunkten
      das Modell trainieren und dann in die Zukunft vorhersagen.
    - Dadurch entsteht eine Zeitreihe von "Pseudo-Echtzeit"-Vorhersagen,
      die wir mit den echten Werten vergleichen können.
    """
    print("Starte historical_forecasts-Backtest ...")

    # 1) Historische Forecasts erzeugen (Simulation von Echtzeit)
    hist_fc_raw = model.historical_forecasts(
        series=series,
        past_covariates=past_covariates,
        start=cfg.start,                 # ab 70% der Daten
        forecast_horizon=cfg.forecast_horizon,
        stride=cfg.stride,
        retrain=cfg.retrain,
        last_points_only=False,          # vollständige Pfade behalten
        verbose=True,
    )
    # -------------------------------------------------------
    # 1a. Forecast-Output in EINEN TimeSeries konvertieren
    # -------------------------------------------------------
    hist_fc = _to_single_series_from_hfc(hist_fc_raw)


    # 2) Fehler-Metriken berechnen (RMSE, MAPE)
    #    Wichtig: hist_fc und series haben denselben Index, aber hist_fc deckt
    #    nur den Teil nach "start" ab.

    # -------------------------------------------------------
    #    Gemeinsame Zeitachse finden
    #    Wichtig: Manche Forecasts beginnen später → schneiden!
    # -------------------------------------------------------
    common_actual = series.slice_intersect(hist_fc)
    common_fc = hist_fc.slice_intersect(series)

    err_rmse = rmse(common_actual, common_fc)
    err_smape = smape(common_actual, common_fc)

    print(f"RMSE (historical_forecasts): {err_rmse:.4f}")
    print(f"sMAPE (historical_forecasts): {err_smape:.2f}%")

    # 3) Optional: Darts-internes backtest()-Convenience-Interface
    print("Berechne zusätzlich model.backtest(...):")

    avg_smape = model.backtest(
        series=series,
        past_covariates=past_covariates,
        start=cfg.start,
        forecast_horizon=cfg.forecast_horizon,
        stride=cfg.stride,
        retrain=cfg.retrain,
        last_points_only=False,
        metric=smape,
    )
    print(f"Durchschnittlicher MAPE über alle historischen Forecasts: {avg_smape:.2f}%")

    return {
        "historical_forecasts": hist_fc,
        "rmse": err_rmse,
        "smape": err_smape,
        "avg_smape_backtest": avg_smape,
    }


# --------------------------------------------------------------------
# Einfache "Trading"-Simulation auf Basis der Vorhersage
# --------------------------------------------------------------------

def trading_simulation_long_short(
    actual: TimeSeries,
    forecast: TimeSeries,
    annualization_factor: int = 252,
):
    """
    Sehr einfache Long/Short-Trading-Simulation auf Basis einer Vorhersage.

    Annahmen:
    - `actual`: Zeitreihe eines Preises (z.B. Close) – echte Werte.
    - `forecast`: Modellvorhersage desselben Preises über denselben Zeitraum.
    - Wir sind immer entweder "long" (+1) oder "short" (-1).
    - Signal: Wenn das Modell steigende Preise erwartet → long,
              sonst → short.

    Wichtiger Hinweis:
    - Diese Funktion ist bewusst stark vereinfacht und dient
      didaktischen Zwecken (Vorlesung, Übungen).
    """

    # 1) Auf gemeinsamen Zeitraum beschränken
    actual = actual.slice_intersect(forecast)
    forecast = forecast.slice_intersect(actual)

    # 2) In pandas-Serien umwandeln
    p_actual: pd.Series = actual.to_series()
    p_forecast: pd.Series = forecast.to_series()

    # 3) Offensichtlich ungültige Werte entfernen:
    #    - Preise <= 0
    #    - nicht-endliche Werte (NaN, Inf)
    mask_valid_price = (
        np.isfinite(p_actual.values) &
        (p_actual.values > 0.0) &
        np.isfinite(p_forecast.values)
    )

    p_actual = p_actual[mask_valid_price]
    p_forecast = p_forecast[mask_valid_price]

    if len(p_actual) < 2:
        print("Zu wenige gültige Datenpunkte für Trading-Simulation.")
        return {
            "equity_curve": np.array([]),
            "strategy_returns": np.array([]),
            "total_return": np.nan,
            "sharpe_like": np.nan,
            "max_drawdown": np.nan,
        }

    # 4) Tatsächliche prozentuale Returns (Buy & Hold-Basis)
    returns = p_actual.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 5) Modell-Return (Erwartung des Modells, 1-Tages-Horizont, gelaggt)
    forecast_ret = (
        p_forecast.pct_change()
        .shift(1)  # Modell-Info einen Tag vorher
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    # 6) Handels-Signal: +1 (long) wenn erwarteter Return > 0, -1 (short) sonst
    position = np.where(forecast_ret > 0, 1.0, -1.0)

    # 7) Strategie-Return: Position * tatsächlicher Return
    strat_ret = position * returns.values

    # 7a) Nur endliche Werte verwenden
    mask_finite = np.isfinite(strat_ret)
    strat_ret = strat_ret[mask_finite]

    if len(strat_ret) == 0:
        print("Alle Strategie-Returns sind nicht endlich (NaN/Inf).")
        return {
            "equity_curve": np.array([]),
            "strategy_returns": np.array([]),
            "total_return": np.nan,
            "sharpe_like": np.nan,
            "max_drawdown": np.nan,
        }

    # 8) Equity-Kurve (angenommen, Startkapital = 1.0)
    equity_curve = (1.0 + strat_ret).cumprod()

    # 9) Kennzahlen
    total_return = equity_curve[-1] - 1.0
    mean_ret = np.mean(strat_ret)
    std_ret = np.std(strat_ret) + 1e-12  # numerische Stabilität
    sharpe = (mean_ret / std_ret) * np.sqrt(annualization_factor)

    # Max Drawdown
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    max_dd = drawdowns.min()

    print("Trading-Simulation (Long/Short basierend auf Forecast):")
    print(f"Gesamtrendite: {total_return * 100:.2f}%")
    print(f"Sharpe-ähnlicher Wert: {sharpe:.2f}")
    print(f"Maximaler Drawdown: {max_dd * 100:.2f}%")

    return {
        "equity_curve": equity_curve,
        "strategy_returns": strat_ret,
        "total_return": total_return,
        "sharpe_like": sharpe,
        "max_drawdown": max_dd,
    }


# --------------------------------------------------------------------
# TFT Feature Importance / Explainability
# --------------------------------------------------------------------

def explain_tft_variables(
    model,
    background_series: TimeSeries,
    background_past_covariates: Optional[TimeSeries] = None,
    max_nr_series: int = 1,
):
    """
    Nutzt den TFTExplainer aus Darts, um Feature Importances / Variable Selection
    für einen trainierten TFTModel zu visualisieren.

    Wichtig:
    - model muss ein TFTModel sein.
    - background_series sollte eine (skalierte) Serie sein, auf der das Modell trainiert wurde.
    - background_past_covariates (optional) sind die korrespondierenden Kovariaten.
    """
    print("Starte TFT-Explainability (Variable Selection / Feature Importances) ...")

    explainer = TFTExplainer(
        model=model,
        background_series=[background_series],             # Liste von Serien
        background_past_covariates=[background_past_covariates] if background_past_covariates is not None else None,
    )

    expl_result = explainer.explain()
    # Optional: Zugriff auf numerische Werte der Importances
    fi_dict = expl_result.get_feature_importances()
    # fi_dict ist ein Dict mit DataFrames für Encoder/Decoder/Static

    print("Verfügbare Feature-Importance-Keys:", list(fi_dict.keys()))
    print("Beispiel – Encoder Importances (erste Zeilen):")
    encoder_df = fi_dict.get("encoder_importance")
    if encoder_df is not None:
        if isinstance(encoder_df, list):
            print(encoder_df[0].head())
        else:
            print(encoder_df.head())

    return {
        "explainer": explainer,
        "expl_result": expl_result,
        "feature_importances": fi_dict,
    }
