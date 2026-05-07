"""
Feature engineering:
  1. Per-day technical indicators (RSI, MACD, Bollinger, ATR, OBV, beta, vol)
  2. Per-stock behavioral fingerprints (7-feature vector for clustering)

Outputs (in data_cache/):
    technical_features.parquet     — long-format per-stock-day technicals
    behavioral_fingerprints.parquet — one row per ticker, 7 normalized features
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import CACHE_DIR


# ─────────────────────────────────────────────────────────────────────────────
#  TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast,   adjust=False).mean()
    ema_slow = close.ewm(span=slow,   adjust=False).mean()
    macd     = ema_fast - ema_slow
    sig      = macd.ewm(span=signal,  adjust=False).mean()
    return macd, sig, macd - sig


def _bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    ma  = close.rolling(n).mean()
    std = close.rolling(n).std()
    upper = ma + k * std
    lower = ma - k * std
    width = (upper - lower) / ma
    return upper, lower, width


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift()
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, min_periods=n, adjust=False).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def _rolling_beta(stock_ret: pd.Series, market_ret: pd.Series, n: int = 60) -> pd.Series:
    """Rolling beta = Cov(stock, market) / Var(market) over n days."""
    cov = stock_ret.rolling(n).cov(market_ret)
    var = market_ret.rolling(n).var()
    return cov / var


def compute_technical_features(df: pd.DataFrame, market_returns: pd.Series) -> pd.DataFrame:
    """
    Compute per-stock-day technical indicators.

    Input:  long fact table from src.data (must include open/high/low/adj_close/volume/raw_daily_return)
    Output: same length DataFrame with technical indicator columns added
    """
    pieces: list[pd.DataFrame] = []
    market_ret = market_returns  # Series indexed by Date

    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("Date").copy()
        c, h, l, v = g["adj_close"], g["high"], g["low"], g["volume"]

        g["ma_20"]  = c.rolling(20).mean()
        g["ma_50"]  = c.rolling(50).mean()
        g["rsi_14"] = _rsi(c, 14)

        macd, sig, hist = _macd(c)
        g["macd"], g["macd_signal"], g["macd_hist"] = macd, sig, hist

        upper, lower, width = _bollinger(c)
        g["bollinger_upper"], g["bollinger_lower"], g["bollinger_width"] = upper, lower, width

        g["atr_14"]         = _atr(h, l, c, 14)
        g["obv"]            = _obv(c, v)
        g["volatility_20d"] = g["raw_daily_return"].rolling(20).std()

        # Beta needs alignment with market series by date
        stock_ret = g.set_index("Date")["raw_daily_return"]
        beta = _rolling_beta(stock_ret, market_ret.reindex(stock_ret.index), 60)
        g["beta_60d"] = beta.values

        pieces.append(g)

    out = pd.concat(pieces, ignore_index=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  BEHAVIORAL FINGERPRINTS (one row per stock)
# ─────────────────────────────────────────────────────────────────────────────
FINGERPRINT_FEATURES = [
    "mean_excess_return",
    "volatility",
    "mean_beta",
    "mean_rsi",
    "mean_bollinger_width",
    "max_drawdown",
    "momentum_score",
]


def _max_drawdown(prices: pd.Series) -> float:
    running_max = prices.cummax()
    dd = prices / running_max - 1
    return dd.min()


def compute_behavioral_fingerprints(features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate each stock's 5 years of daily data into a single 7-feature behavioral vector.

    Returns:
        raw_fp:    one row per ticker, raw (un-scaled) features for interpretation
        scaled_fp: one row per ticker, StandardScaler-normalized for clustering
    """
    rows: list[dict] = []
    for ticker, g in features_df.groupby("ticker", sort=False):
        g = g.sort_values("Date")
        prices = g["adj_close"]
        excess = g["excess_daily_return"].dropna()

        # Momentum: average of 252-day rolling returns (12-month trailing)
        mom = (prices / prices.shift(252) - 1).mean()

        rows.append({
            "ticker":               ticker,
            "mean_excess_return":   excess.mean(),
            "volatility":           excess.std(),
            "mean_beta":            g["beta_60d"].mean(),
            "mean_rsi":             g["rsi_14"].mean(),
            "mean_bollinger_width": g["bollinger_width"].mean(),
            "max_drawdown":         _max_drawdown(prices),
            "momentum_score":       mom,
        })

    raw_fp = pd.DataFrame(rows).dropna()

    # Winsorize tail outliers at 1st/99th percentile to keep clustering stable
    for col in FINGERPRINT_FEATURES:
        lo, hi = raw_fp[col].quantile([0.01, 0.99])
        raw_fp[col] = raw_fp[col].clip(lower=lo, upper=hi)

    scaler = StandardScaler()
    scaled = raw_fp.copy()
    scaled[FINGERPRINT_FEATURES] = scaler.fit_transform(raw_fp[FINGERPRINT_FEATURES])

    return raw_fp, scaled


# ─────────────────────────────────────────────────────────────────────────────
#  ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────
def run(df: pd.DataFrame, market_returns: pd.Series, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tech_path  = CACHE_DIR / "technical_features.parquet"
    raw_fp_path = CACHE_DIR / "fingerprints_raw.parquet"
    scaled_fp_path = CACHE_DIR / "fingerprints_scaled.parquet"

    if not force and tech_path.exists() and raw_fp_path.exists() and scaled_fp_path.exists():
        print(f"[features] Loading cached features from {CACHE_DIR}/")
        return (
            pd.read_parquet(tech_path),
            pd.read_parquet(raw_fp_path),
            pd.read_parquet(scaled_fp_path),
        )

    print("[features] Computing technical indicators (this takes 1–2 min)…")
    features_df = compute_technical_features(df, market_returns)

    print("[features] Building behavioral fingerprints (one per stock)…")
    raw_fp, scaled_fp = compute_behavioral_fingerprints(features_df)

    print(f"[features] Caching to {CACHE_DIR}/")
    features_df.to_parquet(tech_path, index=False)
    raw_fp.to_parquet(raw_fp_path, index=False)
    scaled_fp.to_parquet(scaled_fp_path, index=False)

    print(f"[features] Done. Technicals: {len(features_df):,} rows; Fingerprints: {len(raw_fp)} stocks")
    return features_df, raw_fp, scaled_fp
