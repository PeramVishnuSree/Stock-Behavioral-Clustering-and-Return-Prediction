"""
Data pipeline: pulls S&P 500 constituents + price history, builds the long-format fact table.

Outputs (in data_cache/):
    sp500_table.parquet       — 503 × 5 ticker metadata
    fact_table.parquet        — ~640K × 13 stock-day fact table
    market_returns.parquet    — daily S&P 500 returns (benchmark series)
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from . import CACHE_DIR

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
START    = "2019-01-01"
END      = "2024-12-31"
HEADERS  = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


# ── 1. CONSTITUENT LIST ──────────────────────────────────────────────────────
def fetch_sp500_constituents() -> pd.DataFrame:
    """Scrape S&P 500 ticker list and GICS sector labels from Wikipedia."""
    html = requests.get(WIKI_URL, headers=HEADERS, timeout=30).text
    table = pd.read_html(StringIO(html))[0]
    table = table[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry", "Date added"]]
    table.columns = ["ticker", "company", "gics_sector", "gics_sub_industry", "date_added"]
    table["ticker"] = table["ticker"].str.replace(".", "-", regex=False)  # BRK.B → BRK-B for yfinance
    return table


# ── 2. OHLCV DOWNLOAD ────────────────────────────────────────────────────────
def fetch_price_data(tickers: list[str], start: str = START, end: str = END) -> pd.DataFrame:
    """Bulk-download OHLCV via yfinance; returns a MultiIndex (price_type, ticker) DataFrame."""
    return yf.download(
        tickers + ["^GSPC"],
        start=start, end=end,
        auto_adjust=True,
        progress=True,
        group_by="column",
    )


# ── 3. RETURNS COMPUTATION ───────────────────────────────────────────────────
def compute_returns(close: pd.DataFrame, market_close: pd.Series) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Compute raw, market, and excess (market-adjusted) daily returns."""
    raw_returns    = close.pct_change(fill_method=None)
    market_returns = market_close.pct_change(fill_method=None).rename("market")
    excess_returns = raw_returns.subtract(market_returns, axis=0)
    return raw_returns, market_returns, excess_returns


# ── 4. WIDE → LONG (FACT TABLE) ──────────────────────────────────────────────
def _melt(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    return df.reset_index().melt(id_vars="Date", var_name="ticker", value_name=value_name)


def build_fact_table(
    raw: pd.DataFrame,
    sp500_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Transform the wide MultiIndex price DataFrame into a long-format fact table:
    one row per (ticker, date) with returns, OHLCV, market benchmark, and sector metadata.
    """
    close  = raw["Close"].copy()
    volume = raw["Volume"].copy()
    open_  = raw["Open"].copy()
    high   = raw["High"].copy()
    low    = raw["Low"].copy()

    market_close = close.pop("^GSPC")
    for tbl in (volume, open_, high, low):
        if "^GSPC" in tbl.columns:
            tbl.pop("^GSPC")

    raw_returns, market_returns, excess_returns = compute_returns(close, market_close)

    df_close   = _melt(close,          "adj_close")
    df_open    = _melt(open_,          "open")
    df_high    = _melt(high,           "high")
    df_low     = _melt(low,            "low")
    df_volume  = _melt(volume,         "volume")
    df_raw     = _melt(raw_returns,    "raw_daily_return")
    df_excess  = _melt(excess_returns, "excess_daily_return")
    df_market  = market_returns.reset_index()

    df = (df_close
        .merge(df_open,    on=["Date", "ticker"], how="left")
        .merge(df_high,    on=["Date", "ticker"], how="left")
        .merge(df_low,     on=["Date", "ticker"], how="left")
        .merge(df_volume,  on=["Date", "ticker"], how="left")
        .merge(df_raw,     on=["Date", "ticker"], how="left")
        .merge(df_excess,  on=["Date", "ticker"], how="left")
        .merge(df_market,  on="Date",             how="left")
        .merge(sp500_table, on="ticker",          how="left")
    )
    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)

    # Forward 5-day return + binary direction target
    df["forward_5day_return"] = (
        df.groupby("ticker")["adj_close"]
          .transform(lambda x: x.shift(-5) / x - 1)
    )
    df["forward_5day_direction"] = (df["forward_5day_return"] > 0).astype(int)

    df = df.dropna(subset=["adj_close", "raw_daily_return"])
    return df, market_returns


# ── 5. ORCHESTRATION ─────────────────────────────────────────────────────────
def run(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """End-to-end pipeline. Caches results in data_cache/."""
    fact_path   = CACHE_DIR / "fact_table.parquet"
    meta_path   = CACHE_DIR / "sp500_table.parquet"
    market_path = CACHE_DIR / "market_returns.parquet"

    if not force and fact_path.exists() and meta_path.exists() and market_path.exists():
        print(f"[data] Loading cached fact table from {fact_path}")
        df          = pd.read_parquet(fact_path)
        sp500_table = pd.read_parquet(meta_path)
        mkt         = pd.read_parquet(market_path)["market"]
        return df, sp500_table, mkt

    print("[data] Fetching S&P 500 constituents from Wikipedia…")
    sp500_table = fetch_sp500_constituents()
    print(f"[data] {len(sp500_table)} tickers found")

    print(f"[data] Downloading OHLCV {START} → {END} (this takes 2–5 min)…")
    raw = fetch_price_data(sp500_table["ticker"].tolist())

    print("[data] Building fact table…")
    df, mkt = build_fact_table(raw, sp500_table)

    print(f"[data] Caching to {CACHE_DIR}/")
    df.to_parquet(fact_path, index=False)
    sp500_table.to_parquet(meta_path, index=False)
    mkt.to_frame().to_parquet(market_path)

    print(f"[data] Done. Fact table: {len(df):,} rows × {df.shape[1]} cols, {df['ticker'].nunique()} tickers")
    return df, sp500_table, mkt


if __name__ == "__main__":
    run()
