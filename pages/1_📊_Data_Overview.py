"""
Page 1 — Data Overview.
Shows sources, schema, fact table sample, and per-sector ticker counts.
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

CACHE_DIR = Path(__file__).parent.parent / "data_cache"

st.set_page_config(page_title="Data Overview", page_icon="📊", layout="wide")
st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 1280px; }
    h1, h2, h3 { color: #1B3A5C; }
    .pill { display: inline-block; background: #D6E8F7; color: #1B3A5C;
            padding: 0.25rem 0.7rem; border-radius: 999px; font-size: 0.82rem;
            margin-right: 0.4rem; margin-bottom: 0.4rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────────────────────
st.title("📊 Data Overview")
st.markdown("Sources, schema design, and fact table structure.")

# Guard
if not (CACHE_DIR / "fact_table.parquet").exists():
    st.error("Fact table not found. Run `python pipeline.py` first.")
    st.stop()

@st.cache_data
def load_data():
    df = pd.read_parquet(CACHE_DIR / "fact_table.parquet")
    sp500 = pd.read_parquet(CACHE_DIR / "sp500_table.parquet")
    return df, sp500

df, sp500 = load_data()

# ── DATA SOURCES ─────────────────────────────────────────────────────────────
st.markdown("## 1. Data Sources")
sources = pd.DataFrame([
    {"Source": "Yahoo Finance (yfinance)",
     "What it provides": "Daily OHLCV for ~503 S&P 500 tickers, 2019-01-01 → 2024-12-31",
     "Why this source": "Free, no API key, well-maintained Python wrapper."},
    {"Source": "Wikipedia: List of S&P 500 companies",
     "What it provides": "Ticker symbols, company names, GICS Sector, GICS Sub-Industry",
     "Why this source": "Authoritative, frequently updated, scrapeable with pandas.read_html."},
    {"Source": "S&P 500 Index (^GSPC) via Yahoo Finance",
     "What it provides": "Daily index level — used to compute market-adjusted (excess) returns",
     "Why this source": "Required to remove systematic market-wide moves from per-stock signals."},
])
st.dataframe(sources, use_container_width=True, hide_index=True)

# ── STAR SCHEMA ──────────────────────────────────────────────────────────────
st.markdown("## 2. Star Schema")
st.markdown("""
The data warehouse follows a classic star schema. One fact table records every
stock-day observation; three dimension tables describe the stock, the date, and
the technical indicator snapshot.
""")

c1, c2, c3 = st.columns([1, 1.2, 1])
with c2:
    st.markdown(f"""
    <div style="background:#1B3A5C; color:white; padding:1rem; border-radius:8px; text-align:center;">
        <b>daily_returns_fact</b><br/>
        <span style="font-size:0.85rem; opacity:0.85">{len(df):,} rows</span>
    </div>
    """, unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
for col, (name, rows) in zip([c1, c2, c3], [
    ("stock_dim", df["ticker"].nunique()),
    ("time_dim", df["Date"].nunique()),
    ("technical_features_dim", "computed"),
]):
    col.markdown(f"""
    <div style="background:#D6E8F7; color:#1B3A5C; padding:0.8rem; border-radius:8px;
                text-align:center; margin-top:1rem">
        <b>{name}</b><br/>
        <span style="font-size:0.85rem">{rows} rows</span>
    </div>
    """, unsafe_allow_html=True)

# ── FACT TABLE SAMPLE ────────────────────────────────────────────────────────
st.markdown("## 3. Fact Table Sample")
st.markdown(f"**Shape:** {len(df):,} rows × {df.shape[1]} cols  ·  "
             f"**Date range:** {df['Date'].min().date()} → {df['Date'].max().date()}")

display_cols = ["Date", "ticker", "company", "gics_sector", "adj_close", "volume",
                "raw_daily_return", "excess_daily_return", "forward_5day_direction"]
st.dataframe(df[display_cols].head(15), use_container_width=True, hide_index=True)

with st.expander("Show full schema (all columns)"):
    schema = pd.DataFrame({
        "Column":   df.columns,
        "Dtype":    df.dtypes.astype(str).values,
        "Non-null %": (df.notna().sum() / len(df) * 100).round(1).values,
        "Sample":   [str(df[c].dropna().iloc[0])[:40] if df[c].notna().any() else "—" for c in df.columns],
    })
    st.dataframe(schema, use_container_width=True, hide_index=True)

# ── SECTOR DISTRIBUTION ──────────────────────────────────────────────────────
st.markdown("## 4. Sector Distribution")
sec_counts = (sp500.groupby("gics_sector").size()
                    .reset_index(name="count")
                    .sort_values("count", ascending=True))
fig = px.bar(sec_counts, x="count", y="gics_sector", orientation="h",
              color="count", color_continuous_scale="Blues",
              labels={"count": "Number of stocks", "gics_sector": ""})
fig.update_layout(height=420, plot_bgcolor="white", paper_bgcolor="white",
                   coloraxis_showscale=False, margin=dict(l=0, r=20, t=20, b=40),
                   font=dict(family="Helvetica"))
fig.update_xaxes(showgrid=True, gridcolor="#F4F6F9")
fig.update_yaxes(showgrid=False)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Interpretation.** The 11 GICS sectors are unevenly represented in the S&P 500.
Information Technology, Financials, Industrials, and Health Care dominate.
This imbalance matters for clustering: under-represented sectors (Energy,
Materials, Real Estate, Utilities) may be absorbed into broader behavioral
clusters rather than emerging as distinct groups.
""")

st.markdown("---")
st.caption("Source: Yahoo Finance via yfinance, Wikipedia (GICS labels) — pulled at pipeline run time.")
