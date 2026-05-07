"""
Page 2 — Exploratory Data Analysis.
Distributions, sector behavior, correlation analysis demonstrating factor exposure.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
CACHE_DIR = PROJECT_ROOT / "data_cache"

from src import viz  # noqa: E402

st.set_page_config(page_title="EDA", page_icon="📈", layout="wide")
st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 1280px; }
    h1, h2, h3 { color: #1B3A5C; }
    .insight { background: #F4F8FC; border-left: 4px solid #2E6DA4;
               padding: 0.8rem 1.2rem; border-radius: 4px; margin: 0.8rem 0; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Exploratory Data Analysis")
st.markdown("Understanding distributions, sector behavior, and the factor-exposure problem "
             "that motivates clustering on **excess** returns rather than raw returns.")

if not (CACHE_DIR / "technical_features.parquet").exists():
    st.error("Features not found. Run `python pipeline.py` first.")
    st.stop()

@st.cache_data
def load():
    df    = pd.read_parquet(CACHE_DIR / "fact_table.parquet")
    feats = pd.read_parquet(CACHE_DIR / "technical_features.parquet")
    return df, feats

df, feats = load()

# ── 1. RETURN DISTRIBUTIONS ──────────────────────────────────────────────────
st.markdown("## 1. Return Distributions — Raw vs. Excess")
st.markdown("Excess returns subtract daily market movement to isolate stock-specific behavior.")

c1, c2 = st.columns(2)
c1.plotly_chart(viz.return_distribution(df["raw_daily_return"], "Raw daily returns"),
                 use_container_width=True)
c2.plotly_chart(viz.return_distribution(df["excess_daily_return"], "Excess (market-adjusted) returns"),
                 use_container_width=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Raw mean",     f"{df['raw_daily_return'].mean()*100:.3f}%")
c2.metric("Raw std",      f"{df['raw_daily_return'].std()*100:.3f}%")
c3.metric("Excess mean",  f"{df['excess_daily_return'].mean()*100:.3f}%")
c4.metric("Excess std",   f"{df['excess_daily_return'].std()*100:.3f}%")

st.markdown("""
<div class="insight">
<b>Insight.</b> Excess return distribution is centered closer to zero with thinner tails —
the market component has been stripped away, leaving only stock-specific signal. This is
exactly the input we want for clustering, because we don't want stocks to group together
just because they all rode the same market wave.
</div>
""", unsafe_allow_html=True)

# ── 2. SECTOR-LEVEL BEHAVIOR ─────────────────────────────────────────────────
st.markdown("## 2. Sector-Level Behavior")
st.plotly_chart(viz.sector_volatility_box(df), use_container_width=True)
st.markdown("""
<div class="insight">
<b>Insight.</b> Energy, Communication Services, and Consumer Discretionary stocks exhibit
the highest volatility ranges. Utilities and Consumer Staples are the most stable —
classic "defensive" sectors. The wide IQR within most sectors hints that GICS labels
mask significant intra-sector behavioral diversity.
</div>
""", unsafe_allow_html=True)

st.plotly_chart(viz.sector_cumulative_returns(df), use_container_width=True)
st.markdown("""
<div class="insight">
<b>Insight.</b> Information Technology and Communication Services dominate cumulative
returns over 2019–2024 — both heavily exposed to the AI boom. Energy collapses in 2020
(COVID demand shock) then recovers sharply in 2022 (war + inflation). The divergence
across sectors validates the choice of GICS as a baseline comparison for clustering.
</div>
""", unsafe_allow_html=True)

# ── 3. CORRELATION HEATMAPS — THE FACTOR EXPOSURE PROBLEM ────────────────────
st.markdown("## 3. The Factor-Exposure Problem")
st.markdown("Side-by-side correlation heatmaps for a sample of 60 stocks. "
             "If GICS sectors describe true behavioral groups, raw-return correlations should "
             "show clean blocks along the diagonal. Excess-return heatmaps should weaken those "
             "blocks if sector labels are largely capturing market beta.")

c1, c2 = st.columns(2)
c1.plotly_chart(viz.correlation_heatmap(df, "raw_daily_return",
                  "Raw return correlation (60 stocks, sector-sorted)"),
                  use_container_width=True)
c2.plotly_chart(viz.correlation_heatmap(df, "excess_daily_return",
                  "Excess return correlation (same 60 stocks)"),
                  use_container_width=True)

st.markdown("""
<div class="insight">
<b>Insight.</b> The raw-return heatmap shows pervasive positive correlation — the entire matrix
trends red because most days, most stocks move with the market. The excess-return heatmap reveals
genuine structure: tight correlation blocks within Tech and Energy, but largely independent
behavior across most pairs. <b>This is exactly why we cluster on excess returns:
the raw-return signal is dominated by market-wide moves, not per-stock behavior.</b>
</div>
""", unsafe_allow_html=True)

# ── 4. TECHNICAL INDICATOR EDA ───────────────────────────────────────────────
st.markdown("## 4. Technical Indicators")
st.plotly_chart(viz.rsi_distribution(feats), use_container_width=True)
st.markdown("""
<div class="insight">
<b>Insight.</b> RSI clusters around 50 (neutral momentum) for the broad market, with a
modest right skew during the bull-market portions of the window. Values above 70 ("overbought")
and below 30 ("oversold") are relatively rare — they're the signal points that traders watch.
</div>
""", unsafe_allow_html=True)

# ── 5. SAMPLE STOCK CHART ────────────────────────────────────────────────────
st.markdown("## 5. Inspect a Single Stock")
ticker = st.selectbox("Ticker", sorted(feats["ticker"].unique()), index=0)
sub = feats[feats["ticker"] == ticker].sort_values("Date")

fig = go.Figure()
fig.add_trace(go.Scatter(x=sub["Date"], y=sub["adj_close"], name="Adj Close",
                          line=dict(color="#2E6DA4", width=2)))
fig.add_trace(go.Scatter(x=sub["Date"], y=sub["ma_20"], name="MA(20)",
                          line=dict(color="#F0AD4E", width=1.5, dash="dot")))
fig.add_trace(go.Scatter(x=sub["Date"], y=sub["ma_50"], name="MA(50)",
                          line=dict(color="#D9534F", width=1.5, dash="dash")))
fig.add_trace(go.Scatter(x=sub["Date"], y=sub["bollinger_upper"], name="Bollinger upper",
                          line=dict(color="#888", width=1), showlegend=False))
fig.add_trace(go.Scatter(x=sub["Date"], y=sub["bollinger_lower"], name="Bollinger lower",
                          line=dict(color="#888", width=1),
                          fill="tonexty", fillcolor="rgba(180,180,180,0.12)", showlegend=False))
fig.update_layout(title=f"{ticker} — Price & technical overlays",
                   height=440, plot_bgcolor="white", paper_bgcolor="white",
                   font=dict(family="Helvetica"),
                   margin=dict(l=40, r=20, t=50, b=40))
fig.update_xaxes(showgrid=True, gridcolor="#F4F6F9")
fig.update_yaxes(showgrid=True, gridcolor="#F4F6F9", title="Price ($)")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("All charts computed from the same fact table cached in data_cache/.")
