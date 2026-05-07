"""
Streamlit dashboard — landing page.

Run with:  streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

CACHE_DIR = Path(__file__).parent / "data_cache"

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Behavioral Clustering & Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GLOBAL STYLE ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }
    h1, h2, h3 { color: #1B3A5C; font-family: Helvetica, Arial, sans-serif; }
    .hero-title  { font-size: 2.6rem; font-weight: 700; color: #1B3A5C; margin-bottom: 0.2rem; }
    .hero-sub    { font-size: 1.1rem; color: #5a6c7d; margin-bottom: 2rem; }
    .pill        { display: inline-block; background: #D6E8F7; color: #1B3A5C;
                   padding: 0.25rem 0.7rem; border-radius: 999px; font-size: 0.82rem;
                   margin-right: 0.4rem; margin-bottom: 0.4rem; font-weight: 500; }
    .stat-card   { background: white; padding: 1.2rem 1.4rem; border-radius: 8px;
                   border: 1px solid #E1E8F0; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
    .stat-num    { font-size: 2.0rem; font-weight: 700; color: #2E6DA4; line-height: 1; }
    .stat-lbl    { font-size: 0.85rem; color: #6c757d; margin-top: 0.2rem; }
    .quote-box   { background: #F4F8FC; border-left: 4px solid #2E6DA4;
                   padding: 1rem 1.4rem; border-radius: 4px; font-style: italic;
                   color: #1B3A5C; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)


# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Stock Behavioral Clustering &amp; Return Prediction</div>',
             unsafe_allow_html=True)
st.markdown('<div class="hero-sub">DATA 255 Final Project — exploring whether data-driven '
             'behavioral groupings reveal market structure that GICS sectors miss, and '
             'whether those groupings improve short-term return prediction.</div>',
             unsafe_allow_html=True)

st.markdown(
    '<span class="pill">503 stocks</span>'
    '<span class="pill">5 years of daily data</span>'
    '<span class="pill">3 clustering algorithms</span>'
    '<span class="pill">4 classification models</span>'
    '<span class="pill">Plotly + Streamlit</span>',
    unsafe_allow_html=True,
)
st.write("")


# ── RESEARCH QUESTION ────────────────────────────────────────────────────────
st.markdown("""
<div class="quote-box">
    <b>Research question.</b> Do S&amp;P 500 stocks form distinct behavioral clusters
    that diverge from their official GICS sector classifications when grouped by
    market-adjusted price behavior and risk profile — and can technical indicators,
    enriched by behavioral cluster membership, predict short-term return direction?
</div>
""", unsafe_allow_html=True)


# ── PIPELINE STATUS ──────────────────────────────────────────────────────────
st.markdown("### Pipeline status")

required = {
    "Fact table":            "fact_table.parquet",
    "Technical features":    "technical_features.parquet",
    "Behavioral fingerprints": "fingerprints_scaled.parquet",
    "Cluster assignments":   "cluster_assignments.parquet",
    "Classification metrics":"classification_metrics.parquet",
}

cols = st.columns(5)
all_ready = True
for col, (name, fname) in zip(cols, required.items()):
    path = CACHE_DIR / fname
    ready = path.exists()
    all_ready &= ready
    icon = "✅" if ready else "⏳"
    color = "#5CB85C" if ready else "#F0AD4E"
    col.markdown(
        f'<div class="stat-card">'
        f'<div style="font-size: 1.3rem">{icon}</div>'
        f'<div style="font-weight:600; color:{color}; margin-top:0.4rem">{name}</div>'
        f'<div class="stat-lbl">{fname}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

if not all_ready:
    st.warning(
        "⚠️ Some artifacts are missing. Run the pipeline first:\n\n"
        "```bash\npython pipeline.py\n```\n\n"
        "This downloads 5 years of S&P 500 OHLCV data and computes all features, clusters, and models. "
        "First run takes 5–10 minutes; subsequent runs use cache."
    )
else:
    st.success("✓ All artifacts ready. Use the sidebar to navigate the dashboard.")


# ── SUMMARY STATS ────────────────────────────────────────────────────────────
if all_ready:
    st.markdown("### Quick stats")
    df = pd.read_parquet(CACHE_DIR / "fact_table.parquet")
    metrics_df = pd.read_parquet(CACHE_DIR / "classification_metrics.parquet")
    cluster_metrics = pd.read_parquet(CACHE_DIR / "cluster_metrics.parquet")
    fp = pd.read_parquet(CACHE_DIR / "fingerprints_scaled.parquet")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(f'<div class="stat-card"><div class="stat-num">{df["ticker"].nunique()}</div>'
                f'<div class="stat-lbl">Tickers analyzed</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="stat-card"><div class="stat-num">{len(df):,}</div>'
                f'<div class="stat-lbl">Stock-day rows</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="stat-card"><div class="stat-num">{df["gics_sector"].nunique()}</div>'
                f'<div class="stat-lbl">GICS sectors</div></div>', unsafe_allow_html=True)
    best_ari = cluster_metrics["ari_vs_sector"].max()
    c4.markdown(f'<div class="stat-card"><div class="stat-num">{best_ari:.2f}</div>'
                f'<div class="stat-lbl">Best ARI vs. sectors</div></div>', unsafe_allow_html=True)
    best_auc = metrics_df["roc_auc"].max()
    c5.markdown(f'<div class="stat-card"><div class="stat-num">{best_auc:.3f}</div>'
                f'<div class="stat-lbl">Best ROC-AUC</div></div>', unsafe_allow_html=True)


# ── NAV GUIDE ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Navigate the project")

nav_cols = st.columns(5)
pages = [
    ("📊", "Data Overview",  "Sources, schema, fact table preview"),
    ("📈", "EDA",            "Distributions, sector behavior, correlation"),
    ("🎯", "Clustering",     "K-Means, Hierarchical, DBSCAN — vs. GICS"),
    ("🔮", "Prediction",     "4 classifiers, with/without cluster feature"),
    ("📋", "Methodology",    "Design choices, architecture, references"),
]
for col, (icon, name, desc) in zip(nav_cols, pages):
    col.markdown(
        f'<div class="stat-card" style="height: 130px">'
        f'<div style="font-size: 1.7rem">{icon}</div>'
        f'<div style="font-weight:600; color:#1B3A5C; margin-top:0.4rem">{name}</div>'
        f'<div class="stat-lbl" style="margin-top:0.3rem">{desc}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

st.markdown("---")
st.caption("Vishnu Peram · DATA 255 · San José State University · Spring 2026")
