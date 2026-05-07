"""
Page 5 — Methodology & architecture decisions.
Explains the design choices, pipeline architecture, and provides reading list.
"""
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Methodology", page_icon="📋", layout="wide")
st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 1100px; }
    h1, h2, h3 { color: #1B3A5C; }
    .step-card  { background: white; padding: 1.2rem 1.4rem; border-radius: 8px;
                  border: 1px solid #E1E8F0; margin: 0.6rem 0; }
    .step-num   { display: inline-block; background: #1B3A5C; color: white;
                  width: 28px; height: 28px; border-radius: 50%; text-align: center;
                  line-height: 28px; font-weight: 700; margin-right: 0.6rem; }
    .design-decision { background: #FFF8E7; border-left: 4px solid #F0AD4E;
                       padding: 0.8rem 1.2rem; border-radius: 4px; margin: 0.8rem 0; }
    code { background: #F4F6F9; padding: 2px 6px; border-radius: 3px; font-size: 0.85em; }
</style>
""", unsafe_allow_html=True)

st.title("📋 Methodology & Architecture")
st.markdown("Design choices, pipeline architecture, and references.")

# ── RESEARCH QUESTION ────────────────────────────────────────────────────────
st.markdown("## Research Question")
st.markdown("""
> Do S&P 500 stocks form distinct behavioral clusters that diverge from their
> official GICS sector classifications when grouped by market-adjusted price
> behavior and risk profile — and can technical indicators, enriched by
> behavioral cluster membership, predict short-term return direction?

Two linked sub-questions:
- **RQ1 (Clustering)** — Do behavioral clusters align with GICS sectors?
- **RQ2 (Prediction)** — Does cluster membership improve return-direction prediction?
""")

# ── PIPELINE ARCHITECTURE ────────────────────────────────────────────────────
st.markdown("## Pipeline Architecture")
st.markdown("""
The project is structured as a **caching pipeline**: each stage reads from cache when
possible and writes parquet artifacts when it has to compute. This means the dashboard
loads in seconds (read-only) while the heavy computation happens once in `pipeline.py`.
""")

steps = [
    ("Data ingestion (src/data.py)",
     "Pull S&P 500 constituent list from Wikipedia, OHLCV data from Yahoo Finance "
     "via yfinance. Compute raw, market, and excess returns. Build long-format fact table."),
    ("Feature engineering (src/features.py)",
     "Compute per-stock-day technical indicators (RSI, MACD, Bollinger, ATR, OBV, beta, vol) "
     "and per-stock 7-feature behavioral fingerprints (winsorized + StandardScaler-normalized)."),
    ("Clustering (src/clustering.py)",
     "Run K-Means with elbow + silhouette diagnostics across K=2..15, Hierarchical (Ward), "
     "and DBSCAN. Compute Adjusted Rand Index against GICS sectors. Project to 2D via PCA."),
    ("Classification (src/classification.py)",
     "Train 4 models × 2 variants (with/without cluster label) on temporal train/test split. "
     "Save best Random Forest and feature importances."),
    ("Dashboard (app.py + pages/)",
     "Streamlit multi-page app reads cached artifacts and renders interactive Plotly charts."),
]
for i, (title, body) in enumerate(steps, 1):
    st.markdown(f"""
    <div class="step-card">
        <div><span class="step-num">{i}</span><b>{title}</b></div>
        <div style="margin-top:0.5rem; color:#444">{body}</div>
    </div>
    """, unsafe_allow_html=True)

# ── KEY DESIGN DECISIONS ─────────────────────────────────────────────────────
st.markdown("## Key Design Decisions")

decisions = [
    ("Cluster on excess returns, not raw returns",
     "Raw returns are dominated by market-wide moves (the 'AI boom' problem — tech, energy, and "
     "real estate stocks all move together because they all benefit from a single macro narrative). "
     "By subtracting the S&P 500 daily return before clustering, we get behavioral fingerprints "
     "that reflect stock-specific risk character, not shared factor exposure."),
    ("Aggregate per-stock fingerprints over the full 5-year window",
     "Clustering on per-day features would overweight recent regime; clustering on full-period "
     "averages captures structural character that persists across bull and bear markets. "
     "Five-year window covers pre-COVID, COVID crash, recovery, rate hikes, and AI boom."),
    ("Winsorize fingerprint features at 1st/99th percentile",
     "A handful of extreme outliers (typically meme stocks or post-IPO specials) would otherwise "
     "drag cluster centroids and inflate variance. Winsorizing keeps the structure stable without "
     "discarding any tickers."),
    ("K=5 for K-Means",
     "Chosen by inspecting elbow + silhouette curves and testing against interpretability. "
     "K=5 produces clusters with clear archetypal labels (defensive, high-beta growth, etc.) "
     "without splitting hairs that don't generalize."),
    ("Temporal train/test split (2019-2022 train, 2023-2024 test)",
     "Random shuffling time-series data leaks information. The 2023-2024 holdout includes "
     "the AI boom period — a stress test for whether the model generalizes to a new regime."),
    ("Price-normalize technical features before training",
     "MAs, Bollinger bands, and OBV scale with stock price level. Without normalization, "
     "high-priced stocks like NVDA would dominate. We convert MAs/Bollinger to ratios "
     "relative to adj_close and log-scale volume/OBV."),
    ("With/without cluster label experiment design",
     "Train every model twice — identical setup except for one column. Any AUC delta is "
     "attributable to the cluster label, isolating its causal contribution."),
]
for title, body in decisions:
    st.markdown(f"""
    <div class="design-decision">
        <b>🎯 {title}</b><br/>
        <span style="color:#444">{body}</span>
    </div>
    """, unsafe_allow_html=True)

# ── EVALUATION FRAMEWORK ─────────────────────────────────────────────────────
st.markdown("## Evaluation Framework")
st.markdown("""
| Metric | Part | Purpose |
|---|---|---|
| **ROC-AUC** | Prediction | Primary metric — threshold-independent classification quality |
| **Precision / Recall / F1** | Prediction | False positive cost matters in financial decisions |
| **Confusion matrix** | Prediction | Diagnostic for error types per model |
| **Silhouette score** | Clustering | How well-separated each cluster is internally |
| **Inertia (elbow)** | Clustering | Within-cluster sum of squares — supports K selection |
| **Adjusted Rand Index** | Clustering | Agreement with GICS sectors (0=independent, 1=identical) |
| **Naive baseline** | Both | Always-up accuracy ≈ 53% — model must beat this meaningfully |
""")

# ── REFERENCES ───────────────────────────────────────────────────────────────
st.markdown("## References & Reading List")

with st.expander("Foundational finance concepts"):
    st.markdown("""
    - [Investopedia — Technical Analysis Overview](https://www.investopedia.com/technical-analysis-4689657)
    - [Investopedia — RSI Explained](https://www.investopedia.com/terms/r/rsi.asp)
    - [Investopedia — MACD Explained](https://www.investopedia.com/terms/m/macd.asp)
    - [Investopedia — Bollinger Bands](https://www.investopedia.com/terms/b/bollingerbands.asp)
    - [Investopedia — GICS Sector Classification](https://www.investopedia.com/terms/g/gics.asp)
    - [Investopedia — Beta (Market Sensitivity)](https://www.investopedia.com/terms/b/beta.asp)
    - [Investopedia — Systematic vs. Unsystematic Risk](https://www.investopedia.com/terms/s/systematicrisk.asp)
    - Fama, E.F. (1970). "Efficient Capital Markets: A Review of Theory and Empirical Work." Journal of Finance.
    """)

with st.expander("Machine learning libraries & docs"):
    st.markdown("""
    - [scikit-learn — Clustering Guide](https://scikit-learn.org/stable/modules/clustering.html)
    - [scikit-learn — Classification Guide](https://scikit-learn.org/stable/supervised_learning.html)
    - [XGBoost Documentation](https://xgboost.readthedocs.io/)
    - [pandas — pct_change](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pct_change.html)
    """)

with st.expander("Data sources"):
    st.markdown("""
    - [yfinance — GitHub](https://github.com/ranaroussi/yfinance)
    - [Yahoo Finance — S&P 500](https://finance.yahoo.com/quote/%5EGSPC/)
    - [Wikipedia — List of S&P 500 companies](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
    - [FRED — Federal Reserve Economic Data](https://fred.stlouisfed.org/) (optional macro context)
    """)

with st.expander("Visualization"):
    st.markdown("""
    - [Plotly Python — Documentation](https://plotly.com/python/)
    - [Streamlit — Documentation](https://docs.streamlit.io/)
    """)

# ── REPRODUCIBILITY ──────────────────────────────────────────────────────────
st.markdown("## Reproducibility")
st.code("""# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (5–10 min on first run, instant after)
python pipeline.py

# Force a fresh run, ignoring cache
python pipeline.py --force

# Launch the dashboard
streamlit run app.py
""", language="bash")

st.markdown("---")
st.caption("Vishnu Peram · DATA 255 · San José State University · Spring 2026")
