"""
Page 3 — Clustering.
K-Means, Hierarchical, DBSCAN — and the comparison vs. GICS sectors.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
CACHE_DIR = PROJECT_ROOT / "data_cache"

from src import viz  # noqa: E402
from src.features import FINGERPRINT_FEATURES  # noqa: E402

st.set_page_config(page_title="Clustering", page_icon="🎯", layout="wide")
st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 1280px; }
    h1, h2, h3 { color: #1B3A5C; }
    .insight { background: #F4F8FC; border-left: 4px solid #2E6DA4;
               padding: 0.8rem 1.2rem; border-radius: 4px; margin: 0.8rem 0; }
    .stat-card { background: white; padding: 1rem 1.2rem; border-radius: 8px;
                 border: 1px solid #E1E8F0; }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Behavioral Clustering")
st.markdown("Group S&P 500 stocks by 5-year behavioral fingerprints and compare against GICS sectors.")

needed = ["cluster_assignments.parquet", "cluster_diagnostics.parquet",
           "cluster_metrics.parquet", "fingerprints_raw.parquet"]
if not all((CACHE_DIR / f).exists() for f in needed):
    st.error("Clustering artifacts not found. Run `python pipeline.py` first.")
    st.stop()

@st.cache_data
def load():
    return {
        "assign":    pd.read_parquet(CACHE_DIR / "cluster_assignments.parquet"),
        "diag":      pd.read_parquet(CACHE_DIR / "cluster_diagnostics.parquet"),
        "metrics":   pd.read_parquet(CACHE_DIR / "cluster_metrics.parquet"),
        "raw_fp":    pd.read_parquet(CACHE_DIR / "fingerprints_raw.parquet"),
        "pca":       pd.read_parquet(CACHE_DIR / "pca_projection.parquet"),
    }

D = load()
assign = D["assign"]

# ── 1. THE FINGERPRINT FEATURES ──────────────────────────────────────────────
st.markdown("## 1. Behavioral Fingerprint")
st.markdown(f"""
Each stock is summarized into a single 7-feature vector aggregated across 5 years
of trading history. Features are winsorized at 1st/99th percentile and standardized
before clustering.
""")

c1, c2 = st.columns([1.4, 1])
with c1:
    desc = pd.DataFrame({
        "Feature": FINGERPRINT_FEATURES,
        "Captures": [
            "Average alpha (return above market)",
            "Std dev of daily excess returns",
            "Average 60-day rolling beta — market sensitivity",
            "Average RSI level — momentum tendency",
            "Average Bollinger band width — typical volatility regime",
            "Worst peak-to-trough loss — tail risk",
            "Average 12-month rolling return — trend behavior",
        ],
    })
    st.dataframe(desc, use_container_width=True, hide_index=True)
with c2:
    st.markdown(f"""
    <div class="stat-card">
        <div style="font-size:1.6rem; font-weight:700; color:#2E6DA4">{len(D['raw_fp'])}</div>
        <div style="color:#666">Stocks fingerprinted</div>
        <hr style="margin:0.6rem 0; border:none; border-top:1px solid #E1E8F0">
        <div style="font-size:1.6rem; font-weight:700; color:#2E6DA4">{len(FINGERPRINT_FEATURES)}</div>
        <div style="color:#666">Features per stock</div>
    </div>
    """, unsafe_allow_html=True)

# ── 2. K-MEANS DIAGNOSTICS ───────────────────────────────────────────────────
st.markdown("## 2. Choosing K — Elbow & Silhouette")
st.plotly_chart(viz.elbow_silhouette(D["diag"]), use_container_width=True)
best_k = int(D["diag"].iloc[D["diag"]["silhouette"].idxmax()]["k"])
st.markdown(f"""
<div class="insight">
<b>Reading the chart.</b> Inertia (blue) drops sharply through K=4–6 then flattens — the classic
"elbow." Silhouette (amber) peaks at K={best_k}. We use <b>K=5</b> as a balance between
structural clarity and interpretability.
</div>
""", unsafe_allow_html=True)

# ── 3. PCA SCATTER ───────────────────────────────────────────────────────────
st.markdown("## 3. PCA Projection")
st.markdown("Project the 7-D fingerprint space onto its top 2 principal components for visualization. "
             "Toggle between coloring by behavioral cluster or by GICS sector.")

color_choice = st.radio("Color by", ["Behavioral cluster", "GICS sector"], horizontal=True)
key = "kmeans" if color_choice == "Behavioral cluster" else "gics_sector"
st.plotly_chart(viz.pca_cluster_scatter(assign, color_by=key), use_container_width=True)

evr1, evr2 = D["pca"][["evr1", "evr2"]].iloc[0]
st.caption(f"PC1 explains {evr1:.1%} of variance · PC2 explains {evr2:.1%} · "
            f"combined: {(evr1+evr2):.1%}")

# ── 4. CLUSTER × SECTOR HEATMAP ──────────────────────────────────────────────
st.markdown("## 4. How well do clusters match GICS sectors?")
st.plotly_chart(viz.cluster_sector_heatmap(assign), use_container_width=True)

ari_kmeans = float(D["metrics"].loc[D["metrics"]["algorithm"] == "kmeans", "ari_vs_sector"].iloc[0])
st.markdown(f"""
<div class="insight">
<b>Adjusted Rand Index (K-Means vs. GICS): {ari_kmeans:.3f}.</b>
ARI ranges from 0 (independent) to 1 (identical). Our value indicates {"weak" if ari_kmeans < 0.2 else "moderate"} agreement —
behavioral clusters cut <i>across</i> sectors rather than replicating them. Tech stocks
spread across multiple clusters; defensive sectors (Utilities, Staples, REITs) often pool
into a single "low-volatility" cluster regardless of industry.
</div>
""", unsafe_allow_html=True)

# ── 5. CLUSTER PROFILES ──────────────────────────────────────────────────────
st.markdown("## 5. Cluster Profiles")
st.plotly_chart(viz.cluster_profile_radar(D["raw_fp"], assign, FINGERPRINT_FEATURES),
                 use_container_width=True)

profiles = (D["raw_fp"].merge(assign[["ticker", "kmeans"]], on="ticker")
                       .groupby("kmeans")[FINGERPRINT_FEATURES].mean()
                       .round(3))
st.markdown("**Centroid values per cluster (raw, un-normalized):**")
st.dataframe(profiles, use_container_width=True)

# ── 6. ALGORITHM COMPARISON ──────────────────────────────────────────────────
st.markdown("## 6. All Three Algorithms — Side by Side")
st.dataframe(
    D["metrics"].rename(columns={
        "algorithm": "Algorithm", "n_clusters": "Clusters",
        "n_outliers": "Outliers", "silhouette": "Silhouette",
        "ari_vs_sector": "ARI vs. GICS",
    }).round(3),
    use_container_width=True, hide_index=True,
)

# ── 7. STOCK LOOKUP ──────────────────────────────────────────────────────────
st.markdown("## 7. Stock Lookup")
selected = st.selectbox("Pick a ticker:", sorted(assign["ticker"].unique()))
row = assign[assign["ticker"] == selected].iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric("GICS sector", row["gics_sector"])
c2.metric("K-Means cluster", f"Cluster {row['kmeans']}")
c3.metric("Hierarchical cluster", f"Cluster {row['hierarchical']}")

peers = assign[assign["kmeans"] == row["kmeans"]]["ticker"].tolist()
peers.remove(selected)
st.markdown(f"**Behavioral peers** (same K-Means cluster, {len(peers)} stocks):")
st.write(", ".join(peers[:30]) + (f" … and {len(peers)-30} more" if len(peers) > 30 else ""))

st.markdown("---")
st.caption("Clustering inputs: 7-feature behavioral fingerprints, StandardScaler-normalized, K=5.")
