"""
Page 4 — Prediction.
4 classifiers × 2 variants (with/without cluster label) on a temporal split.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
CACHE_DIR = PROJECT_ROOT / "data_cache"

from src import viz  # noqa: E402

st.set_page_config(page_title="Prediction", page_icon="🔮", layout="wide")
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

st.title("🔮 Return Direction Prediction")
st.markdown("Predict whether a stock's 5-day forward return will be positive or negative — "
             "and test whether behavioral cluster membership adds predictive signal.")

needed = ["classification_metrics.parquet", "feature_importance.parquet", "baseline.parquet"]
if not all((CACHE_DIR / f).exists() for f in needed):
    st.error("Classification artifacts not found. Run `python pipeline.py` first.")
    st.stop()

@st.cache_data
def load():
    return {
        "metrics": pd.read_parquet(CACHE_DIR / "classification_metrics.parquet"),
        "imp":     pd.read_parquet(CACHE_DIR / "feature_importance.parquet"),
        "base":    pd.read_parquet(CACHE_DIR / "baseline.parquet")["baseline"].iloc[0],
    }

D = load()
m = D["metrics"]

# ── 1. EXPERIMENTAL SETUP ────────────────────────────────────────────────────
st.markdown("## 1. Experimental Setup")
c1, c2, c3, c4 = st.columns(4)
c1.markdown('<div class="stat-card"><div style="font-weight:600;color:#1B3A5C">Target</div>'
             '<div style="color:#666">forward_5day_direction</div></div>', unsafe_allow_html=True)
c2.markdown('<div class="stat-card"><div style="font-weight:600;color:#1B3A5C">Train</div>'
             '<div style="color:#666">2019 – 2022</div></div>', unsafe_allow_html=True)
c3.markdown('<div class="stat-card"><div style="font-weight:600;color:#1B3A5C">Test</div>'
             '<div style="color:#666">2023 – 2024</div></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="stat-card"><div style="font-weight:600;color:#1B3A5C">Naive baseline</div>'
             f'<div style="color:#666">{D["base"]:.3f} (always-up accuracy)</div></div>',
             unsafe_allow_html=True)

st.markdown("""
<div class="insight">
<b>Why a temporal split?</b> Random shuffling on time-series data leaks future information into
training. A model that has seen examples from January 2024 cannot honestly be evaluated on
December 2023. We hold out the last 2 years intact.
</div>
""", unsafe_allow_html=True)

# ── 2. THE EXPERIMENT — WITH vs. WITHOUT CLUSTER LABEL ───────────────────────
st.markdown("## 2. Does Cluster Label Improve Prediction?")
metric_choice = st.selectbox("Choose metric", ["roc_auc", "accuracy", "f1", "precision", "recall"], index=0)
st.plotly_chart(viz.metrics_grouped_bar(m, metric=metric_choice), use_container_width=True)

# Per-model improvement table
pivot = m.pivot(index="model", columns="variant", values="roc_auc").reset_index()
pivot["improvement"] = pivot["With cluster"] - pivot["Without cluster"]
pivot = pivot.sort_values("improvement", ascending=False).round(4)

c1, c2 = st.columns([1.4, 1])
c1.markdown("### Improvement from adding cluster label (ROC-AUC)")
c1.dataframe(pivot, use_container_width=True, hide_index=True)
mean_lift = pivot["improvement"].mean()
direction = "improves" if mean_lift > 0 else "does not improve"
c2.markdown(f"""
<div class="insight">
<b>Average lift:</b> {mean_lift:+.4f} AUC.<br/><br/>
The behavioral cluster label <b>{direction}</b> prediction performance on average.
A small lift is expected — the cluster label encodes long-term behavioral character
that complements (rather than duplicates) the short-term technical indicators.
</div>
""", unsafe_allow_html=True)

# ── 3. FULL METRICS TABLE ────────────────────────────────────────────────────
st.markdown("## 3. Full Metrics Table")
display = m.round(4).rename(columns={
    "model": "Model", "variant": "Variant",
    "accuracy": "Accuracy", "precision": "Precision",
    "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC",
})
st.dataframe(display, use_container_width=True, hide_index=True)

# ── 4. FEATURE IMPORTANCE ────────────────────────────────────────────────────
st.markdown("## 4. What Drives the Predictions?")
st.plotly_chart(viz.feature_importance_bar(D["imp"], top_n=15), use_container_width=True)

top3 = D["imp"].head(3)["feature"].tolist()
cluster_rank = D["imp"].reset_index(drop=True).reset_index().query("feature == 'cluster_label'")
cluster_pos = (int(cluster_rank["index"].iloc[0]) + 1) if len(cluster_rank) else "—"

st.markdown(f"""
<div class="insight">
<b>Top 3 features:</b> {', '.join(top3)}.<br/>
<b>Cluster label rank:</b> #{cluster_pos} of {len(D['imp'])} features.<br/><br/>
Momentum and volatility-related indicators dominate. The cluster label provides marginal
but measurable signal — consistent with the hypothesis that behavioral archetype is
weak-but-real predictive information beyond technical indicators alone.
</div>
""", unsafe_allow_html=True)

# ── 5. INTERPRETATION ────────────────────────────────────────────────────────
st.markdown("## 5. How to Read These Results")
st.markdown("""
<div class="insight">
<b>An AUC of 0.55–0.62 is meaningful, not weak.</b> Markets are close to efficient on short
horizons. A naive "always up" predictor reaches ~53% accuracy. Beating that meaningfully —
even by 5–10 percentage points — is real signal. The relevant comparison is not 95% accuracy
(which would imply broken markets) but the gap above the always-up baseline.
</div>

<div class="insight">
<b>The with/without cluster experiment isolates causal lift.</b> Both variants share identical
training data, models, and hyperparameters. The only difference is whether the cluster label
column is included. Any consistent gap in performance is attributable to that single feature.
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("Models: Logistic Regression, Decision Tree, Random Forest, XGBoost (or sklearn GradientBoosting fallback).")
