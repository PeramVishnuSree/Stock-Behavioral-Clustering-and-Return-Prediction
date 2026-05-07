# Stock Behavioral Clustering & Return Prediction

DATA 255 Final Project — San José State University, Spring 2026.

This project asks whether S&P 500 stocks form data-driven behavioral clusters that
diverge from their official GICS sector classifications, and whether those cluster
labels improve short-term return-direction prediction.

## What's inside

```
code/
├── pipeline.py             # end-to-end orchestrator (run once)
├── app.py                  # Streamlit dashboard entry point
├── requirements.txt
├── src/
│   ├── data.py             # Yahoo Finance + Wikipedia → fact table
│   ├── features.py         # technical indicators + behavioral fingerprints
│   ├── clustering.py       # K-Means / Hierarchical / DBSCAN + ARI
│   ├── classification.py   # 4 models × 2 variants (with/without cluster)
│   └── viz.py              # Plotly chart helpers
├── pages/
│   ├── 1_📊_Data_Overview.py
│   ├── 2_📈_EDA.py
│   ├── 3_🎯_Clustering.py
│   ├── 4_🔮_Prediction.py
│   └── 5_📋_Methodology.py
├── data_cache/             # parquet artifacts (generated, gitignored)
└── EDA.ipynb               # original prototype notebook
```

## Setup

```bash
# 1. Create virtual environment & install dependencies
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the pipeline

```bash
# Pulls 5 years of OHLCV for ~503 S&P 500 tickers, computes features, runs
# clustering and classification. First run takes 5–10 minutes; subsequent
# runs use the parquet cache and are instant.
python pipeline.py

# Force a fresh run, ignoring all caches:
python pipeline.py --force
```

Artifacts land in `data_cache/`:

| File | Contents |
|---|---|
| `fact_table.parquet` | Long-format stock-day fact table (~743K rows; 503 → 501 tickers after delisting drops) |
| `sp500_table.parquet` | Ticker → GICS sector mapping |
| `market_returns.parquet` | Daily S&P 500 index returns |
| `technical_features.parquet` | Per-day RSI, MACD, Bollinger, ATR, OBV, beta, vol |
| `fingerprints_raw.parquet` | Per-stock 7-feature behavioral vector (raw) |
| `fingerprints_scaled.parquet` | Same, StandardScaler-normalized for clustering |
| `cluster_assignments.parquet` | KMeans / Hierarchical / DBSCAN labels per ticker |
| `cluster_diagnostics.parquet` | K-Means inertia + silhouette across K=2..15 |
| `pca_projection.parquet` | 2D PCA coordinates per ticker |
| `cluster_metrics.parquet` | Silhouette + ARI vs. GICS for each algorithm |
| `classification_metrics.parquet` | All 4 models × 2 variants (with/without cluster) |
| `feature_importance.parquet` | Random Forest feature ranking |
| `rf_model.joblib`, `boost_model.joblib` | Persisted models |

## Launching the dashboard

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` with five pages:

1. **Data Overview** — sources, schema, fact table preview
2. **EDA** — distributions, sector behavior, factor-exposure correlation analysis
3. **Clustering** — K-Means / Hierarchical / DBSCAN, GICS comparison, PCA scatter
4. **Prediction** — model performance, with/without cluster experiment, feature importance
5. **Methodology** — design choices, architecture, references

## Deployment to Streamlit Cloud

1. Push this repo to GitHub.
2. At [streamlit.io/cloud](https://streamlit.io/cloud), connect the repo and set the entry point to `app.py`.
3. The cloud worker runs `pipeline.py` automatically on first boot if `data_cache/` is empty (or commit the cache directory to ship pre-computed artifacts and skip the data pull).

## Key design choices

- **Cluster on excess (market-adjusted) returns**, not raw returns — strips away the
  systemic market component and prevents the AI-boom factor from distorting clusters.
- **Train-only behavioral fingerprints** — fingerprints aggregate the 2019–2022 window only,
  preventing data leakage when the cluster label is later used as a classification feature
  on the 2023–2024 test set. Run `python fix_leakage_and_baseline.py` after `pipeline.py`
  to regenerate clusters and classification metrics with the leakage-free, one-hot-encoded variant.
- **Temporal train/test split** (years < 2023 / 2023–2024) — never random shuffle on time series.
- **One-hot encode the cluster label** for classification — cluster IDs are nominal, not ordinal.
- **Price-normalize technical features** before classification so high-priced stocks
  don't dominate the model.
- **Test-set baseline** — the naive "always-up" baseline is computed on the test set only
  (2023–2024), which is the comparison the model has to beat to be meaningful.
- **Cache aggressively with parquet** — pipeline runs once, dashboard reads forever.

### Ticker count reconciliation

- **503** tickers scraped from Wikipedia
- **501** tickers had usable price data (2 delisted: SNDK, Q)
- **495** tickers survived training-window fingerprint requirements for clustering

See the **Methodology** page in the dashboard for full rationale, and `REPORT.md` for detailed results.
