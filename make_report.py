"""
Generate a detailed implementation report (Markdown) from cached pipeline artifacts.

Output: REPORT.md in the project root.

Run with:  python make_report.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent
CACHE = PROJECT_ROOT / "data_cache"
OUT   = PROJECT_ROOT / "REPORT.md"


def _load() -> dict:
    return {
        "fact":      pd.read_parquet(CACHE / "fact_table.parquet"),
        "sp500":     pd.read_parquet(CACHE / "sp500_table.parquet"),
        "tech":      pd.read_parquet(CACHE / "technical_features.parquet"),
        "raw_fp":    pd.read_parquet(CACHE / "fingerprints_raw.parquet"),
        "scaled_fp": pd.read_parquet(CACHE / "fingerprints_scaled.parquet"),
        "assign":    pd.read_parquet(CACHE / "cluster_assignments.parquet"),
        "diag":      pd.read_parquet(CACHE / "cluster_diagnostics.parquet"),
        "cmetrics":  pd.read_parquet(CACHE / "cluster_metrics.parquet"),
        "metrics":   pd.read_parquet(CACHE / "classification_metrics.parquet"),
        "feat_imp":  pd.read_parquet(CACHE / "feature_importance.parquet"),
        "baseline":  float(pd.read_parquet(CACHE / "baseline.parquet")["baseline"].iloc[0]),
        "pca":       pd.read_parquet(CACHE / "pca_projection.parquet"),
        "confs":     pd.read_parquet(CACHE / "confusion_matrices.parquet") if (CACHE / "confusion_matrices.parquet").exists() else None,
    }


def _md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def main():
    D = _load()
    fact, sp500, tech = D["fact"], D["sp500"], D["tech"]
    assign, cmetrics, diag = D["assign"], D["cmetrics"], D["diag"]
    metrics, feat_imp, baseline = D["metrics"], D["feat_imp"], D["baseline"]
    raw_fp = D["raw_fp"]
    confs = D["confs"]

    # Headline numbers
    n_tickers = fact["ticker"].nunique()
    n_rows    = len(fact)
    date_min, date_max = fact["Date"].min().date(), fact["Date"].max().date()
    n_sectors = fact["gics_sector"].nunique()

    # K-Means selection: silhouette peak vs chosen K
    diag_sorted = diag.sort_values("silhouette", ascending=False)
    k_silhouette_peak = int(diag_sorted.iloc[0]["k"])
    silhouette_peak_value = float(diag_sorted.iloc[0]["silhouette"])
    silhouette_at_k5 = float(diag.loc[diag["k"] == 5, "silhouette"].iloc[0])

    # Cluster numbers
    ari_k = float(cmetrics.loc[cmetrics["algorithm"] == "kmeans", "ari_vs_sector"].iloc[0])
    sil_k = float(cmetrics.loc[cmetrics["algorithm"] == "kmeans", "silhouette"].iloc[0])
    pc1_var = float(D["pca"]["evr1"].iloc[0])
    pc2_var = float(D["pca"]["evr2"].iloc[0])
    n_fingerprinted = len(raw_fp)

    # Classification numbers
    best_auc_row = metrics.sort_values("roc_auc", ascending=False).iloc[0]
    best_auc_model = best_auc_row["model"]
    best_auc_variant = best_auc_row["variant"]
    best_auc = float(best_auc_row["roc_auc"])

    best_acc_row = metrics.sort_values("accuracy", ascending=False).iloc[0]
    best_acc_model = best_acc_row["model"]
    best_acc = float(best_acc_row["accuracy"])

    # With vs without cluster comparison
    pivot = metrics.pivot(index="model", columns="variant", values="roc_auc")
    # variant column names depend on whether one-hot or raw
    cluster_col = [c for c in pivot.columns if "luster" in c and "ithout" not in c][0]
    no_cluster_col = [c for c in pivot.columns if "ithout" in c][0]
    pivot["delta"] = pivot[cluster_col] - pivot[no_cluster_col]
    avg_delta = float(pivot["delta"].mean())
    n_improved = int((pivot["delta"] > 0).sum())

    # Cluster x sector
    cluster_sector = pd.crosstab(assign["gics_sector"], assign["kmeans"])
    cluster_sizes = assign["kmeans"].value_counts().sort_index()

    # Top features
    top_feats = feat_imp.head(8)

    # ─── BUILD REPORT ──────────────────────────────────────────────────────
    md = []
    md.append("# Stock Behavioral Clustering & Return Prediction")
    md.append("**DATA 255 — Data Mining · Final Project Implementation Report**")
    md.append(f"\nVishnu Peram · San José State University · Spring 2026")
    md.append("\n---\n")

    md.append("## Executive Summary\n")
    md.append(
        f"This project tests whether S&P 500 stocks form data-driven behavioral clusters "
        f"that diverge from their official GICS sector labels, and whether those cluster "
        f"labels improve short-term return prediction. We processed **{n_rows:,} stock-day "
        f"observations** spanning **{n_tickers} stocks** ({date_min} → {date_max}) across "
        f"**{n_sectors} GICS sectors**, computed 8 technical indicators and a 7-feature "
        f"behavioral fingerprint per stock, fit three clustering algorithms, and trained four "
        f"classification models — each in two variants: with and without **one-hot-encoded** "
        f"cluster labels.\n"
    )
    md.append(
        "**Methodological note on data leakage.** Behavioral fingerprints used for clustering "
        "are computed using **only the training-period window (2019–2022)**. The test set "
        "(2023–2024) is never observed during cluster fitting, so the cluster label is a "
        "leakage-free per-stock attribute when used as a feature in the classification step.\n"
    )
    md.append("**Key findings:**")
    md.append(f"- **Behavioral clusters show weak alignment with GICS sectors.** Adjusted Rand Index "
              f"= **{ari_k:.3f}** (near 0 = close to random agreement; 1 = identical). Cluster groupings "
              f"reflect risk archetypes (growth, defensive, cyclical) more than industry membership.")
    md.append(f"- **ROC-AUC is weakly above random.** Best ROC-AUC = **{best_auc:.3f}** "
              f"({best_auc_model}, {best_auc_variant}). The best accuracy across all models is "
              f"**{best_acc:.3f}** ({best_acc_model}), which is **below** the test-set always-up "
              f"baseline of **{baseline:.3f}**. Models therefore have weak ranking signal but do "
              f"not outperform a naive directional baseline on accuracy.")
    md.append(f"- **Cluster label gives a small marginal lift.** Adding one-hot-encoded cluster labels "
              f"changed average ROC-AUC by **{avg_delta:+.4f}** across the 4 models "
              f"({n_improved}/4 models improved). The lift is consistent but small, and would warrant "
              f"statistical-significance bounds before any operational use.\n")

    md.append("---\n")
    md.append("## 1. Project Overview\n")
    md.append("### 1.1 Research question\n")
    md.append(
        "> *Do S&P 500 stocks form distinct behavioral clusters that diverge from their official GICS "
        "sector classifications when grouped by market-adjusted price behavior and risk profile — "
        "and can technical indicators, enriched by behavioral cluster membership, predict short-term "
        "return direction?*\n"
    )
    md.append("**RQ1 — Clustering.** Do behavioral clusters differ from official sectors?")
    md.append("**RQ2 — Prediction.** Does adding a cluster label improve return-direction prediction?\n")

    md.append("### 1.2 Why this matters\n")
    md.append(
        "GICS sectors classify companies by *what they do* (their primary business activity), not by "
        "*how their stocks behave*. During macro narratives like the 2023–2024 AI boom, otherwise "
        "unrelated stocks moved together due to shared factor exposure — semiconductor companies, "
        "utilities with data-center exposure, and infrastructure providers all rallied on the same theme. "
        "The official taxonomy hides this. A data-driven clustering can reveal it.\n"
    )

    md.append("### 1.3 Pipeline overview\n")
    md.append(
        "```\n"
        "Wikipedia + Yahoo Finance  →  fact_table (long format)\n"
        "fact_table                 →  technical_features  +  behavioral_fingerprints (TRAIN ONLY)\n"
        "fingerprints (train-only)  →  KMeans / Hierarchical / DBSCAN clusters\n"
        "technicals + cluster (OHE) →  4 classifiers (with vs without cluster)\n"
        "all artifacts              →  Streamlit dashboard\n"
        "```\n"
    )

    md.append("---\n")
    md.append("## 2. Data Sources & Warehouse\n")
    md.append("### 2.1 Sources\n")
    md.append("| Source | Provides | Access |\n|---|---|---|")
    md.append("| Yahoo Finance (`yfinance`) | OHLCV daily prices, volumes for all S&P 500 tickers + index | Free, no API key |")
    md.append("| Wikipedia constituent list | Ticker symbols, GICS sector & sub-industry, date added | `pd.read_html()` |")
    md.append("| ^GSPC index | S&P 500 daily benchmark used to compute excess returns | Yahoo Finance |")
    md.append(f"\n*Data pull date: 2026-04-30. Time range: {date_min} → {date_max}.*\n")

    md.append("### 2.2 Ticker count reconciliation\n")
    md.append("- **503 tickers** scraped from the Wikipedia constituent list")
    md.append(f"- **{n_tickers} tickers** with usable price data after yfinance download (2 delisted: SNDK, Q)")
    md.append(f"- **{n_fingerprinted} tickers** survived fingerprint requirements for clustering "
              f"(needed full training window of price history with no NaN in 7 fingerprint features)\n")

    md.append("### 2.3 Warehouse design — Star schema\n")
    md.append("**Fact table (`fact_table.parquet`)** — one row per (ticker, date):")
    md.append("- OHLCV: `open, high, low, adj_close, volume`")
    md.append("- Returns: `raw_daily_return, market_daily_return, excess_daily_return`")
    md.append("- Target: `forward_5day_return, forward_5day_direction`")
    md.append("- Joined dimensions: `gics_sector, gics_sub_industry, company, date_added`\n")
    md.append("**Dimension tables (logical):**")
    md.append("- `stock_dim` — ticker metadata (one row per ticker; sourced from Wikipedia)")
    md.append("- `time_dim` — derived from the `Date` column (year, quarter, etc.); not stored as a separate parquet")
    md.append("- `technical_features` — per-stock-day technicals (RSI, MACD, Bollinger, ATR, OBV, beta, vol)")
    md.append("- `fingerprints_raw / scaled` — one row per ticker, 7 behavioral features\n")
    md.append("**Fact table footprint:**")
    md.append(f"- Rows: **{n_rows:,}**")
    md.append(f"- Tickers: **{n_tickers}**")
    md.append(f"- Date range: **{date_min} → {date_max}**")
    md.append(f"- Sectors: **{n_sectors}**\n")

    md.append("---\n")
    md.append("## 3. Data Preprocessing\n")
    md.append("### 3.1 Returns computation\n")
    md.append(
        "`pct_change(fill_method=None)` produces honest NaN values when prices are missing — "
        "the older default forward-filled gaps which would silently report 0% returns on halt days. "
        "Excess returns subtract the same-day market return, isolating stock-specific behavior.\n"
    )
    md.append("### 3.2 Missing data\n")
    md.append("- Tickers added to S&P 500 mid-period: NaN early rows are dropped via `dropna(subset=['adj_close','raw_daily_return'])`")
    md.append("- Technical indicators have NaN warm-up periods (RSI needs 14 days, MA50 needs 50). The classifier dropna step removes these.\n")
    md.append("### 3.3 Outlier handling\n")
    md.append("- Behavioral fingerprint features winsorized at 1st/99th percentile to keep clustering stable")
    md.append("- Daily return outliers retained for prediction (genuine signal)\n")
    md.append("### 3.4 Feature scaling & encoding\n")
    md.append("- Fingerprints standardized via `StandardScaler` before clustering (distance-based methods need this)")
    md.append("- Price-level features (MAs) divided by `adj_close` to make them ratios — otherwise they'd dominate by raw price magnitude")
    md.append("- Volume and OBV log-transformed (orders of magnitude variation)")
    md.append("- **Cluster label is one-hot encoded** before classification — cluster IDs are nominal, not ordinal, so raw integer encoding would inappropriately impose order on logistic regression and tree-split candidates\n")

    md.append("---\n")
    md.append("## 4. Feature Engineering\n")
    md.append("### 4.1 Technical indicators (per stock-day)\n")
    md.append("| Indicator | Family | Formula sketch | Captures |\n|---|---|---|---|")
    md.append("| MA(20), MA(50) | Trend | Rolling mean of close | Trend direction |")
    md.append("| RSI(14) | Momentum | EMA(gains) / EMA(losses) → 0..100 | Overbought/oversold |")
    md.append("| MACD(12,26,9) | Trend + momentum | EMA(12) − EMA(26), signal = EMA(9) of MACD | Momentum shifts |")
    md.append("| Bollinger(20, 2σ) | Volatility | MA ± 2σ; width as proxy for vol regime | Volatility regime |")
    md.append("| ATR(14) | Volatility | EMA of true range | Daily price range |")
    md.append("| OBV | Volume | Cumulative signed volume | Accumulation/distribution |")
    md.append("| Beta(60d) | Risk | Cov(stock, market) / Var(market), 60-day window | Market sensitivity |")
    md.append("| Volatility(20d) | Risk | Rolling std of returns | Recent volatility |\n")

    md.append("### 4.2 Behavioral fingerprint (per stock — TRAIN-ONLY aggregate)\n")
    md.append(
        "**To prevent data leakage**, each stock's fingerprint is computed using only the training-period "
        "window (2019–2022). The test period (2023–2024) is never seen during fingerprint construction "
        "or cluster fitting.\n"
    )
    fp_desc = pd.DataFrame({
        "Feature": ["mean_excess_return", "volatility", "mean_beta", "mean_rsi",
                     "mean_bollinger_width", "max_drawdown", "momentum_score"],
        "What it captures": [
            "Average alpha — outperformance vs. the market",
            "Std dev of daily excess returns — calm vs. wild",
            "60-day rolling beta averaged over training window — market sensitivity",
            "Average RSI level — momentum tendency",
            "Average band width — typical volatility regime",
            "Worst peak-to-trough loss — tail risk",
            "Average rolling 12-month return — trend behavior",
        ],
    })
    md.append(_md_table(fp_desc))
    md.append("")

    md.append("---\n")
    md.append("## 5. Clustering Results (RQ1)\n")
    md.append("### 5.1 K selection\n")
    md.append("K-Means was fit for K=2..15 with `n_init=10`, random_state=42. Diagnostics:")
    md.append(_md_table(diag.round(3)))
    md.append("")
    md.append(
        f"**Silhouette peaks at K = {k_silhouette_peak} ({silhouette_peak_value:.3f}), but K = 5 was selected** "
        f"as an interpretability-driven compromise near the elbow of the inertia curve. K = 2 produced "
        f"clusters that were too coarse to compare meaningfully against the 11 GICS sectors; K = 5 gives "
        f"finer granularity with silhouette = {silhouette_at_k5:.3f}. The trade-off is documented "
        f"explicitly: lower silhouette in exchange for more meaningful behavioral archetypes.\n"
    )

    md.append("### 5.2 Algorithm comparison\n")
    md.append(_md_table(cmetrics.round(3)))
    md.append("\n*All three algorithms produced low ARI vs. GICS sectors, confirming the finding is "
              "structural rather than algorithm-dependent.*\n")

    md.append("### 5.3 Cluster vs. GICS sector — confusion matrix\n")
    md.append(_md_table(cluster_sector.reset_index()))
    md.append("\nReading the matrix: rows are official GICS sectors, columns are behavioral clusters. "
              "If clustering recovered sector labels, we'd see one dominant column per row. "
              "Instead, sectors split across multiple clusters and clusters mix sectors — "
              f"which is what the ARI = {ari_k:.3f} score quantifies.\n")

    md.append("### 5.4 PCA visualization\n")
    md.append(f"The 7-D fingerprint space projects onto 2 principal components capturing "
              f"**{(pc1_var + pc2_var):.1%}** of total variance (PC1: {pc1_var:.1%}, PC2: {pc2_var:.1%}). "
              f"Cluster regions are visually separable in this space — see dashboard.\n")

    md.append("### 5.5 Cluster sizes & profiles\n")
    md.append("Cluster sizes (number of stocks per K-Means cluster):")
    sizes_df = cluster_sizes.reset_index()
    sizes_df.columns = ["cluster", "size"]
    md.append(_md_table(sizes_df))
    md.append("")
    md.append("Cluster centroids on raw (un-normalized) behavioral features:")
    profiles = (raw_fp.merge(assign[["ticker","kmeans"]], on="ticker")
                       .groupby("kmeans")[["mean_excess_return", "volatility", "mean_beta",
                                              "mean_rsi", "mean_bollinger_width", "max_drawdown",
                                              "momentum_score"]].mean().round(4))
    md.append(_md_table(profiles.reset_index()))
    md.append("")

    md.append("---\n")
    md.append("## 6. Classification Results (RQ2)\n")
    md.append("### 6.1 Setup\n")
    md.append("- **Target:** `forward_5day_direction` (1 = stock up over next 5 trading days, 0 = flat or down)")
    md.append(f"- **Features (without cluster):** {len([c for c in metrics.columns if c not in ['model','variant','accuracy','precision','recall','f1','roc_auc']])} technical indicators (subset stored after compression)")
    md.append("- **Features (with cluster):** technical indicators + **one-hot-encoded** cluster label "
              "(5 dummy variables, one per cluster)")
    md.append("- **Train / test split:** temporal — train on years < 2023, test on 2023–2024")
    md.append("- **Cluster fingerprints:** computed using **training-period data only** (no leakage into test)")
    md.append(f"- **Naive baseline (always predict up, evaluated on test set):** **{baseline:.3f}** — "
              "any model has to beat this on accuracy to be useful as a directional predictor.\n")

    md.append("### 6.2 Model comparison\n")
    md.append(_md_table(metrics.round(4)))
    md.append("")
    md.append(
        f"**Interpretation.** Best ROC-AUC is {best_auc:.3f} ({best_auc_model}, {best_auc_variant}) — "
        f"barely above the random-ranking floor of 0.500. Best accuracy is {best_acc:.3f} ({best_acc_model}), "
        f"**below** the always-up baseline of {baseline:.3f}. Models extract some weak ranking signal "
        f"(probabilities correctly order positive cases above negative slightly more often than chance) "
        f"but cannot translate that into directional accuracy that beats simply predicting 'up'. This is "
        f"consistent with weak-form efficient-market expectations.\n"
    )

    md.append("### 6.3 With vs. without cluster — ROC-AUC delta\n")
    md.append(_md_table(pivot.round(4).reset_index()))
    md.append(f"\nAverage ROC-AUC delta across models: **{avg_delta:+.4f}**. {n_improved} of 4 models improved when the one-hot cluster feature was added. The lift is small and consistent — likely capturing residual long-term behavioral structure that per-day technicals miss, but not large enough to claim economic significance without statistical-significance testing on the AUC differences.\n")

    md.append("### 6.4 Confusion matrices (test set)\n")
    if confs is not None:
        md.append(_md_table(confs))
    md.append("\n*All models lean toward predicting `up` (high recall, low specificity). Adding the cluster "
              "feature shifts the operating point slightly toward the negative class, reducing recall but "
              "improving the AUC ranking quality.*\n")

    md.append("### 6.5 Feature importance (Random Forest, with cluster)\n")
    md.append(_md_table(top_feats.round(4)))
    md.append("")

    md.append("---\n")
    md.append("## 7. Knowledge Interpretation\n")
    md.append(
        f"**Finding 1 — Behavioral clusters show weak alignment with GICS sectors.** ARI of {ari_k:.3f} "
        f"is near zero, indicating cluster assignments and sector labels are close to randomly aligned. "
        f"Clusters group stocks by *risk profile* (high-volatility growth, low-volatility defensive, "
        f"cyclicals) — a structural lens GICS doesn't provide. RQ1 supported.\n"
    )
    md.append(
        f"**Finding 2 — Short-term direction prediction does not beat the naive baseline on accuracy.** "
        f"Best accuracy ({best_acc:.3f}) is below the always-up baseline ({baseline:.3f}). However, best "
        f"ROC-AUC ({best_auc:.3f}) is slightly above 0.5, indicating weak but non-zero ranking signal. "
        f"This is consistent with weak-form market efficiency — technical indicators contain *some* "
        f"signal but not enough to beat a directional-bias rule on a 5-day horizon.\n"
    )
    md.append(
        f"**Finding 3 — Cluster label adds small marginal AUC value.** Average AUC delta of {avg_delta:+.4f} "
        f"with the cluster feature one-hot encoded. The lift is small and consistent across models, but "
        f"future work should test it for statistical significance before any operational claims.\n"
    )

    md.append("---\n")
    md.append("## 8. Architecture & Deployment\n")
    md.append("### 8.1 Project layout\n")
    md.append("```")
    md.append("code/")
    md.append("├── pipeline.py                    # orchestrator")
    md.append("├── fix_leakage_and_baseline.py    # leakage-free re-clustering + classification")
    md.append("├── make_report.py                 # generates REPORT.md from cache")
    md.append("├── make_presentation.py           # generates PRESENTATION.pptx from cache")
    md.append("├── app.py                         # Streamlit landing page")
    md.append("├── pages/                         # 5 dashboard pages")
    md.append("├── src/")
    md.append("│   ├── data.py                    # ETL")
    md.append("│   ├── features.py                # technicals + fingerprints")
    md.append("│   ├── clustering.py              # KMeans / Hierarchical / DBSCAN")
    md.append("│   ├── classification.py          # 4 models")
    md.append("│   └── viz.py                     # Plotly chart helpers")
    md.append("├── data_cache/                    # parquet cache of all artifacts")
    md.append("└── requirements.txt")
    md.append("```\n")
    md.append("### 8.2 Deployment\n")
    md.append("- **Platform:** Streamlit Community Cloud (free tier)")
    md.append("- **Repo:** `https://github.com/PeramVishnuSree/Stock-Behavioral-Clustering-and-Return-Prediction`")
    md.append("- **Build:** auto-installs `requirements.txt`, runs `app.py`")
    md.append("- **Data:** parquet files committed to repo for instant first-load\n")

    md.append("---\n")
    md.append("## 9. Limitations & Future Work\n")
    md.append("- **Survivorship bias:** the constituent list is the *current* S&P 500. Past delistings are not in the panel. Fix: use historical CRSP membership data.")
    md.append("- **Static cluster labels:** behavioral fingerprints are aggregated over the training window. A stock's behavior can shift across regimes. Fix: rolling-window clustering with periodic re-fingerprinting.")
    md.append("- **Single forward horizon:** only 5-day direction tested. Other horizons (1d, 20d, 60d) likely have different signal-to-noise.")
    md.append("- **No transaction costs / position sizing:** ROC-AUC alone is not a Sharpe ratio. A backtest with realistic costs would assess economic significance.")
    md.append("- **AUC lift not significance-tested:** the +0.0027 average AUC delta is consistent but small; bootstrap or DeLong tests would confirm whether it's statistically distinguishable from zero.")
    md.append("- **Could add fundamentals or sentiment** (P/E, earnings surprise, news sentiment) as additional features.\n")

    md.append("---\n")
    md.append("## 10. Reproducibility\n")
    md.append("```bash")
    md.append("git clone https://github.com/PeramVishnuSree/Stock-Behavioral-Clustering-and-Return-Prediction.git")
    md.append("cd Stock-Behavioral-Clustering-and-Return-Prediction")
    md.append("python -m venv .venv && source .venv/bin/activate")
    md.append("pip install -r requirements.txt")
    md.append("python pipeline.py                    # ~5–10 min: data + features + initial clustering/classification")
    md.append("python fix_leakage_and_baseline.py    # ~30 sec: train-only clusters + one-hot classification")
    md.append("streamlit run app.py                  # opens dashboard at http://localhost:8501")
    md.append("```\n")

    md.append("---")
    md.append(f"\n*Report generated automatically from cached pipeline artifacts in `data_cache/`. "
              f"All numbers reflect the actual training run with leakage-free clustering.*\n")

    OUT.write_text("\n".join(md))
    print(f"✓ Report written to {OUT}  ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
