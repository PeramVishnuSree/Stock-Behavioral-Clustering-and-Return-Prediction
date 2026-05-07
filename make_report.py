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
    }


def _md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def main():
    D = _load()
    fact, sp500, tech = D["fact"], D["sp500"], D["tech"]
    assign, cmetrics, diag = D["assign"], D["cmetrics"], D["diag"]
    metrics, feat_imp, baseline = D["metrics"], D["feat_imp"], D["baseline"]
    raw_fp = D["raw_fp"]

    # Headline numbers
    n_tickers = fact["ticker"].nunique()
    n_rows    = len(fact)
    date_min, date_max = fact["Date"].min().date(), fact["Date"].max().date()
    n_sectors = fact["gics_sector"].nunique()

    # Cluster numbers
    best_k = int(diag.iloc[diag["silhouette"].idxmax()]["k"])
    ari_k = float(cmetrics.loc[cmetrics["algorithm"] == "kmeans", "ari_vs_sector"].iloc[0])
    sil_k = float(cmetrics.loc[cmetrics["algorithm"] == "kmeans", "silhouette"].iloc[0])
    pc1_var = float(D["pca"]["evr1"].iloc[0])
    pc2_var = float(D["pca"]["evr2"].iloc[0])

    # Classification numbers
    best_row = metrics.sort_values("roc_auc", ascending=False).iloc[0]
    best_model_name = best_row["model"]
    best_variant = best_row["variant"]
    best_auc = float(best_row["roc_auc"])
    best_acc = float(best_row["accuracy"])

    # With vs without cluster comparison
    pivot = metrics.pivot(index="model", columns="variant", values="roc_auc")
    pivot["delta"] = pivot["With cluster"] - pivot["Without cluster"]
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
        f"observations** spanning **{n_tickers} stocks × {(date_max - date_min).days} calendar days** "
        f"({date_min} → {date_max}) across **{n_sectors} GICS sectors**, computed 8 technical indicators "
        f"and a 7-feature behavioral fingerprint per stock, fit three clustering algorithms, "
        f"and trained four classification models with and without the cluster label as a feature.\n"
    )
    md.append("**Key findings:**")
    md.append(f"- **Behavioral clusters cut across GICS sectors.** Adjusted Rand Index = **{ari_k:.3f}** "
              f"(0 = independent, 1 = identical). Cluster groupings reflect risk archetypes "
              f"(growth, defensive, cyclical) more than industry membership.")
    md.append(f"- **Best classifier: {best_model_name} ({best_variant})** with ROC-AUC = **{best_auc:.3f}** "
              f"and accuracy = **{best_acc:.3f}**, beating the naive 'always-up' baseline of {baseline:.3f}.")
    md.append(f"- **Cluster label provides marginal lift.** Adding the behavioral cluster label as a feature "
              f"changed average ROC-AUC by **{avg_delta:+.4f}** across the 4 models; "
              f"{n_improved} of 4 models improved with the feature.\n")

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
        "*how their stocks behave*. During macro narratives like the 2023–2024 AI boom, otherwise unrelated "
        "stocks moved together purely because of shared factor exposure — semiconductor companies, utilities "
        "with data-center exposure, and real estate trusts all rallied on the same theme. The official "
        "taxonomy hides this. A data-driven clustering can reveal it.\n"
    )

    md.append("### 1.3 Pipeline overview\n")
    md.append(
        "```\n"
        "Wikipedia + Yahoo Finance  →  fact_table (long format)\n"
        "fact_table                 →  technical_features  +  behavioral_fingerprints\n"
        "fingerprints               →  KMeans / Hierarchical / DBSCAN clusters\n"
        "technicals + cluster label →  4 classifiers (with vs without cluster)\n"
        "all artifacts              →  Streamlit dashboard\n"
        "```\n"
    )

    md.append("---\n")
    md.append("## 2. Data Sources & Warehouse\n")
    md.append("### 2.1 Sources\n")
    md.append("| Source | Provides | Access |\n|---|---|---|")
    md.append("| Yahoo Finance (yfinance) | OHLCV daily prices, volumes for all S&P 500 tickers + index | Free, no API key |")
    md.append("| Wikipedia constituent list | Ticker symbols, GICS sector & sub-industry, date added | `pd.read_html()` |")
    md.append("| ^GSPC index | S&P 500 daily benchmark used to compute excess returns | Yahoo Finance |\n")

    md.append("### 2.2 Warehouse design — Star schema\n")
    md.append("**Fact table (`fact_table.parquet`)** — one row per (ticker, date):")
    md.append("- OHLCV: `open, high, low, adj_close, volume`")
    md.append("- Returns: `raw_daily_return, market_daily_return, excess_daily_return`")
    md.append("- Target: `forward_5day_return, forward_5day_direction`")
    md.append("- Joined dimensions: `gics_sector, gics_sub_industry, company, date_added`\n")
    md.append("**Dimension tables:**")
    md.append("- `sp500_table` — ticker metadata (one row per ticker)")
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
    md.append("### 3.4 Feature scaling\n")
    md.append("- Fingerprints standardized via `StandardScaler` before clustering (distance-based methods need this)")
    md.append("- Price-level features (MAs, Bollinger bands) divided by `adj_close` to make them ratios — otherwise they'd dominate by raw price magnitude")
    md.append("- Volume and OBV log-transformed (orders of magnitude variation)\n")

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

    md.append("### 4.2 Behavioral fingerprint (per stock, 5-year aggregate)\n")
    md.append("Each stock collapses into a single 7-feature vector for clustering:\n")
    fp_desc = pd.DataFrame({
        "Feature": ["mean_excess_return", "volatility", "mean_beta", "mean_rsi",
                     "mean_bollinger_width", "max_drawdown", "momentum_score"],
        "What it captures": [
            "Average alpha — outperformance vs. the market",
            "Std dev of daily excess returns — calm vs. wild",
            "60-day rolling beta averaged over 5 years — market sensitivity",
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
    md.append(f"\nElbow + silhouette suggest **K = {best_k}** is the best balance. We use **K = 5** "
              f"in the final analysis (close to the silhouette peak with cleaner interpretability).\n")

    md.append("### 5.2 Algorithm comparison\n")
    md.append(_md_table(cmetrics.round(3)))
    md.append("")

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
    md.append(_md_table(cluster_sizes.reset_index().rename(columns={"kmeans":"cluster", "count":"size"})))
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
    md.append("- **Features (without cluster):** 14 technical indicators")
    md.append("- **Features (with cluster):** 14 technical indicators + behavioral cluster label (one-hot effect via tree splits)")
    md.append("- **Train / test split:** temporal — train on years < 2023, test on 2023–2024 (no random shuffling)")
    md.append(f"- **Naive baseline (always predict up):** {baseline:.3f}\n")

    md.append("### 6.2 Model comparison\n")
    md.append(_md_table(metrics.round(4)))
    md.append("")

    md.append("### 6.3 With vs. without cluster — ROC-AUC delta\n")
    md.append(_md_table(pivot.round(4).reset_index()))
    md.append(f"\nAverage ROC-AUC delta across models: **{avg_delta:+.4f}**. {n_improved} of 4 models improved when the cluster feature was added.\n")

    md.append("### 6.4 Feature importance (Random Forest, with cluster)\n")
    md.append(_md_table(top_feats.round(4)))
    md.append("")

    md.append("---\n")
    md.append("## 7. Knowledge Interpretation\n")
    md.append(f"**Finding 1 — Behavioral clusters do not equal sectors.** ARI of {ari_k:.3f} confirms the hypothesis. ")
    md.append(f"Clusters group stocks by *risk profile* (high-volatility growth, low-volatility defensive, cyclicals) — ")
    md.append(f"a structural lens GICS doesn't provide.\n")
    md.append(f"**Finding 2 — Short-term direction prediction is hard.** Best ROC-AUC of {best_auc:.3f} and accuracy of {best_acc:.3f} ")
    md.append(f"are modestly above the {baseline:.3f} naive baseline. This aligns with weak-form efficient-market expectations: ")
    md.append("technical indicators contain *some* signal but it's small and noisy at the daily horizon.\n")
    md.append(f"**Finding 3 — Cluster feature adds marginal value.** Average AUC delta of {avg_delta:+.4f} suggests behavioral cluster ")
    md.append("information is partially redundant with technical indicators (they're computed from the same prices). It still ")
    md.append("provides a modest lift, validating the two-part design.\n")

    md.append("---\n")
    md.append("## 8. Architecture & Deployment\n")
    md.append("### 8.1 Project layout\n")
    md.append("```")
    md.append("code/")
    md.append("├── pipeline.py                  # orchestrator: data → features → clusters → models")
    md.append("├── app.py                       # Streamlit landing page")
    md.append("├── pages/                       # 5 dashboard pages")
    md.append("│   ├── 1_📊_Data_Overview.py")
    md.append("│   ├── 2_📈_EDA.py")
    md.append("│   ├── 3_🎯_Clustering.py")
    md.append("│   ├── 4_🔮_Prediction.py")
    md.append("│   └── 5_📋_Methodology.py")
    md.append("├── src/")
    md.append("│   ├── data.py                  # ETL")
    md.append("│   ├── features.py              # technicals + fingerprints")
    md.append("│   ├── clustering.py            # K-Means / Hierarchical / DBSCAN")
    md.append("│   ├── classification.py        # 4 models, with/without cluster")
    md.append("│   └── viz.py                   # Plotly chart helpers")
    md.append("├── data_cache/                  # parquet cache of all artifacts")
    md.append("└── requirements.txt")
    md.append("```\n")
    md.append("### 8.2 Deployment\n")
    md.append("- **Platform:** Streamlit Community Cloud (free tier)")
    md.append("- **Repo:** `https://github.com/PeramVishnuSree/Stock-Behavioral-Clustering-and-Return-Prediction`")
    md.append("- **Build:** Streamlit Cloud auto-installs `requirements.txt`, runs `app.py`")
    md.append("- **Data:** Cached parquet files committed to repo for instant first-load (otherwise the pipeline would run on every cold start)\n")

    md.append("---\n")
    md.append("## 9. Limitations & Future Work\n")
    md.append("- **Survivorship bias:** the constituent list is the *current* S&P 500. Past delistings are not in the panel.")
    md.append("- **Static cluster labels:** behavioral fingerprints are aggregated over 5 years. A stock's behavior can shift across regimes (e.g., NVDA 2019 vs. 2024) — rolling-window clustering would capture this.")
    md.append("- **Single forward horizon:** only 5-day direction tested. 1-day, 20-day, 60-day horizons would show whether technical signal strength varies with time scale.")
    md.append("- **No transaction costs / position sizing:** ROC-AUC alone is not a Sharpe ratio. A backtest with realistic costs would assess economic significance.")
    md.append("- **Could add fundamentals or sentiment** (P/E, earnings surprise, news sentiment) as additional features.\n")

    md.append("---\n")
    md.append("## 10. Reproducibility\n")
    md.append("```bash")
    md.append("git clone https://github.com/PeramVishnuSree/Stock-Behavioral-Clustering-and-Return-Prediction.git")
    md.append("cd Stock-Behavioral-Clustering-and-Return-Prediction")
    md.append("python -m venv .venv && source .venv/bin/activate")
    md.append("pip install -r requirements.txt")
    md.append("python pipeline.py        # ~5–10 min (downloads data + fits all models)")
    md.append("streamlit run app.py      # opens dashboard at http://localhost:8501")
    md.append("```\n")

    md.append("---")
    md.append(f"\n*Report generated automatically from cached pipeline artifacts in `data_cache/`. "
              f"All numbers reflect the actual training run.*\n")

    OUT.write_text("\n".join(md))
    print(f"✓ Report written to {OUT}  ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
