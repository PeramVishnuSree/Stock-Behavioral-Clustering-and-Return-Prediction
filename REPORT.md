# Stock Behavioral Clustering & Return Prediction
**DATA 255 — Data Mining · Final Project Implementation Report**

Vishnu Peram · San José State University · Spring 2026

---

## Executive Summary

This project tests whether S&P 500 stocks form data-driven behavioral clusters that diverge from their official GICS sector labels, and whether those cluster labels improve short-term return prediction. We processed **743,837 stock-day observations** spanning **501 stocks** (2019-01-03 → 2024-12-30) across **11 GICS sectors**, computed 8 technical indicators and a 7-feature behavioral fingerprint per stock, fit three clustering algorithms, and trained four classification models — each in two variants: with and without **one-hot-encoded** cluster labels.

**Methodological note on data leakage.** Behavioral fingerprints used for clustering are computed using **only the training-period window (2019–2022)**. The test set (2023–2024) is never observed during cluster fitting, so the cluster label is a leakage-free per-stock attribute when used as a feature in the classification step.

**Key findings:**
- **Behavioral clusters show weak alignment with GICS sectors.** Adjusted Rand Index = **0.045** (near 0 = close to random agreement; 1 = identical). Cluster groupings reflect risk archetypes (growth, defensive, cyclical) more than industry membership.
- **ROC-AUC is weakly above random.** Best ROC-AUC = **0.507** (Decision Tree, With cluster (one-hot)). The best accuracy across all models is **0.533** (Logistic Regression), which is **below** the test-set always-up baseline of **0.538**. Models therefore have weak ranking signal but do not outperform a naive directional baseline on accuracy.
- **Cluster label gives a small marginal lift.** Adding one-hot-encoded cluster labels changed average ROC-AUC by **+0.0049** across the 4 models (4/4 models improved). The lift is consistent but small, and would warrant statistical-significance bounds before any operational use.

---

## 1. Project Overview

### 1.1 Research question

> *Do S&P 500 stocks form distinct behavioral clusters that diverge from their official GICS sector classifications when grouped by market-adjusted price behavior and risk profile — and can technical indicators, enriched by behavioral cluster membership, predict short-term return direction?*

**RQ1 — Clustering.** Do behavioral clusters differ from official sectors?
**RQ2 — Prediction.** Does adding a cluster label improve return-direction prediction?

### 1.2 Why this matters

GICS sectors classify companies by *what they do* (their primary business activity), not by *how their stocks behave*. During macro narratives like the 2023–2024 AI boom, otherwise unrelated stocks moved together due to shared factor exposure — semiconductor companies, utilities with data-center exposure, and infrastructure providers all rallied on the same theme. The official taxonomy hides this. A data-driven clustering can reveal it.

### 1.3 Pipeline overview

```
Wikipedia + Yahoo Finance  →  fact_table (long format)
fact_table                 →  technical_features  +  behavioral_fingerprints (TRAIN ONLY)
fingerprints (train-only)  →  KMeans / Hierarchical / DBSCAN clusters
technicals + cluster (OHE) →  4 classifiers (with vs without cluster)
all artifacts              →  Streamlit dashboard
```

---

## 2. Data Sources & Warehouse

### 2.1 Sources

| Source | Provides | Access |
|---|---|---|
| Yahoo Finance (`yfinance`) | OHLCV daily prices, volumes for all S&P 500 tickers + index | Free, no API key |
| Wikipedia constituent list | Ticker symbols, GICS sector & sub-industry, date added | `pd.read_html()` |
| ^GSPC index | S&P 500 daily benchmark used to compute excess returns | Yahoo Finance |

*Data pull date: 2026-04-30. Time range: 2019-01-03 → 2024-12-30.*

### 2.2 Ticker count reconciliation

- **503 tickers** scraped from the Wikipedia constituent list
- **501 tickers** with usable price data after yfinance download (2 delisted: SNDK, Q)
- **495 tickers** survived fingerprint requirements for clustering (needed full training window of price history with no NaN in 7 fingerprint features)

### 2.3 Warehouse design — Star schema

**Fact table (`fact_table.parquet`)** — one row per (ticker, date):
- OHLCV: `open, high, low, adj_close, volume`
- Returns: `raw_daily_return, market_daily_return, excess_daily_return`
- Target: `forward_5day_return, forward_5day_direction`
- Joined dimensions: `gics_sector, gics_sub_industry, company, date_added`

**Dimension tables (logical):**
- `stock_dim` — ticker metadata (one row per ticker; sourced from Wikipedia)
- `time_dim` — derived from the `Date` column (year, quarter, etc.); not stored as a separate parquet
- `technical_features` — per-stock-day technicals (RSI, MACD, Bollinger, ATR, OBV, beta, vol)
- `fingerprints_raw / scaled` — one row per ticker, 7 behavioral features

**Fact table footprint:**
- Rows: **743,837**
- Tickers: **501**
- Date range: **2019-01-03 → 2024-12-30**
- Sectors: **11**

---

## 3. Data Preprocessing

### 3.1 Returns computation

`pct_change(fill_method=None)` produces honest NaN values when prices are missing — the older default forward-filled gaps which would silently report 0% returns on halt days. Excess returns subtract the same-day market return, isolating stock-specific behavior.

### 3.2 Missing data

- Tickers added to S&P 500 mid-period: NaN early rows are dropped via `dropna(subset=['adj_close','raw_daily_return'])`
- Technical indicators have NaN warm-up periods (RSI needs 14 days, MA50 needs 50). The classifier dropna step removes these.

### 3.3 Outlier handling

- Behavioral fingerprint features winsorized at 1st/99th percentile to keep clustering stable
- Daily return outliers retained for prediction (genuine signal)

### 3.4 Feature scaling & encoding

- Fingerprints standardized via `StandardScaler` before clustering (distance-based methods need this)
- Price-level features (MAs) divided by `adj_close` to make them ratios — otherwise they'd dominate by raw price magnitude
- Volume and OBV log-transformed (orders of magnitude variation)
- **Cluster label is one-hot encoded** before classification — cluster IDs are nominal, not ordinal, so raw integer encoding would inappropriately impose order on logistic regression and tree-split candidates

---

## 4. Feature Engineering

### 4.1 Technical indicators (per stock-day)

| Indicator | Family | Formula sketch | Captures |
|---|---|---|---|
| MA(20), MA(50) | Trend | Rolling mean of close | Trend direction |
| RSI(14) | Momentum | EMA(gains) / EMA(losses) → 0..100 | Overbought/oversold |
| MACD(12,26,9) | Trend + momentum | EMA(12) − EMA(26), signal = EMA(9) of MACD | Momentum shifts |
| Bollinger(20, 2σ) | Volatility | MA ± 2σ; width as proxy for vol regime | Volatility regime |
| ATR(14) | Volatility | EMA of true range | Daily price range |
| OBV | Volume | Cumulative signed volume | Accumulation/distribution |
| Beta(60d) | Risk | Cov(stock, market) / Var(market), 60-day window | Market sensitivity |
| Volatility(20d) | Risk | Rolling std of returns | Recent volatility |

### 4.2 Behavioral fingerprint (per stock — TRAIN-ONLY aggregate)

**To prevent data leakage**, each stock's fingerprint is computed using only the training-period window (2019–2022). The test period (2023–2024) is never seen during fingerprint construction or cluster fitting.

| Feature              | What it captures                                                       |
|:---------------------|:-----------------------------------------------------------------------|
| mean_excess_return   | Average alpha — outperformance vs. the market                          |
| volatility           | Std dev of daily excess returns — calm vs. wild                        |
| mean_beta            | 60-day rolling beta averaged over training window — market sensitivity |
| mean_rsi             | Average RSI level — momentum tendency                                  |
| mean_bollinger_width | Average band width — typical volatility regime                         |
| max_drawdown         | Worst peak-to-trough loss — tail risk                                  |
| momentum_score       | Average rolling 12-month return — trend behavior                       |

---

## 5. Clustering Results (RQ1)

### 5.1 K selection

K-Means was fit for K=2..15 with `n_init=10`, random_state=42. Diagnostics:
|   k |   inertia |   silhouette |
|----:|----------:|-------------:|
|   2 |  2334.31  |        0.393 |
|   3 |  1819.7   |        0.342 |
|   4 |  1479.07  |        0.276 |
|   5 |  1223.01  |        0.256 |
|   6 |  1085.39  |        0.251 |
|   7 |   961.125 |        0.245 |
|   8 |   895.876 |        0.226 |
|   9 |   837.648 |        0.218 |
|  10 |   788.391 |        0.216 |
|  11 |   741.451 |        0.219 |
|  12 |   709.822 |        0.214 |
|  13 |   685.411 |        0.217 |
|  14 |   661.78  |        0.216 |
|  15 |   642.707 |        0.202 |

**Silhouette peaks at K = 2 (0.393), but K = 5 was selected** as an interpretability-driven compromise near the elbow of the inertia curve. K = 2 produced clusters that were too coarse to compare meaningfully against the 11 GICS sectors; K = 5 gives finer granularity with silhouette = 0.256. The trade-off is documented explicitly: lower silhouette in exchange for more meaningful behavioral archetypes.

### 5.2 Algorithm comparison

| algorithm    |   n_clusters |   n_outliers |   silhouette |   ari_vs_sector |
|:-------------|-------------:|-------------:|-------------:|----------------:|
| kmeans       |            5 |            0 |        0.256 |           0.045 |
| hierarchical |            5 |            0 |        0.285 |           0.026 |
| dbscan       |            3 |          111 |        0.207 |           0.014 |

*All three algorithms produced low ARI vs. GICS sectors, confirming the finding is structural rather than algorithm-dependent.*

### 5.3 Cluster vs. GICS sector — confusion matrix

| gics_sector            |   0 |   1 |   2 |   3 |   4 |
|:-----------------------|----:|----:|----:|----:|----:|
| Communication Services |   1 |   9 |   2 |   7 |   4 |
| Consumer Discretionary |   5 |  16 |   7 |   4 |  16 |
| Consumer Staples       |   0 |   4 |   0 |  26 |   5 |
| Energy                 |  15 |   4 |   0 |   1 |   1 |
| Financials             |   1 |  27 |   2 |  18 |  28 |
| Health Care            |   4 |   7 |   1 |  25 |  20 |
| Industrials            |   5 |  13 |   3 |  25 |  31 |
| Information Technology |  14 |  19 |   2 |   6 |  30 |
| Materials              |   6 |   6 |   0 |   9 |   5 |
| Real Estate            |   0 |   9 |   0 |  17 |   5 |
| Utilities              |   0 |   2 |   1 |  26 |   1 |

Reading the matrix: rows are official GICS sectors, columns are behavioral clusters. If clustering recovered sector labels, we'd see one dominant column per row. Instead, sectors split across multiple clusters and clusters mix sectors — which is what the ARI = 0.045 score quantifies.

### 5.4 PCA visualization

The 7-D fingerprint space projects onto 2 principal components capturing **85.3%** of total variance (PC1: 50.8%, PC2: 34.5%). Cluster regions are visually separable in this space — see dashboard.

### 5.5 Cluster sizes & profiles

Cluster sizes (number of stocks per K-Means cluster):
|   cluster |   size |
|----------:|-------:|
|         0 |     51 |
|         1 |    116 |
|         2 |     18 |
|         3 |    164 |
|         4 |    146 |

Cluster centroids on raw (un-normalized) behavioral features:
|   kmeans |   mean_excess_return |   volatility |   mean_beta |   mean_rsi |   mean_bollinger_width |   max_drawdown |   momentum_score |
|---------:|---------------------:|-------------:|------------:|-----------:|-----------------------:|---------------:|-----------------:|
|        0 |               0.0011 |       0.029  |      1.4028 |    53.4464 |                 0.2162 |        -0.6558 |           0.4932 |
|        1 |              -0      |       0.0203 |      1.1117 |    51.9362 |                 0.1526 |        -0.57   |           0.1125 |
|        2 |              -0.0005 |       0.0379 |      1.4819 |    49.3053 |                 0.2612 |        -0.816  |          -0.0914 |
|        3 |               0.0001 |       0.015  |      0.6567 |    53.4573 |                 0.106  |        -0.3821 |           0.1143 |
|        4 |               0.0005 |       0.0159 |      1.0588 |    54.6376 |                 0.1315 |        -0.4474 |           0.2587 |

---

## 6. Classification Results (RQ2)

### 6.1 Setup

- **Target:** `forward_5day_direction` (1 = stock up over next 5 trading days, 0 = flat or down)
- **Features (without cluster):** 11 technical indicators (MA-20, MA-50, RSI, MACD, MACD-hist, Bollinger width, ATR, OBV, beta-60d, volatility-20d, volume)
- **Features (with cluster):** technical indicators + **one-hot-encoded** cluster label (5 dummy variables, one per cluster)
- **Train / test split:** temporal — train on years < 2023, test on 2023–2024
- **Cluster fingerprints:** computed using **training-period data only** (no leakage into test)
- **Naive baseline (always predict up, evaluated on test set):** **0.538** — any model has to beat this on accuracy to be useful as a directional predictor.

### 6.2 Model comparison

| model               | variant                |   accuracy |   precision |   recall |     f1 |   roc_auc |
|:--------------------|:-----------------------|-----------:|------------:|---------:|-------:|----------:|
| Logistic Regression | Without cluster        |     0.5329 |      0.537  |   0.9531 | 0.6869 |    0.4908 |
| Decision Tree       | Without cluster        |     0.5308 |      0.5376 |   0.9093 | 0.6757 |    0.4998 |
| Random Forest       | Without cluster        |     0.532  |      0.5395 |   0.8845 | 0.6702 |    0.503  |
| XGBoost             | Without cluster        |     0.525  |      0.5398 |   0.7906 | 0.6415 |    0.5054 |
| Logistic Regression | With cluster (one-hot) |     0.5305 |      0.537  |   0.9191 | 0.6779 |    0.4986 |
| Decision Tree       | With cluster (one-hot) |     0.5272 |      0.5385 |   0.8428 | 0.6572 |    0.5071 |
| Random Forest       | With cluster (one-hot) |     0.5282 |      0.5389 |   0.8485 | 0.6591 |    0.5061 |
| XGBoost             | With cluster (one-hot) |     0.5211 |      0.5401 |   0.7357 | 0.6229 |    0.5067 |

**Interpretation.** Best ROC-AUC is 0.507 (Decision Tree, With cluster (one-hot)) — barely above the random-ranking floor of 0.500. Best accuracy is 0.533 (Logistic Regression), **below** the always-up baseline of 0.538. Models extract some weak ranking signal (probabilities correctly order positive cases above negative slightly more often than chance) but cannot translate that into directional accuracy that beats simply predicting 'up'. This is consistent with weak-form efficient-market expectations.

### 6.3 With vs. without cluster — ROC-AUC delta

| model               |   With cluster (one-hot) |   Without cluster |   delta |
|:--------------------|-------------------------:|------------------:|--------:|
| Decision Tree       |                   0.5071 |            0.4998 |  0.0073 |
| Logistic Regression |                   0.4986 |            0.4908 |  0.0078 |
| Random Forest       |                   0.5061 |            0.503  |  0.0031 |
| XGBoost             |                   0.5067 |            0.5054 |  0.0012 |

Average ROC-AUC delta across models: **+0.0049**. 4 of 4 models improved when the one-hot cluster feature was added. The lift is small and consistent — likely capturing residual long-term behavioral structure that per-day technicals miss, but not large enough to claim economic significance without statistical-significance testing on the AUC differences.

### 6.4 Confusion matrices (test set)

| model               | variant                |    tn |     fp |    fn |     tp |
|:--------------------|:-----------------------|------:|-------:|------:|-------:|
| Logistic Regression | Without cluster        |  5076 | 109589 |  6248 | 127082 |
| Decision Tree       | Without cluster        | 10389 | 104276 | 12090 | 121240 |
| Random Forest       | Without cluster        | 14013 | 100652 | 15403 | 117927 |
| XGBoost             | Without cluster        | 24782 |  89883 | 27921 | 105409 |
| Logistic Regression | With cluster (one-hot) |  9025 | 105640 | 10792 | 122538 |
| Decision Tree       | With cluster (one-hot) | 18378 |  96287 | 20959 | 112371 |
| Random Forest       | With cluster (one-hot) | 17858 |  96807 | 20204 | 113126 |
| XGBoost             | With cluster (one-hot) | 31127 |  83538 | 35237 |  98093 |

*All models lean toward predicting `up` (high recall, low specificity). Adding the cluster feature shifts the operating point slightly toward the negative class, reducing recall but improving the AUC ranking quality.*

### 6.5 Feature importance (Random Forest, with cluster)

| feature         |   importance |
|:----------------|-------------:|
| beta_60d        |       0.1084 |
| volatility_20d  |       0.1006 |
| obv             |       0.0929 |
| ma_50           |       0.0914 |
| bollinger_width |       0.0904 |
| atr_14          |       0.0846 |
| macd_hist       |       0.084  |
| ma_20           |       0.0817 |

---

## 7. Knowledge Interpretation

**Finding 1 — Behavioral clusters show weak alignment with GICS sectors.** ARI of 0.045 is near zero, indicating cluster assignments and sector labels are close to randomly aligned. Clusters group stocks by *risk profile* (high-volatility growth, low-volatility defensive, cyclicals) — a structural lens GICS doesn't provide. RQ1 supported.

**Finding 2 — Short-term direction prediction does not beat the naive baseline on accuracy.** Best accuracy (0.533) is below the always-up baseline (0.538). However, best ROC-AUC (0.507) is slightly above 0.5, indicating weak but non-zero ranking signal. This is consistent with weak-form market efficiency — technical indicators contain *some* signal but not enough to beat a directional-bias rule on a 5-day horizon.

**Finding 3 — Cluster label adds small marginal AUC value.** Average AUC delta of +0.0049 with the cluster feature one-hot encoded. The lift is small and consistent across models, but future work should test it for statistical significance before any operational claims.

---

## 8. Architecture & Deployment

### 8.1 Project layout

```
code/
├── pipeline.py                    # orchestrator
├── fix_leakage_and_baseline.py    # leakage-free re-clustering + classification
├── make_report.py                 # generates REPORT.md from cache
├── make_presentation.py           # generates PRESENTATION.pptx from cache
├── app.py                         # Streamlit landing page
├── pages/                         # 5 dashboard pages
├── src/
│   ├── data.py                    # ETL
│   ├── features.py                # technicals + fingerprints
│   ├── clustering.py              # KMeans / Hierarchical / DBSCAN
│   ├── classification.py          # 4 models
│   └── viz.py                     # Plotly chart helpers
├── data_cache/                    # parquet cache of all artifacts
└── requirements.txt
```

### 8.2 Deployment

- **Platform:** Streamlit Community Cloud (free tier)
- **Repo:** `https://github.com/PeramVishnuSree/Stock-Behavioral-Clustering-and-Return-Prediction`
- **Build:** auto-installs `requirements.txt`, runs `app.py`
- **Data:** parquet files committed to repo for instant first-load

---

## 9. Limitations & Future Work

- **Survivorship bias:** the constituent list is the *current* S&P 500. Past delistings are not in the panel. Fix: use historical CRSP membership data.
- **Static cluster labels:** behavioral fingerprints are aggregated over the training window. A stock's behavior can shift across regimes. Fix: rolling-window clustering with periodic re-fingerprinting.
- **Single forward horizon:** only 5-day direction tested. Other horizons (1d, 20d, 60d) likely have different signal-to-noise.
- **No transaction costs / position sizing:** ROC-AUC alone is not a Sharpe ratio. A backtest with realistic costs would assess economic significance.
- **AUC lift not significance-tested:** the +0.0049 average AUC delta is consistent but small; bootstrap or DeLong tests would confirm whether it's statistically distinguishable from zero.
- **Could add fundamentals or sentiment** (P/E, earnings surprise, news sentiment) as additional features.

---

## 10. Reproducibility

```bash
git clone https://github.com/PeramVishnuSree/Stock-Behavioral-Clustering-and-Return-Prediction.git
cd Stock-Behavioral-Clustering-and-Return-Prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python pipeline.py                    # ~5–10 min: data + features + initial clustering/classification
python fix_leakage_and_baseline.py    # ~30 sec: train-only clusters + one-hot classification
streamlit run app.py                  # opens dashboard at http://localhost:8501
```

---

*Report generated automatically from cached pipeline artifacts in `data_cache/`. All numbers reflect the actual training run with leakage-free clustering.*
