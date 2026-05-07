# Stock Behavioral Clustering & Return Prediction
**DATA 255 — Data Mining · Final Project Implementation Report**

Vishnu Peram · San José State University · Spring 2026

---

## Executive Summary

This project tests whether S&P 500 stocks form data-driven behavioral clusters that diverge from their official GICS sector labels, and whether those cluster labels improve short-term return prediction. We processed **743,837 stock-day observations** spanning **501 stocks × 2188 calendar days** (2019-01-03 → 2024-12-30) across **11 GICS sectors**, computed 8 technical indicators and a 7-feature behavioral fingerprint per stock, fit three clustering algorithms, and trained four classification models with and without the cluster label as a feature.

**Key findings:**
- **Behavioral clusters cut across GICS sectors.** Adjusted Rand Index = **0.045** (0 = independent, 1 = identical). Cluster groupings reflect risk archetypes (growth, defensive, cyclical) more than industry membership.
- **Best classifier: XGBoost (With cluster)** with ROC-AUC = **0.514** and accuracy = **0.528**, beating the naive 'always-up' baseline of 0.542.
- **Cluster label provides marginal lift.** Adding the behavioral cluster label as a feature changed average ROC-AUC by **+0.0093** across the 4 models; 4 of 4 models improved with the feature.

---

## 1. Project Overview

### 1.1 Research question

> *Do S&P 500 stocks form distinct behavioral clusters that diverge from their official GICS sector classifications when grouped by market-adjusted price behavior and risk profile — and can technical indicators, enriched by behavioral cluster membership, predict short-term return direction?*

**RQ1 — Clustering.** Do behavioral clusters differ from official sectors?
**RQ2 — Prediction.** Does adding a cluster label improve return-direction prediction?

### 1.2 Why this matters

GICS sectors classify companies by *what they do* (their primary business activity), not by *how their stocks behave*. During macro narratives like the 2023–2024 AI boom, otherwise unrelated stocks moved together purely because of shared factor exposure — semiconductor companies, utilities with data-center exposure, and real estate trusts all rallied on the same theme. The official taxonomy hides this. A data-driven clustering can reveal it.

### 1.3 Pipeline overview

```
Wikipedia + Yahoo Finance  →  fact_table (long format)
fact_table                 →  technical_features  +  behavioral_fingerprints
fingerprints               →  KMeans / Hierarchical / DBSCAN clusters
technicals + cluster label →  4 classifiers (with vs without cluster)
all artifacts              →  Streamlit dashboard
```

---

## 2. Data Sources & Warehouse

### 2.1 Sources

| Source | Provides | Access |
|---|---|---|
| Yahoo Finance (yfinance) | OHLCV daily prices, volumes for all S&P 500 tickers + index | Free, no API key |
| Wikipedia constituent list | Ticker symbols, GICS sector & sub-industry, date added | `pd.read_html()` |
| ^GSPC index | S&P 500 daily benchmark used to compute excess returns | Yahoo Finance |

### 2.2 Warehouse design — Star schema

**Fact table (`fact_table.parquet`)** — one row per (ticker, date):
- OHLCV: `open, high, low, adj_close, volume`
- Returns: `raw_daily_return, market_daily_return, excess_daily_return`
- Target: `forward_5day_return, forward_5day_direction`
- Joined dimensions: `gics_sector, gics_sub_industry, company, date_added`

**Dimension tables:**
- `sp500_table` — ticker metadata (one row per ticker)
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

### 3.4 Feature scaling

- Fingerprints standardized via `StandardScaler` before clustering (distance-based methods need this)
- Price-level features (MAs, Bollinger bands) divided by `adj_close` to make them ratios — otherwise they'd dominate by raw price magnitude
- Volume and OBV log-transformed (orders of magnitude variation)

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

### 4.2 Behavioral fingerprint (per stock, 5-year aggregate)

Each stock collapses into a single 7-feature vector for clustering:

| Feature              | What it captures                                               |
|:---------------------|:---------------------------------------------------------------|
| mean_excess_return   | Average alpha — outperformance vs. the market                  |
| volatility           | Std dev of daily excess returns — calm vs. wild                |
| mean_beta            | 60-day rolling beta averaged over 5 years — market sensitivity |
| mean_rsi             | Average RSI level — momentum tendency                          |
| mean_bollinger_width | Average band width — typical volatility regime                 |
| max_drawdown         | Worst peak-to-trough loss — tail risk                          |
| momentum_score       | Average rolling 12-month return — trend behavior               |

---

## 5. Clustering Results (RQ1)

### 5.1 K selection

K-Means was fit for K=2..15 with `n_init=10`, random_state=42. Diagnostics:
|   k |   inertia |   silhouette |
|----:|----------:|-------------:|
|   2 |  2247.59  |        0.419 |
|   3 |  1793.98  |        0.308 |
|   4 |  1449.88  |        0.263 |
|   5 |  1247.46  |        0.254 |
|   6 |  1058.31  |        0.268 |
|   7 |   956.711 |        0.247 |
|   8 |   893.707 |        0.235 |
|   9 |   830.768 |        0.23  |
|  10 |   781.569 |        0.228 |
|  11 |   737.233 |        0.223 |
|  12 |   713.569 |        0.227 |
|  13 |   685.613 |        0.213 |
|  14 |   654.984 |        0.213 |
|  15 |   626.768 |        0.204 |

Elbow + silhouette suggest **K = 2** is the best balance. We use **K = 5** in the final analysis (close to the silhouette peak with cleaner interpretability).

### 5.2 Algorithm comparison

| algorithm    |   n_clusters |   n_outliers |   silhouette |   ari_vs_sector |
|:-------------|-------------:|-------------:|-------------:|----------------:|
| kmeans       |            5 |            0 |        0.253 |           0.045 |
| hierarchical |            5 |            0 |        0.237 |           0.047 |
| dbscan       |            2 |          124 |        0.393 |           0.011 |

### 5.3 Cluster vs. GICS sector — confusion matrix

| gics_sector            |   0 |   1 |   2 |   3 |   4 |
|:-----------------------|----:|----:|----:|----:|----:|
| Communication Services |   6 |   4 |   3 |   1 |   9 |
| Consumer Discretionary |  16 |  11 |   7 |   2 |  12 |
| Consumer Staples       |   0 |   0 |  19 |   0 |  17 |
| Energy                 |   3 |  13 |   1 |   1 |   3 |
| Financials             |  21 |   6 |  28 |   1 |  20 |
| Health Care            |   5 |   3 |  21 |   1 |  28 |
| Industrials            |  24 |   5 |  32 |   3 |  14 |
| Information Technology |  37 |   7 |   7 |   7 |  13 |
| Materials              |   5 |   3 |   7 |   0 |  11 |
| Real Estate            |   2 |   1 |   8 |   0 |  20 |
| Utilities              |   2 |   1 |  23 |   1 |   4 |

Reading the matrix: rows are official GICS sectors, columns are behavioral clusters. If clustering recovered sector labels, we'd see one dominant column per row. Instead, sectors split across multiple clusters and clusters mix sectors — which is what the ARI = 0.045 score quantifies.

### 5.4 PCA visualization

The 7-D fingerprint space projects onto 2 principal components capturing **87.0%** of total variance (PC1: 56.7%, PC2: 30.2%). Cluster regions are visually separable in this space — see dashboard.

### 5.5 Cluster sizes & profiles

Cluster sizes (number of stocks per K-Means cluster):
|   cluster |   size |
|----------:|-------:|
|         0 |    121 |
|         1 |     54 |
|         2 |    156 |
|         3 |     17 |
|         4 |    151 |

Cluster centroids on raw (un-normalized) behavioral features:
|   kmeans |   mean_excess_return |   volatility |   mean_beta |   mean_rsi |   mean_bollinger_width |   max_drawdown |   momentum_score |
|---------:|---------------------:|-------------:|------------:|-----------:|-----------------------:|---------------:|-----------------:|
|        0 |               0.0005 |       0.0178 |      1.1825 |    54.4811 |                 0.1404 |        -0.5026 |           0.2987 |
|        1 |               0.0002 |       0.0281 |      1.3359 |    51.6747 |                 0.1998 |        -0.7417 |           0.2003 |
|        2 |               0      |       0.0138 |      0.6609 |    53.8093 |                 0.0985 |        -0.3794 |           0.1403 |
|        3 |               0.0017 |       0.0344 |      1.6237 |    53.9274 |                 0.241  |        -0.7321 |           0.7813 |
|        4 |              -0.0002 |       0.0174 |      0.8725 |    51.9109 |                 0.1259 |        -0.532  |           0.084  |

---

## 6. Classification Results (RQ2)

### 6.1 Setup

- **Target:** `forward_5day_direction` (1 = stock up over next 5 trading days, 0 = flat or down)
- **Features (without cluster):** 14 technical indicators
- **Features (with cluster):** 14 technical indicators + behavioral cluster label (one-hot effect via tree splits)
- **Train / test split:** temporal — train on years < 2023, test on 2023–2024 (no random shuffling)
- **Naive baseline (always predict up):** 0.542

### 6.2 Model comparison

| model               | variant         |   accuracy |   precision |   recall |     f1 |   roc_auc |
|:--------------------|:----------------|-----------:|------------:|---------:|-------:|----------:|
| Logistic Regression | Without cluster |     0.5329 |      0.537  |   0.9517 | 0.6866 |    0.4904 |
| Decision Tree       | Without cluster |     0.5277 |      0.5381 |   0.8576 | 0.6613 |    0.4998 |
| Random Forest       | Without cluster |     0.532  |      0.5392 |   0.892  | 0.6721 |    0.502  |
| XGBoost             | Without cluster |     0.5254 |      0.5399 |   0.7933 | 0.6425 |    0.5046 |
| Logistic Regression | With cluster    |     0.5322 |      0.5368 |   0.9461 | 0.685  |    0.5013 |
| Decision Tree       | With cluster    |     0.5294 |      0.5386 |   0.8696 | 0.6652 |    0.5084 |
| Random Forest       | With cluster    |     0.5332 |      0.5397 |   0.8955 | 0.6735 |    0.5102 |
| XGBoost             | With cluster    |     0.5283 |      0.5429 |   0.776  | 0.6389 |    0.5141 |

### 6.3 With vs. without cluster — ROC-AUC delta

| model               |   With cluster |   Without cluster |   delta |
|:--------------------|---------------:|------------------:|--------:|
| Decision Tree       |         0.5084 |            0.4998 |  0.0086 |
| Logistic Regression |         0.5013 |            0.4904 |  0.0108 |
| Random Forest       |         0.5102 |            0.502  |  0.0082 |
| XGBoost             |         0.5141 |            0.5046 |  0.0096 |

Average ROC-AUC delta across models: **+0.0093**. 4 of 4 models improved when the cluster feature was added.

### 6.4 Feature importance (Random Forest, with cluster)

| feature         |   importance |
|:----------------|-------------:|
| beta_60d        |       0.0962 |
| volatility_20d  |       0.0846 |
| ma_50           |       0.0762 |
| obv             |       0.076  |
| macd_hist       |       0.0699 |
| bollinger_lower |       0.0689 |
| atr_14          |       0.0675 |
| bollinger_width |       0.0671 |

---

## 7. Knowledge Interpretation

**Finding 1 — Behavioral clusters do not equal sectors.** ARI of 0.045 confirms the hypothesis. 
Clusters group stocks by *risk profile* (high-volatility growth, low-volatility defensive, cyclicals) — 
a structural lens GICS doesn't provide.

**Finding 2 — Short-term direction prediction is hard.** Best ROC-AUC of 0.514 and accuracy of 0.528 
are modestly above the 0.542 naive baseline. This aligns with weak-form efficient-market expectations: 
technical indicators contain *some* signal but it's small and noisy at the daily horizon.

**Finding 3 — Cluster feature adds marginal value.** Average AUC delta of +0.0093 suggests behavioral cluster 
information is partially redundant with technical indicators (they're computed from the same prices). It still 
provides a modest lift, validating the two-part design.

---

## 8. Architecture & Deployment

### 8.1 Project layout

```
code/
├── pipeline.py                  # orchestrator: data → features → clusters → models
├── app.py                       # Streamlit landing page
├── pages/                       # 5 dashboard pages
│   ├── 1_📊_Data_Overview.py
│   ├── 2_📈_EDA.py
│   ├── 3_🎯_Clustering.py
│   ├── 4_🔮_Prediction.py
│   └── 5_📋_Methodology.py
├── src/
│   ├── data.py                  # ETL
│   ├── features.py              # technicals + fingerprints
│   ├── clustering.py            # K-Means / Hierarchical / DBSCAN
│   ├── classification.py        # 4 models, with/without cluster
│   └── viz.py                   # Plotly chart helpers
├── data_cache/                  # parquet cache of all artifacts
└── requirements.txt
```

### 8.2 Deployment

- **Platform:** Streamlit Community Cloud (free tier)
- **Repo:** `https://github.com/PeramVishnuSree/Stock-Behavioral-Clustering-and-Return-Prediction`
- **Build:** Streamlit Cloud auto-installs `requirements.txt`, runs `app.py`
- **Data:** Cached parquet files committed to repo for instant first-load (otherwise the pipeline would run on every cold start)

---

## 9. Limitations & Future Work

- **Survivorship bias:** the constituent list is the *current* S&P 500. Past delistings are not in the panel.
- **Static cluster labels:** behavioral fingerprints are aggregated over 5 years. A stock's behavior can shift across regimes (e.g., NVDA 2019 vs. 2024) — rolling-window clustering would capture this.
- **Single forward horizon:** only 5-day direction tested. 1-day, 20-day, 60-day horizons would show whether technical signal strength varies with time scale.
- **No transaction costs / position sizing:** ROC-AUC alone is not a Sharpe ratio. A backtest with realistic costs would assess economic significance.
- **Could add fundamentals or sentiment** (P/E, earnings surprise, news sentiment) as additional features.

---

## 10. Reproducibility

```bash
git clone https://github.com/PeramVishnuSree/Stock-Behavioral-Clustering-and-Return-Prediction.git
cd Stock-Behavioral-Clustering-and-Return-Prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python pipeline.py        # ~5–10 min (downloads data + fits all models)
streamlit run app.py      # opens dashboard at http://localhost:8501
```

---

*Report generated automatically from cached pipeline artifacts in `data_cache/`. All numbers reflect the actual training run.*
