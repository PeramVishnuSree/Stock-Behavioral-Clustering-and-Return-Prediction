# Anticipated Questions & Answers
## Stock Behavioral Clustering & Return Prediction — DATA 255

This document anticipates the questions you're most likely to face during the Q&A portion of your presentation, with detailed, defensible answers. Questions are grouped by topic.

---

## A. Research Question & Motivation

### Q1. Why did you pick this topic?
The S&P 500 is publicly accessible, well-instrumented data. The research question — "do behavioral groupings differ from official sector labels?" — has a clear yes/no answer that data mining is uniquely suited to address: clustering produces the alternative grouping; ARI quantifies how much it differs. It also forces an honest engagement with two real-world problems: (1) factor-exposure noise (the AI boom example), which motivates the use of excess returns; and (2) weak-form market efficiency, which sets a realistic ceiling on prediction accuracy.

### Q2. What's the practical value of this work?
Three things. First, **portfolio diversification:** if you hold 5 stocks all in different GICS sectors but all in the same behavioral cluster, you're not actually diversified. Second, **risk modeling:** behavioral archetypes (high-vol growth, low-vol defensive) are how real risk managers think; this maps stocks to those archetypes empirically. Third, **feature engineering for ML:** the cluster label is shown to add a small but real predictive lift on top of technicals — useful in any longer downstream pipeline.

### Q3. Why 2019–2024 and not a longer window?
Five years is enough to span five distinct market regimes (pre-COVID bull, COVID crash, recovery, 2022 hiking cycle, AI boom) without going back so far that companies' business models have fundamentally changed (think Tesla in 2015 vs. 2024). Rolling earlier introduces survivorship bias and makes the sector mappings less stable. If anything, a follow-up could test whether 10-year fingerprints produce more or less stable clusters.

---

## B. Data & Warehouse

### Q4. Why use Yahoo Finance? Isn't Bloomberg / CRSP more rigorous?
Yahoo Finance (via `yfinance`) is free, programmable, and good enough for daily OHLCV data on actively traded large-caps. CRSP is gold-standard for academic research because it includes delisted stocks (avoiding survivorship bias) and corrects for distribution events more carefully — but it's behind a paywall. For a class project demonstrating the methodology, Yahoo is the correct trade-off: my pipeline is reproducible by anyone, anywhere, with zero authentication.

### Q5. What's survivorship bias and how does it affect your results?
Survivorship bias is the problem of analyzing only companies that *survived* to the present day. The current S&P 500 list excludes companies that were dropped during the period (e.g., bankruptcies, acquisitions). This biases results upward — you only see winners. My analysis explicitly documents this as a limitation. To fix, I'd need to use the historical S&P 500 constituent membership over time (available on CRSP / WRDS).

### Q6. Why a star schema specifically?
Star schemas are the standard for analytical (OLAP) workloads. Single fact table joined to dimensions makes filtering by date / sector / ticker trivially fast and produces clean SQL even though I'm using parquet files. It also matches the conceptual model of the problem: a stock-day observation is the unit of analysis; everything else (sector metadata, time attributes, technical indicator snapshots) is a dimension describing it.

### Q7. Why `pct_change(fill_method=None)` instead of the default?
The default (`fill_method='pad'`) forward-fills missing prices before computing the percentage change. So if a stock was halted on Wednesday, the default would treat Thursday's price as a 0% return rather than a missing observation. That's a silent data corruption. `fill_method=None` is honest: missing data stays missing, and downstream code can decide how to handle it.

---

## C. Preprocessing

### Q8. Why use *excess* returns for clustering?
This is the methodological centerpiece. Raw returns are dominated by the market component — when the market goes up 1% on Tuesday, almost every stock's raw return rises that day, regardless of whether the underlying behavior is fundamentally similar. If you cluster on raw returns, you'd find that clusters mostly reflect "high beta vs low beta" — which is interesting but not what we set out to study. Excess returns (`stock_return − market_return`) strip out the market component, leaving only the stock-specific component. Now clustering reveals what's actually distinctive about each stock's character.

### Q9. Why winsorize fingerprint features but not classification features?
Different goals. Clustering cares about *typical* behavior; outlier days (a 25% earnings move) corrupt the centroid. Winsorizing at 1st/99th percentile removes a few extreme observations without changing the central tendency. For prediction, *every* observation is a training example — outliers might contain real signal (if a stock is overbought, it might fall back). Removing them throws away information.

### Q10. Why log-transform OBV and volume?
They span orders of magnitude. Apple trades 50M shares some days and 200M others; tiny stocks trade 100K. A linear feature with this range would dominate distance computations and tree splits purely by magnitude. `np.log1p()` compresses the dynamic range so volume contributes proportionally to its information content.

---

## D. Feature Engineering

### Q11. Why these specific 8 technical indicators?
They cover the four standard families used in quantitative finance: trend (MA, MACD), momentum (RSI), volatility (Bollinger, ATR), and risk/volume (beta, OBV). Adding more would risk multicollinearity (many indicators are derivatives of each other); fewer would miss signal. This is a defensible standard set used in countless papers and trading systems.

### Q12. Why a 7-feature behavioral fingerprint and not all 8 indicators averaged?
The behavioral fingerprint asks "what is this stock's character over 5 years?" — that's a one-row-per-stock question. Averaging daily indicators is one way to summarize, but I picked specific aggregations that capture different aspects: mean (typical level), std (variability), max drawdown (tail risk), momentum score (trend behavior). Each one answers a different question. Adding all 8 indicator means would create collinearity (RSI mean ≈ momentum score statistically).

### Q13. How do you compute beta?
Beta is `Cov(stock_return, market_return) / Var(market_return)` over a rolling 60-day window. The 60-day choice is a trade-off: shorter windows are more responsive but noisier; longer windows are smoother but less timely. 60 days is standard practice in equity research.

### Q14. What does RSI of 70 mean?
RSI is a 0–100 momentum oscillator computed over 14 days. RSI > 70 conventionally indicates overbought (recent gains have outpaced recent losses); RSI < 30 indicates oversold. The thresholds are heuristic, not mathematically derived. Empirically, mean reversion from these extremes is a well-documented but weak signal.

---

## E. Clustering — Part 1

### Q15. Why K = 5?
The elbow plot shows inertia drops sharply through K=4–6, then flattens. Silhouette peaks within that range. K=5 is a clean compromise: enough granularity to separate behavioral types but few enough to interpret. Choosing K=10 would over-fragment; K=3 would over-summarize.

### Q16. The silhouette is only ~0.25 — isn't that bad?
Silhouette of 0.25 in financial data is decent. Values > 0.5 typically require well-separated synthetic data. Real markets have continuous distributions of stock characteristics — you don't expect cleanly separated groups, you expect gradients. The fact that silhouette is positive *at all* confirms genuine structure exists.

### Q17. ARI of 0.044 — interpret?
ARI ranges from 0 (independent labels) to 1 (identical). 0.044 means K-Means clusters are essentially independent of GICS sector labels. This is the headline result of RQ1: behavioral clustering reveals structure that the official sector labels do *not* capture. Tech stocks split across multiple clusters; defensives from different sectors pool into one.

### Q18. Why use three different clustering algorithms?
Robustness check. K-Means assumes spherical clusters; Hierarchical builds trees with different merge rules; DBSCAN is density-based with a noise category. If they all produced wildly different ARIs, our finding would be fragile. Instead they agree (low ARI in all three). This is a methodological belt-and-suspenders.

### Q19. DBSCAN found 124 outliers — what are they?
These are stocks with no nearby neighbors in the 7-D fingerprint space. Likely candidates: highly idiosyncratic stocks (meme stocks like GME, AMC), recent IPOs with thin price history, biotech with extreme volatility. They're not noise to be discarded — they're an interesting tail that deserves separate analysis.

### Q20. How do you visualize a 7-dimensional space?
Principal Component Analysis (PCA) finds the 2 directions in 7-D space that capture the most variance. Plotting on those 2 axes shows ~40-50% of the variance — which means visual cluster separation is approximate, but real. A stock that looks "alone" in PCA space isn't necessarily alone in 7-D, but trends are preserved.

---

## F. Classification — Part 2

### Q21. Why predict 5-day forward return direction and not price?
Direction (binary: up/down) is more tractable than precise price prediction. Most technical signals are noisy; predicting the right direction more often than chance is already valuable and doesn't require getting the magnitude right. 5 days is short enough to show technical-signal effects but long enough to be less dominated by intraday microstructure noise.

### Q22. Why temporal split, not random shuffle?
Time series data has serial correlation: today's prices depend on yesterday's. If you randomly shuffle, your test set contains observations that are temporally interleaved with training observations — the model can learn time-specific patterns that "leak" forward in time. Temporal split (train on past, test on future) simulates how the model would actually be used: trained on history, deployed on the unknown future.

### Q23. ROC-AUC of 0.51 — that's barely above chance. Is this a failure?
No, this is the realistic outcome. The Efficient Market Hypothesis (Fama 1970) predicts that publicly available information cannot be used to systematically beat the market. Technical indicators are public, so the EMH predicts AUC near 0.50. Getting AUC of 0.51-0.52 with a small-but-real edge is actually consistent with the literature — short-horizon technical signal exists but is weak. The rigor is in honestly characterizing this, not pretending to beat 0.65.

### Q24. Why does adding the cluster label only marginally improve AUC?
Because the cluster label is *derived from* the same OHLCV data that produces the technical indicators. There's no new information — the cluster is essentially a compressed summary of long-term behavioral properties that the per-day technicals partially recover anyway. A small lift is what theory predicts. To get a bigger lift, you'd add information *not* in OHLCV (fundamentals, news sentiment).

### Q25. Why these four models?
They span a complexity spectrum: Logistic Regression (linear baseline), Decision Tree (interpretable non-linear), Random Forest (ensemble of trees, handles interactions), XGBoost (boosted ensemble, typically strongest). Comparing them gauges whether non-linear modeling adds value (it does, marginally) and whether ensembles beat single trees (they do).

### Q26. Why do all your models have similar AUC?
This is a finding, not a flaw. When the signal is weak, model choice matters less than feature choice. All four models converge to roughly the same modest AUC because the underlying signal is at that level. If one model was dramatically better, it would suggest the others were misconfigured.

### Q27. Did you tune hyperparameters?
I used reasonable defaults documented in the code (`n_estimators=200, max_depth=12` for RF, `learning_rate=0.05, n_estimators=300` for XGB). Extensive tuning could improve AUC by 0.005-0.010 but wouldn't change the headline conclusion. To be rigorous, hyperparameters should be tuned via cross-validation on the training set only — never the test set.

### Q28. What about overfitting?
Tree depth is capped (`max_depth=8` for tree, `max_depth=12` for RF) and Random Forest aggregates many trees, both of which reduce overfitting. The temporal train/test split is the key guardrail — you can't accidentally peek at the future. If the model were overfit, training AUC would be much higher than test AUC (which it isn't, by inspection).

---

## G. Visualization & Dashboard

### Q29. Why Streamlit?
Streamlit auto-converts Python scripts into interactive dashboards with no JavaScript. Free deployment via Streamlit Community Cloud, fast iteration, and good Plotly integration make it ideal for student projects where you want a polished UI without becoming a frontend developer.

### Q30. Why Plotly and not Matplotlib?
Plotly produces interactive charts (hover, zoom, toggle) which is what users want in a dashboard. It also exports to PNG cleanly via Kaleido for the slide deck. Matplotlib is fine for static publication figures but feels dead in a web dashboard.

### Q31. Did you commit the data cache to the repo?
Yes — about 200 MB of parquet files. This means Streamlit Cloud loads the dashboard instantly without re-running the 5–10 minute pipeline on every cold start. The trade-off is repo size, which is fine for a class project on a free tier.

---

## H. Methodology Defense

### Q32. What's the biggest weakness of your approach?
Three honest weaknesses, in order of importance:
1. **Survivorship bias** — only stocks currently in the S&P 500 are included.
2. **Static behavioral fingerprints** — a stock's behavior changes across regimes (NVDA in 2019 was not NVDA in 2024). Rolling-window fingerprints would capture this.
3. **Single horizon** — only 5-day prediction tested. The signal-to-noise ratio likely varies with horizon.

### Q33. Could you have used deep learning?
Yes, and it's listed as future work. LSTMs or transformers on raw OHLCV sequences would let the model discover its own features rather than relying on hand-engineered technicals. However, deep models need much more data (tens of thousands of training examples per ticker is borderline), are harder to interpret, and are unlikely to dramatically outperform on a weak-signal task. They're better suited for higher-frequency intraday data.

### Q34. Why didn't you include macro features (rates, inflation)?
Two reasons. First, scope: keeping features to per-stock indicators makes the analysis cleanly attributable to stock behavior. Adding macro features would make every stock-day observation share the same macro context, which doesn't change *which* stock outperforms — it changes *all* of them together. Second, complexity: it's a clean future-work extension rather than core methodology.

### Q35. What if I told you the AI boom is just one period — your results might not replicate in other regimes?
Valid concern. The 5-year period intentionally spans multiple regimes precisely to mitigate this. If clustering only worked in one regime, the 5-year aggregate would dilute the signal. The fact that we still get ARI = 0.04 suggests the disagreement between behavioral and sector groupings holds across regimes. A robust follow-up: re-run on a different 5-year window (2014-2018) and compare.

---

## I. Reproducibility & Engineering

### Q36. Walk me through running this from scratch.
```bash
git clone <repo>
cd Stock-Behavioral-Clustering-and-Return-Prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python pipeline.py     # 5–10 min
streamlit run app.py
```
That's it. The pipeline is idempotent (caches mean re-running is fast); the dashboard reads from the same cache.

### Q37. How did you organize the code?
Separation of concerns. `src/data.py` handles ETL. `src/features.py` handles feature engineering. `src/clustering.py` and `src/classification.py` handle modeling. `src/viz.py` handles plot helpers. `pipeline.py` orchestrates. `app.py` + `pages/` is the dashboard. Each module has a `run()` function with caching, so they can be invoked independently or chained.

### Q38. Why parquet and not CSV?
Parquet is columnar, compressed, and preserves dtypes. The fact table is 743K rows × 16 cols — as CSV it'd be ~150 MB; as parquet it's ~45 MB and loads ~10× faster. For analytical workloads parquet is the obvious choice.

---

## J. Critical / Adversarial Questions

### Q39. Aren't you just demonstrating that a noisy method finds noisy structure?
The ARI test is precisely the rebuttal. If clustering were finding noise, the ARI would still be near zero, but so would the silhouette. With silhouette around 0.25 *and* ARI around 0.04, we have evidence of genuine structure that's distinct from sectors. If the structure were noise, both metrics would be near zero.

### Q40. Why should anyone trust ML predictions on financial data given the field's reproducibility crisis?
Several reasons specific to my methodology:
- **Temporal split** (no future peeking)
- **Honest reporting** of weak AUC values rather than overclaiming
- **Multiple models** to avoid cherry-picking
- **Public, reproducible data**

The AUC of 0.51 is the result you should *expect* from an honest analysis of weak-form efficient markets. A paper claiming 0.65 AUC on similar data would be the suspicious one.

### Q41. What if your bug-fixed code has subtle bugs you haven't caught?
Two safeguards: (1) unit-testable building blocks — each technical indicator is a small function that can be sanity-checked against known values (e.g., RSI of a constant series should be 50); (2) cross-checking — every model converged to a similar AUC, which would be unlikely if upstream features were systematically broken.

### Q42. What's the single number you'd point to that summarizes the project?
ARI of 0.044 — because it answers RQ1 directly. The classification ROC-AUC of 0.514 answers RQ2. Together they say: yes, behavioral clusters differ structurally from sectors; and yes, but barely, knowing the cluster helps prediction. Honest, defensible answers to both halves of the question.

---

*This document is intended as a personal Q&A reference. Read it once before the presentation; you'll only need to recall about half of it on the day.*
