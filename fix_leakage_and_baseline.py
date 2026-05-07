"""
Fix #1 (data leakage): Refit clustering using ONLY training-period data
                        (2019-2022), then assign all stocks to those clusters.
Fix #2 (one-hot encoding): One-hot encode cluster labels for classification.
Fix #3 (baseline): Compute always-up baseline on the test set, not full set.
Fix #4 (confusion matrix): Save confusion matrices for the best model.

Outputs (overwrites in data_cache/):
    cluster_assignments.parquet  — uses train-only fitted clusters
    cluster_metrics.parquet      — recomputed
    pca_projection.parquet       — recomputed on train-only space
    classification_metrics.parquet — re-run with one-hot cluster labels
    baseline.parquet             — TEST-SET baseline (proper)
    feature_importance.parquet   — re-run RF importances
    confusion_matrices.parquet   — NEW: per-model confusion matrices
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, adjusted_rand_score, silhouette_score,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

PROJECT_ROOT = Path(__file__).parent
CACHE = PROJECT_ROOT / "data_cache"

FINGERPRINT_FEATURES = [
    "mean_excess_return", "volatility", "mean_beta",
    "mean_rsi", "mean_bollinger_width", "max_drawdown", "momentum_score",
]
TECHNICAL_FEATURES = [
    "ma_20", "ma_50",
    "rsi_14", "macd", "macd_hist",
    "bollinger_width",
    "atr_14", "obv",
    "beta_60d", "volatility_20d", "volume",
]


# ── HELPERS ─────────────────────────────────────────────────────────────────
def _max_drawdown(prices: pd.Series) -> float:
    rm = prices.cummax()
    dd = prices / rm - 1
    return dd.min()


def fingerprint_one_stock(g: pd.DataFrame) -> dict:
    """Compute the 7 behavioral features from a stock's daily series."""
    g = g.sort_values("Date")
    prices = g["adj_close"]
    excess = g["excess_daily_return"].dropna()
    mom = (prices / prices.shift(252) - 1).mean()
    return {
        "mean_excess_return":   excess.mean(),
        "volatility":           excess.std(),
        "mean_beta":            g["beta_60d"].mean(),
        "mean_rsi":             g["rsi_14"].mean(),
        "mean_bollinger_width": g["bollinger_width"].mean(),
        "max_drawdown":         _max_drawdown(prices),
        "momentum_score":       mom,
    }


def winsorize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        lo, hi = out[c].quantile([0.01, 0.99])
        out[c] = out[c].clip(lo, hi)
    return out


def _evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_proba),
    }


def _models() -> dict:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, n_jobs=-1),
        "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                            subsample=0.8, eval_metric="logloss",
                                            random_state=42, n_jobs=-1)
    else:
        models["Gradient Boosting"] = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    return models


def _normalize_by_close(features_df: pd.DataFrame) -> pd.DataFrame:
    out = features_df.copy()
    out["ma_20"]    = out["ma_20"] / out["adj_close"]
    out["ma_50"]    = out["ma_50"] / out["adj_close"]
    out["volume"]   = np.log1p(out["volume"].clip(lower=0))
    out["obv"]      = np.sign(out["obv"]) * np.log1p(out["obv"].abs())
    return out


# ── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("Loading cached technical features…")
    feats = pd.read_parquet(CACHE / "technical_features.parquet")
    sp500 = pd.read_parquet(CACHE / "sp500_table.parquet")
    print(f"  rows={len(feats):,}  tickers={feats['ticker'].nunique()}")

    SPLIT_YEAR = 2023

    # ─── 1) Train-only fingerprints ─────────────────────────────────────────
    print(f"\nComputing TRAIN-ONLY fingerprints (Date < {SPLIT_YEAR})…")
    train_feats = feats[feats["Date"].dt.year < SPLIT_YEAR]
    rows = []
    for ticker, g in train_feats.groupby("ticker", sort=False):
        fp = fingerprint_one_stock(g)
        fp["ticker"] = ticker
        rows.append(fp)
    raw_fp = pd.DataFrame(rows).dropna()
    raw_fp = winsorize(raw_fp, FINGERPRINT_FEATURES)
    print(f"  train-only fingerprints: {len(raw_fp)} stocks")

    scaler = StandardScaler().fit(raw_fp[FINGERPRINT_FEATURES])
    scaled_X = scaler.transform(raw_fp[FINGERPRINT_FEATURES])
    scaled_fp = raw_fp.copy()
    scaled_fp[FINGERPRINT_FEATURES] = scaled_X

    # ─── 2) Refit clustering on train-only ──────────────────────────────────
    print("\nFitting K-Means K=5 on train-only fingerprints…")
    km = KMeans(n_clusters=5, n_init=20, random_state=42).fit(scaled_X)
    km_labels = km.predict(scaled_X)

    # K-Means diagnostics for elbow plot (re-run on train-only)
    print("Recomputing K-Means diagnostics K=2..15 on train-only…")
    diag_rows = []
    for k in range(2, 16):
        kk = KMeans(n_clusters=k, n_init=10, random_state=42).fit(scaled_X)
        sil = silhouette_score(scaled_X, kk.labels_)
        diag_rows.append({"k": k, "inertia": kk.inertia_, "silhouette": sil})
    diag = pd.DataFrame(diag_rows)

    print("Fitting Hierarchical (Ward) on train-only…")
    Z = linkage(scaled_X, method="ward")
    hc_labels = fcluster(Z, t=5, criterion="maxclust") - 1

    print("Fitting DBSCAN on train-only…")
    db_labels = DBSCAN(eps=0.9, min_samples=5).fit_predict(scaled_X)

    # PCA projection
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled_X)
    evr = pca.explained_variance_ratio_

    # Cluster vs sector
    sector_lookup = sp500.set_index("ticker")["gics_sector"]
    sectors = sector_lookup.reindex(raw_fp["ticker"].values).fillna("Unknown")
    sector_codes = pd.Categorical(sectors).codes

    metrics_rows = []
    for name, labels in [("kmeans", km_labels), ("hierarchical", hc_labels), ("dbscan", db_labels)]:
        valid = labels >= 0
        sil = silhouette_score(scaled_X[valid], labels[valid]) if valid.sum() > 1 and len(set(labels[valid])) > 1 else np.nan
        ari = adjusted_rand_score(sector_codes, labels)
        metrics_rows.append({
            "algorithm": name,
            "n_clusters": int(len(set(labels[labels >= 0]))),
            "n_outliers": int((labels == -1).sum()),
            "silhouette": sil,
            "ari_vs_sector": ari,
        })
    cluster_metrics = pd.DataFrame(metrics_rows)
    print("\nCluster metrics (train-only fingerprints):")
    print(cluster_metrics.round(4).to_string(index=False))

    # ─── 3) Save cluster artifacts ──────────────────────────────────────────
    assignments = pd.DataFrame({
        "ticker":       raw_fp["ticker"].values,
        "kmeans":       km_labels,
        "hierarchical": hc_labels,
        "dbscan":       db_labels,
        "gics_sector":  sectors.values,
        "pca1":         coords[:, 0],
        "pca2":         coords[:, 1],
    })

    pca_df = pd.DataFrame({"pca1": coords[:, 0], "pca2": coords[:, 1],
                            "evr1": evr[0], "evr2": evr[1], "ticker": raw_fp["ticker"].values})

    assignments.to_parquet(CACHE / "cluster_assignments.parquet", index=False)
    diag.to_parquet(CACHE / "cluster_diagnostics.parquet", index=False)
    cluster_metrics.to_parquet(CACHE / "cluster_metrics.parquet", index=False)
    pca_df.to_parquet(CACHE / "pca_projection.parquet", index=False)
    raw_fp.to_parquet(CACHE / "fingerprints_raw.parquet", index=False)
    scaled_fp.to_parquet(CACHE / "fingerprints_scaled.parquet", index=False)
    np.save(CACHE / "linkage_matrix.npy", Z)
    print("✓ Saved leakage-free clustering artifacts")

    # ─── 4) Classification with one-hot cluster + train-only labels ─────────
    print("\nPreparing classification dataset with one-hot encoded clusters…")
    df = feats.merge(
        assignments[["ticker", "kmeans"]].rename(columns={"kmeans": "cluster_label"}),
        on="ticker", how="left",
    )
    df = _normalize_by_close(df)
    df = df.dropna(subset=TECHNICAL_FEATURES + ["forward_5day_direction", "cluster_label"])
    df["cluster_label"] = df["cluster_label"].astype(int)

    # One-hot encode cluster
    n_clusters = int(df["cluster_label"].max()) + 1
    cluster_oh_cols = [f"cluster_{i}" for i in range(n_clusters)]
    for i in range(n_clusters):
        df[f"cluster_{i}"] = (df["cluster_label"] == i).astype(int)

    train = df[df["Date"].dt.year < SPLIT_YEAR]
    test  = df[df["Date"].dt.year >= SPLIT_YEAR]
    y_train, y_test = train["forward_5day_direction"], test["forward_5day_direction"]

    # Test-set baseline (always predict up)
    test_baseline = float(y_test.mean())  # if predict 1 always, accuracy = mean(y_test)
    print(f"\nTest-set always-up baseline: {test_baseline:.4f}")
    print(f"Train rows={len(train):,}  Test rows={len(test):,}")

    # ─── 5) Train both variants ─────────────────────────────────────────────
    base_cols    = TECHNICAL_FEATURES
    cluster_cols = TECHNICAL_FEATURES + cluster_oh_cols  # ONE-HOT now

    rows = []
    confs = []
    feat_imp_rows = None

    for variant, cols in [("Without cluster", base_cols), ("With cluster (one-hot)", cluster_cols)]:
        X_train, X_test = train[cols].copy(), test[cols].copy()
        scaler_x = StandardScaler().fit(X_train)
        X_train_s = scaler_x.transform(X_train)
        X_test_s  = scaler_x.transform(X_test)

        for name, model in _models().items():
            print(f"  [{variant}] training {name}…")
            X_tr, X_te = (X_train_s, X_test_s) if name == "Logistic Regression" else (X_train.values, X_test.values)
            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)
            y_proba = model.predict_proba(X_te)[:, 1]
            metrics = _evaluate(y_test, y_pred, y_proba)
            metrics.update({"model": name, "variant": variant})
            rows.append(metrics)

            cm = confusion_matrix(y_test, y_pred)
            confs.append({
                "model": name, "variant": variant,
                "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
            })

            if variant == "With cluster (one-hot)" and name == "Random Forest":
                feat_imp_rows = pd.DataFrame({
                    "feature": cols,
                    "importance": model.feature_importances_,
                }).sort_values("importance", ascending=False)

    metrics_df = pd.DataFrame(rows)[["model", "variant", "accuracy", "precision", "recall", "f1", "roc_auc"]]
    confs_df = pd.DataFrame(confs)

    print("\nClassification metrics (leakage-free clusters, one-hot encoded):")
    print(metrics_df.round(4).to_string(index=False))
    print("\nConfusion matrices (test set):")
    print(confs_df.to_string(index=False))

    # ─── 6) Save classification artifacts ───────────────────────────────────
    metrics_df.to_parquet(CACHE / "classification_metrics.parquet", index=False)
    confs_df.to_parquet(CACHE / "confusion_matrices.parquet", index=False)
    feat_imp_rows.to_parquet(CACHE / "feature_importance.parquet", index=False)
    pd.DataFrame({"baseline": [test_baseline]}).to_parquet(CACHE / "baseline.parquet")

    print("\n✓ All artifacts updated. Test-set baseline:", test_baseline)


if __name__ == "__main__":
    main()
