"""
Classification module — Part 2 of the project.

Trains 4 models (Logistic Regression, Decision Tree, Random Forest, XGBoost)
to predict 5-day forward return direction. Critical experiment: trains each model
WITH and WITHOUT the behavioral cluster label feature, to test whether
clustering output adds predictive signal beyond raw technical indicators.

Outputs (in data_cache/):
    classification_metrics.parquet — per-model metrics, with/without cluster
    feature_importance.parquet     — Random Forest importances
    rf_model.joblib                — best Random Forest (with cluster label)
    xgb_model.joblib               — best XGBoost (with cluster label)
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import CACHE_DIR


# Try XGBoost; fall back to sklearn GradientBoostingClassifier if not installed
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


TECHNICAL_FEATURES = [
    "ma_20", "ma_50",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bollinger_upper", "bollinger_lower", "bollinger_width",
    "atr_14", "obv",
    "beta_60d", "volatility_20d",
    "volume",
]


def _normalize_by_close(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Some features (MAs, Bollinger bands, OBV) scale with the stock's price level
    and would dominate the model. Convert them to ratios relative to adj_close.
    """
    out = features_df.copy()
    out["ma_20"]            = out["ma_20"] / out["adj_close"]
    out["ma_50"]            = out["ma_50"] / out["adj_close"]
    out["bollinger_upper"]  = out["bollinger_upper"] / out["adj_close"]
    out["bollinger_lower"]  = out["bollinger_lower"] / out["adj_close"]
    # log-scale volume and OBV which span orders of magnitude
    out["volume"] = np.log1p(out["volume"].clip(lower=0))
    out["obv"]    = np.sign(out["obv"]) * np.log1p(out["obv"].abs())
    return out


def _temporal_split(df: pd.DataFrame, split_year: int = 2023) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["Date"].dt.year < split_year]
    test  = df[df["Date"].dt.year >= split_year]
    return train, test


def _evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_proba),
    }


def prepare_dataset(features_df: pd.DataFrame, cluster_assignments: pd.DataFrame) -> pd.DataFrame:
    """Join cluster labels onto the technical-features fact table and drop NaN rows."""
    df = features_df.merge(
        cluster_assignments[["ticker", "kmeans"]].rename(columns={"kmeans": "cluster_label"}),
        on="ticker", how="left",
    )
    df = _normalize_by_close(df)
    df = df.dropna(subset=TECHNICAL_FEATURES + ["forward_5day_direction", "cluster_label"])
    return df


def _models() -> dict:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, n_jobs=-1),
        "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=12,
                                                       n_jobs=-1, random_state=42),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                            subsample=0.8, eval_metric="logloss",
                                            random_state=42, n_jobs=-1)
    else:
        models["Gradient Boosting"] = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    return models


def train_and_evaluate(df: pd.DataFrame, split_year: int = 2023) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Trains every model twice — with and without the cluster_label feature — and
    returns a comparison table + the trained "with cluster" RF and XGB models.
    """
    train, test = _temporal_split(df, split_year)

    base_X_cols    = TECHNICAL_FEATURES
    cluster_X_cols = TECHNICAL_FEATURES + ["cluster_label"]
    y_train, y_test = train["forward_5day_direction"], test["forward_5day_direction"]

    rows = []
    saved_models: dict = {}
    feat_imp = None

    for variant, cols in [("Without cluster", base_X_cols), ("With cluster", cluster_X_cols)]:
        X_train, X_test = train[cols].copy(), test[cols].copy()

        # Logistic regression benefits from scaling — fit a scaler once for it
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        for name, model in _models().items():
            print(f"[classification] {variant} — training {name}…")
            X_tr, X_te = (X_train_scaled, X_test_scaled) if name == "Logistic Regression" else (X_train.values, X_test.values)
            model.fit(X_tr, y_train)
            y_pred  = model.predict(X_te)
            y_proba = model.predict_proba(X_te)[:, 1]
            metrics = _evaluate(y_test, y_pred, y_proba)
            metrics.update({"model": name, "variant": variant})
            rows.append(metrics)

            # Save the "with cluster" Random Forest for feature-importance + persistence
            if variant == "With cluster":
                if name == "Random Forest":
                    saved_models["rf"] = model
                    feat_imp = pd.DataFrame({
                        "feature": cols,
                        "importance": model.feature_importances_,
                    }).sort_values("importance", ascending=False)
                if name in ("XGBoost", "Gradient Boosting"):
                    saved_models["boost"] = model

    metrics_df = pd.DataFrame(rows)[
        ["model", "variant", "accuracy", "precision", "recall", "f1", "roc_auc"]
    ]
    return metrics_df, saved_models, feat_imp


def baseline_accuracy(df: pd.DataFrame) -> float:
    """Naive baseline: always predict 'up'."""
    return df["forward_5day_direction"].mean()


# ─────────────────────────────────────────────────────────────────────────────
def run(features_df: pd.DataFrame, cluster_assignments: pd.DataFrame, force: bool = False):
    metrics_path = CACHE_DIR / "classification_metrics.parquet"
    feat_path    = CACHE_DIR / "feature_importance.parquet"
    rf_path      = CACHE_DIR / "rf_model.joblib"
    boost_path   = CACHE_DIR / "boost_model.joblib"
    baseline_path = CACHE_DIR / "baseline.parquet"

    if not force and all(p.exists() for p in [metrics_path, feat_path, baseline_path]):
        print("[classification] Loading cached results…")
        return {
            "metrics":     pd.read_parquet(metrics_path),
            "importance":  pd.read_parquet(feat_path),
            "baseline":    pd.read_parquet(baseline_path)["baseline"].iloc[0],
        }

    print("[classification] Preparing dataset…")
    ds = prepare_dataset(features_df, cluster_assignments)
    base = baseline_accuracy(ds)
    print(f"[classification] Naive 'always up' baseline accuracy: {base:.3f}")

    metrics_df, saved_models, feat_imp = train_and_evaluate(ds)

    print(f"[classification] Caching results to {CACHE_DIR}/")
    metrics_df.to_parquet(metrics_path, index=False)
    feat_imp.to_parquet(feat_path, index=False)
    pd.DataFrame({"baseline": [base]}).to_parquet(baseline_path, index=False)
    if "rf" in saved_models:
        joblib.dump(saved_models["rf"], rf_path)
    if "boost" in saved_models:
        joblib.dump(saved_models["boost"], boost_path)

    print(f"[classification] Done.\n{metrics_df.to_string(index=False)}")
    return {"metrics": metrics_df, "importance": feat_imp, "baseline": base}
