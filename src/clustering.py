"""
Clustering module — Part 1 of the project.

Runs K-Means (with elbow + silhouette diagnostics), Hierarchical (with Ward linkage),
and DBSCAN (for outlier detection). Compares results against GICS sector labels
using Adjusted Rand Index.

Outputs (in data_cache/):
    cluster_assignments.parquet   — ticker → KMeans / Hierarchical / DBSCAN labels
    cluster_diagnostics.parquet   — K-Means elbow & silhouette scores per K
    pca_projection.parquet        — 2D PCA coords for visualization
    cluster_metrics.parquet       — ARI, silhouette, etc. summary
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

from . import CACHE_DIR
from .features import FINGERPRINT_FEATURES


# ── K-MEANS ─────────────────────────────────────────────────────────────────
def kmeans_diagnostics(X: np.ndarray, k_range: range = range(2, 16)) -> pd.DataFrame:
    """Compute inertia + silhouette for each K to support elbow analysis."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        sil = silhouette_score(X, km.labels_)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


def fit_kmeans(X: np.ndarray, k: int) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(X)


# ── HIERARCHICAL ────────────────────────────────────────────────────────────
def fit_hierarchical(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (labels, linkage_matrix)."""
    Z = linkage(X, method="ward")
    labels = fcluster(Z, t=k, criterion="maxclust") - 1  # 0-indexed
    return labels, Z


# ── DBSCAN ──────────────────────────────────────────────────────────────────
def fit_dbscan(X: np.ndarray, eps: float = 0.9, min_samples: int = 5) -> np.ndarray:
    """Returns labels with -1 = noise/outlier."""
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)


# ── COMPARISON / METRICS ────────────────────────────────────────────────────
def compute_metrics(X: np.ndarray, labels: np.ndarray, sector_codes: np.ndarray) -> dict:
    """Silhouette + Adjusted Rand Index against GICS sector labels."""
    valid = labels >= 0  # ignore DBSCAN noise
    metrics: dict = {}
    if valid.sum() > 1 and len(set(labels[valid])) > 1:
        metrics["silhouette"] = silhouette_score(X[valid], labels[valid])
    else:
        metrics["silhouette"] = np.nan
    metrics["ari_vs_sector"] = adjusted_rand_score(sector_codes, labels)
    metrics["n_clusters"] = len(set(labels[labels >= 0]))
    metrics["n_outliers"] = int((labels == -1).sum())
    return metrics


def confusion_matrix(sectors: pd.Series, clusters: pd.Series) -> pd.DataFrame:
    """Cross-tab of GICS sectors × cluster labels."""
    return pd.crosstab(sectors, clusters, rownames=["GICS Sector"], colnames=["Cluster"])


# ── PCA PROJECTION ──────────────────────────────────────────────────────────
def project_2d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduce 7-D fingerprint to 2D via PCA. Returns (coords, explained_variance_ratio_)."""
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    return coords, pca.explained_variance_ratio_


# ─────────────────────────────────────────────────────────────────────────────
#  ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────
def run(scaled_fp: pd.DataFrame, sp500_table: pd.DataFrame, k: int = 5, force: bool = False):
    """
    Full clustering pipeline. Picks K=5 by default (a reasonable structural choice
    based on prior runs); the elbow plot is produced regardless so you can revisit.
    """
    out_assignments = CACHE_DIR / "cluster_assignments.parquet"
    out_diag        = CACHE_DIR / "cluster_diagnostics.parquet"
    out_pca         = CACHE_DIR / "pca_projection.parquet"
    out_metrics     = CACHE_DIR / "cluster_metrics.parquet"
    out_linkage     = CACHE_DIR / "linkage_matrix.npy"

    if not force and all(p.exists() for p in [out_assignments, out_diag, out_pca, out_metrics, out_linkage]):
        print("[clustering] Loading cached results…")
        return {
            "assignments": pd.read_parquet(out_assignments),
            "diagnostics": pd.read_parquet(out_diag),
            "pca":         pd.read_parquet(out_pca),
            "metrics":     pd.read_parquet(out_metrics),
            "linkage":     np.load(out_linkage),
        }

    X = scaled_fp[FINGERPRINT_FEATURES].values
    tickers = scaled_fp["ticker"].values

    # Sector codes for ARI
    sector_lookup = sp500_table.set_index("ticker")["gics_sector"]
    sectors = sector_lookup.reindex(tickers).fillna("Unknown")
    sector_codes = pd.Categorical(sectors).codes

    print("[clustering] Running K-Means diagnostics (K=2..15)…")
    diag = kmeans_diagnostics(X)

    print(f"[clustering] Fitting K-Means with k={k}…")
    km_labels = fit_kmeans(X, k)

    print(f"[clustering] Fitting Hierarchical (Ward) with k={k}…")
    hc_labels, Z = fit_hierarchical(X, k)

    print("[clustering] Fitting DBSCAN…")
    db_labels = fit_dbscan(X)

    print("[clustering] Projecting to 2D via PCA…")
    coords, evr = project_2d(X)

    assignments = pd.DataFrame({
        "ticker":       tickers,
        "kmeans":       km_labels,
        "hierarchical": hc_labels,
        "dbscan":       db_labels,
        "gics_sector":  sectors.values,
        "pca1":         coords[:, 0],
        "pca2":         coords[:, 1],
    })

    metrics_rows = []
    for name, labels in [("kmeans", km_labels), ("hierarchical", hc_labels), ("dbscan", db_labels)]:
        m = compute_metrics(X, labels, sector_codes)
        m["algorithm"] = name
        metrics_rows.append(m)
    metrics_df = pd.DataFrame(metrics_rows)[["algorithm", "n_clusters", "n_outliers", "silhouette", "ari_vs_sector"]]

    pca_df = pd.DataFrame({"pca1": coords[:, 0], "pca2": coords[:, 1],
                            "evr1": evr[0], "evr2": evr[1], "ticker": tickers})

    print(f"[clustering] Caching results to {CACHE_DIR}/")
    assignments.to_parquet(out_assignments, index=False)
    diag.to_parquet(out_diag, index=False)
    pca_df.to_parquet(out_pca, index=False)
    metrics_df.to_parquet(out_metrics, index=False)
    np.save(out_linkage, Z)

    print(f"[clustering] Done. Metrics:\n{metrics_df.to_string(index=False)}")
    return {
        "assignments": assignments,
        "diagnostics": diag,
        "pca":         pca_df,
        "metrics":     metrics_df,
        "linkage":     Z,
    }
