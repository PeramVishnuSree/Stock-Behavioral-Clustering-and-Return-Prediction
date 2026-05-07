"""
End-to-end orchestrator. Runs the full project pipeline:

    Wikipedia + Yahoo Finance  →  fact table
    fact table                 →  technical features + behavioral fingerprints
    fingerprints               →  K-Means / Hierarchical / DBSCAN clusters
    technicals + cluster label →  4 classifiers (with/without cluster)

All artifacts are cached as parquet/joblib in data_cache/.

Run with:
    python pipeline.py            # uses cache where available
    python pipeline.py --force    # re-runs everything from scratch
"""
import argparse

from src import data, features, clustering, classification


def main(force: bool = False):
    print("\n" + "═"*70)
    print("  STAGE 1 — DATA")
    print("═"*70)
    df, sp500_table, market_returns = data.run(force=force)

    print("\n" + "═"*70)
    print("  STAGE 2 — FEATURES")
    print("═"*70)
    features_df, raw_fp, scaled_fp = features.run(df, market_returns, force=force)

    print("\n" + "═"*70)
    print("  STAGE 3 — CLUSTERING")
    print("═"*70)
    cluster_results = clustering.run(scaled_fp, sp500_table, k=5, force=force)

    print("\n" + "═"*70)
    print("  STAGE 4 — CLASSIFICATION")
    print("═"*70)
    classification.run(features_df, cluster_results["assignments"], force=force)

    print("\n" + "═"*70)
    print("  ✓ PIPELINE COMPLETE — artifacts in data_cache/")
    print("═"*70)
    print("\nNext: launch the dashboard with → streamlit run app.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                          help="Recompute every stage (ignore cache)")
    main(force=parser.parse_args().force)
