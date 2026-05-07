"""
Plotly visualization helpers — used across all Streamlit pages.

Theme: clean, professional, financial-dashboard aesthetic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── DESIGN TOKENS ────────────────────────────────────────────────────────────
NAVY   = "#1B3A5C"
BLUE   = "#2E6DA4"
TEAL   = "#17A2B8"
AMBER  = "#F0AD4E"
CRIMSON= "#D9534F"
GRAY   = "#6C757D"
LIGHT  = "#F4F6F9"

PALETTE = ["#2E6DA4", "#17A2B8", "#F0AD4E", "#D9534F", "#5CB85C",
           "#8E44AD", "#1ABC9C", "#E67E22", "#34495E", "#16A085", "#C0392B"]


def _apply_theme(fig: go.Figure, title: str | None = None, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left",
                   font=dict(size=16, color=NAVY, family="Helvetica")),
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Helvetica", color="#333333", size=12),
        margin=dict(l=50, r=30, t=50 if title else 30, b=50),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=LIGHT, borderwidth=1),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Helvetica"),
    )
    fig.update_xaxes(showgrid=True, gridcolor=LIGHT, zeroline=False, linecolor=LIGHT)
    fig.update_yaxes(showgrid=True, gridcolor=LIGHT, zeroline=False, linecolor=LIGHT)
    return fig


# ── EDA CHARTS ──────────────────────────────────────────────────────────────
def return_distribution(returns: pd.Series, title: str) -> go.Figure:
    fig = go.Figure(data=[go.Histogram(
        x=returns.dropna(), nbinsx=120,
        marker=dict(color=BLUE, line=dict(width=0)),
        name="returns",
    )])
    fig.add_vline(x=0, line_dash="dash", line_color=GRAY)
    fig.update_xaxes(title="Daily return")
    fig.update_yaxes(title="Frequency")
    return _apply_theme(fig, title)


def sector_volatility_box(df: pd.DataFrame) -> go.Figure:
    sec_vol = (df.groupby(["ticker", "gics_sector"])["raw_daily_return"]
                 .std().reset_index()
                 .rename(columns={"raw_daily_return": "volatility"}))
    sec_vol["volatility"] *= np.sqrt(252)  # annualize for interpretability
    order = sec_vol.groupby("gics_sector")["volatility"].median().sort_values().index.tolist()
    fig = px.box(sec_vol, x="gics_sector", y="volatility", category_orders={"gics_sector": order},
                 color="gics_sector", color_discrete_sequence=PALETTE)
    fig.update_traces(marker=dict(size=4))
    fig.update_xaxes(title="GICS Sector", tickangle=-30)
    fig.update_yaxes(title="Annualized volatility")
    fig.update_layout(showlegend=False)
    return _apply_theme(fig, "Volatility distribution by GICS sector", height=480)


def sector_cumulative_returns(df: pd.DataFrame) -> go.Figure:
    sector_avg = (df.groupby(["Date", "gics_sector"])["raw_daily_return"]
                    .mean().reset_index())
    sector_avg["cum_return"] = (
        sector_avg.groupby("gics_sector")["raw_daily_return"]
                  .transform(lambda r: (1 + r).cumprod() - 1)
    )
    fig = px.line(sector_avg, x="Date", y="cum_return", color="gics_sector",
                  color_discrete_sequence=PALETTE)
    fig.update_yaxes(title="Cumulative return", tickformat=".0%")
    fig.update_xaxes(title="")
    return _apply_theme(fig, "Cumulative sector returns 2019–2024", height=460)


def correlation_heatmap(df: pd.DataFrame, value_col: str, title: str, sample: int = 60) -> go.Figure:
    """Pairwise correlation heatmap of a sample of stocks, ordered by sector."""
    pivot = df.pivot_table(index="Date", columns="ticker", values=value_col)
    sectors = df.drop_duplicates("ticker").set_index("ticker")["gics_sector"]
    sample_tickers = (sectors.dropna()
                             .reset_index()
                             .groupby("gics_sector").head(max(1, sample // sectors.nunique()))
                             ["ticker"].tolist())
    sample_tickers = [t for t in sample_tickers if t in pivot.columns][:sample]
    sample_tickers.sort(key=lambda t: sectors.get(t, ""))

    corr = pivot[sample_tickers].corr()
    fig = px.imshow(corr.values,
                    x=sample_tickers, y=sample_tickers,
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    aspect="auto")
    fig.update_xaxes(title="", showticklabels=False)
    fig.update_yaxes(title="", showticklabels=False)
    return _apply_theme(fig, title, height=460)


def rsi_distribution(features_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(data=[go.Histogram(
        x=features_df["rsi_14"].dropna(), nbinsx=80,
        marker=dict(color=TEAL, line=dict(width=0)))])
    fig.add_vline(x=30, line_dash="dash", line_color=CRIMSON, annotation_text="Oversold (30)")
    fig.add_vline(x=70, line_dash="dash", line_color=CRIMSON, annotation_text="Overbought (70)")
    fig.update_xaxes(title="RSI(14)")
    fig.update_yaxes(title="Frequency")
    return _apply_theme(fig, "Distribution of 14-day RSI across all stocks")


# ── CLUSTERING CHARTS ───────────────────────────────────────────────────────
def elbow_silhouette(diag: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=diag["k"], y=diag["inertia"],
                              mode="lines+markers", name="Inertia",
                              line=dict(color=BLUE, width=2),
                              marker=dict(size=8)),
                   secondary_y=False)
    fig.add_trace(go.Scatter(x=diag["k"], y=diag["silhouette"],
                              mode="lines+markers", name="Silhouette",
                              line=dict(color=AMBER, width=2, dash="dot"),
                              marker=dict(size=8)),
                   secondary_y=True)
    fig.update_xaxes(title="K (number of clusters)")
    fig.update_yaxes(title="Inertia (lower=tighter)", secondary_y=False, color=BLUE)
    fig.update_yaxes(title="Silhouette (higher=better)", secondary_y=True, color=AMBER)
    return _apply_theme(fig, "K-Means: elbow + silhouette diagnostics")


def pca_cluster_scatter(assignments: pd.DataFrame, color_by: str = "kmeans") -> go.Figure:
    df = assignments.copy()
    df[color_by] = df[color_by].astype(str)
    fig = px.scatter(df, x="pca1", y="pca2", color=color_by,
                     hover_data=["ticker", "gics_sector"],
                     color_discrete_sequence=PALETTE)
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white"), opacity=0.85))
    fig.update_xaxes(title="PC1")
    fig.update_yaxes(title="PC2")
    title = "PCA projection — colored by behavioral cluster" if color_by == "kmeans" else \
            "PCA projection — colored by GICS sector"
    return _apply_theme(fig, title, height=520)


def cluster_sector_heatmap(assignments: pd.DataFrame) -> go.Figure:
    """Confusion-style heatmap: rows=GICS sectors, cols=clusters."""
    ct = pd.crosstab(assignments["gics_sector"], assignments["kmeans"].astype(str))
    fig = px.imshow(ct.values, x=[f"Cluster {c}" for c in ct.columns], y=ct.index,
                    color_continuous_scale="Blues", aspect="auto",
                    labels=dict(color="Stock count"))
    for i, sector in enumerate(ct.index):
        for j, cluster in enumerate(ct.columns):
            v = ct.iloc[i, j]
            if v > 0:
                fig.add_annotation(x=j, y=i, text=str(v), showarrow=False,
                                    font=dict(color="white" if v > ct.values.max()/2 else NAVY, size=11))
    fig.update_xaxes(title="")
    fig.update_yaxes(title="")
    return _apply_theme(fig, "GICS Sector × Behavioral Cluster", height=520)


def cluster_profile_radar(raw_fp: pd.DataFrame, assignments: pd.DataFrame,
                            features: list[str]) -> go.Figure:
    """Radar chart of standardized cluster centroids — shows each cluster's character."""
    merged = raw_fp.merge(assignments[["ticker", "kmeans"]], on="ticker")
    centroids = merged.groupby("kmeans")[features].mean()
    # Standardize each feature column to [0,1] for radar comparability
    norm = (centroids - centroids.min()) / (centroids.max() - centroids.min() + 1e-9)

    fig = go.Figure()
    for c in norm.index:
        fig.add_trace(go.Scatterpolar(
            r=norm.loc[c].values,
            theta=features,
            fill="toself", name=f"Cluster {c}",
            line=dict(color=PALETTE[c % len(PALETTE)]),
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
        title=dict(text="Behavioral cluster fingerprints (relative scale)",
                   x=0.0, xanchor="left", font=dict(size=16, color=NAVY)),
        height=520, paper_bgcolor="white",
        font=dict(family="Helvetica", color="#333"),
    )
    return fig


# ── CLASSIFICATION CHARTS ───────────────────────────────────────────────────
def metrics_grouped_bar(metrics_df: pd.DataFrame, metric: str = "roc_auc") -> go.Figure:
    fig = px.bar(metrics_df, x="model", y=metric, color="variant",
                  barmode="group",
                  color_discrete_map={"Without cluster": GRAY, "With cluster": BLUE})
    fig.update_yaxes(title=metric.upper(), range=[0.45, max(0.7, metrics_df[metric].max() + 0.05)])
    fig.update_xaxes(title="")
    return _apply_theme(fig, f"Model performance — {metric.upper()}: with vs. without cluster feature")


def feature_importance_bar(feat_imp: pd.DataFrame, top_n: int = 15) -> go.Figure:
    f = feat_imp.head(top_n).iloc[::-1]
    fig = go.Figure(data=[go.Bar(
        x=f["importance"], y=f["feature"], orientation="h",
        marker=dict(color=BLUE, line=dict(width=0)),
    )])
    fig.update_xaxes(title="Importance")
    fig.update_yaxes(title="")
    return _apply_theme(fig, "Random Forest feature importance (top features)", height=460)
