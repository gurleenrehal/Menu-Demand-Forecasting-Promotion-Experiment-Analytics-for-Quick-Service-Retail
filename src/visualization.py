"""
visualization.py
Reusable chart functions used across the notebooks and to generate the static
figures shipped in reports/figures/. Kept dependency-light (matplotlib +
seaborn only) so the repo runs without a Tableau/PowerBI license; see
dashboard/README.md for how the same KPIs map onto a BI tool.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")


def plot_monthly_revenue(df: pd.DataFrame, save_path: str = None):
    monthly = df.groupby(df["transaction_date"].dt.to_period("M"))["revenue"].sum()
    monthly.index = monthly.index.astype(str)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(monthly.index, monthly.values, marker="o", color="#6f4e37")
    ax.set_title("Monthly Revenue Trend")
    ax.set_ylabel("Revenue ($)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
    return fig


def plot_revenue_by(df: pd.DataFrame, group_col: str, save_path: str = None):
    grp = df.groupby(group_col)["revenue"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(7, max(4, 0.4 * len(grp))))
    grp.plot(kind="barh", ax=ax, color="#8b5e34")
    ax.set_title(f"Total Revenue by {group_col}")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
    return fig


def plot_model_comparison(results_df: pd.DataFrame, metric: str = "MAE", save_path: str = None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(results_df["model"], results_df[metric], color="#5b3a29")
    ax.set_title(f"Model Comparison - Test {metric}")
    ax.set_ylabel(metric)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
    return fig


def plot_feature_importance(importance: pd.Series, top_n: int = 10, save_path: str = None):
    top = importance.head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    top.plot(kind="barh", ax=ax, color="#9c6b30")
    ax.set_title(f"Feature Importance (top {top_n})")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
    return fig
