"""
statistical_analysis.py
Runs the hypothesis tests that justify the business framing of the project:
1. Welch's t-test  -> does average per-transaction revenue differ Weekday vs Weekend?
2. One-way ANOVA   -> does average per-transaction revenue differ across the 3 stores?
3. Chi-square test -> is product-category mix associated with store location?
4. Pearson corr    -> relationship between unit price and quantity purchased per transaction.
"""
import pandas as pd
from scipy import stats
import numpy as np


def transaction_level(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse line items to one row per transaction (qty-weighted revenue, first category)."""
    tx = df.groupby(["transaction_id", "transaction_date", "store_location", "is_weekend"]).agg(
        revenue=("revenue", "sum"),
        items=("transaction_qty", "sum"),
    ).reset_index()
    return tx


def weekday_weekend_ttest(tx: pd.DataFrame) -> dict:
    weekday = tx.loc[~tx["is_weekend"], "revenue"]
    weekend = tx.loc[tx["is_weekend"], "revenue"]
    t_stat, p_val = stats.ttest_ind(weekend, weekday, equal_var=False)  # Welch's t-test
    return {
        "test": "Welch's two-sample t-test",
        "h0": "Mean transaction revenue is equal on weekends and weekdays.",
        "h1": "Mean transaction revenue differs between weekends and weekdays.",
        "n_weekday": len(weekday), "n_weekend": len(weekend),
        "mean_weekday": round(weekday.mean(), 4), "mean_weekend": round(weekend.mean(), 4),
        "t_stat": round(t_stat, 4), "p_value": p_val,
        "significant_at_0.05": bool(p_val < 0.05),
    }


def store_anova(tx: pd.DataFrame) -> dict:
    groups = [g["revenue"].values for _, g in tx.groupby("store_location")]
    f_stat, p_val = stats.f_oneway(*groups)
    means = tx.groupby("store_location")["revenue"].mean().round(4).to_dict()
    return {
        "test": "One-way ANOVA",
        "h0": "Mean transaction revenue is equal across all three store locations.",
        "h1": "At least one store's mean transaction revenue differs.",
        "f_stat": round(f_stat, 4), "p_value": p_val,
        "group_means": means,
        "significant_at_0.05": bool(p_val < 0.05),
    }


def category_store_chi_square(df: pd.DataFrame) -> dict:
    contingency = pd.crosstab(df["store_location"], df["product_category"])
    chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
    return {
        "test": "Chi-square test of independence",
        "h0": "Product-category mix purchased is independent of store location.",
        "h1": "Product-category mix purchased is associated with store location.",
        "chi2_stat": round(chi2, 4), "dof": dof, "p_value": p_val,
        "significant_at_0.05": bool(p_val < 0.05),
    }


def price_quantity_correlation(df: pd.DataFrame) -> dict:
    r, p_val = stats.pearsonr(df["unit_price"], df["transaction_qty"])
    return {
        "test": "Pearson correlation (unit_price vs transaction_qty)",
        "r": round(r, 4), "p_value": p_val,
        "significant_at_0.05": bool(p_val < 0.05),
    }


def run(df: pd.DataFrame) -> dict:
    tx = transaction_level(df)
    return {
        "weekday_weekend_ttest": weekday_weekend_ttest(tx),
        "store_anova": store_anova(tx),
        "category_store_chi_square": category_store_chi_square(df),
        "price_quantity_correlation": price_quantity_correlation(df),
    }


if __name__ == "__main__":
    from data_preprocessing import run as preprocess
    df, _ = preprocess()
    results = run(df)
    for k, v in results.items():
        print(k, "->", v)
