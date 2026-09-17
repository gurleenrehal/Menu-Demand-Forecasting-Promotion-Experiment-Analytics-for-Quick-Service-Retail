"""
feature_engineering.py
Builds a daily store x category demand table and engineers the lag/rolling/
calendar features used for next-day demand forecasting.
"""
import pandas as pd
import numpy as np


def build_daily_demand(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby(["transaction_date", "store_location", "product_category"])
        .agg(units_sold=("transaction_qty", "sum"), revenue=("revenue", "sum"))
        .reset_index()
    )
    # complete the calendar so every store x category has one row per day (0 units on gaps)
    all_dates = pd.date_range(daily["transaction_date"].min(), daily["transaction_date"].max(), freq="D")
    stores = daily["store_location"].unique()
    cats = daily["product_category"].unique()
    idx = pd.MultiIndex.from_product([all_dates, stores, cats],
                                      names=["transaction_date", "store_location", "product_category"])
    daily = daily.set_index(["transaction_date", "store_location", "product_category"]).reindex(idx, fill_value=0).reset_index()
    return daily


def add_calendar_features(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["day_of_week"] = daily["transaction_date"].dt.dayofweek
    daily["is_weekend"] = daily["day_of_week"] >= 5
    daily["month"] = daily["transaction_date"].dt.month
    daily["day_of_month"] = daily["transaction_date"].dt.day
    return daily


def add_lag_and_rolling_features(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.sort_values(["store_location", "product_category", "transaction_date"]).copy()
    key = ["store_location", "product_category"]
    daily["lag_1"] = daily.groupby(key)["units_sold"].shift(1)
    daily["lag_7"] = daily.groupby(key)["units_sold"].shift(7)
    daily["_shifted"] = daily["lag_1"]
    daily["rolling_mean_7"] = daily.groupby(key)["_shifted"].transform(lambda s: s.rolling(7).mean())
    daily["rolling_mean_14"] = daily.groupby(key)["_shifted"].transform(lambda s: s.rolling(14).mean())
    daily["rolling_std_7"] = daily.groupby(key)["_shifted"].transform(lambda s: s.rolling(7).std())
    daily = daily.drop(columns=["_shifted"])
    return daily.dropna().reset_index(drop=True)


def run(df: pd.DataFrame) -> pd.DataFrame:
    daily = build_daily_demand(df)
    daily = add_calendar_features(daily)
    daily = add_lag_and_rolling_features(daily)
    daily = pd.get_dummies(daily, columns=["store_location", "product_category"], drop_first=False)
    return daily


if __name__ == "__main__":
    from data_preprocessing import run as preprocess
    df, _ = preprocess()
    feat = run(df)
    print(feat.shape)
    print(feat.columns.tolist())
    feat.to_csv("data/daily_demand_features.csv", index=False)
