"""
data_preprocessing.py
Loads the raw Maven Roasters transaction log, validates it, and engineers
calendar/revenue fields used by every downstream module.
"""
import pandas as pd
import numpy as np

RAW_PATH = "data/coffee_shop_sales.csv"


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["transaction_date"])
    return df


def basic_quality_report(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "date_range": (df["transaction_date"].min(), df["transaction_date"].max()),
        "stores": df["store_location"].unique().tolist(),
        "categories": df["product_category"].unique().tolist(),
    }


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()
    df = df.dropna(subset=["transaction_qty", "unit_price", "transaction_date"])
    df = df[(df["transaction_qty"] > 0) & (df["unit_price"] > 0)]
    return df


def add_calendar_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["transaction_time"] = pd.to_datetime(df["transaction_time"], format="%H:%M:%S")
    df["hour"] = df["transaction_time"].dt.hour
    df["day_of_week"] = df["transaction_date"].dt.day_name()
    df["is_weekend"] = df["transaction_date"].dt.dayofweek >= 5
    df["month"] = df["transaction_date"].dt.month
    df["week"] = df["transaction_date"].dt.isocalendar().week.astype(int)
    df["revenue"] = df["transaction_qty"] * df["unit_price"]
    return df


def run(path: str = RAW_PATH) -> pd.DataFrame:
    raw = load_raw(path)
    report = basic_quality_report(raw)
    cleaned = clean(raw)
    enriched = add_calendar_fields(cleaned)
    return enriched, report


if __name__ == "__main__":
    df, report = run()
    print(report)
    df.to_csv("data/coffee_shop_sales_clean.csv", index=False)
