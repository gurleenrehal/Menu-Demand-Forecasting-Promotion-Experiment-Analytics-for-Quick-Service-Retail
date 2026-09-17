"""
evaluation.py
Thin, reusable metric layer so modeling.py and the notebooks don't duplicate
scoring logic. Regression metrics only, since every task in this project
(demand forecasting) is a regression problem.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_report(y_true: pd.Series, y_pred: np.ndarray, model_name: str = "model") -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    denom = y_true.replace(0, np.nan)
    mape = float((np.abs((y_true - y_pred) / denom)).dropna().mean() * 100)
    return {"model": model_name, "MAE": round(mae, 3), "RMSE": round(rmse, 3),
            "R2": round(r2, 3), "MAPE_%": round(mape, 2)}


def compare_models(results: list) -> pd.DataFrame:
    return pd.DataFrame(results).sort_values("MAE")


def error_by_segment(df: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray, segment_col: str) -> pd.DataFrame:
    """Breaks down absolute error by a categorical segment (e.g. store, category) -
    used for the error-analysis / model-limitations section of the report."""
    tmp = df[[segment_col]].copy()
    tmp["abs_error"] = np.abs(y_true.values - y_pred)
    return tmp.groupby(segment_col)["abs_error"].mean().sort_values(ascending=False)
