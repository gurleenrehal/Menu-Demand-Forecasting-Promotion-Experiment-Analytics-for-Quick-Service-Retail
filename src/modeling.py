"""
modeling.py
Trains and compares models that forecast next-day units_sold for a given
store x product-category pair, using a time-based (not random) split so no
future information leaks into training - the correct protocol for time series.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

TARGET = "units_sold"
DROP_COLS = ["transaction_date", "revenue", TARGET]


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    df = df.sort_values("transaction_date")
    cutoff = df["transaction_date"].quantile(1 - test_frac)
    train = df[df["transaction_date"] <= cutoff]
    test = df[df["transaction_date"] > cutoff]
    return train, test


def get_xy(df: pd.DataFrame):
    X = df.drop(columns=DROP_COLS)
    y = df[TARGET]
    return X, y


def naive_baseline(test: pd.DataFrame) -> dict:
    # baseline: predict today's demand = last week's demand for the same store/category (lag_7)
    y_true = test[TARGET]
    y_pred = test["lag_7"]
    return score(y_true, y_pred, "Baseline (naive: lag_7)")


def score(y_true, y_pred, name) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    denom = y_true.replace(0, np.nan)
    mape = float((np.abs((y_true - y_pred) / denom)).dropna().mean() * 100)
    return {"model": name, "MAE": round(mae, 3), "RMSE": round(rmse, 3), "MAPE_%": round(mape, 2)}


def cross_validated_score(model, X_train, y_train, n_splits=5) -> float:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes = []
    for tr_idx, val_idx in tscv.split(X_train):
        model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        pred = model.predict(X_train.iloc[val_idx])
        maes.append(mean_absolute_error(y_train.iloc[val_idx], pred))
    return float(np.mean(maes))


def run(df: pd.DataFrame) -> dict:
    train, test = time_split(df)
    X_train, y_train = get_xy(train)
    X_test, y_test = get_xy(test)

    results = [naive_baseline(test)]
    cv_scores = {}

    lr = LinearRegression()
    cv_scores["Linear Regression"] = cross_validated_score(lr, X_train, y_train)
    lr.fit(X_train, y_train)
    results.append(score(y_test, lr.predict(X_test), "Linear Regression"))

    rf = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=3,
                                random_state=42, n_jobs=-1)
    cv_scores["Random Forest"] = cross_validated_score(rf, X_train, y_train)
    rf.fit(X_train, y_train)
    results.append(score(y_test, rf.predict(X_test), "Random Forest"))

    xgb_model = xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42)
    cv_scores["XGBoost"] = cross_validated_score(xgb_model, X_train, y_train)
    xgb_model.fit(X_train, y_train)
    results.append(score(y_test, xgb_model.predict(X_test), "XGBoost"))

    feature_importance = pd.Series(xgb_model.feature_importances_, index=X_train.columns) \
        .sort_values(ascending=False)

    return {
        "test_results": pd.DataFrame(results),
        "cv_mae_by_model": cv_scores,
        "feature_importance_xgb": feature_importance,
        "fitted_models": {"linear_regression": lr, "random_forest": rf, "xgboost": xgb_model},
        "train": train, "test": test, "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
    }


if __name__ == "__main__":
    from data_preprocessing import run as preprocess
    from feature_engineering import run as engineer
    df, _ = preprocess()
    feat = engineer(df)
    out = run(feat)
    print(out["test_results"].to_string(index=False))
    print("CV MAE:", out["cv_mae_by_model"])
    print(out["feature_importance_xgb"].head(10))
