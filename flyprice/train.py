"""Train and compare three regressors on the flights dataset, tracked with MLflow.

Models: Linear Regression (baseline), Random Forest, XGBoost.
The target is log1p(price) during training; metrics are computed back in USD.
The best model (lowest test MAE) is saved to models/best_model.joblib with its metadata.

Usage:
    python -m flyprice.train                       # full dataset
    python -m flyprice.train --sample 20000        # faster run
    MLFLOW_TRACKING_URI=http://localhost:5000 python -m flyprice.train
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor

from flyprice.features import CATEGORICAL, FEATURES, NUMERIC, TARGET, load_dataset

ROOT = Path(__file__).resolve().parent.parent


def linear_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("model", LinearRegression())])


def tree_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL),
    ])


def build_models(seed: int) -> dict[str, tuple[Pipeline, dict]]:
    rf_params = {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 2, "n_jobs": -1, "random_state": seed}
    xgb_params = {"n_estimators": 600, "max_depth": 8, "learning_rate": 0.05, "subsample": 0.9,
                  "colsample_bytree": 0.9, "n_jobs": -1, "random_state": seed}
    return {
        "linear_regression": (linear_pipeline(), {}),
        "random_forest": (Pipeline([("pre", tree_preprocessor()),
                                    ("model", RandomForestRegressor(**rf_params))]), rf_params),
        "xgboost": (Pipeline([("pre", tree_preprocessor()), ("model", XGBRegressor(**xgb_params))]), xgb_params),
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "flights_dataset_100000.csv"))
    parser.add_argument("--sample", type=int, default=0, help="use only N rows (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(ROOT / "models"))
    args = parser.parse_args()

    df = load_dataset(args.data)
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=args.seed)
    X, y = df[FEATURES], df[TARGET].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed)

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("flight-price-prediction")
    results = {}
    best_name, best_model, best_mae = None, None, float("inf")

    for name, (pipe, params) in build_models(args.seed).items():
        model = TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": name, "n_train": len(X_train), "n_test": len(X_test), **params})
            start = time.perf_counter()
            model.fit(X_train, y_train)
            train_s = time.perf_counter() - start
            metrics = evaluate(y_test, model.predict(X_test))
            metrics["train_seconds"] = round(train_s, 2)
            mlflow.log_metrics(metrics)
        results[name] = metrics
        print(f"{name:18s} MAE={metrics['mae']:8.2f}  RMSE={metrics['rmse']:8.2f}  "
              f"MAPE={metrics['mape']:.3f}  R2={metrics['r2']:.4f}  ({train_s:.1f}s)")
        if metrics["mae"] < best_mae:
            best_name, best_model, best_mae = name, model, metrics["mae"]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, out / "best_model.joblib")
    meta = {"best_model": best_name, "features": FEATURES, "metrics": results,
            "n_rows": len(df), "test_size": 0.2, "seed": args.seed}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    if out.resolve() == (ROOT / "models").resolve():  # only the main run updates the published results
        (ROOT / "results").mkdir(exist_ok=True)
        (ROOT / "results" / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(f"\nBest model: {best_name} -> {out / 'best_model.joblib'}")


if __name__ == "__main__":
    main()
