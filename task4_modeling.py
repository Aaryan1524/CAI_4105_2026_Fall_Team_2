"""Training-only, reproducible model-improvement experiments for Task 4."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class ApartmentFeatures(TransformerMixin, BaseEstimator):
    """Optional row-wise features: no targets or fitted population statistics."""

    def __init__(self, enabled=False):
        self.enabled = enabled

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X.copy()
        if self.enabled:
            # +1 includes studios and prevents division by zero.
            result["area_per_room"] = result["square_feet"] / (result["bedrooms"] + 1)
            result["area_per_bathroom"] = result["square_feet"] / result["bathrooms"].clip(lower=0.5)
            result["bathrooms_per_room"] = result["bathrooms"] / (result["bedrooms"] + 1)
            result["log_square_feet"] = np.log1p(result["square_feet"].clip(lower=0))
            # Rotated coordinate axes let trees represent diagonal geographic splits.
            # Fixed formulas, not locations learned from the test data.
            result["location_sum"] = result["latitude"] + result["longitude"]
            result["location_difference"] = result["latitude"] - result["longitude"]
        return result


def numeric_columns(frame):
    return frame.select_dtypes(include="number").columns.tolist()


def categorical_columns(frame):
    # Handles both pandas object columns and newer string dtypes.
    return frame.select_dtypes(exclude="number").columns.tolist()


def make_pipeline(estimator):
    preprocessing = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric_columns),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical_columns),
        ],
        sparse_threshold=0,
    )
    return Pipeline([
        ("features", ApartmentFeatures()),
        ("preprocessor", preprocessing),
        ("model", estimator),
    ])


def search_spaces():
    """Explicit Cartesian grids, each including feature engineering on/off."""
    return {
        "Random Forest": (
            make_pipeline(RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=1)),
            {
                "features__enabled": [False, True],
                "model__max_depth": [None, 16, 28],
                "model__min_samples_leaf": [1, 3],
                "model__max_features": [0.7, 1.0],
            },
        ),
        "Gradient Boosting": (
            make_pipeline(GradientBoostingRegressor(random_state=42, min_samples_leaf=5)),
            {
                "features__enabled": [False, True],
                "model__n_estimators": [400, 800],
                "model__learning_rate": [0.03, 0.07, 0.1],
                "model__max_depth": [2, 4],
            },
        ),
        "Histogram Gradient Boosting": (
            make_pipeline(HistGradientBoostingRegressor(
                max_iter=500, early_stopping=False, random_state=42,
            )),
            {
                "features__enabled": [False, True],
                "model__learning_rate": [0.05, 0.1],
                "model__max_leaf_nodes": [15, 31],
                "model__l2_regularization": [0.0, 10.0, 100.0],
            },
        ),
    }


def search_table(search):
    result = pd.DataFrame(search.cv_results_)
    table = pd.DataFrame({
        "Rank": result["rank_test_score"],
        "Train RMSE": -result["mean_train_score"],
        "CV RMSE": -result["mean_test_score"],
        "CV Std": result["std_test_score"],
        "Fit time (s)": result["mean_fit_time"],
        "Parameters": result["params"],
    })
    table["Gap"] = table["CV RMSE"] - table["Train RMSE"]
    return table.sort_values(["Rank", "CV RMSE"]).reset_index(drop=True)


def refine_boosting(X, y, cv, results_dir, n_jobs=4):
    """Expand the initial boosting boundary, then nest this refinement grid."""
    import json
    from time import perf_counter
    from joblib import parallel_backend
    from threadpoolctl import threadpool_limits
    from sklearn.base import clone
    from sklearn.metrics import root_mean_squared_error
    from sklearn.model_selection import GridSearchCV, KFold

    pipeline = make_pipeline(GradientBoostingRegressor(
        random_state=42, min_samples_leaf=5,
    ))
    grid = {
        "features__enabled": [False, True],
        "model__n_estimators": [800, 1200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [4, 6],
    }

    def make_search():
        return GridSearchCV(
            clone(pipeline), grid, cv=cv, scoring="neg_root_mean_squared_error",
            n_jobs=n_jobs, pre_dispatch=n_jobs, refit=True,
            return_train_score=True, error_score="raise",
        )

    start = perf_counter()
    refined_search = make_search()
    with parallel_backend("threading"), threadpool_limits(limits=1):
        refined_search.fit(X, y)
    table = search_table(refined_search)
    table.to_csv(results_dir / "gradient_boosting_refinement_grid.csv", index=False)
    print(f"Refined boosting tuning RMSE: {-refined_search.best_score_:.2f}", flush=True)
    print(f"Refined parameters: {refined_search.best_params_}", flush=True)

    rows = []
    selected_parameters = []
    oof = np.full(len(y), np.nan)
    for fold, (train_index, validation_index) in enumerate(cv.split(X, y), start=1):
        search = make_search().set_params(cv=KFold(n_splits=3, shuffle=True, random_state=123))
        with parallel_backend("threading"), threadpool_limits(limits=1):
            search.fit(X.iloc[train_index], y.iloc[train_index])
            predictions = search.predict(X.iloc[validation_index])
            fitted = search.predict(X.iloc[train_index])
        oof[validation_index] = predictions
        rows.append({
            "Fold": fold,
            "Train RMSE": root_mean_squared_error(y.iloc[train_index], fitted),
            "Outer RMSE": root_mean_squared_error(y.iloc[validation_index], predictions),
            "Inner best RMSE": -search.best_score_,
        })
        selected_parameters.append(search.best_params_)
        print(f"Refined boosting outer fold {fold}: RMSE {rows[-1]['Outer RMSE']:.2f}", flush=True)

    assert np.isfinite(oof).all()
    nested = pd.DataFrame(rows)
    nested.to_csv(results_dir / "refinement_nested_cv.csv", index=False)
    summary = {
        "family": "Gradient Boosting",
        "tuning_cv_rmse": float(-refined_search.best_score_),
        "best_parameters": refined_search.best_params_,
        "model_parameters": refined_search.best_estimator_.named_steps["model"].get_params(),
        "mean_outer_rmse": float(nested["Outer RMSE"].mean()),
        "std_outer_rmse": float(nested["Outer RMSE"].std(ddof=0)),
        "pooled_oof_rmse": float(root_mean_squared_error(y, oof)),
        "mean_train_rmse": float(nested["Train RMSE"].mean()),
        "outer_fold_best_parameters": selected_parameters,
        "total_seconds": perf_counter() - start,
        "note": "Conditional refinement-grid check. Family and refinement ranges were chosen from previous training CV results; selection bias remains.",
    }
    (results_dir / "refinement_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Refined nested mean outer RMSE: {summary['mean_outer_rmse']:.2f} +/- {summary['std_outer_rmse']:.2f}", flush=True)
    return refined_search, table, nested, summary
