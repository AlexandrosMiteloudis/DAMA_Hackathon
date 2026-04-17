
from __future__ import annotations

from collections import defaultdict, Counter
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold


def evaluate_nested_cv(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    estimator: BaseEstimator,
    param_grid: dict | list[dict],
    metrics: list[str] | dict[str, str | Callable],
    refit_metric: str,
    outer_splits: int = 5,
    inner_splits: int = 3,
    is_classification: bool = True,
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Perform Nested Cross-Validation with unbiased evaluation and global model ranking.
    """

    # 1. Ensure NumPy arrays for safe CV indexing & force y to 1D shape
    X_proc = X.to_numpy() if isinstance(X, (pd.DataFrame, pd.Series)) else np.asarray(X)
    y_proc = y.to_numpy() if isinstance(y, (pd.DataFrame, pd.Series)) else np.asarray(y)
    y_proc = np.squeeze(y_proc)

    # 2. Metric and Scorer setup
    if isinstance(metrics, list):
        scoring = {m: m for m in metrics}
    elif isinstance(metrics, dict):
        scoring = metrics.copy()
    else:
        raise TypeError("`metrics` must be a list of strings or a dictionary.")

    if refit_metric not in scoring:
        raise ValueError(f"`refit_metric` '{refit_metric}' must be included in `metrics`.")

    scorers = {name: get_scorer(s) for name, s in scoring.items()}
    metric_names = list(scoring.keys())

    # 3. Cross-Validation setup
    CV_Logic = StratifiedKFold if is_classification else KFold
    outer_cv = CV_Logic(n_splits=outer_splits, shuffle=True, random_state=random_state)
    inner_cv = CV_Logic(n_splits=inner_splits, shuffle=True, random_state=random_state)

    # 4. Result containers
    outer_scores = {m: [] for m in metric_names}
    best_params_per_fold = []
    fold_details = []
    
    outer_combo_scores = defaultdict(lambda: {"model": "", "params": {}, "scores": []})

    def _get_model_name(est: BaseEstimator) -> str:
        if hasattr(est, "named_steps") and "model" in est.named_steps:
            return est.named_steps["model"].__class__.__name__
        return est.__class__.__name__

    def _format_value(val: Any) -> str:
        if isinstance(val, type):
            return val.__name__
        repr_str = repr(val)
        if " at 0x" in repr_str and "<" in repr_str and ">" in repr_str:
            return val.__class__.__name__
        return repr_str

    def _make_key(model_name: str, params: dict) -> str:
        safe_params = {k: _format_value(v) for k, v in sorted(params.items())}
        return f"{model_name}__{str(safe_params)}"

    # 5. Main Nested CV Loop
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_proc, y_proc), 1):
        X_train, X_test = X_proc[train_idx], X_proc[test_idx]
        y_train, y_test = y_proc[train_idx], y_proc[test_idx]

        search = GridSearchCV(
            estimator=clone(estimator),
            param_grid=param_grid,
            scoring=scoring,
            refit=refit_metric,
            cv=inner_cv,
            n_jobs=n_jobs,
            error_score="raise",
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        best_params_per_fold.append(search.best_params_)

        fold_scores = {}
        for name, scorer in scorers.items():
            score = float(scorer(best_model, X_test, y_test))
            outer_scores[name].append(score)
            fold_scores[name] = round(score, 4)

        for params in search.cv_results_["params"]:
            temp_model = clone(estimator).set_params(**params)
            temp_model.fit(X_train, y_train)
            
            model_name = _get_model_name(temp_model)
            score = float(scorers[refit_metric](temp_model, X_test, y_test))
            key = _make_key(model_name, params)
            
            outer_combo_scores[key]["model"] = model_name
            outer_combo_scores[key]["params"] = params
            outer_combo_scores[key]["scores"].append(score)

        fold_details.append({
            "fold": fold_idx,
            "best_params": search.best_params_,
            "scores": fold_scores,
            "best_inner_score": round(float(search.best_score_), 4),
        })

        if verbose:
            print(f"[Fold {fold_idx}] Outer Generalization Scores: {fold_scores}")

    # 6. Aggregation and Ranking
    metrics_summary = {
        m: {
            "mean": round(float(np.mean(scores)), 4),
            "std": round(float(np.std(scores)), 4),
        }
        for m, scores in outer_scores.items()
    }

    model_ranking = sorted(
        [
            {
                "model": val["model"],
                "params": val["params"],
                "mean_outer_score": round(float(np.mean(val["scores"])), 4),
                "std_outer_score": round(float(np.std(val["scores"])), 4),
            }
            for key, val in outer_combo_scores.items()
        ],
        key=lambda x: x["mean_outer_score"],
        reverse=True
    )

    # 7. Prepare the Best Model (Unfitted)
    best_overall_params = model_ranking[0]["params"]
    ready_to_train_model = clone(estimator).set_params(**best_overall_params)

    return {
        "metrics": metrics_summary,
        "model_ranking": model_ranking,
        "best_params_per_fold": best_params_per_fold,
        "fold_details": fold_details,
        "best_model": ready_to_train_model, 
        "config": {
            "outer_splits": outer_splits,
            "inner_splits": inner_splits,
            "refit_metric": refit_metric
        }
    }
