"""Module for evaluating machine learning models using Nested Cross-Validation."""

import inspect
import logging
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = True
except ImportError:
    HAS_STRATIFIED_GROUP_KFOLD = False

logger = logging.getLogger(__name__)


# =============================================================================
# Private Helpers
# =============================================================================

def _build_stratified_group_kfold(n_splits: int, random_state: int):
    """Safely instantiate StratifiedGroupKFold."""
    sig = inspect.signature(StratifiedGroupKFold.__init__)
    if "shuffle" in sig.parameters:
        return StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
    return StratifiedGroupKFold(n_splits=n_splits)


def _params_to_stable_key(params: dict) -> str:
    """Convert a hyperparameter dictionary to a stable, canonical string key."""
    return str(dict(sorted(params.items())))


def _validate_inputs(
    outer_splits: int,
    inner_splits: int,
    repeats: int,
    refit_metric: str,
    metric_names: list,
    X: np.ndarray,
    y: np.ndarray,
    groups,
) -> None:
    """Validate all critical inputs before any CV logic runs."""
    if not isinstance(outer_splits, int) or outer_splits < 2:
        raise ValueError(f"outer_splits must be int >= 2, got {outer_splits}.")
    if not isinstance(inner_splits, int) or inner_splits < 2:
        raise ValueError(f"inner_splits must be int >= 2, got {inner_splits}.")
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError(f"repeats must be an int >= 1, got {repeats}.")
    if refit_metric not in metric_names:
        raise ValueError(
            f"refit_metric '{refit_metric}' must be one of: {metric_names}."
        )
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same number of samples. "
            f"Got X: {len(X)}, y: {len(y)}."
        )
    if groups is not None and len(groups) != len(X):
        raise ValueError(
            f"groups must have the same number of samples as X. "
            f"Got groups: {len(groups)}, X: {len(X)}."
        )


def _coerce_to_numpy(*arrays):
    """Convert pandas DataFrames/Series to NumPy arrays in-place."""
    result =[]
    for arr in arrays:
        if isinstance(arr, (pd.DataFrame, pd.Series)):
            result.append(arr.to_numpy())
        else:
            result.append(arr)
    return tuple(result)


def _build_scorers(scoring: dict) -> dict:
    """Pre-build callable scorer objects from string names or callables."""
    scorers = {}
    for name, scorer in scoring.items():
        if isinstance(scorer, str):
            try:
                scorers[name] = get_scorer(scorer)
            except ValueError:
                raise ValueError(
                    f"Metric '{name}' with value '{scorer}' is not a valid "
                    f"sklearn scorer string."
                )
        else:
            scorers[name] = scorer
    return scorers


def _build_outer_cv(
    is_classification: bool,
    outer_splits: int,
    repeats: int,
    random_state: int,
    groups,
):
    """Instantiate the correct outer CV splitter for the given configuration."""
    if groups is not None:
        if is_classification and HAS_STRATIFIED_GROUP_KFOLD:
            return _build_stratified_group_kfold(outer_splits, random_state)
        return GroupKFold(n_splits=outer_splits)

    if repeats > 1:
        if is_classification:
            return RepeatedStratifiedKFold(
                n_splits=outer_splits,
                n_repeats=repeats,
                random_state=random_state,
            )
        return RepeatedKFold(
            n_splits=outer_splits,
            n_repeats=repeats,
            random_state=random_state,
        )

    if is_classification:
        return StratifiedKFold(
            n_splits=outer_splits, shuffle=True, random_state=random_state
        )

    return KFold(
        n_splits=outer_splits, shuffle=True, random_state=random_state
    )


def _build_inner_cv(
    is_classification: bool,
    inner_splits: int,
    random_state: int,
    groups,
):
    """Instantiate the correct inner CV splitter for the given configuration."""
    if groups is not None:
        if is_classification and HAS_STRATIFIED_GROUP_KFOLD:
            return _build_stratified_group_kfold(inner_splits, random_state)
        return GroupKFold(n_splits=inner_splits)

    if is_classification:
        return StratifiedKFold(
            n_splits=inner_splits, shuffle=True, random_state=random_state
        )

    return KFold(
        n_splits=inner_splits, shuffle=True, random_state=random_state
    )


def _warn_if_low_folds(outer_splits: int, repeats: int) -> None:
    """Emit a UserWarning when the outer fold count is too low."""
    total_outer = outer_splits * repeats
    if total_outer < 5:
        warnings.warn(
            f"Total outer evaluations = {total_outer}. "
            f"This is below the recommended minimum of 5. "
            f"Generalisation estimates may be unstable.",
            UserWarning,
            stacklevel=3,
        )


def _aggregate_metrics(outer_scores: dict) -> dict:
    aggregated = {}
    for metric_name, scores in outer_scores.items():
        arr = np.array(scores, dtype=float)
        aggregated[metric_name] = {
            "fold_scores": [round(float(s), 4) for s in arr],
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr, ddof=1)), 4) if len(arr) > 1 else 0.0,
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
        }
    return aggregated


def _aggregate_combo_comparison(all_combos_tracker: dict, refit_metric: str) -> list:
    combo_list = []
    for params_str, scores in all_combos_tracker.items():
        arr = np.array(scores, dtype=float)
        combo_list.append({
            "params": params_str,
            f"mean_inner_{refit_metric}": round(float(np.mean(arr)), 4),
            f"std_inner_{refit_metric}": round(float(np.std(arr, ddof=1)), 4) if len(arr) > 1 else 0.0,
            "n_outer_folds_seen": int(len(arr)),
        })
    return sorted(combo_list, key=lambda x: x[f"mean_inner_{refit_metric}"], reverse=True)


def _aggregate_hyperparameter_stability(best_params_per_fold: list) -> dict:
    """Summarise how consistently hyperparameters were selected across folds."""
    param_counts = Counter(
        _params_to_stable_key(p) for p in best_params_per_fold
    )
    total_folds = len(best_params_per_fold)

    return {
        "selection_frequency": dict(param_counts),
        "selection_frequency_pct": {
            k: round(v / total_folds, 4) for k, v in param_counts.items()
        },
        "most_frequent": (
            param_counts.most_common(1)[0][0] if param_counts else None
        ),
        "most_frequent_count": (
            param_counts.most_common(1)[0][1] if param_counts else 0
        ),
    }


def _print_fold_summary(
    fold_idx: int,
    fold_scores: dict,
    best_params: dict,
    best_inner_score: float,
    refit_metric: str,
    metric_names: list,
) -> None:
    """Print a single outer fold result to stdout."""
    score_str = " | ".join([f"{m}: {fold_scores[m]:.4f}" for m in metric_names])
    print(f"Fold {fold_idx:02d} | {score_str}")
    print(f"         Best params      : {best_params}")
    print(f"         Best inner {refit_metric}: {best_inner_score:.4f}")
    print("-" * 80)


def _print_final_summary(
    results: dict,
    metric_names: list,
    best_params_per_fold: list,
) -> None:
    """Print an end-of-run summary table to stdout."""
    print("=" * 80)
    print("NESTED CV COMPLETE — OUTER GENERALISATION SCORES")
    print("=" * 80)
    for m in metric_names:
        r = results["metrics"][m]
        print(
            f"  {m:25s} | "
            f"mean: {r['mean']:.4f} ± {r['std']:.4f} "
            f"[min: {r['min']:.4f}, max: {r['max']:.4f}]"
        )
    print("=" * 80)
    stability = results["hyperparameter_stability"]
    most_freq = stability["most_frequent"]
    freq_count = stability["most_frequent_count"]
    total = len(best_params_per_fold)
    print(f"  Most selected params ({freq_count}/{total} folds): {most_freq}")
    print("=" * 80)


# =============================================================================
# Public API
# =============================================================================

def evaluate_nested_cv(
    X,
    y,
    estimator,
    param_grid,
    metrics,
    refit_metric: str,
    outer_splits: int = 5,
    inner_splits: int = 3,
    is_classification: bool = True,
    random_state: int = 42,
    groups=None,
    repeats: int = 1,
    n_jobs: int = -1,
    verbose: bool = True,
    return_fold_estimators: bool = False,
) -> dict:
    """Evaluate a model using Nested Cross-Validation."""
    if isinstance(metrics, list):
        scoring = {m: m for m in metrics}
    elif isinstance(metrics, dict):
        scoring = metrics.copy()
    else:
        raise TypeError(
            "metrics must be a list of scorer strings or a dict mapping "
            f"names to scorer strings/callables. Got: {type(metrics)}."
        )

    metric_names = list(scoring.keys())
    X, y, groups = _coerce_to_numpy(X, y, groups)

    _validate_inputs(
        outer_splits=outer_splits,
        inner_splits=inner_splits,
        repeats=repeats,
        refit_metric=refit_metric,
        metric_names=metric_names,
        X=X,
        y=y,
        groups=groups,
    )

    _warn_if_low_folds(outer_splits, repeats)
    scorers = _build_scorers(scoring)

    outer_cv = _build_outer_cv(
        is_classification=is_classification,
        outer_splits=outer_splits,
        repeats=repeats,
        random_state=random_state,
        groups=groups,
    )
    inner_cv = _build_inner_cv(
        is_classification=is_classification,
        inner_splits=inner_splits,
        random_state=random_state,
        groups=groups,
    )

    logger.debug(
        "Nested CV config | outer=%s inner=%s refit=%s metrics=%s",
        outer_cv, inner_cv, refit_metric, metric_names
    )

    outer_scores = {m:[] for m in metric_names}
    best_params_per_fold =[]
    all_combos_tracker = defaultdict(list)
    fold_details = []
    fold_estimators =[]

    outer_split_iter = (
        outer_cv.split(X, y, groups)
        if groups is not None
        else outer_cv.split(X, y)
    )

    if verbose:
        print(f"\nStarting Nested CV | Optimising for: {refit_metric}")
        print("=" * 80)

    for fold_idx, (train_idx, test_idx) in enumerate(outer_split_iter, 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_groups = groups[train_idx] if groups is not None else None

        search = GridSearchCV(
            estimator=clone(estimator),
            param_grid=param_grid,
            scoring=scoring,
            refit=refit_metric,
            cv=inner_cv,
            n_jobs=n_jobs,
            return_train_score=False,
            error_score="raise",
        )

        try:
            if train_groups is not None:
                search.fit(X_train, y_train, groups=train_groups)
            else:
                search.fit(X_train, y_train)
        except Exception as exc:
            raise RuntimeError(
                f"GridSearchCV failed on outer fold {fold_idx}. "
                f"Original error: {exc}"
            ) from exc

        best_model = search.best_estimator_
        best_params = search.best_params_
        best_params_per_fold.append(best_params)

        if return_fold_estimators:
            fold_estimators.append(best_model)

        score_key = f"mean_test_{refit_metric}"
        for params, score in zip(
            search.cv_results_["params"], search.cv_results_[score_key]
        ):
            all_combos_tracker[_params_to_stable_key(params)].append(score)

        fold_scores = {}
        for metric_name, scorer in scorers.items():
            try:
                score = scorer(best_model, X_test, y_test)
            except Exception as exc:
                raise RuntimeError(
                    f"Scorer '{metric_name}' failed on outer fold {fold_idx}. "
                    f"Original error: {exc}"
                ) from exc

            outer_scores[metric_name].append(round(float(score), 4))
            fold_scores[metric_name] = round(float(score), 4)

        fold_details.append({
            "fold": fold_idx,
            "best_params": best_params,
            "outer_test_scores": fold_scores,
            "best_inner_score": round(float(search.best_score_), 4),
        })

        if verbose:
            _print_fold_summary(
                fold_idx=fold_idx,
                fold_scores=fold_scores,
                best_params=best_params,
                best_inner_score=float(search.best_score_),
                refit_metric=refit_metric,
                metric_names=metric_names,
            )

    results = {
        "metrics": _aggregate_metrics(outer_scores),
        "combo_comparison": _aggregate_combo_comparison(
            all_combos_tracker, refit_metric
        ),
        "best_params_per_fold": best_params_per_fold,
        "hyperparameter_stability": _aggregate_hyperparameter_stability(
            best_params_per_fold
        ),
        "fold_details": fold_details,
        "config": {
            "outer_splits": outer_splits,
            "inner_splits": inner_splits,
            "repeats": repeats,
            "is_classification": is_classification,
            "refit_metric": refit_metric,
            "metrics": metric_names,
            "used_groups": groups is not None,
            "random_state": random_state,
            "n_jobs": n_jobs,
        },
    }

    if return_fold_estimators:
        results["fold_estimators"] = fold_estimators

    if verbose:
        _print_final_summary(results, metric_names, best_params_per_fold)

    return results