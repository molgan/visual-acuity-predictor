from dataclasses import dataclass
from typing import Literal

import pandas as pd

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    RandomizedSearchCV,
)

from src.modeling.model_inspection import get_selected_features
from src.modeling.estimators import get_inner_estimator


SearchType = Literal["random", "grid"]


@dataclass
class TuningResult:
    search: RandomizedSearchCV | GridSearchCV
    results: pd.DataFrame
    candidates: pd.DataFrame


def _add_n_selected_features(
    search: GridSearchCV | RandomizedSearchCV,
    results: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    coef_threshold: float = 1e-8,
    treat_sparse_model_as_selector: bool = False,
) -> pd.DataFrame:
    results = results.copy()
    feature_names = list(X.columns)

    n_selected_features = []

    for candidate_index in results.index:
        params = search.cv_results_["params"][candidate_index]

        model = clone(search.estimator)
        model.set_params(**params)
        model.fit(X, y)
        
        inner_estimator = get_inner_estimator(model)

        selected_features = get_selected_features(
            estimator=inner_estimator,
            feature_names=feature_names,
            treat_sparse_model_as_selector=treat_sparse_model_as_selector,
            coef_threshold=coef_threshold,
        )

        n_selected_features.append(
            len(selected_features)
            if selected_features is not None
            else None
        )

    results["n_selected_features"] = n_selected_features

    return results


def _summarize_search_results(
    search: GridSearchCV | RandomizedSearchCV,
    score_tolerance: float = 0.02,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = pd.DataFrame(search.cv_results_).copy()

    results["generalization_gap"] = results["mean_train_score"] - results["mean_test_score"]
    results["abs_generalization_gap"] = results["generalization_gap"].abs()

    param_cols = [
        col for col in results.columns
        if col.startswith("param_")
    ]

    display_cols = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "generalization_gap",
        *param_cols,
    ]

    results = results.sort_values(
        ["mean_test_score", "abs_generalization_gap"],
        ascending=[False, True]
    )

    best_score = results["mean_test_score"].max()
    candidates = results[
        results["mean_test_score"] >= best_score - score_tolerance
    ].sort_values(
        ["mean_test_score", "abs_generalization_gap"],
        ascending=[False, True]
    )

    return results[display_cols], candidates[display_cols]


def tune_model(
    estimator: BaseEstimator,
    param_space: dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    scoring: str = "r2",
    search_type: SearchType = "random",
    n_splits: int = 5,
    n_iter: int = 50,
    random_state: int = 42,
    score_tolerance: float = 0.02,
    add_n_selected_features: bool = False,
    treat_sparse_model_as_selector: bool = False,
    coef_threshold: float = 1e-8,
    n_jobs: int = -1,
) -> TuningResult:
    cv = GroupKFold(n_splits=n_splits)

    common_kwargs = dict(
        estimator=estimator,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=True,
        refit=True,
    )

    if search_type == "random":
        search = RandomizedSearchCV(
            param_distributions=param_space,
            n_iter=n_iter,
            random_state=random_state,
            **common_kwargs,
        )
    elif search_type == "grid":
        search = GridSearchCV(
            param_grid=param_space,
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unknown search_type: {search_type}")

    search.fit(X, y, groups=groups)

    results, candidates = _summarize_search_results(
        search=search,
        score_tolerance=score_tolerance,
    )

    if add_n_selected_features:
        candidate_indices = candidates.index

        results = _add_n_selected_features(
            search=search,
            results=results,
            X=X,
            y=y,
            coef_threshold=coef_threshold,
            treat_sparse_model_as_selector=treat_sparse_model_as_selector,
        )

        candidates = results.loc[candidate_indices]

    return TuningResult(
        search=search,
        results=results,
        candidates=candidates,
    )