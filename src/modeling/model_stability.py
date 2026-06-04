from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn import config_context

from src.modeling.model_inspection import get_selected_features


@dataclass
class ModelStabilityResult:
    scores: pd.DataFrame
    scores_summary: pd.DataFrame
    feature_selection: pd.DataFrame | None
    feature_selection_summary: pd.DataFrame | None


def _summarize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        col
        for col in ["split_id", "groups_val", "selected_features"]
        if col in scores.columns
    ]

    numeric_scores = (
        scores
        .drop(columns=columns_to_drop)
        .dropna(axis=1, how="all")
    )

    summary = numeric_scores.agg([
        "mean",
        "std",
        "min",
        lambda x: x.quantile(0.25),
        "median",
        lambda x: x.quantile(0.75),
        "max",
    ]).T

    summary.columns = [
        "mean",
        "std",
        "min",
        "q25",
        "median",
        "q75",
        "max",
    ]

    return summary


def _summarize_feature_selection(feature_selection: pd.DataFrame) -> pd.DataFrame:
    return (
        feature_selection
        .groupby("feature")
        .agg(selection_frequency=("selected", "mean"))
        .reset_index()
    )


def _get_pipeline_model(estimator: BaseEstimator) -> BaseEstimator:
    if isinstance(estimator, Pipeline):
        return estimator.named_steps.get("model")

    return estimator


def _get_fit_params(
    estimator: BaseEstimator,
    groups_train: pd.Series,
) -> dict:
    model = _get_pipeline_model(estimator)

    if isinstance(model, ElasticNetCV):
        return {"groups": groups_train}

    return {}


def _get_selected_hyperparams(estimator: BaseEstimator) -> dict:
    model = _get_pipeline_model(estimator)

    if isinstance(model, ElasticNetCV):
        return {
            "selected_alpha": getattr(model, "alpha_", None),
            "selected_l1_ratio": getattr(model, "l1_ratio_", None),
        }
    
    return {}


def evaluate_model_stability(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 100,
    val_size: float = 0.2,
    random_state: int = 42,
    collect_feature_selection: bool = True,
    treat_sparse_model_as_selector: bool = False,
    coef_threshold: float = 1e-8,
) -> ModelStabilityResult:
    splitter = GroupShuffleSplit(
        n_splits=n_splits,
        test_size=val_size,
        random_state=random_state,
    )

    score_rows = []
    feature_selection_rows = []

    feature_names = list(X.columns)

    for split_id, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups)):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        
        groups_train = groups.iloc[train_idx]

        model = clone(estimator)
        fit_params = _get_fit_params(estimator=model, groups_train=groups_train)
        if fit_params:
            with config_context(enable_metadata_routing=True):
                model.fit(X_train, y_train, **fit_params)
        else:
            model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        r2_train = r2_score(y_train, y_train_pred)
        r2_val = r2_score(y_val, y_val_pred)

        mae_train = mean_absolute_error(y_train, y_train_pred)
        mae_val = mean_absolute_error(y_val, y_val_pred)

        rmse_train = root_mean_squared_error(y_train, y_train_pred)
        rmse_val = root_mean_squared_error(y_val, y_val_pred)

        selected_features = None

        if collect_feature_selection:
            selected_features = get_selected_features(
                estimator=model,
                feature_names=feature_names,
                treat_sparse_model_as_selector=treat_sparse_model_as_selector,
                coef_threshold=coef_threshold,
            )

            if selected_features is not None:
                selected_set = set(selected_features)

                for feature in feature_names:
                    feature_selection_rows.append({
                        "split_id": split_id,
                        "feature": feature,
                        "selected": feature in selected_set,
                    })

        score_rows.append({
            "split_id": split_id,

            **_get_selected_hyperparams(model),
            
            "r2_train": r2_train,
            "r2_val": r2_val,
            "r2_gap": r2_train - r2_val,

            "mae_train": mae_train,
            "mae_val": mae_val,
            "mae_gap": mae_val - mae_train,

            "rmse_train": rmse_train,
            "rmse_val": rmse_val,
            "rmse_gap": rmse_val - rmse_train,

            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_groups_train": groups.iloc[train_idx].nunique(),
            "n_groups_val": groups.iloc[val_idx].nunique(),

            "groups_val": groups.iloc[val_idx].unique().tolist(),

            "target_std_train": y_train.std(),
            "target_range_train": y_train.max() - y_train.min(),

            "target_std_val": y_val.std(),
            "target_range_val": y_val.max() - y_val.min(),
            "target_mean_val": y_val.mean(),
            "target_median_val": y_val.median(),

            "n_selected_features": (
                len(selected_features)
                if selected_features is not None
                else None
            ),
            "selected_features": selected_features,
        })

    scores = pd.DataFrame(score_rows)
    scores_summary = _summarize_scores(scores)

    feature_selection = None
    feature_selection_summary = None

    if feature_selection_rows:
        feature_selection = pd.DataFrame(feature_selection_rows)
        feature_selection_summary = _summarize_feature_selection(feature_selection)

    return ModelStabilityResult(
        scores=scores,
        scores_summary=scores_summary,
        feature_selection=feature_selection,
        feature_selection_summary=feature_selection_summary,
    )


def compare_candidates_by_metric(
    stability_results: dict,
    metric: str,
    lower_is_better: bool,
) -> pd.DataFrame:
    rows = []

    for candidate_name, stability in stability_results.items():
        scores_summary = stability.scores_summary

        rows.append({
            "candidate": candidate_name,
            "median_val": scores_summary.loc[f"{metric}_val", "median"],
            "q25_val": scores_summary.loc[f"{metric}_val", "q25"],
            "q75_val": scores_summary.loc[f"{metric}_val", "q75"],
            "mean_val": scores_summary.loc[f"{metric}_val", "mean"],
            "std_val": scores_summary.loc[f"{metric}_val", "std"],
            "median_train": scores_summary.loc[f"{metric}_train", "median"],
            "mean_train": scores_summary.loc[f"{metric}_train", "mean"],
            "median_gap": scores_summary.loc[f"{metric}_gap", "median"],
            "q75_gap": scores_summary.loc[f"{metric}_gap", "q75"],
            "n_selected_features_median":(
                np.nan 
                if "n_selected_features" not in scores_summary.index 
                else scores_summary.loc["n_selected_features", "median"]
            ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["median_val", "median_gap"],
            ascending=[lower_is_better, True],
        )
        .reset_index(drop=True)
        .assign(rank=lambda df: df.index + 1)
    )


def compare_candidates_by_metrics(
    stability_results: dict,
) -> pd.DataFrame:
    rows = []

    for candidate_name, stability in stability_results.items():
        scores_summary = stability.scores_summary

        rows.append({
            "candidate": candidate_name,

            "mae_median_val": scores_summary.loc["mae_val", "median"],
            "mae_median_gap": scores_summary.loc["mae_gap", "median"],

            "rmse_median_val": scores_summary.loc["rmse_val", "median"],
            "rmse_median_gap": scores_summary.loc["rmse_gap", "median"],

            "r2_median_val": scores_summary.loc["r2_val", "median"],
            "r2_median_gap": scores_summary.loc["r2_gap", "median"],

            "n_selected_features_median": (
                np.nan
                if "n_selected_features" not in scores_summary.index
                else scores_summary.loc["n_selected_features", "median"]
            ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "mae_median_val",
                "mae_median_gap",
                "n_selected_features_median",
            ],
            ascending=[True, True, True],
        )
        .reset_index(drop=True)
    )


def compare_candidates_by_feature_selection(
    stability_results: dict,
    min_frequency: float = 0.0,
) -> pd.DataFrame:

    candidate_tables = []

    for candidate_name, stability in stability_results.items():
        fs = stability.feature_selection_summary

        if fs is None:
            continue

        candidate_fs = (
            fs
            .query("selection_frequency > @min_frequency")
            .loc[:, ["feature", "selection_frequency"]]
            .set_index("feature")
            .rename(
                columns={
                    "selection_frequency": candidate_name
                }
            )
        )

        candidate_tables.append(candidate_fs)

    if not candidate_tables:
        return pd.DataFrame()

    result = (
        pd.concat(candidate_tables, axis=1)
        .fillna(0)
    )

    candidate_cols = result.columns

    result["max_selection_frequency"] = result[candidate_cols].max(axis=1)

    return (
        result
        .sort_values(
            "max_selection_frequency",
            ascending=False,
        )
        .reset_index()
    )