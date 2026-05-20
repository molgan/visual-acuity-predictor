from dataclasses import dataclass

import pandas as pd

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GroupShuffleSplit

from src.modeling.model_inspection import get_model_coefficients


@dataclass
class CoefficientStabilityResult:
    coefficients: pd.DataFrame
    coefficients_summary: pd.DataFrame


def _summarize_by_condition(
    coefficients: pd.DataFrame,
    condition_col: str,
    suffix: str,
) -> pd.DataFrame:
    subset = coefficients[coefficients[condition_col]].copy()

    if subset.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                f"mean_coef_when_{suffix}",
                f"median_coef_when_{suffix}",
                f"std_coef_when_{suffix}",
                f"positive_frequency_when_{suffix}",
                f"negative_frequency_when_{suffix}",
                f"sign_stability_when_{suffix}",
            ]
        )

    summary = (
        subset
        .groupby("feature")
        .agg(
            **{
                f"mean_coef_when_{suffix}": ("coef", "mean"),
                f"median_coef_when_{suffix}": ("coef", "median"),
                f"std_coef_when_{suffix}": ("coef", "std"),
                f"positive_frequency_when_{suffix}": ("positive", "mean"),
                f"negative_frequency_when_{suffix}": ("negative", "mean"),
            }
        )
        .reset_index()
    )

    summary[f"sign_stability_when_{suffix}"] = summary[
        [
            f"positive_frequency_when_{suffix}",
            f"negative_frequency_when_{suffix}",
        ]
    ].max(axis=1)

    return summary


def _summarize_coefficients(
    coefficients: pd.DataFrame,
    n_splits: int,
) -> pd.DataFrame:
    frequency_summary = (
        coefficients
        .groupby("feature")
        .agg(
            n_in_model=("in_model", "sum"),
            n_nonzero=("nonzero", "sum"),
        )
        .reset_index()
    )

    frequency_summary["n_splits"] = n_splits
    frequency_summary["frequency_in_model"] = frequency_summary["n_in_model"] / n_splits
    frequency_summary["frequency_nonzero"] = frequency_summary["n_nonzero"] / n_splits

    in_model_summary = _summarize_by_condition(
        coefficients=coefficients,
        condition_col="in_model",
        suffix="in_model",
    )

    nonzero_summary = _summarize_by_condition(
        coefficients=coefficients,
        condition_col="nonzero",
        suffix="nonzero",
    )

    summary = (
        frequency_summary
        .merge(in_model_summary, on="feature", how="left")
        .merge(nonzero_summary, on="feature", how="left")
    )

    summary = summary.sort_values(
        [
            "frequency_nonzero",
            "frequency_in_model",
        ],
        ascending=[False, False],
        na_position="last",
    )

    return summary


def evaluate_coefficient_stability(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 100,
    val_size: float = 0.2,
    random_state: int = 42,
    coef_threshold: float = 1e-8,
) -> CoefficientStabilityResult:
    splitter = GroupShuffleSplit(
        n_splits=n_splits,
        test_size=val_size,
        random_state=random_state,
    )

    rows = []
    feature_names = list(X.columns)

    for split_id, (train_idx, _) in enumerate(splitter.split(X, y, groups)):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        model = clone(estimator)
        model.fit(X_train, y_train)

        model_coefficients = get_model_coefficients(
            estimator=model,
            feature_names=feature_names,
        )

        if model_coefficients is None:
            raise ValueError(
                "No coefficients were collected. "
                "Make sure the estimator has a model with 1D coef_."
            )

        coef_by_feature = dict(
            zip(
                model_coefficients["feature"],
                model_coefficients["coef"],
            )
        )

        for feature in feature_names:
            coef = float(coef_by_feature.get(feature, 0.0))

            in_model = feature in coef_by_feature
            nonzero = abs(coef) > coef_threshold

            rows.append(
                {
                    "split_id": split_id,
                    "feature": feature,
                    "coef": coef,
                    "in_model": in_model,
                    "nonzero": nonzero,
                    "positive": coef > coef_threshold,
                    "negative": coef < -coef_threshold,
                }
            )

    coefficients = pd.DataFrame(rows)

    coefficients_summary = _summarize_coefficients(
        coefficients=coefficients,
        n_splits=n_splits,
    )

    return CoefficientStabilityResult(
        coefficients=coefficients,
        coefficients_summary=coefficients_summary,
    )


def get_coefficient_stability_view(
    coef_stability,
    mode: str = "nonzero",
    min_frequency: float = 0.0,
):
    if mode not in {"nonzero", "in_model"}:
        raise ValueError("mode must be 'nonzero' or 'in_model'")

    cols = [
        "feature",
        f"frequency_{mode}",
        f"mean_coef_when_{mode}",
        f"median_coef_when_{mode}",
        f"std_coef_when_{mode}",
        f"sign_stability_when_{mode}",
    ]

    return (
        coef_stability.coefficients_summary
        .filter(items=cols)
        .query(f"frequency_{mode} > @min_frequency")
        .sort_values(
            [f"frequency_{mode}", f"sign_stability_when_{mode}"],
            ascending=[False, False],
        )
    )