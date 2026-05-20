import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from src.modeling.estimators import get_inner_estimator


def get_selected_features(
    estimator: BaseEstimator,
    feature_names: list[str],
    treat_sparse_model_as_selector: bool = False,
    coef_threshold: float = 1e-8,
) -> list[str] | None:
    estimator = get_inner_estimator(estimator)

    if isinstance(estimator, Pipeline):
        if "feature_selector" in estimator.named_steps:
            selector = estimator.named_steps["feature_selector"]

            if hasattr(selector, "selected_features_"):
                return list(selector.selected_features_)

            if hasattr(selector, "support_"):
                return list(np.asarray(feature_names)[selector.support_])

        model = estimator.named_steps.get("model")

    else:
        model = estimator

    if not treat_sparse_model_as_selector:
        return None

    if model is None:
        return None

    if not hasattr(model, "coef_"):
        return None

    coefs = np.asarray(model.coef_)

    if coefs.ndim != 1:
        raise ValueError(f"Only single-target models are supported. Got coef shape={coefs.shape}.")

    if len(coefs) != len(feature_names):
        raise ValueError(f"Coefficient count does not match feature_names. Got {len(coefs)} coefficients for {len(feature_names)} features.")

    return [
        feature
        for feature, value in zip(feature_names, coefs)
        if abs(value) > coef_threshold
    ]


def get_model_coefficients(
    estimator: BaseEstimator,
    feature_names: list[str],
) -> pd.DataFrame | None:
    estimator = get_inner_estimator(estimator)

    if isinstance(estimator, Pipeline):
        model = estimator.named_steps.get("model")

        if "feature_selector" in estimator.named_steps:
            selected_features = get_selected_features(
                estimator=estimator,
                feature_names=feature_names,
                treat_sparse_model_as_selector=False,
            )

            if selected_features is None:
                return None

            model_feature_names = selected_features
        else:
            model_feature_names = feature_names

    else:
        model = estimator
        model_feature_names = feature_names

    if model is None:
        return None

    if not hasattr(model, "coef_"):
        return None

    coefs = np.asarray(model.coef_)

    if coefs.ndim != 1:
        raise ValueError(f"Only single-target models are supported. Got coef shape={coefs.shape}.")

    if len(coefs) != len(model_feature_names):
        raise ValueError(f"Coefficient count does not match model feature names. Got {len(coefs)} coefficients for {len(model_feature_names)} features.")

    return pd.DataFrame({
        "feature": model_feature_names,
        "coef": coefs,
    })