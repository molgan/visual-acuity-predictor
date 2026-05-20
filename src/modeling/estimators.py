from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor


def get_inner_estimator(estimator: BaseEstimator) -> BaseEstimator:
    if isinstance(estimator, TransformedTargetRegressor):
        return getattr(estimator, "regressor_", estimator.regressor)

    return estimator