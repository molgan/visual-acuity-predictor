import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import ElasticNet, ElasticNetCV


class ElasticNetFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        alpha: float = 0.1,
        l1_ratio: float = 0.4,
        max_iter: int = 10000,
        coef_threshold: float = 1e-8,
        random_state: int | None = 42,
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.coef_threshold = coef_threshold
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names_in_ = list(X.columns)

        self.selector_model_ = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )

        self.selector_model_.fit(X, y)

        self.selected_features_ = [
            feature
            for feature, coef_value in zip(
                self.feature_names_in_,
                self.selector_model_.coef_,
            )
            if abs(coef_value) > self.coef_threshold
        ]

        if not self.selected_features_:
            raise ValueError("ElasticNet did not select any features.")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "selected_features_"):
            raise ValueError("ElasticNetFeatureSelector must be fitted first.")

        return X[self.selected_features_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if not hasattr(self, "selected_features_"):
            raise ValueError("Transformer has not been fitted yet.")

        return np.array(self.selected_features_, dtype=object)
    

class ElasticNetCVFeatureSelector(BaseEstimator, TransformerMixin):
    # ВИКОРИСТОВУЄ CV=5, ТОБТО НЕМАЄ GROUP!!!!
    def __init__(
        self,
        l1_ratio=(0.1, 0.5, 0.7, 0.9, 1.0),
        cv=5,
        random_state=42,
        max_iter=10000,
        coef_threshold=1e-8,
    ):
        self.l1_ratio = l1_ratio
        self.cv = cv
        self.random_state = random_state
        self.max_iter = max_iter
        self.coef_threshold = coef_threshold

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names_in_ = list(X.columns)

        self.selector_model_ = ElasticNetCV(
            l1_ratio=self.l1_ratio,
            cv=self.cv,
            random_state=self.random_state,
            max_iter=self.max_iter,
        )

        self.selector_model_.fit(X, y)

        coef = self.selector_model_.coef_

        self.selected_features_ = [
            feature
            for feature, coef_value in zip(self.feature_names_in_, coef)
            if abs(coef_value) > self.coef_threshold
        ]

        if not self.selected_features_:
            raise ValueError("ElasticNetCV did not select any features.")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "selected_features_"):
            raise ValueError("ElasticNetFeatureSelector must be fitted first.")

        return X[self.selected_features_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if not hasattr(self, "selected_features_"):
            raise ValueError("Transformer has not been fitted yet.")

        return np.array(self.selected_features_, dtype=object)