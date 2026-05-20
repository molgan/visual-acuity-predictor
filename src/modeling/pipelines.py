from dataclasses import asdict

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import RFECV
from sklearn.compose import TransformedTargetRegressor

from src.modeling.stepwise_selector import StepwiseFeatureSelector
from src.modeling.elasticnet_selector import ElasticNetFeatureSelector
from src.modeling.configs import (
    StepwiseConfig, 
    ElasticNetConfig, 
    ElasticNetSearchConfig,
    RandomForestConfig
)


def make_dummy_pipeline(
    strategy: str = "mean",
) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
        ("model", DummyRegressor(strategy=strategy)),
    ])


def make_linear_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
        ("model", LinearRegression()),
    ])


def make_elasticnet_pipeline(
    elasticnet: ElasticNetConfig = ElasticNetConfig(), 
) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
        ("scaler", StandardScaler().set_output(transform="pandas")),
        ("model", ElasticNet(**asdict(elasticnet))),
    ])


def make_elasticnetcv_pipeline(
    elasticnet_search: ElasticNetSearchConfig = ElasticNetSearchConfig(), 
    n_splits: int = 5, 
    random_state: int = 42,
) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
        ("scaler", StandardScaler().set_output(transform="pandas")),
        ("model", ElasticNetCV(
            **elasticnet_search.elasticnetcv_params, 
            cv = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        )),
    ])


def make_elasticnetselector_linear_pipeline( 
    elasticnet: ElasticNetConfig = ElasticNetConfig(), 
) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
        ("scaler", StandardScaler().set_output(transform="pandas")),
        ("feature_selector", ElasticNetFeatureSelector(**asdict(elasticnet))), 
        ("model", LinearRegression()),
    ])


def make_stepwise_linear_pipeline(
    stepwise: StepwiseConfig = StepwiseConfig(),
) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
        ("feature_selector", StepwiseFeatureSelector(**asdict(stepwise)),),
        ("model", LinearRegression()),
    ])


def make_rfecv_linear_pipeline(
    n_splits: int = 5, 
    random_state: int = 42,
) -> Pipeline:
    cv = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
        ("scaler", StandardScaler().set_output(transform="pandas")),
        ("feature_selector", RFECV(estimator=LinearRegression(), cv=cv, scoring="r2")),
        ("model", LinearRegression()),
    ])


def make_elasticnetselector_elasticnet_pipeline( 
    elasticnet_selector: ElasticNetConfig = ElasticNetConfig(), 
    elasticnet: ElasticNetConfig = ElasticNetConfig(), 
) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
        ("scaler", StandardScaler().set_output(transform="pandas")),
        ("feature_selector", ElasticNetFeatureSelector(**asdict(elasticnet_selector))), 
        ("model", ElasticNet(**asdict(elasticnet))),
    ])


def make_randomforest_pipeline(
    rf: RandomForestConfig = RandomForestConfig(),
    random_state: int = 42,
) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
        ("model", RandomForestRegressor(**asdict(rf), random_state=random_state)),
    ])


def make_rfecv_randomforest_pipeline(
    rf: RandomForestConfig = RandomForestConfig(),
    n_splits: int = 5, 
    random_state: int = 42,
) -> Pipeline:
    cv = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
        (
            "feature_selector",
            RFECV(
                estimator=RandomForestRegressor(**asdict(rf), random_state=random_state),
                cv=cv,
                scoring="r2",
                step=1,
            ),
        ),
        ("model", RandomForestRegressor(**asdict(rf), random_state=random_state)),
    ])


def make_log_target_pipeline(
    pipeline: Pipeline,
) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log,
        inverse_func=np.exp,
        check_inverse=True,
    )


def make_log10_target_pipeline(
    pipeline: Pipeline,
) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=lambda y: -np.log10(y),
        inverse_func=lambda y: np.power(10, -y),
        check_inverse=True,
    )


def build_pipeline(
    method: str,
    random_state: int = 42,
    n_splits: int = 5,
    stepwise: StepwiseConfig = StepwiseConfig(),
    elasticnet: ElasticNetConfig = ElasticNetConfig(),
    rf: RandomForestConfig = RandomForestConfig()
) -> Pipeline:
    if method == "Dummy": 
        return make_dummy_pipeline()
    elif method == "LR": 
        return make_linear_pipeline()
    elif method == "EN":
        return make_elasticnet_pipeline(elasticnet)
    elif method == "ENselector_LR":
        return make_elasticnetselector_linear_pipeline(elasticnet)
    elif method == "Stepwise_LR": 
        return make_stepwise_linear_pipeline(stepwise)
    elif method == "RF":
        return make_randomforest_pipeline(rf=rf, random_state=random_state)
    elif method == "RFECV_RF":
        return make_rfecv_randomforest_pipeline(rf=rf, n_splits=n_splits, random_state=random_state)

    raise ValueError(f"Unknown method: {method}")