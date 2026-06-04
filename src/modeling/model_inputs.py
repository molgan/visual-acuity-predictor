from dataclasses import dataclass
from collections.abc import Sequence

import pandas as pd

from src.modeling.splitters import make_group_train_test_split


@dataclass(frozen=True)
class ModelInputs:
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series | None
    metadata: pd.DataFrame | None

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)

    @property
    def n_features(self) -> int:
        return self.X.shape[1]
    

@dataclass(frozen=True)
class TrainTestModelInputs:
    train: ModelInputs
    test: ModelInputs


def prepare_model_inputs(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    include_feature_cols: Sequence[str] | None = None,
    exclude_feature_cols: Sequence[str] | None = None,
) -> ModelInputs:
    required_cols = {target_col, group_col}

    missing_required_cols = required_cols - set(df.columns)
    if missing_required_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_required_cols)}")

    if df[target_col].isna().any():
        raise ValueError(f"Target column contains missing values: {target_col}")

    if df[group_col].isna().any():
        raise ValueError(f"Group column contains missing values: {group_col}")

    protected_cols = {target_col, group_col}

    if include_feature_cols is None:
        feature_cols = [
            col
            for col in df.columns
            if col not in protected_cols
        ]
    else:
        include_feature_cols = list(include_feature_cols)

        missing_include_cols = set(include_feature_cols) - set(df.columns)
        if missing_include_cols:
            raise ValueError(f"Unknown include_feature_cols: {sorted(missing_include_cols)}")

        protected_include_cols = set(include_feature_cols) & protected_cols
        if protected_include_cols:
            raise ValueError(f"Target/group columns cannot be used as features: {sorted(protected_include_cols)}")

        feature_cols = include_feature_cols

    if exclude_feature_cols is not None:
        exclude_feature_cols = list(exclude_feature_cols)

        missing_exclude_cols = set(exclude_feature_cols) - set(df.columns)
        if missing_exclude_cols:
            raise ValueError(f"Unknown exclude_feature_cols: {sorted(missing_exclude_cols)}")

        protected_exclude_cols = set(exclude_feature_cols) & protected_cols
        if protected_exclude_cols:
            raise ValueError(f"Target/group columns cannot be excluded as features: {sorted(protected_exclude_cols)}")

        feature_cols = [
            col
            for col in feature_cols
            if col not in exclude_feature_cols
        ]

    if not feature_cols:
        raise ValueError("No feature columns remain after filtering.")

    metadata_cols = [
        col
        for col in df.columns
        if col not in feature_cols and col not in protected_cols
    ]

    metadata = df[metadata_cols] if metadata_cols else None

    return ModelInputs(
        X=df[feature_cols],
        y=df[target_col],
        groups=df[group_col],
        metadata=metadata,
    )


def prepare_train_test_model_inputs(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    exclude_feature_cols: Sequence[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainTestModelInputs:
    df_train, df_test = make_group_train_test_split(
        df=df,
        group_col=group_col,
        test_size=test_size,
        random_state=random_state,
    )

    inputs_train = prepare_model_inputs(
        df=df_train,
        target_col=target_col,
        group_col=group_col,
        exclude_feature_cols=exclude_feature_cols,
    )

    inputs_test = prepare_model_inputs(
        df=df_test,
        target_col=target_col,
        group_col=group_col,
        exclude_feature_cols=exclude_feature_cols,
        include_feature_cols=inputs_train.feature_names,
    )

    return TrainTestModelInputs(
        train=inputs_train,
        test=inputs_test,
    )