import json
from pathlib import Path

import pandas as pd

from src.data.config import make_config, DEFAULT_DATASET_DATE
from src.data.columns import PATIENT_COL, EYE_COL, EXAM_COL, DIAGNOSIS_COL, BINARY_MAPPINGS
from src.data.column_groups import (
    get_ophthalmology_columns,
    get_ophthalmology_map_columns,
    get_psychology_columns,
    get_demographic_columns,
)
from src.data.ml_dataset_specs import (
    SPECS,
    MLDatasetSpec,
    get_excluded_features_for_target,
)


MAX_MISSING_RATE_FOR_ML = 0.5


def get_feature_columns(df: pd.DataFrame, spec: MLDatasetSpec) -> list[str]:
    ophth_cols = get_ophthalmology_columns(df)
    map_cols = get_ophthalmology_map_columns(df)
    psych_cols = get_psychology_columns(df)
    demographic_cols = get_demographic_columns(df)

    ophth_without_maps = [
        col for col in ophth_cols
        if col not in map_cols
    ]

    if spec.feature_set == "ophthalmology_without_maps":
        feature_cols = demographic_cols + [DIAGNOSIS_COL] + ophth_without_maps

    elif spec.feature_set == "ophthalmology_with_maps":
        feature_cols = demographic_cols + ophth_without_maps + map_cols

    elif spec.feature_set == "ophthalmology_without_maps_with_psychology":
        feature_cols = demographic_cols + ophth_without_maps + psych_cols

    elif spec.feature_set == "ophthalmology_with_maps_psychology":
        feature_cols = demographic_cols + ophth_without_maps + map_cols + psych_cols

    else:
        raise ValueError(f"Unknown feature_set: {spec.feature_set}")

    feature_cols = [
        col for col in feature_cols
        if col != spec.target_col
    ]

    return feature_cols


def validate_columns_exist(df: pd.DataFrame, cols: list[str]) -> None:
    missing_cols = [col for col in cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")


def validate_ml_dataset(df: pd.DataFrame, feature_cols: list[str], spec: MLDatasetSpec) -> None:
    if df.empty:
        raise ValueError(f"No rows found for exam={spec.exam}")

    if not feature_cols:
        raise ValueError(f"Feature list is empty for: {spec.dataset_name}")

    validate_columns_exist(df, [spec.target_col] + feature_cols)

    duplicates = df[[PATIENT_COL, EYE_COL, EXAM_COL]].duplicated().sum()

    if duplicates > 0:
        raise ValueError(
            f"Є дублікати {PATIENT_COL}-{EYE_COL}-{EXAM_COL}: {duplicates}"
        )


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col, mapping in BINARY_MAPPINGS.items():
        if col not in df.columns:
            continue

        unique_values = set(df[col].dropna().unique())
        unknown_values = unique_values - set(mapping.keys())

        if unknown_values:
            raise ValueError(
                f"Unknown values in column '{col}': {sorted(unknown_values)}"
            )

        df[col] = df[col].map(mapping)

    return df


def exclude_forbidden_features(feature_cols: list[str], spec: MLDatasetSpec) -> tuple[list[str], dict]:
    excluded = get_excluded_features_for_target(spec.target_col)

    actually_excluded = [
        col for col in excluded
        if col in feature_cols
    ]

    kept_cols = [
        col for col in feature_cols
        if col not in actually_excluded
    ]

    info = {
        "excluded_feature_columns_by_target": actually_excluded,
        "n_excluded_feature_columns_by_target": len(actually_excluded),
    }

    return kept_cols, info


def drop_columns_by_missing_rate(df: pd.DataFrame, cols: list[str], threshold: float) -> tuple[list[str], dict]:
    missing_rates = (df[cols].isna().mean())

    dropped_cols = missing_rates[missing_rates > threshold].index.tolist()

    kept_cols = [
        col for col in cols
        if col not in dropped_cols
    ]

    info = {
        "missing_rate_threshold": threshold,
        "dropped_columns_by_missing_rate": {
            col: float(missing_rates[col])
            for col in dropped_cols
        },
        "n_dropped_missing_columns": len(dropped_cols),
    }

    return kept_cols, info


def drop_constant_columns(df: pd.DataFrame, cols: list[str]) -> tuple[list[str], dict]:
    nunique = df[cols].nunique(dropna=True)

    dropped_cols = nunique[nunique <= 1].index.tolist()

    kept_cols = [
        col for col in cols
        if col not in dropped_cols
    ]

    info = {
        "dropped_constant_columns": dropped_cols,
        "n_dropped_constant_columns": len(dropped_cols),
    }

    return kept_cols, info


def detect_near_constant_columns(df: pd.DataFrame, cols: list[str], dominant_rate_threshold: float = 0.95) -> dict:
    near_constant = {}

    for col in cols:
        value_counts = df[col].dropna().value_counts(normalize=True)

        if value_counts.empty:
            continue

        dominant_rate = float(value_counts.iloc[0])

        if dominant_rate >= dominant_rate_threshold:
            near_constant[col] = dominant_rate

    return {
        "near_constant_threshold": dominant_rate_threshold,
        "near_constant_columns": near_constant,
        "n_near_constant_columns": len(near_constant),
    }


def build_ml_dataset(df: pd.DataFrame, spec: MLDatasetSpec) -> tuple[pd.DataFrame, dict]:
    df_exam = df[df[EXAM_COL] == spec.exam].copy()

    n_rows_before_dropna = int(df_exam.shape[0])
    df_exam = df_exam.dropna(subset=[spec.target_col])
    n_rows_after_dropna = int(df_exam.shape[0])

    df_exam = encode_binary_columns(df_exam)

    feature_cols = get_feature_columns(df=df_exam, spec=spec)
    feature_cols, target_exclusion_info = exclude_forbidden_features(
        feature_cols=feature_cols,
        spec=spec,
    )
    feature_cols, missing_info = drop_columns_by_missing_rate(
        df = df_exam, 
        cols=feature_cols, 
        threshold=MAX_MISSING_RATE_FOR_ML
    )
    feature_cols, constant_info = drop_constant_columns(
        df=df_exam, 
        cols=feature_cols
    )
    near_constant_info = detect_near_constant_columns(
        df=df_exam, 
        cols=feature_cols, 
        dominant_rate_threshold=0.95
    )

    validate_ml_dataset(df=df_exam, feature_cols=feature_cols, spec=spec)

    ml_cols = [PATIENT_COL, EYE_COL, spec.target_col] + feature_cols

    df_ml = df_exam[ml_cols].copy()

    metadata = {
        "dataset_name": spec.dataset_name,
        "target_col": spec.target_col,
        "feature_set": spec.feature_set,
        "exam": spec.exam,
        "n_rows_before_dropna_target": n_rows_before_dropna,
        "n_rows_after_dropna_target": n_rows_after_dropna,
        "n_dropped_missing_target": n_rows_before_dropna - n_rows_after_dropna,
        "n_unique_patients": int(df_ml[PATIENT_COL].nunique()),
        "n_features": int(len(feature_cols)),
        "feature_columns": feature_cols,
        **target_exclusion_info ,
        **missing_info,
        **constant_info,
        **near_constant_info,
    }

    return df_ml, metadata


def save_ml_dataset(df: pd.DataFrame, metadata: dict, dataset_path: Path, metadata_path: Path) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_excel(dataset_path, index=False)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print(f"Saved dataset: {dataset_path}")
    print(f"Saved metadata: {metadata_path}")


def select_specs(specs: list[MLDatasetSpec], dataset_names: list[str] | None = None) -> list[MLDatasetSpec]:
    if dataset_names is None:
        return specs

    selected_specs = [
        spec for spec in specs
        if spec.dataset_name in dataset_names
    ]

    found_names = {spec.dataset_name for spec in selected_specs}
    missing_names = set(dataset_names) - found_names

    if missing_names:
        raise ValueError(f"Unknown dataset_name values: {sorted(missing_names)}")

    return selected_specs


def main(dataset_date: str = DEFAULT_DATASET_DATE, dataset_names: list[str] | None = None) -> None:
    config = make_config(dataset_date)

    df_long = pd.read_excel(config.long_path)

    selected_specs = select_specs(SPECS, dataset_names)

    for spec in selected_specs:
        df_ml, metadata = build_ml_dataset(df=df_long, spec=spec)

        dataset_path = config.ml_data_dir / f"{spec.dataset_name}_{dataset_date}.xlsx"
        metadata_path = config.ml_data_dir / f"{spec.dataset_name}_{dataset_date}_metadata.json"

        save_ml_dataset(df=df_ml, metadata=metadata, dataset_path=dataset_path, metadata_path=metadata_path)


if __name__ == "__main__":
    main()
