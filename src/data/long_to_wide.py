import pandas as pd
from pathlib import Path

from src.config import make_config, DEFAULT_DATASET_DATE
from src.column_groups import get_patient_level_columns
from src.columns import (
    PATIENT_COL,
    EXAM_COL,
    EYE_COL,
    EXAM_DATE_COL,
    DIAGNOSIS_COL,
)


def extract_patients(df: pd.DataFrame, patient_level_cols: list[str]) -> pd.DataFrame:
    patients = (
        df[[PATIENT_COL] + patient_level_cols]
        .groupby(PATIENT_COL, as_index=False)
        .first()
    )

    return patients


def extract_exam_dates(df: pd.DataFrame) -> pd.DataFrame:
    dates = (
        df[[PATIENT_COL, EXAM_COL, EXAM_DATE_COL]]
        .dropna(subset=[EXAM_COL])
        .groupby([PATIENT_COL, EXAM_COL], as_index=False)
        .first()
    )

    dates_wide = dates.pivot(
        index=PATIENT_COL,
        columns=EXAM_COL,
        values=EXAM_DATE_COL,
    )

    dates_wide.columns = [f"{EXAM_DATE_COL}_exam{int(col)}" for col in dates_wide.columns]

    return dates_wide.reset_index()


def extract_diagnosis(df: pd.DataFrame) -> pd.DataFrame:
    diagnosis = (
        df[df[EXAM_COL] == 1][[PATIENT_COL, EYE_COL, DIAGNOSIS_COL]]
        .dropna(subset=[DIAGNOSIS_COL])
        .groupby([PATIENT_COL, EYE_COL], as_index=False)
        .first()
    )

    return diagnosis


def build_eye_wide(df: pd.DataFrame, patient_level_cols: list[str]) -> pd.DataFrame:
    service_cols = set(
        patient_level_cols
        + [
            PATIENT_COL,
            EXAM_COL,
            EYE_COL,
            EXAM_DATE_COL,
            DIAGNOSIS_COL,
        ]
    )

    value_cols = [col for col in df.columns if col not in service_cols]

    eye_long = df[[PATIENT_COL, EYE_COL, EXAM_COL] + value_cols].copy()

    duplicates = eye_long.duplicated(
        subset=[PATIENT_COL, EYE_COL, EXAM_COL],
        keep=False
    )

    if duplicates.any():
        raise ValueError("Є дублікати Пацієнт-Око-Обстеження, pivot_table не сформовано")

    eye_wide = eye_long.pivot_table(
        index=[PATIENT_COL, EYE_COL],
        columns=EXAM_COL,
        values=value_cols,
        aggfunc="first",
    )

    eye_wide.columns = [
        f"{feature}_exam{int(visit)}"
        for feature, visit in eye_wide.columns
    ]

    visits = sorted(df[EXAM_COL].dropna().unique())

    ordered_cols = [
        f"{col}_exam{int(v)}"
        for col in value_cols
        for v in visits
    ]

    eye_wide = eye_wide.reindex(columns=ordered_cols)

    return eye_wide.reset_index()


def reorder_wide_columns(df: pd.DataFrame, patient_level_cols: list[str]) -> pd.DataFrame:
    id_cols = [PATIENT_COL, EYE_COL]

    diagnosis_cols = [DIAGNOSIS_COL]

    date_cols = sorted(
        [
            col for col in df.columns
            if col.startswith(f"{EXAM_DATE_COL}_exam")
        ],
        key=lambda x: int(x.split("exam")[-1])
    )

    patient_cols = [
        col for col in patient_level_cols
        if col in df.columns
    ]

    used_cols = set(id_cols + diagnosis_cols + date_cols + patient_cols)
    ophthalmology_cols = [col for col in df.columns if col not in used_cols]

    ordered_cols = (
        id_cols
        + diagnosis_cols
        + date_cols
        + ophthalmology_cols
        + patient_cols
    )

    return df[ordered_cols]


def build_wide_from_clean(df: pd.DataFrame) -> pd.DataFrame:
    patient_level_cols = get_patient_level_columns(df)

    patients = extract_patients(df, patient_level_cols)
    dates_wide = extract_exam_dates(df)
    diagnosis = extract_diagnosis(df)
    eye_wide = build_eye_wide(df, patient_level_cols)

    wide = (
        eye_wide
        .merge(diagnosis, on=[PATIENT_COL, EYE_COL], how="left")
        .merge(patients, on=PATIENT_COL, how="left")
        .merge(dates_wide, on=PATIENT_COL, how="left")
    )

    wide = reorder_wide_columns(wide, patient_level_cols)

    return wide


def validate_wide_dataset(df_long: pd.DataFrame, df_wide: pd.DataFrame) -> None:
    print(f"Shape: {df_wide.shape}")
    print(df_wide[EYE_COL].value_counts(dropna=False))

    duplicated_eyes = df_wide[[PATIENT_COL, EYE_COL]].duplicated().sum()

    if duplicated_eyes > 0:
        raise ValueError(
            f"У wide-датасеті є дублікати Пацієнт-Око: {duplicated_eyes}"
        )

    expected_rows = df_long[[PATIENT_COL, EYE_COL]].drop_duplicates().shape[0]
    actual_rows = df_wide.shape[0]

    if expected_rows != actual_rows:
        raise ValueError(
            f"Кількість рядків не збігається: очікувалось {expected_rows}, отримано {actual_rows}"
        )

    missing_eye = df_wide[EYE_COL].isna().sum()

    if missing_eye > 0:
        raise ValueError(
            f"У wide-датасеті є пропущені значення в колонці {EYE_COL}: {missing_eye}"
        )

    print("Wide dataset validation passed")


def main(dataset_date: str = DEFAULT_DATASET_DATE) -> None:
    config = make_config(dataset_date)

    df_long = pd.read_excel(config.long_path)

    df_wide = build_wide_from_clean(df_long)

    validate_wide_dataset(df_long, df_wide)

    config.wide_path.parent.mkdir(parents=True, exist_ok=True)
    df_wide.to_excel(config.wide_path, index=False)

    print(f"Wide dataset saved to: {config.wide_path}")


if __name__ == "__main__":
    main() 