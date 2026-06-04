import pandas as pd

from src.data.columns import (
    OPHTH_START,
    OPHTH_END,
    PSYCH_START,
    PSYCH_END,
    BIRTH_DATE_COL,
    AGE_COL,
    SEX_COL,
    NOSOGENY_COL,
)


def get_ophthalmology_columns(df: pd.DataFrame) -> list[str]:
    if OPHTH_START not in df.columns:
        return []

    if OPHTH_END not in df.columns:
        return []

    return list(df.loc[:, OPHTH_START:OPHTH_END].columns.str.strip())


def get_ophthalmology_map_columns(df: pd.DataFrame) -> list[str]:
    ophth_cols = get_ophthalmology_columns(df)

    return [
        col for col in ophth_cols
        if col.startswith("1-") or col.startswith("2-")
    ]


def get_psychology_columns(df: pd.DataFrame) -> list[str]:
    if PSYCH_START not in df.columns:
        return []

    if PSYCH_END not in df.columns:
        return []

    return list(df.loc[:, PSYCH_START:PSYCH_END].columns.str.strip())


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    base_cols = [AGE_COL]

    ophthalmology_cols = get_ophthalmology_columns(df)
    
    psych_cols = get_psychology_columns(df)
    psych_cols = [c for c in psych_cols if c != NOSOGENY_COL]

    cols = base_cols + ophthalmology_cols + psych_cols

    return [c for c in cols if c in df.columns]
    

def get_patient_level_columns(df: pd.DataFrame) -> list[str]:
    base_cols = [
        BIRTH_DATE_COL,
        AGE_COL,
        SEX_COL,
    ]

    psych_cols = get_psychology_columns(df)

    return [col for col in base_cols + psych_cols if col in df.columns]


def get_demographic_columns(df: pd.DataFrame) -> list[str]:
    cols = [AGE_COL, SEX_COL]
    return [col for col in cols if col in df.columns]