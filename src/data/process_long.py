import pandas as pd
import numpy as np
import re
from pathlib import Path

from src.data.config import make_config, DEFAULT_DATASET_DATE
from src.data.columns import (
    PATIENT_COL,
    EXAM_COL,
    EYE_COL,
    PATIENT_RAW_COL,
    EXAM_RAW_COL,
    EXAM_DATE_COL,
    BIRTH_DATE_COL,
    AGE_COL,
    SEX_COL,
    DIAGNOSIS_COL,
    NOSOGENY_COL,
    ALLOWED_EYES,
    ALLOWED_SEX,
    ALLOWED_DIAGNOSIS,
    ALLOWED_NOSOGENY,
)
from src.data.column_groups import (
    get_ophthalmology_columns,
    get_numeric_columns,
    get_patient_level_columns,
)


NON_BLOCKING_CHECKS = {
    "bad_exam_date_order", 
    "bad_age", 
    "identical_right_left_eyes",
    "empty_columns",
}


def clean_column_names(df: pd.DataFrame) -> None:
    df.columns = df.columns.str.strip()


def parse_exam(value) -> pd.Series:
    if pd.isna(value):
        return pd.Series([np.nan, np.nan, np.nan])

    value = str(value).strip().lower()
    value = re.sub(r"\s+", "", value)

    match = re.match(r"^(\d+)\.(\d+)([ds])$", value)

    if match is None:
        return pd.Series([np.nan, np.nan, np.nan])

    return pd.Series([int(match.group(1)), int(match.group(2)), match.group(3)])


def prepare_identifier_columns(df: pd.DataFrame) -> None:
    df.rename(columns={PATIENT_COL: PATIENT_RAW_COL, EXAM_COL: EXAM_RAW_COL}, inplace=True)
    df[PATIENT_RAW_COL] = df[PATIENT_RAW_COL].ffill()
    df[[PATIENT_COL, EXAM_COL, EYE_COL]] = (df[EXAM_RAW_COL].apply(parse_exam))


def check_exam_parsing(df: pd.DataFrame) -> pd.DataFrame:
    bad = df[
        df[PATIENT_COL].isna() 
        | df[EXAM_COL].isna() 
        | df[EYE_COL].isna()
    ].copy()

    if len(bad) == 0:
        return pd.DataFrame()

    bad["problem"] = f"Не вдалося розпарсити {EXAM_RAW_COL}"

    return bad[[PATIENT_RAW_COL, EXAM_RAW_COL, PATIENT_COL, EXAM_COL, EYE_COL, "problem"]].copy()


def check_patient_ids(df: pd.DataFrame) -> pd.DataFrame:
    temp = df[
        df[PATIENT_RAW_COL].notna() 
        | df[PATIENT_COL].notna()
    ].copy()

    temp[PATIENT_RAW_COL + "_num"] = pd.to_numeric(temp[PATIENT_RAW_COL], errors="coerce")

    # не число
    bad_non_numeric_raw = temp[
        temp[PATIENT_RAW_COL].notna()
        & temp[PATIENT_RAW_COL + "_num"].isna()
    ].copy()

    # не ціле число (типу 123.5)
    bad_non_integer_raw = temp[
        temp[PATIENT_RAW_COL + "_num"].notna()
        & (temp[PATIENT_RAW_COL + "_num"] % 1 != 0)
    ].copy()

    # не співпадає з parsed ID
    bad_mismatch = temp[
        temp[PATIENT_RAW_COL + "_num"].notna()
        & temp[PATIENT_COL].notna()
        & (temp[PATIENT_RAW_COL + "_num"].astype("Int64") != temp[PATIENT_COL])
    ].copy()

    bad = pd.concat(
        [
            bad_non_numeric_raw.assign(problem=f"{PATIENT_RAW_COL} не є числом"),
            bad_non_integer_raw.assign(problem=f"{PATIENT_RAW_COL} не є цілим числом"),
            bad_mismatch.assign(problem=f"{PATIENT_RAW_COL} не збігається з {PATIENT_COL} з {EXAM_COL}"),
        ],
        ignore_index=True,
    )

    if len(bad) == 0:
        return pd.DataFrame()

    return bad[[PATIENT_RAW_COL, EXAM_RAW_COL, PATIENT_COL, EXAM_COL, EYE_COL, "problem"]].copy()


def check_eye_values(df: pd.DataFrame) -> pd.DataFrame:
    bad = df[
        df[EYE_COL].isna()
        | ~df[EYE_COL].isin(ALLOWED_EYES)
    ].copy()

    if len(bad) == 0:
        return pd.DataFrame()

    allowed_str = ", ".join(str(v) for v in sorted(ALLOWED_EYES))
    bad["problem"] = f"У колонці '{EYE_COL}' є пропуск або значення не з множини {allowed_str}"

    return bad[[PATIENT_RAW_COL, EXAM_RAW_COL, PATIENT_COL, EXAM_COL, EYE_COL, "problem"]].copy()
    

def check_eye_exam_uniqueness(df: pd.DataFrame) -> pd.DataFrame:
    bad = df[df.duplicated(subset=[PATIENT_COL, EXAM_COL, EYE_COL], keep=False)].copy()

    if len(bad) == 0:
        return pd.DataFrame()

    bad["problem"] = f"Дублікат комбінації {PATIENT_COL} + {EXAM_COL} + {EYE_COL}"

    return (
        bad[[PATIENT_RAW_COL, EXAM_RAW_COL, PATIENT_COL, EXAM_COL, EYE_COL, "problem"]]
        .sort_values([PATIENT_COL, EXAM_COL, EYE_COL])
        .copy()
    )


def check_exam_eye_structure(df: pd.DataFrame) -> pd.DataFrame:
    eye_sets = (
        df.groupby([PATIENT_COL, EXAM_COL])[EYE_COL]
        .apply(lambda x: set(x.dropna()))
        .reset_index(name="eyes")
    )

    bad = eye_sets[
        ~eye_sets["eyes"].apply(lambda x: x == ALLOWED_EYES)
    ].copy()

    if len(bad) == 0:
        return pd.DataFrame()

    allowed_str = ", ".join(str(v) for v in sorted(ALLOWED_EYES))
    bad["problem"] = f"Для комбінації {PATIENT_COL} + {EXAM_COL} немає рівно двох очей {allowed_str}"

    return bad[[PATIENT_COL, EXAM_COL, "eyes", "problem"]].copy()


def check_date_parsing(df: pd.DataFrame) -> pd.DataFrame:
    problems = []

    for col in [EXAM_DATE_COL, BIRTH_DATE_COL]:
        if col not in df.columns:
            continue

        raw_values = df[col]
        parsed = pd.to_datetime(raw_values, errors="coerce", dayfirst=True)
        bad_mask = raw_values.notna() & parsed.isna()

        if bad_mask.any():
            bad_rows = df.loc[bad_mask, [PATIENT_RAW_COL, EXAM_RAW_COL, col]].copy()
            bad_rows["column"] = col
            bad_rows.rename(columns={col: EXAM_DATE_COL}, inplace=True)
            bad_rows["problem"] = "Помилка в даті"
            problems.append(bad_rows)

    if problems:
        return pd.concat(problems, ignore_index=True)

    return pd.DataFrame()


def check_exam_dates_consistency(df: pd.DataFrame) -> pd.DataFrame:
    if EXAM_DATE_COL not in df.columns:
        return pd.DataFrame()

    date_check = (
        df.dropna(subset=[EXAM_DATE_COL])
        .groupby([PATIENT_COL, EXAM_COL])[EXAM_DATE_COL]
        .nunique()
        .reset_index(name="n_dates")
    )

    mismatches = date_check[date_check["n_dates"] > 1]

    if len(mismatches) == 0:
        return pd.DataFrame()

    bad = (
        df[[PATIENT_COL, EXAM_COL, EYE_COL, EXAM_DATE_COL]]
        .merge(mismatches[[PATIENT_COL, EXAM_COL]], on=[PATIENT_COL, EXAM_COL])
        .sort_values([PATIENT_COL, EXAM_COL, EYE_COL])
        .copy()
    )

    bad["problem"] = f"Для комбінації {PATIENT_COL} + {EXAM_COL} вказано різні дати"

    return bad


def check_exam_date_order(df: pd.DataFrame) -> pd.DataFrame:
    if EXAM_DATE_COL not in df.columns:
        return pd.DataFrame()

    temp = (
        df[[PATIENT_COL, EXAM_COL, EXAM_DATE_COL]]
        .dropna(subset=[PATIENT_COL, EXAM_COL, EXAM_DATE_COL])
        .drop_duplicates()
        .copy()
    )

    temp[EXAM_DATE_COL] = pd.to_datetime(temp[EXAM_DATE_COL], errors="coerce", dayfirst=True)

    temp = (
        temp
        .dropna(subset=[EXAM_DATE_COL])
        .sort_values([PATIENT_COL, EXAM_COL])
        .copy()
    )

    temp["previous_exam"] = temp.groupby(PATIENT_COL)[EXAM_COL].shift(1)
    temp["previous_date"] = temp.groupby(PATIENT_COL)[EXAM_DATE_COL].shift(1)

    bad = temp[
        temp["previous_date"].notna()
        & (temp[EXAM_DATE_COL] < temp["previous_date"])
    ].copy()

    if len(bad) == 0:
        return pd.DataFrame()

    bad["problem"] = "Дати обстежень ідуть не у порядку номерів обстежень"

    return bad[[PATIENT_COL, EXAM_COL, EXAM_DATE_COL, "previous_exam", "previous_date", "problem"]].copy()


def check_birth_before_exam(df: pd.DataFrame) -> pd.DataFrame:
    if BIRTH_DATE_COL not in df.columns or EXAM_DATE_COL not in df.columns:
        return pd.DataFrame()

    temp = df[[PATIENT_COL, EYE_COL, EXAM_COL, BIRTH_DATE_COL, EXAM_DATE_COL]].copy()
    temp[BIRTH_DATE_COL] = pd.to_datetime(temp[BIRTH_DATE_COL], errors="coerce", dayfirst=True)
    temp[EXAM_DATE_COL] = pd.to_datetime(temp[EXAM_DATE_COL], errors="coerce", dayfirst=True)

    bad = temp[
        temp[BIRTH_DATE_COL].notna()
        & temp[EXAM_DATE_COL].notna()
        & (temp[BIRTH_DATE_COL] >= temp[EXAM_DATE_COL])
    ].copy()

    if len(bad) == 0:
        return pd.DataFrame()

    bad["problem"] = "Дата народження пізніше дати обстеження"

    return bad[[PATIENT_COL, EYE_COL, EXAM_COL, BIRTH_DATE_COL, EXAM_DATE_COL, "problem"]].copy()


def calculate_age(birth_date: pd.Series, exam_date: pd.Series) -> pd.Series:
    age = exam_date.dt.year - birth_date.dt.year

    before_birthday = (
        (exam_date.dt.month < birth_date.dt.month)
        | (
            (exam_date.dt.month == birth_date.dt.month)
            & (exam_date.dt.day < birth_date.dt.day)
        )
    )

    age = age - before_birthday.astype(int)

    return age


def check_age_consistency(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [PATIENT_COL, EXAM_COL, EXAM_DATE_COL, BIRTH_DATE_COL, AGE_COL]

    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame()

    temp = df[required_cols].copy()

    temp[EXAM_DATE_COL] = pd.to_datetime(temp[EXAM_DATE_COL], errors="coerce", dayfirst=True)
    temp[BIRTH_DATE_COL] = pd.to_datetime(temp[BIRTH_DATE_COL], errors="coerce", dayfirst=True)
    temp[AGE_COL] = pd.to_numeric(temp[AGE_COL].astype("string").str.replace(",", ".", regex=False), errors="coerce")

    temp = temp[temp[EXAM_COL] == 1].copy()

    temp = temp[
        temp[EXAM_DATE_COL].notna()
        & temp[BIRTH_DATE_COL].notna()
        & temp[AGE_COL].notna()
    ].copy()

    if len(temp) == 0:
        return pd.DataFrame()

    temp["age_calculated"] = calculate_age(temp[BIRTH_DATE_COL], temp[EXAM_DATE_COL])
    temp["age_diff"] = temp["age_calculated"] - temp[AGE_COL]

    bad = temp[temp["age_diff"].abs() > 1].copy()

    if len(bad) == 0:
        return pd.DataFrame()

    bad["problem"] = "Вік не відповідає даті народження та даті обстеження"

    return bad[[PATIENT_COL, AGE_COL, "age_calculated", "age_diff", BIRTH_DATE_COL, EXAM_DATE_COL, "problem"]].copy()


def check_patient_level_consistency(df: pd.DataFrame, patient_level_cols: list[str]) -> pd.DataFrame: 
    problems = []

    for col in patient_level_cols:
        nunique = df.groupby(PATIENT_COL)[col].nunique(dropna=True)
        bad_patients = nunique[nunique > 1]

        for patient_id in bad_patients.index:
            values = df.loc[df[PATIENT_COL] == patient_id, col].dropna().unique()

            problems.append({
                PATIENT_COL: patient_id,
                "Ознака": col,
                "Значення": list(values),
                "problem": f"Ознака '{col}' має різні значення для одного пацієнта",
            })

    if len(problems) == 0:
        return pd.DataFrame()

    return pd.DataFrame(problems)


def check_patient_level_coded_category(df: pd.DataFrame, col: str, allowed_values: set[int | float], exam: int = 1) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame({"problem": [f"Стовпець '{col}' відсутній"]})

    temp = df[df[EXAM_COL] == exam][[PATIENT_COL, col]].copy()
    temp["value_numeric"] = pd.to_numeric(temp[col], errors="coerce")

    summary = (
        temp.groupby(PATIENT_COL)
        .agg(
            raw_values=(col, lambda x: list(x.dropna().unique())),
            has_valid=("value_numeric", lambda x: x.isin(allowed_values).any()),
            has_invalid=("value_numeric", lambda x: ((x.notna()) & ~x.isin(allowed_values)).any()),
        )
        .reset_index()
    )

    bad = summary[(~summary["has_valid"]) | summary["has_invalid"]].copy()

    if len(bad) == 0:
        return pd.DataFrame()

    allowed_str = ", ".join(str(v) for v in sorted(allowed_values))
    bad["problem"] = (
        f"Ознака '{col}' відсутня або містить некоректні значення "
        f"(допустимо лише {allowed_str})"
    )

    return bad[[PATIENT_COL, "raw_values", "problem"]].copy()


def check_diagnosis_values(df: pd.DataFrame) -> pd.DataFrame:
    if DIAGNOSIS_COL not in df.columns:
        return pd.DataFrame({"problem": [f"Стовпець '{DIAGNOSIS_COL}' відсутній"]})

    allowed = {x.lower() for x in ALLOWED_DIAGNOSIS }

    temp = df[df[EXAM_COL] == 1][[PATIENT_COL, EYE_COL, EXAM_COL, DIAGNOSIS_COL]].copy()

    temp["diagnosis_normalized"] = (
        temp[DIAGNOSIS_COL]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    bad = temp[
        temp["diagnosis_normalized"].isna()
        | ~temp["diagnosis_normalized"].isin(allowed)
    ].copy()

    if len(bad) == 0:
        return pd.DataFrame()

    allowed_str = ", ".join(str(v) for v in sorted(ALLOWED_DIAGNOSIS))
    bad["problem"] = f"Для першого обстеження діагноз відсутній або не належить до множини {allowed_str}"

    return bad[[PATIENT_COL, EYE_COL, EXAM_COL, DIAGNOSIS_COL, "problem"]].copy()
    

def check_identical_right_left_eyes(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = get_ophthalmology_columns(df)

    if not feature_cols:
        return pd.DataFrame()

    rows = []

    for (patient, exam), group in df.groupby([PATIENT_COL, EXAM_COL]):
        if set(group[EYE_COL].dropna()) != ALLOWED_EYES:
            continue

        first_eye, second_eye = sorted(ALLOWED_EYES)
        first = group[group[EYE_COL] == first_eye][feature_cols].iloc[0]
        second = group[group[EYE_COL] == second_eye][feature_cols].iloc[0]

        if first.isna().all() and second.isna().all():
            continue
            
        comparison = first.eq(second) | (first.isna() & second.isna())

        if comparison.all():
            rows.append({
                PATIENT_COL: patient,
                EXAM_COL: exam,
                "n_compared_features": len(feature_cols),
                "problem": "Праве і ліве око повністю однакові за офтальмологічними ознаками",
            })

    return pd.DataFrame(rows)


def check_numeric_columns(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    problems = []

    for col in numeric_cols:
        if col not in df.columns:
            problems.append(pd.DataFrame({"column": [col], "problem": ["Стовпець відсутній"]}))
            continue

        raw = df[col]

        parsed = (
            raw.astype("string")
            .str.replace(",", ".", regex=False)
            .str.strip()
        )

        numeric = pd.to_numeric(parsed, errors="coerce")
        bad_mask = raw.notna() & numeric.isna()

        if bad_mask.any():
            bad_rows = df.loc[bad_mask, [PATIENT_COL, EYE_COL, EXAM_COL, col]].copy()

            bad_rows["column"] = col
            bad_rows.rename(columns={col: "raw_value"}, inplace=True)

            bad_rows["problem"] = "Значення не може бути перетворене у число"

            problems.append(bad_rows)

    if problems:
        return pd.concat(problems, ignore_index=True)

    return pd.DataFrame()


def check_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    mask = df.isna().all()
    cols = df.columns[mask]

    if len(cols) == 0:
        return pd.DataFrame()

    return pd.DataFrame({
        "column": cols,
        "n_rows": len(df),
        "dtype": df.dtypes[mask].astype(str).values,
        "problem": "Стовпець містить лише пропуски (100% NaN). Він буде автоматично видалений!",
    })


def drop_identifier_columns(df: pd.DataFrame) -> None:
    cols_to_drop = [
        PATIENT_RAW_COL,
        EXAM_RAW_COL,
        "№ історії",
        "Прізвище",
        "Ім'я",
        "По батькові",
    ]

    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)


def drop_empty_columns(df: pd.DataFrame) -> None:
    cols_to_drop = df.columns[df.isna().all()].tolist()
    df.drop(columns=cols_to_drop, inplace=True)


def normalize_date_columns(df: pd.DataFrame) -> None:
    for col in [EXAM_DATE_COL, BIRTH_DATE_COL]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)


def normalize_diagnosis_column(df: pd.DataFrame) -> None:
    if DIAGNOSIS_COL not in df.columns:
        return

    df[DIAGNOSIS_COL] = (
        df[DIAGNOSIS_COL]
        .astype("string")
        .str.strip()
        .str.capitalize()
    )

    
def normalize_numeric_columns(df: pd.DataFrame, numeric_cols: list[str]) -> None:
    for col in numeric_cols:
        if col not in df.columns:
            continue

        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
            .str.replace(",", ".", regex=False)
            .pipe(pd.to_numeric, errors="coerce")
        )


# Дублює patient-level ознаки на всі рядки пацієнта
def propagate_patient_level_features(df: pd.DataFrame, patient_level_cols: list[str]) -> pd.DataFrame:
    for col in patient_level_cols:
        # беремо перше ненульове значення для кожного пацієнта
        first_values = (
            df.groupby(PATIENT_COL)[col]
            .transform(lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan)
        )

        df[col] = first_values

    return df


# Дублює дату обстеження на всі рядки в межах одного Пацієнт + Обстеження (тобто на обидва ока)
def propagate_exam_date(df: pd.DataFrame) -> pd.DataFrame:
    df[EXAM_DATE_COL] = (
        df.groupby([PATIENT_COL, EXAM_COL])[EXAM_DATE_COL]
        .transform(lambda x: x.dropna().iloc[0] if x.notna().any() else pd.NaT)
    )

    return df


def make_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "column": df.columns,
        "n_missing": df.isna().sum().values,
        "percent_missing": (df.isna().mean().values * 100).round(2),
        "dtype": df.dtypes.astype(str).values,
    })


def make_missing_report_by_exam(df: pd.DataFrame) -> pd.DataFrame:
    if EXAM_COL not in df.columns:
        return pd.DataFrame()

    result = pd.DataFrame({"column": df.columns})

    for exam, group in df.groupby(EXAM_COL, sort=True):
        missing_n = group.isna().sum().reindex(df.columns)
        missing_pct = (group.isna().mean() * 100).round(2).reindex(df.columns)

        result[f"n_missing_exam_{exam}"] = missing_n.values
        result[f"percent_missing_exam_{exam}"] = missing_pct.values

    return result

    
def make_numeric_summary(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    existing_cols = [col for col in numeric_cols if col in df.columns]

    if not existing_cols:
        return pd.DataFrame()

    temp = df[existing_cols].copy()

    for col in existing_cols:
        temp[col] = (
            temp[col]
            .astype("string")
            .str.replace(",", ".", regex=False)
            .pipe(pd.to_numeric, errors="coerce")
        )

    return temp.describe().T.reset_index().rename(columns={"index": "column"})


def make_numeric_summary_by_exam(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    if EXAM_COL not in df.columns:
        return pd.DataFrame()

    existing_cols = [col for col in numeric_cols if col in df.columns]

    if not existing_cols:
        return pd.DataFrame()

    result = pd.DataFrame({"column": existing_cols})

    for exam, group in df.groupby(EXAM_COL, sort=True):
        temp = group[existing_cols].copy()

        for col in existing_cols:
            temp[col] = (
                temp[col]
                .astype("string")
                .str.strip()
                .str.replace(",", ".", regex=False)
                .pipe(pd.to_numeric, errors="coerce")
            )

        summary = temp.describe().T.reindex(existing_cols)

        for stat in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
            result[f"{stat}_exam_{exam}"] = summary[stat].values

    return result


def format_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%d.%m.%Y")

    return df


def save_quality_report(
    df: pd.DataFrame,
    report_path: Path, 
    validation_results: dict[str, pd.DataFrame], 
    numeric_cols: list[str]
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        for sheet_name, result in validation_results.items():
            if result is not None and len(result) > 0:
                result_to_save = format_for_excel(result)
                result_to_save.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        
        make_missing_report(df).to_excel(writer, sheet_name="missing", index=False)
        make_missing_report_by_exam(df).to_excel(writer, sheet_name="missing_by_exam", index=False)
        
        make_numeric_summary(df, numeric_cols).to_excel(writer, sheet_name="numeric_summary", index=False)
        make_numeric_summary_by_exam(df, numeric_cols).to_excel(writer, sheet_name="numeric_summary_by_exam", index=False)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    clean_column_names(df)
    prepare_identifier_columns(df)
    
    return df


def validate(df: pd.DataFrame, numeric_cols: list[str], patient_level_cols: list[str]) -> dict[str, pd.DataFrame]:
    df = df.copy()

    validation_results = {
        "bad_exam_parsing": check_exam_parsing(df),
        "bad_patient_ids": check_patient_ids(df),
        "bad_eye_values": check_eye_values(df),
        "duplicate_eye_exam": check_eye_exam_uniqueness(df),
        "bad_exam_eye_structure": check_exam_eye_structure(df),

        "bad_date_parsing": check_date_parsing(df),
        "bad_exam_dates": check_exam_dates_consistency(df),
        "bad_exam_date_order": check_exam_date_order(df),
        "bad_birth_exam_dates": check_birth_before_exam(df),

        "bad_patient_level": check_patient_level_consistency(df, patient_level_cols),
        "bad_age": check_age_consistency(df),
        "bad_sex_values": check_patient_level_coded_category(df, SEX_COL, ALLOWED_SEX),
        "bad_nosogeny_values": check_patient_level_coded_category(df, NOSOGENY_COL, ALLOWED_NOSOGENY),
        "bad_diagnosis_values": check_diagnosis_values(df),
        "identical_right_left_eyes": check_identical_right_left_eyes(df),
        "bad_numeric_values": check_numeric_columns(df, numeric_cols),
        "empty_columns": check_empty_columns(df),
    }

    return validation_results
    

def clean(df: pd.DataFrame, numeric_cols: list[str], patient_level_cols:list[str]) -> pd.DataFrame:
    df = df.copy()

    drop_identifier_columns(df)
    drop_empty_columns(df)
    normalize_diagnosis_column(df)
    normalize_numeric_columns(df, numeric_cols)
    propagate_patient_level_features(df, patient_level_cols)
    propagate_exam_date(df)
    normalize_date_columns(df)

    return df


def main(dataset_date: str = DEFAULT_DATASET_DATE) -> None:
    config = make_config(dataset_date)

    df_raw = pd.read_excel(config.raw_path, sheet_name=config.sheet_name)

    df_prepared = prepare(df_raw)

    numeric_columns = get_numeric_columns(df_prepared)
    patient_level_cols = get_patient_level_columns(df_prepared)

    validation_results = validate(df_prepared, numeric_columns, patient_level_cols)
    df_clean = clean(df_prepared, numeric_columns, patient_level_cols)

    save_quality_report(
        df=df_clean,
        report_path=config.quality_report_path,
        validation_results=validation_results,
        numeric_cols=numeric_columns,
    )

    has_errors = any(
        len(validation_results[name]) > 0 
        for name in validation_results 
        if name not in NON_BLOCKING_CHECKS
    )

    if has_errors:
        raise ValueError(f"Валідацію не пройдено. Потрібно перевірити файл: {config.quality_report_path}")
    
    config.long_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean_to_save = format_for_excel(df_clean)
    df_clean_to_save.to_excel(config.long_path, index=False)

    print(f"Clean long dataset saved to: {config.long_path}")
    print(f"Quality report saved to: {config.quality_report_path}")


if __name__ == "__main__":
    main()