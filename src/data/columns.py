PATIENT_COL = "Пацієнт"
EXAM_COL = "Обстеження"
EYE_COL = "Око"

PATIENT_RAW_COL = "Пацієнт_raw"
EXAM_RAW_COL = "Обстеження_raw"

EXAM_DATE_COL = "Дата"
BIRTH_DATE_COL = "Дата народження"
AGE_COL = "Вік"
SEX_COL = "Стать: 1 - ч, 2 - ж"

DIAGNOSIS_COL = "Діагноз"
NOSOGENY_COL = "Нозогенія"

ALLOWED_EYES = frozenset({"d", "s"})
ALLOWED_SEX = frozenset({1, 2})
ALLOWED_DIAGNOSIS = frozenset({"Здорова", "Слабкого", "Середнього", "Високого"})
ALLOWED_NOSOGENY = frozenset({1, 2, 3, 4, 5, 6, 7})

OPHTH_START = "Гострота зору некорегована далека відстань"
OPHTH_END = "Cup Area mm2"
PSYCH_START = "L"
PSYCH_END = "Екзальтована"

BINARY_MAPPINGS = {
    EYE_COL: {"d": 1, "s": 0},
    SEX_COL: {1: 1, 2: 0},
}