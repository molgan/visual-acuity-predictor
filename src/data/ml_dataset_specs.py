from dataclasses import dataclass

UCVA_TARGET = "Гострота зору некорегована далека відстань"

EXCLUDED_FEATURES_BY_TARGET = {
    UCVA_TARGET: [
        "Корекція sph",
        "Корекція cyl",
        "Корекція cyl ax",
        "Гострота зору максимально корегована далека відстань",
        "Гострота зору некорегована далека відстань циклоплегія",
        "Корекція циклоплегія sph",
        "Корекція циклоплегія cyl",
        "Корекція циклоплегія cyl ax",
        "Гострота зору максимально корегована далека відстань циклоплегія",
        "Гострота зору некорегована близька відстань",
        "Гострота зору максимально корегована близька відстань",
        # "Очний тиск",  
    ],
}


def get_excluded_features_for_target(target_col: str) -> list[str]:
    return EXCLUDED_FEATURES_BY_TARGET.get(target_col, [])


@dataclass(frozen=True)
class MLDatasetSpec:
    target_col: str
    feature_set: str
    dataset_name: str
    exam: int = 1


SPECS = [
    MLDatasetSpec(
        target_col=UCVA_TARGET,
        feature_set="ophthalmology_without_maps",
        dataset_name="ucva_exam1_base",
        exam=1,
    ),
    MLDatasetSpec(
        target_col=UCVA_TARGET,
        feature_set="ophthalmology_with_maps",
        dataset_name="ucva_exam1_with_maps",
        exam=1,
    ),
    MLDatasetSpec(
        target_col=UCVA_TARGET,
        feature_set="ophthalmology_without_maps_with_psychology",
        dataset_name="ucva_exam1_without_maps_with_psychology",
        exam=1,
    ),
    MLDatasetSpec(
        target_col=UCVA_TARGET,
        feature_set="ophthalmology_with_maps_psychology",
        dataset_name="ucva_exam1_with_maps_psychology",
        exam=1,
    ),
]