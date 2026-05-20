from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATASET_DATE = "2024-11-02"


@dataclass
class Config:
    raw_path: Path
    sheet_name: str
    long_path: Path
    wide_path: Path
    quality_report_path: Path
    ml_data_dir: Path


def make_config(dataset_date: str = DEFAULT_DATASET_DATE) -> Config:
    return Config(
        raw_path=Path(f"data/raw/ophthalmology_raw_{dataset_date}.xlsx"),
        sheet_name="База",
        long_path=Path(f"data/processed/ophthalmology_long_{dataset_date}.xlsx"),
        wide_path=Path(f"data/processed/ophthalmology_wide_{dataset_date}.xlsx"),
        quality_report_path=Path(f"reports/quality_report_{dataset_date}.xlsx"),
        ml_data_dir=Path("data/ml"),
    )