from dataclasses import dataclass, field
from collections.abc import Sequence
import numpy as np


@dataclass(frozen=True)
class StepwiseConfig:
    alpha_add: float = 0.05
    alpha_del: float = 0.08
    corr_threshold: float = 0.8
    min_samples: int = 10


@dataclass(frozen=True)
class ElasticNetConfig:
    alpha: float = 0.35
    l1_ratio: float = 0.1
    max_iter: int = 10000


@dataclass(frozen=True)
class ElasticNetSearchConfig:
    alphas: Sequence[float] = field(
        default_factory=lambda: np.logspace(-4, 2, 100)
    )

    l1_ratios: Sequence[float] = field(
        default_factory=lambda: np.linspace(0.05, 1.0, 20)
    )

    max_iter: int = 10000

    @property
    def param_space(self):
        return {
            "model__alpha": self.alphas,
            "model__l1_ratio": self.l1_ratios,
        }


    @property
    def elasticnetcv_params(self):
        return {
            "alphas": self.alphas,
            "l1_ratio": self.l1_ratios,
            "max_iter": self.max_iter,
        }


@dataclass(frozen=True)
class RandomForestConfig:
    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_leaf: int = 1
    min_samples_split: int = 2
    max_features: str | float | int | None = 1.0
    n_jobs: int = -1


@dataclass(frozen=True)
class RandomForestTuningConfig:
    n_estimators: Sequence[int] = field(
        default_factory=lambda: [100, 200, 500]
    )

    max_depth: Sequence[int | None] = field(
        default_factory=lambda: [None, 3, 5, 8, 12, 20]
    )

    min_samples_leaf: Sequence[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 12]
    )

    min_samples_split: Sequence[int] = field(
        default_factory=lambda: [2, 4, 8, 16]
    )

    max_features: Sequence[int | float | str] = field(
        default_factory=lambda: [0.3, 0.5, 0.7, 1.0, "sqrt"]
    )

    @property
    def param_space(self):
        return {
            "model__n_estimators": self.n_estimators,
            "model__max_depth": self.max_depth,
            "model__min_samples_leaf": self.min_samples_leaf,
            "model__min_samples_split": self.min_samples_split,
            "model__max_features": self.max_features,
        }

def prefix_param_space(param_space: dict, prefix: str) -> dict:
    return {
        f"{prefix}__{name}": values
        for name, values in param_space.items()
    }