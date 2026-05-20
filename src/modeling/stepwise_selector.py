import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.base import BaseEstimator, TransformerMixin


def forward_step(
    X: pd.DataFrame,
    y: pd.Series,
    selected: list[str],
    remaining: set[str],
    alpha_add: float,
    corr_threshold: float,
    min_samples: int,
    first_step: bool,
) -> str | None:
    """
    Один крок forward selection.
    Повертає найкращу ознаку для додавання або None.
    """

    candidates = []

    for candidate in remaining:
        candidate_features = selected + [candidate]

        if X.shape[0] < min_samples:
            continue

        if has_strong_correlation(
            X=X[candidate_features],
            selected=selected,
            candidate=candidate,
            threshold=corr_threshold,
        ):
            continue

        X_model = sm.add_constant(X[candidate_features], has_constant="add")

        try:
            model = sm.OLS(y, X_model).fit()
        except Exception:
            continue

        pvalue = model.pvalues.get(candidate, np.nan)

        if np.isfinite(pvalue):
            candidates.append((pvalue, candidate))

    if not candidates:
        return None

    candidates.sort()
    best_pvalue, best_candidate = candidates[0]

    if best_pvalue > alpha_add and not first_step:
        return None

    return best_candidate


def backward_step(
    X: pd.DataFrame,
    y: pd.Series,
    selected: list[str],
    alpha_del: float,
    min_samples: int,
) -> str | None:
    """
    Один крок backward elimination.
    Повертає ознаку для видалення або None.
    """

    if not selected:
        return None

    if X.shape[0] < min_samples:
        return None

    X_model = sm.add_constant(X[selected], has_constant="add")

    try:
        model = sm.OLS(y, X_model).fit()
    except Exception:
        return None

    pvalues = model.pvalues.drop(index="const", errors="ignore")

    if pvalues.empty:
        return None

    worst_feature = pvalues.idxmax()
    worst_pvalue = pvalues.max()

    if worst_pvalue > alpha_del:
        return worst_feature

    return None


def has_strong_correlation(
    X: pd.DataFrame,
    selected: list[str],
    candidate: str,
    threshold: float,
) -> bool:
    for feature in selected:
        corr = X[[feature, candidate]].corr().iloc[0, 1]

        if pd.notna(corr) and abs(corr) > threshold:
            return True

    return False


def stepwise_selection(
    X: pd.DataFrame,
    y: pd.Series,
    alpha_add: float = 0.05,
    alpha_del: float = 0.08,
    corr_threshold: float = 0.8,
    min_samples: int = 10,
    initial_features: list[str] | None = None,
) -> list[str]:
    """
    Покроковий відбір ознак.
    Повертає список selected_features.
    """
 
    if X.isna().any().any():
        raise ValueError("X contains missing values.")

    if y.isna().any():
        raise ValueError("y contains missing values.")

    if initial_features is None:
        selected: list[str] = []
    else:
        selected = initial_features.copy()

    unknown_features = set(selected) - set(X.columns)

    if unknown_features:
        raise ValueError(f"Unknown initial features: {unknown_features}")

    remaining = set(X.columns) - set(selected)
    first_step = len(selected) == 0

    while remaining:
        feature_to_add = forward_step(
            X=X,
            y=y,
            selected=selected,
            remaining=remaining,
            alpha_add=alpha_add,
            corr_threshold=corr_threshold,
            min_samples=min_samples,
            first_step=first_step,
        )

        if feature_to_add  is None:
            break

        selected.append(feature_to_add)
        remaining.remove(feature_to_add)
        first_step = False

        feature_to_remove = backward_step(
            X=X,
            y=y,
            selected=selected,
            alpha_del=alpha_del,
            min_samples=min_samples,
        )

        if feature_to_remove is not None:
            selected.remove(feature_to_remove)
            remaining.add(feature_to_remove)

    return selected


class StepwiseFeatureSelector(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible обгортка над stepwise_selection.
    Потрібна для того, щоб stepwise можна було вставити в Pipeline.
    """

    def __init__(
        self,
        alpha_add: float = 0.05,
        alpha_del: float = 0.08,
        corr_threshold: float = 0.8,
        min_samples: int = 10,
        initial_features: list[str] | None = None,

    ):
        self.alpha_add = alpha_add
        self.alpha_del = alpha_del
        self.corr_threshold = corr_threshold
        self.min_samples = min_samples
        self.initial_features = initial_features

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if not isinstance(y, pd.Series):
            y = pd.Series(y, index=X.index, name="target")

        self.feature_names_in_ = list(X.columns)

        self.selected_features_ = stepwise_selection(
            X=X,
            y=y,
            alpha_add=self.alpha_add,
            alpha_del=self.alpha_del,
            corr_threshold=self.corr_threshold,
            min_samples=self.min_samples,
            initial_features=self.initial_features,
        )

        if not self.selected_features_:
            raise ValueError("Stepwise selection did not select any features.")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "selected_features_"):
            raise ValueError("StepwiseFeatureSelector спочатку потрібно навчити через fit().")

        return X[self.selected_features_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if not hasattr(self, "selected_features_"):
            raise ValueError("Transformer has not been fitted yet.")

        return np.array(self.selected_features_, dtype=object)