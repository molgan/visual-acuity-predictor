import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def make_group_train_test_split(
    df: pd.DataFrame,
    group_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_idx, test_idx = next(
        splitter.split(df, groups=df[group_col])
    )
    
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    return train_df, test_df