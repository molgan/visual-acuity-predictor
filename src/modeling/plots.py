import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _find_param_column(
    results: pd.DataFrame,
    param_name: str,
) -> str:
    matches = [
        col
        for col in results.columns
        if col.startswith("param_")
        and col.endswith(param_name)
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one column ending with '{param_name}', "
            f"found {matches}"
        )

    return matches[0]


def plot_elasticnet_heatmap(
    results: pd.DataFrame,
    value_col: str,
    title: str,
    figsize: tuple[float, float] = (10, 6),
):
    data = results.copy()

    alpha_col = _find_param_column(data, "alpha")
    l1_col = _find_param_column(data, "l1_ratio")

    data["alpha"] = data[alpha_col].astype(float).round(7)
    data["l1_ratio"] = data[l1_col].astype(float).round(7)

    heatmap_data = data.pivot_table(
        index="l1_ratio",
        columns="alpha",
        values=value_col,
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        heatmap_data,
        cmap="viridis",
        annot=True,
        fmt=".3g",
        annot_kws={"size": 6},
        cbar_kws={"label": value_col},
        ax=ax,
    )

    ax.set_xlabel("alpha")
    ax.set_ylabel("l1_ratio")
    ax.set_title(title)

    fig.tight_layout()

    return fig, ax