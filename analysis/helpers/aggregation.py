from typing import Optional

import pandas as pd


def subject_level_stats(
        df: pd.DataFrame,
        groupby: list[str],
        value_col: str,
        subject_col: str = "subject",
        agg: str = "mean",
) -> pd.DataFrame:
    """
    Two-stage aggregation: first compute ``agg`` per subject within each group,
    then compute population mean, std, and SEM across subjects.

    Returns a DataFrame with columns ``[*groupby, n, mean, std, sem]``.
    """
    per_subject = (
        df
        .groupby([subject_col] + groupby, observed=True)[value_col]
        .agg(agg)
        .rename("_value")
        .reset_index()
    )
    result = (
        per_subject
        .groupby(groupby, observed=True)["_value"]
        .agg(n="count", mean="mean", std="std", sem="sem")
        .fillna({"std": 0, "sem": 0})
        .reset_index()
    )
    return result


def calc_miss_rate(
        idents: pd.DataFrame,
        groups: list[str],
        category_col: str = "identification_category",
        miss_label: str = "miss",
        valid_labels: Optional[tuple[str, ...]] = ("hit", "miss"),
) -> pd.DataFrame:
    """
    Calculate miss rate per group from an identifications DataFrame.

    Returns a DataFrame indexed by ``groups`` with columns
    ``[n_targets, n_misses, miss_rate]``.
    """
    filtered = idents
    if valid_labels is not None:
        filtered = idents.loc[idents[category_col].isin(valid_labels)]
    result = (
        filtered
        .groupby(groups, observed=True)
        .agg(
            n_targets=(category_col, "count"),
            n_misses=(category_col, lambda x: (x == miss_label).sum()),
        )
        .query("n_targets > 0")
        .fillna({"n_misses": 0})
        .assign(miss_rate=lambda df: df["n_misses"] / df["n_targets"])
    )
    return result
