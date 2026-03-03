from typing import List, Optional, Union, Tuple

import numpy as np
import pandas as pd


def calculate_step_sizes(
        funnel: pd.DataFrame, grouping_cols: List[str], criteria_cols: List[str]
) -> pd.DataFrame:
    """
    Calculate funnel step sizes (counts of True) per group.
    Returns a DataFrame indexed by `grouping_cols` with columns corresponding to `criteria_cols`, where values are the
    count of True rows per group for each criterion.
    Raises KeyError if any grouping column or criteria column is missing, or TypeError if any criteria column is not
    boolean dtype.
    """
    missing_group = [c for c in grouping_cols if c not in funnel.columns]
    if missing_group:
        raise KeyError(f"Missing grouping columns: {missing_group}")
    missing_criteria = [c for c in criteria_cols if c not in funnel.columns]
    if missing_criteria:
        raise KeyError(f"Missing criteria columns: {missing_criteria}")
    for col in criteria_cols:
        if not pd.api.types.is_bool_dtype(funnel[col]):
            raise TypeError(f"Criteria column '{col}' must be boolean dtype, got {funnel[col].dtype}.")
    tmp = funnel[grouping_cols + criteria_cols].copy()
    tmp[criteria_cols] = tmp[criteria_cols].fillna(False).astype(bool)
    tmp["all"] = True
    result = tmp.groupby(grouping_cols, dropna=False, observed=True)[["all"] + criteria_cols].sum()
    return result


def calculate_proportions(
        sizes: pd.DataFrame,
        numerator_col: str,
        denominator_col: str,
        primary_key: Optional[str] = None,
        subgroups: Optional[Union[str, List[str]]] = None,
) -> pd.DataFrame:
    """
    Calculates proportions and aggregates by primary key and optional subgroups.
    Includes 'overall' rows where subgroups are labeled as 'all'.
    """
    subgroups = [subgroups] if isinstance(subgroups, str) else (subgroups or [])
    all_groups = [primary_key] + subgroups
    _validate_inputs(sizes, numerator_col, denominator_col, all_groups)
    props = (
        sizes[all_groups + [numerator_col, denominator_col]]
        .assign(proportion=lambda x: x[numerator_col].astype(float) / x[denominator_col].astype(float))
        .dropna(subset=["proportion"], how="any", axis=0)
    )
    summary = _aggregate(props, [primary_key])
    if not subgroups:
        # early exit if no subgroups, as summary and detailed are the same
        return summary
    # Aggregate within all groups (detailed), and merge with overall results
    detailed = _aggregate(props, all_groups)
    summary, detailed = _fill_subgroups_as_all(summary, detailed, subgroups)

    joint = (
        pd.concat([detailed, summary], ignore_index=True)
        .sort_values(all_groups)
        .reset_index(drop=True)
    )
    return joint


def _validate_inputs(df: pd.DataFrame, num: str, den: str, groups: List[str]):
    """Ensures columns exist and value columns are numeric."""
    missing = [c for c in groups + [num, den] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    if len(groups) != len(set(groups)):
        raise ValueError(f"Grouping columns must be unique, got duplicates: {groups}")
    if num == den:
        raise KeyError(f"Nominator and denominator columns must be different.")
    if any(c in groups for c in [num, den]):
        raise ValueError(f"Grouping columns cannot include the nominator/denominator columns.")
    if not all(pd.api.types.is_numeric_dtype(df[c]) for c in [num, den]):
        raise TypeError(f"Columns '{num}' and '{den}' must be numeric.")


def _aggregate(df: pd.DataFrame, groups: List[str]) -> pd.DataFrame:
    """Performs standard mean/std/sem aggregation."""
    return (
        df.groupby(groups, observed=True)["proportion"]
        .agg(n="count", mean="mean", std="std", sem="sem")
        .fillna({"std": 0, "sem": 0})
        .reset_index()
    )


def _fill_subgroups_as_all(
        summary: pd.DataFrame, detailed: pd.DataFrame, subgroups: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Adds subgroup columns to the summary df and fills them with 'all'."""
    for col in subgroups:
        summary[col] = "all"
        if detailed[col].dtype.name == "category":
            # Sync categories to allow 'all'
            if "all" not in detailed[col].cat.categories:
                detailed[col] = detailed[col].cat.add_categories("all")
            summary[col] = pd.Categorical(
                summary[col],
                categories=detailed[col].cat.categories,
                ordered=detailed[col].cat.ordered
            )
    return summary, detailed
