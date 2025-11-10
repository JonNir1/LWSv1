from typing import Literal, Union, List

import numpy as np
import pandas as pd


def calculate_proportions(
        funnel_sizes: pd.DataFrame,
        nominator: str,
        denominator: str,
        aggregate_by: Literal["trial_category", "target_category", "both"],
        per_subject: bool,
) -> pd.DataFrame:
    if (nominator not in funnel_sizes.columns) or (denominator not in funnel_sizes.columns):
        raise ValueError(f"Funnel Sizes must contain columns `{nominator}` and `{denominator}`.")
    aggregate_by = aggregate_by.lower()
    if aggregate_by not in ["trial_category", "target_category", "both"]:
        raise NotImplementedError(f"Aggregate By `{aggregate_by}` is not supported.")
    if aggregate_by == "trial_category":
        agg_cols = ["trial_category"]
    elif aggregate_by == "target_category":
        agg_cols = ["target_category"]
    else:  # aggregate_by == "both"
        agg_cols = ["trial_category", "target_category"]
    agg_cols = ["subject"] + agg_cols if per_subject else agg_cols
    missing_columns = [col for col in agg_cols if col not in funnel_sizes.columns]
    if missing_columns:
        raise ValueError(f"Funnel Sizes must contain columns: {missing_columns}.")
    prop = (
        funnel_sizes
        .copy()     # avoid modifying the original DataFrame
        .assign(proportion=lambda df: df[nominator].astype(float) / df[denominator].astype(float))
        .dropna(subset=["proportion"], how="any", axis=0)
        .drop(columns=[nominator, denominator])
    )
    return _aggregate_and_sort(prop, agg_cols)


def _aggregate_and_sort(proportions: pd.DataFrame, by: Union[str, List[str]]) -> pd.DataFrame:
    if isinstance(by, str):
        by = [by]
    has_subject = "subject" in by
    # calculated aggregates across all category levels
    overall = proportions.groupby("subject", as_index=False) if has_subject else proportions
    overall = overall.agg(
        n_trials=("proportion", "count"),
        median=("proportion", "median"),
        mean=("proportion", "mean"),
        std=("proportion", "std"),
    )
    overall = overall if has_subject else overall.T
    # calculated aggregates per category level
    aggregated = (
        proportions
        .groupby(by)
        .agg(
            n_trials=("proportion", "count"),
            median=("proportion", "median"),
            mean=("proportion", "mean"),
            std=("proportion", "std"),
        )
        .drop(index="UNKNOWN", level=1, errors="ignore")  # drop UNKNOWN category if exists
        .reset_index(drop=False)
    )
    # concatenate overall and per-category aggregates
    for col in by:
        if col == "subject":
            continue
        aggregated[col] = aggregated[col].cat.add_categories("all")
        overall[col] = pd.Categorical(
            ["all"] * len(overall), categories=aggregated[col].cat.categories.tolist(), ordered=True
        )
    result = (
        pd.concat([overall, aggregated], ignore_index=True)
        .assign(sem=lambda df: df["std"] / np.sqrt(df["n_trials"]))
        .fillna({"std": 0.0, "sem": 0.0})
        .sort_values(by)
        .reset_index(drop=True)
    )
    result = result[by + ["n_trials", "median", "mean", "std", "sem",]]
    return result
