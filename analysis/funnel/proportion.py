from typing import Literal, Union, List

import numpy as np
import pandas as pd
from tqdm import tqdm


def calculate_funnel_sizes(
        funnel_data: pd.DataFrame, steps: List[str], verbose: bool = False
) -> pd.DataFrame:
    GROUPBY_COLUMNS = ["subject", "trial", "target"]
    missing_columns = [col for col in GROUPBY_COLUMNS if col not in funnel_data.columns]
    if missing_columns:
        raise ValueError(f"Funnel Data is missing required columns: {missing_columns}.")
    missing_steps = [step for step in steps if step not in funnel_data.columns]
    if missing_steps:
        raise ValueError(f"Funnel Data is missing required funnel steps: {missing_steps}.")
    metadata_columns = [col for col in funnel_data.columns if col not in steps]
    sizes = []
    for _, group in tqdm(funnel_data.groupby(GROUPBY_COLUMNS), disable=not verbose, desc="Calculating Funnel Sizes"):
        metadata = group[metadata_columns].iloc[0]
        counts = group[steps].sum(axis=0)
        sizes.append(pd.concat([metadata, counts]))
    sizes = pd.concat(sizes, axis=1).T
    return sizes


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
    overall = proportions.groupby("subject", as_index=False) if has_subject else proportions
    overall = overall.agg(
        n_trials=("proportion", "count"),
        median=("proportion", "median"),
        mean=("proportion", "mean"),
        std=("proportion", "std"),
    )
    overall = overall if has_subject else overall.T
    overall[[col for col in by if col != "subject"]] = "all"
    aggregated = proportions.groupby(by, as_index=False).agg(
        n_trials=("proportion", "count"),
        median=("proportion", "median"),
        mean=("proportion", "mean"),
        std=("proportion", "std"),
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
