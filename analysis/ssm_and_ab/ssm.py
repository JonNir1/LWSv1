"""
Shared helpers for the SSM (Subsequent Search Misses) / Self-Induced Attentional Blink notebooks.

Every notebook in this package loads its base table via `load_ssm_funnel()`, so the funnel filtering,
history enrichment (`enrich_funnel_with_history`, absorbed from the retired `visit_classification.py` -
this is its only consumer in the repo) and `visit_type` classification are computed identically everywhere.
Downstream, `add_ssm_predictors()` and `build_category_pair_table()` build the specific predictors each
question needs on top of that shared base.

CODE_REVIEW.md M10 applies here exactly as it does to `time_on_task`/`spatial_effects`/`hit_rate`: the unit
of observation is a visit, and visits nest within trial within subject, so every model fit from this module's
output should be a flat-vs-subject/trial-nested pair, not a flat fit alone.
"""
from enum import Enum, auto

import numpy as np
import pandas as pd

from analysis.helpers.read_data import DataStore, load_analysis_data


class VisitType(Enum):
    LWS_BEFORE_ANY_HIT = auto()
    LWS_AFTER_HIT_SAME = auto()
    LWS_AFTER_HIT_DIFF = auto()
    LWS_AFTER_2HIT_MIXED = auto()
    LWS_AFTER_2HIT_BOTH_DIFF = auto()
    IDENTIFICATION_VISIT = auto()
    TARGET_RETURN = auto()
    OTHER = auto()


def classify_visit(row: pd.Series, hits: pd.DataFrame) -> VisitType:
    trial_hits = hits.loc[
        (hits["subject"] == row["subject"]) & (hits["trial"] == row["trial"])
    ].sort_values("time")
    target_hits = trial_hits.loc[trial_hits["target"] == row["target"]]

    if len(target_hits) == 1:
        hit_time = target_hits["time"].iloc[0]
        if hit_time < row["start_time"]:
            return VisitType.TARGET_RETURN
        if row["start_time"] <= hit_time <= row["end_time"]:
            return VisitType.IDENTIFICATION_VISIT

    if row["is_lws"]:
        n_found = row["num_targets_found_before"]
        if n_found == 0:
            return VisitType.LWS_BEFORE_ANY_HIT
        same_cat_found = row["target_category"] in row["target_categories_found_before"]
        if n_found == 1:
            return VisitType.LWS_AFTER_HIT_SAME if same_cat_found else VisitType.LWS_AFTER_HIT_DIFF
        if n_found == 2:
            return VisitType.LWS_AFTER_2HIT_MIXED if same_cat_found else VisitType.LWS_AFTER_2HIT_BOTH_DIFF

    return VisitType.OTHER


def enrich_funnel_with_history(
        funnel: pd.DataFrame,
        hits: pd.DataFrame,
        fixations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Vectorized enrichment: adds num_targets_found_before, time_since_recent_find,
    same_category_found_before, target_categories_found_before, fixations_since_trial_start,
    fixations_since_last_hit.
    """
    result = funnel.copy()
    trial_key = ["subject", "trial"]

    # pandas merge_asof requires the "on" column sorted *globally*, even when `by` is given - sorting by
    # `trial_key + [on_col]` (group-then-time) is NOT sufficient and raises "left keys must be sorted"
    # under pandas 3.x. `by` handles the per-group matching; the sort just needs to be by the "on" column.
    hits_sorted = hits.sort_values(trial_key + ["time"]).reset_index(drop=True)
    hits_sorted["_hit_rank"] = hits_sorted.groupby(trial_key).cumcount() + 1

    hits_for_asof = hits_sorted[trial_key + ["time", "_hit_rank", "target_category"]].rename(
        columns={"time": "_last_hit_time", "_hit_rank": "_last_hit_rank", "target_category": "_last_hit_cat"}
    ).sort_values("_last_hit_time")
    merged = pd.merge_asof(
        result.sort_values("start_time"),
        hits_for_asof,
        by=trial_key,
        left_on="start_time",
        right_on="_last_hit_time",
        direction="backward",
    )

    merged["num_targets_found_before"] = merged["_last_hit_rank"].fillna(0).astype(int)
    merged["time_since_recent_find"] = np.where(
        merged["_last_hit_time"].notna(),
        merged["start_time"] - merged["_last_hit_time"],
        np.nan,
    )

    # same_cat/_ cats are indexed by `result`'s original (pre-sort) row index, and `merged` still carries
    # that same index (merge_asof reorders rows but doesn't reset it) - join on the index, not columns:
    # merged has no column in common with same_cat's single output column, so `on=<column intersection>`
    # (this function's original form) resolved to `on=[]`, which pandas 3.x raises on rather than silently
    # cross-joining.
    same_cat = _compute_same_category_found(result, hits_sorted, trial_key)
    merged = merged.merge(same_cat, left_index=True, right_index=True, how="left")

    _cats = _collect_prior_categories(result, hits_sorted, trial_key)
    merged = merged.merge(_cats, left_index=True, right_index=True, how="left")

    fix_sorted = fixations.sort_values(
        trial_key + ["eye", "end_time"]
    ).reset_index(drop=True)
    fix_sorted["_fix_cum"] = fix_sorted.groupby(trial_key + ["eye"]).cumcount() + 1

    fix_for_asof = fix_sorted[trial_key + ["eye", "end_time", "_fix_cum"]].rename(
        columns={"end_time": "_fix_end", "_fix_cum": "_fix_count"}
    ).sort_values("_fix_end")
    merged_fix = pd.merge_asof(
        merged.sort_values("start_time"),
        fix_for_asof,
        by=trial_key + ["eye"],
        left_on="start_time",
        right_on="_fix_end",
        direction="backward",
    )
    merged_fix["fixations_since_trial_start"] = merged_fix["_fix_count"].fillna(0).astype(int)

    merged_fix["fixations_since_last_hit"] = _compute_fixations_since_last_hit(
        merged_fix, fix_sorted, trial_key,
    )

    drop_cols = [c for c in merged_fix.columns if c.startswith("_")]
    merged_fix = merged_fix.drop(columns=drop_cols)

    return merged_fix.sort_values(
        trial_key + ["eye", "start_time"]
    ).reset_index(drop=True)


def _compute_same_category_found(
        funnel: pd.DataFrame,
        hits_sorted: pd.DataFrame,
        trial_key: list[str],
) -> pd.DataFrame:
    if "target_category" not in funnel.columns:
        return pd.DataFrame({"same_category_found_before": False}, index=funnel.index)

    keys = trial_key + ["target", "start_time", "target_category"]
    events = funnel[keys].copy()
    events["_idx"] = events.index

    cross = events.merge(
        hits_sorted[trial_key + ["time", "target_category"]].rename(
            columns={"target_category": "_hit_cat", "time": "_hit_time"}
        ),
        on=trial_key,
        how="left",
    )
    cross = cross[cross["_hit_time"] < cross["start_time"]]
    # `target_category` (funnel, ordered ImageCategoryEnum categorical) and `_hit_cat` (hits, built
    # separately from `data.targets`) are both "category" dtype but not the same categories object -
    # pandas refuses to compare two Categorical Series unless their categories match exactly. Compare as
    # strings instead of trying to align the two dtypes.
    cross["_same"] = cross["target_category"].astype(str) == cross["_hit_cat"].astype(str)
    any_same = cross.groupby("_idx")["_same"].any().rename("same_category_found_before")
    result = events[["_idx"]].copy()
    result = result.join(any_same, on="_idx")
    result["same_category_found_before"] = result["same_category_found_before"].fillna(False)
    return result.set_index("_idx")["same_category_found_before"].to_frame()


def _collect_prior_categories(
        funnel: pd.DataFrame,
        hits_sorted: pd.DataFrame,
        trial_key: list[str],
) -> pd.DataFrame:
    keys = trial_key + ["start_time"]
    events = funnel[keys].copy()
    events["_idx"] = events.index

    cross = events.merge(
        hits_sorted[trial_key + ["time", "target_category"]].rename(
            columns={"time": "_hit_time"}
        ),
        on=trial_key,
        how="left",
    )
    cross = cross[cross["_hit_time"] < cross["start_time"]]
    cats = (
        cross.groupby("_idx")["target_category"]
        .apply(list)
        .rename("target_categories_found_before")
    )
    result = events[["_idx"]].set_index("_idx")
    result = result.join(cats)
    result["target_categories_found_before"] = result["target_categories_found_before"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    return result


def _compute_fixations_since_last_hit(
        merged: pd.DataFrame,
        fix_sorted: pd.DataFrame,
        trial_key: list[str],
) -> pd.Series:
    has_hit = merged["_last_hit_time"].notna()
    result = pd.Series(0, index=merged.index, dtype=int)
    if not has_hit.any():
        return result

    subset = merged.loc[has_hit, trial_key + ["eye", "start_time", "_last_hit_time"]].copy()
    subset["_idx"] = subset.index

    fix_key = trial_key + ["eye"]
    joined = subset.merge(
        fix_sorted[fix_key + ["start_time", "end_time"]].rename(
            columns={"start_time": "_fs", "end_time": "_fe"}
        ),
        on=fix_key,
        how="inner",
    )
    valid = joined[(joined["_fs"] > joined["_last_hit_time"]) & (joined["_fe"] < joined["start_time"])]
    counts = valid.groupby("_idx").size()
    result.loc[counts.index] = counts.values
    return result


def build_hits_table(data: DataStore) -> pd.DataFrame:
    """
    One row per hit identification, with `target_category` attached. Shared so "what counts as a hit" (and
    which columns are dropped) can't drift between notebooks.
    """
    targets = data.targets
    hits = (
        data.identifications
        .loc[data.identifications["identification_category"] == "hit"]
        .drop(columns=[
            "identification_category", "left_x", "left_y", "left_pupil", "right_x", "right_y", "right_pupil",
        ], errors="ignore")
        .merge(
            targets[["subject", "trial", "target", "category"]], on=["subject", "trial", "target"], how="left",
        )
        .rename(columns={"category": "target_category"})
        .sort_values(["subject", "trial", "time"])
        .reset_index(drop=True)
    )
    return hits


def load_ssm_funnel(**load_kwargs) -> tuple[DataStore, pd.DataFrame, pd.DataFrame]:
    """
    Base table for every SSM/AB notebook: valid-trial LWS visits, enriched with hit history and
    `visit_type`. Single-target trials are kept - they always have `num_targets_found_before == 0`, so they
    can't inform a prior-hit *contrast*, but they're valid data for the "no prior hit" baseline itself, and
    dropping them would only shrink that baseline's precision for no benefit.

    :return: (DataStore, enriched visit funnel, hits table)
    """
    data, funnel_results = load_analysis_data(funnel_type="lws", event_type="visit", **load_kwargs)
    valid_trials = data.trial_funnel.loc[data.trial_funnel["is_valid_trial"], ["subject", "trial"]]
    funnel_results = funnel_results.merge(valid_trials, on=["subject", "trial"], how="inner")
    funnel_results = funnel_results.drop(columns=[
        "upto_before_identification", "upto_after_identification",
        "upto_not_close_to_trial_end", "upto_not_before_exemplar_visit",
    ], errors="ignore")
    funnel_results = funnel_results.merge(
        data.metadata[["subject", "trial", "num_targets"]], on=["subject", "trial"], how="left",
    )

    hits = build_hits_table(data)
    funnel_results = enrich_funnel_with_history(funnel_results, hits, data.fixations)
    funnel_results["visit_type"] = funnel_results.apply(lambda row: classify_visit(row, hits), axis=1)

    return data, funnel_results, hits


def add_ssm_predictors(df: pd.DataFrame, hits: pd.DataFrame) -> pd.DataFrame:
    """
    Adds `any_prior_hit` (Q1: was *any* other target already identified in this trial before this visit
    started) and `same_icon_as_last_hit` (Q5: is this visit's target the same icon as the most recently
    identified target).
    """
    result = df.copy()
    result["any_prior_hit"] = result["num_targets_found_before"] > 0

    # see enrich_funnel_with_history's comment: merge_asof needs the "on" column sorted globally, not
    # group-then-on, even with `by` given.
    hits_for_asof = hits[["subject", "trial", "time", "target"]].rename(
        columns={"time": "_last_hit_time", "target": "_last_hit_target"}
    ).sort_values("_last_hit_time")
    merged = pd.merge_asof(
        result.sort_values("start_time"),
        hits_for_asof,
        by=["subject", "trial"],
        left_on="start_time",
        right_on="_last_hit_time",
        direction="backward",
    )
    merged["same_icon_as_last_hit"] = np.where(
        merged["_last_hit_time"].notna(),
        merged["target"] == merged["_last_hit_target"],
        False,
    )
    return (
        merged.drop(columns=["_last_hit_time", "_last_hit_target"])
        .sort_values(["subject", "trial", "eye", "start_time"])
        .reset_index(drop=True)
    )


def join_single_prior_hit_category(df: pd.DataFrame, hits: pd.DataFrame) -> pd.DataFrame:
    """
    Q6's row-level join: visits with exactly one prior hit (`num_targets_found_before == 1`, so "the" most
    recent hit is unambiguous), with that hit's category attached as `hit_category` and the visit's own
    target category renamed `miss_category`. Shared by `build_category_pair_table` (the descriptive heatmap)
    and the Q6 model-data builder in `03_target_identity.ipynb`, so the join logic lives in one place.
    """
    single_prior = df.loc[df["num_targets_found_before"] == 1].copy()
    hits_for_asof = hits[["subject", "trial", "time", "target_category"]].rename(
        columns={"time": "_last_hit_time", "target_category": "hit_category"}
    ).sort_values("_last_hit_time")
    merged = pd.merge_asof(
        single_prior.sort_values("start_time"),
        hits_for_asof,
        by=["subject", "trial"],
        left_on="start_time",
        right_on="_last_hit_time",
        direction="backward",
    )
    return merged.rename(columns={"target_category": "miss_category"})


def build_category_pair_table(df: pd.DataFrame, hits: pd.DataFrame, min_cell_n: int = 10) -> pd.DataFrame:
    """
    Q6: one row per (hit_category, miss_category) pair, with visit count and raw LWS rate. `min_cell_n`
    (default 10, a judgment call - see the plan doc) flags which cells are dense enough to support
    inference vs. description-only via the `sufficient_n` column.
    """
    merged = join_single_prior_hit_category(df, hits)
    table = (
        merged.groupby(["hit_category", "miss_category"], observed=True)
        .agg(n_visits=("is_lws", "size"), n_lws=("is_lws", "sum"), lws_rate=("is_lws", "mean"))
        .reset_index()
    )
    table["sufficient_n"] = table["n_visits"] >= min_cell_n
    return table
