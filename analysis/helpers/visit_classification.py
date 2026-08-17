from enum import Enum, auto

import numpy as np
import pandas as pd


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


def get_target_history(event_row: pd.Series, hits: pd.DataFrame) -> pd.Series:
    prior_hits = hits[
        (hits["subject"] == event_row["subject"])
        & (hits["trial"] == event_row["trial"])
        & (hits["time"] < event_row["start_time"])
    ]
    time_since_id = np.nan
    if not prior_hits.empty:
        time_since_id = event_row["start_time"] - prior_hits["time"].max()
    return pd.Series({
        "num_targets_found_before": len(prior_hits),
        "target_categories_found_before": prior_hits["target_category"].tolist(),
        "time_since_recent_find": time_since_id,
    })


def count_fixations_from_trial_onset(
        event_row: pd.Series, fixations: pd.DataFrame,
) -> int:
    return int(fixations.loc[
        (fixations["subject"] == event_row["subject"])
        & (fixations["trial"] == event_row["trial"])
        & (fixations["eye"] == event_row["eye"])
        & (fixations["end_time"] < event_row["start_time"])
    ].shape[0])


def count_fixations_since_last_hit(
        event_row: pd.Series, hits: pd.DataFrame, fixations: pd.DataFrame,
) -> int:
    prior_hits = hits.loc[
        (hits["subject"] == event_row["subject"])
        & (hits["trial"] == event_row["trial"])
        & (hits["time"] < event_row["start_time"])
    ]
    if prior_hits.empty:
        return 0
    last_hit_time = prior_hits["time"].max()
    return int(fixations.loc[
        (fixations["subject"] == event_row["subject"])
        & (fixations["trial"] == event_row["trial"])
        & (fixations["eye"] == event_row["eye"])
        & (fixations["start_time"] > last_hit_time)
        & (fixations["end_time"] < event_row["start_time"])
    ].shape[0])


def enrich_funnel_with_history(
        funnel: pd.DataFrame,
        hits: pd.DataFrame,
        fixations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Vectorized enrichment: adds num_targets_found_before, time_since_recent_find,
    same_category_found_before, fixations_since_trial_start, fixations_since_last_hit.
    """
    result = funnel.copy()
    trial_key = ["subject", "trial"]

    hits_sorted = hits.sort_values(trial_key + ["time"]).reset_index(drop=True)
    hits_sorted["_hit_rank"] = hits_sorted.groupby(trial_key).cumcount() + 1

    merged = pd.merge_asof(
        result.sort_values(trial_key + ["start_time"]),
        hits_sorted[trial_key + ["time", "_hit_rank", "target_category"]].rename(
            columns={"time": "_last_hit_time", "_hit_rank": "_last_hit_rank",
                      "target_category": "_last_hit_cat"}
        ),
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

    same_cat = _compute_same_category_found(result, hits_sorted, trial_key)
    merged = merged.merge(same_cat, on=merged.columns.intersection(same_cat.columns).tolist(), how="left")

    _cats = _collect_prior_categories(result, hits_sorted, trial_key)
    merged = merged.merge(_cats, left_index=True, right_index=True, how="left")

    fix_sorted = fixations.sort_values(
        trial_key + ["eye", "end_time"]
    ).reset_index(drop=True)
    fix_sorted["_fix_cum"] = fix_sorted.groupby(trial_key + ["eye"]).cumcount() + 1

    fix_for_asof = fix_sorted[trial_key + ["eye", "end_time", "_fix_cum"]].rename(
        columns={"end_time": "_fix_end", "_fix_cum": "_fix_count"}
    )
    merged_fix = pd.merge_asof(
        merged.sort_values(trial_key + ["eye", "start_time"]),
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
    cross["_same"] = cross["target_category"] == cross["_hit_cat"]
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
