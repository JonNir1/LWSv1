"""
Functional Visual Field (FVF) estimation.

The FVF is the radius around fixation from which a target can be detected and selected for foveation. It is the
input to an inspection-conditioned d' denominator (`CODE_REVIEW.md` T1), where the count of *plausibly inspected*
items replaces the current "every non-target icon".

Two estimators are provided, because the obvious formulation does not work.

**Why not `P(identified | min eccentricity from any fixation)`.** Marking a target requires foveating it - the
identification is classified as a hit only when gaze is within `ON_TARGET_THRESHOLD_DVA` of the target - so every
hit has minimum eccentricity below that threshold *by construction*. The curve degenerates into a step function at
the on-target threshold and recovers nothing about peripheral detection.

**A - foveation falloff** (`estimate_by_foveation_falloff`). `P(target was ever foveated | minimum distance from
any NON-on-target fixation)`. The outcome is foveation rather than identification, which removes the circularity,
and it separates the two constructs: this curve measures detection-and-selection, while `P(not identified |
foveated)` is the LWS rate.

**B - saccade-launch distance** (`estimate_by_launch_distance`). For each foveated target, the distance from the
*launch* fixation - the one immediately preceding the first on-target fixation - to that target. The subject
selected the target from that distance, so it is a direct sample of the field. FVF is a high percentile of the
distribution.

Neither is authoritative. `_determine_fvf.ipynb` compares them against each other and against
`ON_TARGET_THRESHOLD_DVA`; substantial disagreement is itself a finding.

**C - selection hazard** (`estimate_by_selection_hazard`). For every fixation at which a target was still
unfoveated, `P(the next saccade lands on that target | current distance to it)`. A discrete-time hazard rather than
a per-target outcome, so it does not aggregate over the trial.

**MEASURED ON THE LWS-v1 DATA (2026-08-06): USE C. A AND B BOTH FAIL, FOR OPPOSITE REASONS.**

*A is saturated.* Its predictor - the closest a subject came to a target without foveating it, over the whole trial
- has almost no spread: p50 2.04, p95 3.54, p99 4.93 DVA. With ~195 fixations per trial over a ~34 x 19 DVA array,
essentially every target is approached closely at some point regardless of whether it was detected. `P(foveated)`
descends only from 0.96 to 0.82 and never reaches half its asymptote, so the estimator returns NaN rather than the
edge of the data.

*B measures the wrong thing.* It returns ~13.4 DVA pooled - implausible on an array only ~19 DVA tall. The fixation
preceding a target's first on-target fixation is usually just wherever the subject was scanning, so B approximates
the 95th percentile of saccade amplitude rather than a detection radius.

*C works.* Pooled **4.34 DVA**, per subject 4.11-5.53 - a tight spread, against A's 2.76-4.87 and B's 10.3-16.6.
The hazard falls monotonically from 0.204 to 0.005 over 2.9-14.5 DVA, a 40x range, so the half-point is real rather
than censored. The estimate is 2.5x `ON_TARGET_THRESHOLD_DVA` (1.75), which is the expected relationship: the field
from which a target can be *detected* must exceed the radius within which gaze counts as *on* it, but stay the same
order of magnitude. All three estimators recover a known radius from synthetic data (true 4.0 -> A 3.9, B 3.8,
C 3.9), so the divergence is a property of the real scanpaths, not of the implementations.
"""

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import config as cnfg
import constants as cnst

_KEYS = [cnst.SUBJECT_STR, cnst.TRIAL_STR, cnst.TARGET_STR]
_POOLED = "all"


def target_distance_columns(fixations: pd.DataFrame) -> Dict[str, str]:
    """Map target identifier -> its DVA distance column in the fixations table (e.g. `icon92` -> `icon92_distance_dva`).

    The `icon` prefix is required, not just the suffix: `closest_icon_distance_dva` also ends in `_distance_dva`
    but is a single nearest-target distance, not a per-target one, and would enter the reshape as a phantom target.
    """
    suffix = f"_{cnst.DISTANCE_STR}_dva"
    return {
        col[: -len(suffix)]: col for col in fixations.columns
        if col.endswith(suffix) and col.startswith(cnst.ICON_STR)
    }


def per_target_distances(fixations: pd.DataFrame, on_target_threshold_dva: float) -> pd.DataFrame:
    """
    Reshape the wide per-target distance columns into one row per (subject, trial, target) fixation.

    :return: long-format frame with columns subject, trial, eye, event, start_time, target, distance_dva, on_target.
    """
    dist_cols = target_distance_columns(fixations)
    if not dist_cols:
        raise NotImplementedError(
            "no `*_distance_dva` columns found in the fixations table: they were removed from the persisted events "
            "table and are restored by the deferred `fixations_to_targets()` helper - see CODE_REVIEW.md. "
            "Note that all three estimators also need their 'preceding fixation' logic revisited, since the table "
            "now interleaves saccades and blinks between fixations."
        )
    keep = [c for c in [cnst.SUBJECT_STR, cnst.TRIAL_STR, cnst.EYE_STR, cnst.EVENT_STR, cnst.START_TIME_STR]
            if c in fixations.columns]
    long = fixations.melt(
        id_vars=keep, value_vars=list(dist_cols.values()),
        var_name="_col", value_name=cnst.DISTANCE_DVA_STR,
    )
    inverse = {col: tgt for tgt, col in dist_cols.items()}
    long[cnst.TARGET_STR] = long["_col"].map(inverse)
    long = long.drop(columns=["_col"]).dropna(subset=[cnst.DISTANCE_DVA_STR])
    long["on_target"] = long[cnst.DISTANCE_DVA_STR] <= on_target_threshold_dva
    return long


def estimate_by_foveation_falloff(
        fixations: pd.DataFrame,
        on_target_threshold_dva: float = cnfg.ON_TARGET_THRESHOLD_DVA,
        n_bins: int = 12,
        max_distance_dva: float = 15.0,
        falloff_fraction: float = 0.5,
) -> Tuple[pd.Series, float, pd.DataFrame]:
    """
    Estimator A. Fit `P(target ever foveated | min distance from any non-on-target fixation)` and read off the
    distance at which the curve falls to `falloff_fraction` of its near-fovea asymptote.

    :param n_bins: number of equal-count distance bins used to trace the curve.
    :param max_distance_dva: ignore approach distances beyond this (the array is only ~34 x 19 DVA).
    :param falloff_fraction: fraction of the near-fovea asymptote defining the field edge.

    :return: (per-subject FVF Series indexed by subject, pooled FVF, the binned curve for plotting).
    """
    long = per_target_distances(fixations, on_target_threshold_dva)
    # outcome: was this target ever foveated during the trial?
    foveated = long.groupby(_KEYS, observed=True)["on_target"].any().rename("foveated")
    # predictor: closest the subject came WITHOUT foveating it
    peripheral = long.loc[~long["on_target"]]
    approach = peripheral.groupby(_KEYS, observed=True)[cnst.DISTANCE_DVA_STR].min().rename("approach_dva")
    data = pd.concat([foveated, approach], axis=1).dropna()
    data = data[data["approach_dva"] <= max_distance_dva]

    curve = _binned_curve(data, "approach_dva", "foveated", n_bins)
    pooled = _falloff_point(curve, falloff_fraction)
    per_subject = {}
    for subj, grp in data.groupby(level=0, observed=True):
        if len(grp) < n_bins * 2:      # too few targets to trace a curve
            continue
        per_subject[subj] = _falloff_point(_binned_curve(grp, "approach_dva", "foveated", n_bins), falloff_fraction)
    return pd.Series(per_subject, name="fvf_dva").sort_index(), pooled, curve


def estimate_by_launch_distance(
        fixations: pd.DataFrame,
        on_target_threshold_dva: float = cnfg.ON_TARGET_THRESHOLD_DVA,
        percentile: float = 95.0,
) -> Tuple[pd.Series, float, pd.DataFrame]:
    """
    Estimator B. For each foveated target, the distance from the fixation immediately preceding its first on-target
    fixation. FVF is the `percentile` of that distribution.

    :return: (per-subject FVF Series indexed by subject, pooled FVF, the per-target launch distances).
    """
    long = per_target_distances(fixations, on_target_threshold_dva)
    if cnst.EVENT_STR not in long.columns:
        raise ValueError("fixations table must carry an `event` column to identify the launching fixation")
    long = long.sort_values([cnst.SUBJECT_STR, cnst.TRIAL_STR, cnst.EYE_STR, cnst.TARGET_STR, cnst.EVENT_STR])

    group_keys = [cnst.SUBJECT_STR, cnst.TRIAL_STR, cnst.EYE_STR, cnst.TARGET_STR]
    launches = []
    for keys, grp in long.groupby(group_keys, observed=True):
        on = grp["on_target"].to_numpy()
        if not on.any():
            continue                                    # never foveated - no launch to measure
        first = int(np.flatnonzero(on)[0])
        if first == 0:
            continue                                    # already on target at the first fixation; no launch fixation
        launches.append((*keys, float(grp[cnst.DISTANCE_DVA_STR].iloc[first - 1])))
    launch_df = pd.DataFrame(launches, columns=group_keys + ["launch_dva"])
    if launch_df.empty:
        raise ValueError("no launch fixations found - check the on-target threshold")

    pooled = float(np.percentile(launch_df["launch_dva"], percentile))
    per_subject = (
        launch_df.groupby(cnst.SUBJECT_STR, observed=True)["launch_dva"]
        .apply(lambda s: float(np.percentile(s, percentile)))
        .rename("fvf_dva")
    )
    return per_subject.sort_index(), pooled, launch_df


def selection_opportunities(
        fixations: pd.DataFrame,
        on_target_threshold_dva: float = cnfg.ON_TARGET_THRESHOLD_DVA,
) -> pd.DataFrame:
    """
    One row per (fixation, not-yet-foveated target) pair - the opportunities from which a target could be selected.

    For each (trial, eye, target), every fixation *before* the target's first on-target fixation is an opportunity;
    the last of them is the one the subject actually launched from, and is marked `selected`. Targets never
    foveated contribute opportunities that were all declined.

    This is the unit estimator C needs: it does not aggregate over the trial, so it cannot saturate the way A's
    "closest approach over the whole trial" does.

    :return: columns subject, trial, target, distance_dva, selected.
    """
    long = per_target_distances(fixations, on_target_threshold_dva)
    group_keys = [cnst.SUBJECT_STR, cnst.TRIAL_STR, cnst.EYE_STR, cnst.TARGET_STR]
    if cnst.EVENT_STR not in long.columns:
        raise ValueError("fixations table must carry an `event` column to order fixations within a trial")
    long = long.sort_values(group_keys + [cnst.EVENT_STR])

    parts = []
    for (subject, trial, _eye, target), grp in long.groupby(group_keys, observed=True):
        on = grp["on_target"].to_numpy()
        distances = grp[cnst.DISTANCE_DVA_STR].to_numpy()
        foveated_at = np.flatnonzero(on)
        first = int(foveated_at[0]) if foveated_at.size else len(on)
        if first == 0:
            continue                                    # on target from the first fixation; nothing was selected
        selected = np.zeros(first, dtype=bool)
        if foveated_at.size:
            selected[first - 1] = True                  # the launching fixation
        parts.append(pd.DataFrame({
            cnst.SUBJECT_STR: subject, cnst.TRIAL_STR: trial, cnst.TARGET_STR: target,
            cnst.DISTANCE_DVA_STR: distances[:first], "selected": selected,
        }))
    if not parts:
        raise ValueError("no selection opportunities found - check the on-target threshold")
    return pd.concat(parts, ignore_index=True)


def estimate_by_selection_hazard(
        fixations: pd.DataFrame,
        on_target_threshold_dva: float = cnfg.ON_TARGET_THRESHOLD_DVA,
        n_bins: int = 12,
        max_distance_dva: float = 15.0,
        hazard_fraction: float = 0.5,
) -> Tuple[pd.Series, float, pd.DataFrame]:
    """
    Estimator C. `P(the next saccade lands on this target | current distance to it)`, over every fixation at which
    the target was still unfoveated.

    A discrete-time hazard rather than a per-target outcome. Because most opportunities at any distance are
    declined, the curve has room to fall, which is what A lacks: A asks whether a target was *ever* approached
    closely, and with ~195 fixations per trial the answer is almost always yes.

    FVF is the distance at which the hazard falls to `hazard_fraction` of its near-fovea value.

    :return: (per-subject FVF Series, pooled FVF, the binned hazard curve).
    """
    opportunities = selection_opportunities(fixations, on_target_threshold_dva)
    opportunities = opportunities[opportunities[cnst.DISTANCE_DVA_STR] <= max_distance_dva]

    curve = _binned_curve(opportunities, cnst.DISTANCE_DVA_STR, "selected", n_bins)
    pooled = _falloff_point(curve, hazard_fraction)
    per_subject = {}
    for subject, grp in opportunities.groupby(cnst.SUBJECT_STR, observed=True):
        if len(grp) < n_bins * 10:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)     # per-subject censoring is reported via NaN
            per_subject[subject] = _falloff_point(
                _binned_curve(grp, cnst.DISTANCE_DVA_STR, "selected", n_bins), hazard_fraction
            )
    return pd.Series(per_subject, name="fvf_dva").sort_index(), pooled, curve


def estimate_fvf(
        fixations: pd.DataFrame,
        on_target_threshold_dva: float = cnfg.ON_TARGET_THRESHOLD_DVA,
) -> pd.DataFrame:
    """
    Run all three estimators and return a per-subject comparison table, with the pooled values in an `"all"` row.

    Columns: `foveation_falloff` (A), `launch_distance` (B), `selection_hazard` (C), and `on_target_threshold` for
    reference. On the LWS-v1 data only C is usable - see the module docstring for why A and B fail.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # A's censoring is expected here; it surfaces as NaN
        falloff_by_subject, falloff_pooled, _curve = estimate_by_foveation_falloff(
            fixations, on_target_threshold_dva=on_target_threshold_dva
        )
    launch_by_subject, launch_pooled, _launches = estimate_by_launch_distance(
        fixations, on_target_threshold_dva=on_target_threshold_dva
    )
    hazard_by_subject, hazard_pooled, _hazard_curve = estimate_by_selection_hazard(
        fixations, on_target_threshold_dva=on_target_threshold_dva
    )
    out = pd.concat(
        [
            falloff_by_subject.rename("foveation_falloff"),
            launch_by_subject.rename("launch_distance"),
            hazard_by_subject.rename("selection_hazard"),
        ],
        axis=1,
    )
    out.loc[_POOLED] = [falloff_pooled, launch_pooled, hazard_pooled]
    out["on_target_threshold"] = on_target_threshold_dva
    out.index.name = cnst.SUBJECT_STR
    return out


def _binned_curve(data: pd.DataFrame, predictor: str, outcome: str, n_bins: int) -> pd.DataFrame:
    """Equal-count binning of `predictor`, with the mean of `outcome` per bin."""
    n_bins = max(2, min(n_bins, len(data) // 2))
    bins = pd.qcut(data[predictor], q=n_bins, duplicates="drop")
    curve = (
        data.groupby(bins, observed=True)
        .agg(centre=(predictor, "median"), rate=(outcome, "mean"), n=(outcome, "size"))
        .reset_index(drop=True)
        .sort_values("centre")
    )
    return curve


def _falloff_point(curve: pd.DataFrame, fraction: float) -> float:
    """
    Distance at which `rate` first drops to `fraction` of the near-fovea asymptote, linearly interpolated.

    The asymptote is the rate in the nearest bin - which is below 1 whenever LWS occurs, and that is the point:
    the *level* carries the LWS rate, the *falloff* carries the field size.

    Returns NaN when the curve never falls that far within the observed range. Returning the last bin centre
    instead would look like an estimate while actually being the edge of the data - on the LWS-v1 fixation tables
    the curve only descends from ~0.96 to ~0.82, so every "estimate" would have been that artefact.
    """
    if curve.empty:
        return float("nan")
    asymptote = float(curve["rate"].iloc[0])
    target_rate = asymptote * fraction
    below = curve[curve["rate"] <= target_rate]
    if below.empty:
        warnings.warn(
            f"foveation-falloff curve never drops to {fraction:.0%} of its near-fovea rate "
            f"({asymptote:.3f}); it only reaches {curve['rate'].min():.3f} by "
            f"{curve['centre'].iloc[-1]:.2f} DVA. The predictor is saturated - see the module docstring. "
            f"Returning NaN rather than the edge of the observed range.",
            RuntimeWarning,
        )
        return float("nan")
    first = below.index[0]
    pos = curve.index.get_loc(first)
    if pos == 0:
        return float(curve["centre"].iloc[0])
    lo, hi = curve.iloc[pos - 1], curve.iloc[pos]
    if hi["rate"] == lo["rate"]:
        return float(hi["centre"])
    weight = (lo["rate"] - target_rate) / (lo["rate"] - hi["rate"])
    return float(lo["centre"] + weight * (hi["centre"] - lo["centre"]))
