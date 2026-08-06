"""Empirical checks of CODE_REVIEW findings against the built pickles.

These quantify how often each finding actually bites in this dataset, which is what decides its severity. They skip
when `cnfg.OUTPUT_PATH` is absent, and they print their measurements so the numbers land in the test log.

Run with output visible:
    pytest tests/test_realdata_findings.py -s -m realdata
"""

import numpy as np
import pandas as pd
import pytest

import config as cnfg
from analysis.helpers.read_data import read_data
from data_models.LWSEnums import SignalDetectionCategoryEnum

pytestmark = pytest.mark.realdata

# the peyes default that flags a fixation as an outlier (see H7); the project never sets it
PEYES_FIXATION_MAX_DURATION_MS = 2500


def test_h7_long_fixations_dropped_as_outliers(loaded, capsys):
    """H7: how many fixations does the inherited 2500 ms cap remove, and are on-target ones over-represented?"""
    fixs = loaded.fixations
    too_long = fixs["duration"] > PEYES_FIXATION_MAX_DURATION_MS
    flagged = fixs["outlier_reasons"].map(lambda r: isinstance(r, list) and "max_duration" in r)

    dist_cols = [c for c in fixs.columns if c.startswith("target") and c.endswith("distance_dva")]
    on_target = fixs[dist_cols].le(cnfg.ON_TARGET_THRESHOLD_DVA).any(axis=1)

    with capsys.disabled():
        print(f"\n--- H7: fixation duration cap ({PEYES_FIXATION_MAX_DURATION_MS} ms) ---")
        print(f"fixations total                : {len(fixs):,}")
        print(f"duration > cap                 : {too_long.sum():,} ({100 * too_long.mean():.3f}%)")
        print(f"flagged 'max_duration'         : {flagged.sum():,}")
        print(f"duration percentiles (ms)      : "
              f"p50={fixs['duration'].quantile(.50):.0f}  p95={fixs['duration'].quantile(.95):.0f}  "
              f"p99={fixs['duration'].quantile(.99):.0f}  max={fixs['duration'].max():.0f}")
        if too_long.any():
            print(f"of those over the cap, on-target: {on_target[too_long].sum():,} "
                  f"({100 * on_target[too_long].mean():.1f}%)")
        print(f"on-target rate, all fixations  : {100 * on_target.mean():.1f}%")

    # the cap and the flag must agree - if they diverge, the flag is being set by something else
    assert flagged.sum() == too_long.sum(), "outlier flag disagrees with the duration cap"


def test_c4_false_alarms_shadowing_hits(loaded, capsys):
    """C4: count targets whose earliest identification row is a false alarm that precedes a genuine hit."""
    idents = loaded.identifications
    cat = idents[cnfg.IDENTIFICATION_CATEGORY_STR]

    hits = (
        idents[cat.isin([SignalDetectionCategoryEnum.HIT, SignalDetectionCategoryEnum.REPEATED_HIT])]
        .groupby(["subject", "trial", "target"], observed=True)["time"].min()
    )
    fas = (
        idents[cat == SignalDetectionCategoryEnum.FALSE_ALARM]
        .groupby(["subject", "trial", "target"], observed=True)["time"].min()
    )
    both = pd.concat([hits.rename("hit"), fas.rename("fa")], axis=1).dropna()
    shadowed = both[both["fa"] < both["hit"]]

    with capsys.disabled():
        print("\n--- C4: false alarms shadowing real hits ---")
        print(f"identification rows            : {len(idents):,}")
        print(f"false-alarm rows               : {(cat == SignalDetectionCategoryEnum.FALSE_ALARM).sum():,}")
        print(f"FA rows carrying a target label: "
              f"{idents.loc[cat == SignalDetectionCategoryEnum.FALSE_ALARM, 'target'].notna().sum():,}")
        print(f"targets with both an FA and hit: {len(both):,}")
        print(f"  ... where the FA comes first : {len(shadowed):,}  <- mis-timed identifications")
        if len(shadowed):
            delta = (shadowed["hit"] - shadowed["fa"])
            print(f"  median truncation of the LWS window: {delta.median():.0f} ms "
                  f"(max {delta.max():.0f} ms)")

    assert len(shadowed) == 0, (
        f"{len(shadowed)} target(s) take their identification time from a preceding false alarm"
    )


def test_h2_outlier_exclusion_is_a_noop_for_visits(output_dir, loaded, capsys):
    """H2: `drop_outliers` must change the visit table, not only the fixation table."""
    kept = read_data(output_dir, drop_bad_eye=False, drop_outliers=False, missing="raise")
    dropped = read_data(output_dir, drop_bad_eye=False, drop_outliers=True, missing="raise")

    with capsys.disabled():
        print("\n--- H2: drop_outliers coverage ---")
        print(f"fixations  {len(kept.fixations):,} -> {len(dropped.fixations):,} "
              f"({len(kept.fixations) - len(dropped.fixations):,} dropped)")
        print(f"visits     {len(kept.visits):,} -> {len(dropped.visits):,} "
              f"({len(kept.visits) - len(dropped.visits):,} dropped)")

    assert len(dropped.fixations) < len(kept.fixations), "no fixations dropped - is the flag reaching read_data?"
    assert len(dropped.visits) < len(kept.visits), "drop_outliers had no effect on the visit table"


def test_m1_metadata_is_object_dtype(loaded, capsys):
    """M1: metadata columns come back as object dtype after the concat/transpose round-trip."""
    numeric_cols = ["duration", "num_targets", "num_distractors", "gaze_coverage", "num_actions"]
    present = [c for c in numeric_cols if c in loaded.metadata.columns]
    dtypes = {c: str(loaded.metadata[c].dtype) for c in present}

    with capsys.disabled():
        print("\n--- M1: metadata dtypes ---")
        print(f"  {dtypes}")

    non_numeric = [c for c, d in dtypes.items() if d == "object"]
    assert not non_numeric, f"expected numeric dtypes, got object for: {non_numeric}"
