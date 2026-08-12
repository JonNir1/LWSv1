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
from analysis.helpers.read_data import load_data
from data_models.LWSEnums import SignalDetectionCategoryEnum

pytestmark = pytest.mark.realdata

# the peyes default that flags a fixation as an outlier (see H7); the project never sets it
PEYES_FIXATION_MAX_DURATION_MS = 2500


def test_h7_long_fixation_cap_is_immaterial_in_this_dataset(loaded, capsys):
    """H7: measure the reach of the inherited 2500 ms cap.

    Measured 2026-08-06: 6 of 116,947 fixations (0.005%) exceed the cap, max observed duration 2797 ms. All 6 are
    on-target against a 10.0% base rate - the predicted enrichment direction, but far too few to move any result.
    This test pins the magnitude so a change in the detector or the threshold shows up.
    """
    fixs = loaded.fixations
    too_long = fixs["duration"] > PEYES_FIXATION_MAX_DURATION_MS
    flagged = fixs["outlier_reasons"].map(lambda r: isinstance(r, list) and "max_duration" in r)

    # TODO: restore on-target enrichment check once fixations_to_targets() lands (step 5)

    with capsys.disabled():
        print(f"\n--- H7: fixation duration cap ({PEYES_FIXATION_MAX_DURATION_MS} ms) ---")
        print(f"fixations total                : {len(fixs):,}")
        print(f"duration > cap                 : {too_long.sum():,} ({100 * too_long.mean():.3f}%)")
        print(f"flagged 'max_duration'         : {flagged.sum():,}")
        print(f"duration percentiles (ms)      : "
              f"p50={fixs['duration'].quantile(.50):.0f}  p95={fixs['duration'].quantile(.95):.0f}  "
              f"p99={fixs['duration'].quantile(.99):.0f}  max={fixs['duration'].max():.0f}")

    # the cap and the flag must agree - if they diverge, the flag is being set by something else
    assert flagged.sum() == too_long.sum(), "outlier flag disagrees with the duration cap"
    # magnitude guard: if this ever climbs, T3 stops being cosmetic
    assert too_long.mean() < 0.001, (
        f"{100 * too_long.mean():.3f}% of fixations now exceed the inherited cap - revisit T3"
    )


@pytest.mark.xfail(
    strict=True,
    reason="C4: FIXED IN CODE, but these pickles predate the fix. Measured 2026-08-06 on the stale build - 12 "
    "targets take their identification time from a preceding false alarm, truncating the LWS window by a median "
    "of 3684 ms (max 12904 ms). Remove this marker after re-running the pipeline (needs SEARCH_ARRAY_PATH).",
)
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


def test_drop_outliers_reaches_the_event_table(output_dir, capsys):
    """`drop_outliers` must actually remove rows - it is the only quality filter read_data applies."""
    kept = load_data(output_dir, drop_bad_eye=False, drop_outliers=False, missing="raise")
    dropped = load_data(output_dir, drop_bad_eye=False, drop_outliers=True, missing="raise")

    with capsys.disabled():
        print("\n--- drop_outliers coverage ---")
        for label, k, d in [
            ("all events", kept.eye_movements, dropped.eye_movements),
            ("fixations", kept.fixations, dropped.fixations),
        ]:
            print(f"{label:12s} {len(k):,} -> {len(d):,} ({len(k) - len(d):,} dropped)")

    assert len(dropped.fixations) < len(kept.fixations), "no fixations dropped - is the flag reaching read_data?"
    # the filter now reaches saccades and blinks too, which fixations-only filtering could not
    assert (len(kept.eye_movements) - len(dropped.eye_movements)) > (len(kept.fixations) - len(dropped.fixations))


# --- broken by the eye_movements refactor, pending `fixations_to_targets()` --------------------------------------
# These are `strict` so they turn red - and the markers come off - the moment the helper lands.

_PENDING_DISTANCES = pytest.mark.xfail(
    strict=True, raises=NotImplementedError,
    reason="the per-target distance columns were removed from the events table; restored by the deferred "
           "`fixations_to_targets()` helper - see CODE_REVIEW.md",
)


@_PENDING_DISTANCES
def test_h2_visit_funnel_with_outliers_dropped(data_store):
    """H2: visit-level outlier exclusion (via load_data) produces a non-empty funnel."""
    from pipeline.stage3_classify.build_funnels import build_event_classification_funnel

    funnel = build_event_classification_funnel(data_store, "lws", "visit", exclude="invalid_trials")
    assert len(funnel) > 0


@_PENDING_DISTANCES
def test_h2_fixation_funnel_with_outliers_dropped(data_store):
    """Fixation-level funnel with outliers dropped (via load_data) works."""
    from pipeline.stage3_classify.build_funnels import build_event_classification_funnel

    funnel = build_event_classification_funnel(data_store, "lws", "fixation", exclude="invalid_trials")
    assert len(funnel) > 0


class TestEventTableInvariants:
    """The properties `eye_movements.pkl` must hold, checked against the real build rather than a fixture."""

    FIXATION_ONLY = ["x", "y", "num_fixs_to_strip"]

    def test_the_key_is_unique(self, loaded):
        assert not loaded.eye_movements.duplicated(subset=["subject", "trial", "eye", "event"]).any()

    def test_saccades_and_blinks_survived_to_disk(self, loaded, capsys):
        counts = loaded.eye_movements["event_type"].value_counts()
        with capsys.disabled():
            print(f"\n--- event table composition ---\n{counts.to_string()}")
        assert {"FIXATION", "SACCADE"}.issubset(set(counts.index))

    def test_fixation_only_columns_are_null_elsewhere(self, loaded):
        """A saccade's `center_pixel` is a point the eye crossed, not one it held; writing it into `x`/`y` would
        make every downstream consumer treat it as a gaze position."""
        others = loaded.eye_movements.loc[loaded.eye_movements["event_type"] != "FIXATION"]
        populated = [c for c in self.FIXATION_ONLY if others[c].notna().any()]
        assert not populated, f"non-fixation events carry {populated}"

    def test_fixation_only_columns_are_populated_for_fixations(self, loaded):
        empty = [c for c in self.FIXATION_ONLY if loaded.fixations[c].isna().all()]
        assert not empty, f"fixations are missing {empty}"

    def test_is_outlier_agrees_with_the_reason_list(self, loaded):
        """`read_data`'s filter switched from the list to the bool; they must not be able to disagree."""
        events = loaded.eye_movements
        from_list = events["outlier_reasons"].map(lambda r: isinstance(r, (list, tuple)) and len(r) > 0)
        assert (events["is_outlier"].astype(bool) == from_list).all()

    def test_event_ids_are_contiguous_within_an_eye(self, loaded):
        """The gaps the old fixations table had were the removed saccades. Retaining them closes the gaps, which is
        what makes `event +- 1` the true temporal neighbour."""
        spans = (
            loaded.eye_movements
            .groupby(["subject", "trial", "eye"], observed=True)["event"]
            .agg(["min", "max", "count"])
        )
        assert (spans["min"] == 0).all()
        assert (spans["max"] - spans["min"] + 1 == spans["count"]).all()

    def test_saccades_carry_their_endpoints(self, loaded):
        """`peyes`' `summary()` omits `start_pixel`/`end_pixel`, so they are read off the Event objects. Without
        them a saccade's landing site cannot be recovered - amplitude and azimuth give magnitude and direction only."""
        saccades = loaded.eye_movements.loc[loaded.eye_movements["event_type"] == "SACCADE"]
        assert saccades[["start_x", "start_y", "end_x", "end_y"]].notna().any().all()


@pytest.mark.xfail(
    strict=True,
    reason="M1: measured 2026-08-06 - every numeric metadata column is object dtype after the concat/.T round-trip",
)
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
