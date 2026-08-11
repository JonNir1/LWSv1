"""Identification classification: hit / repeated_hit / false_alarm / miss.

Covers the source half of CODE_REVIEW finding C4 - a false alarm must not carry a target label.
"""

import numpy as np
import pandas as pd
import pytest

import constants as cnst
from data_models.LWSEnums import SignalDetectionCategoryEnum as SDT
from pipeline.align.target_identifications import (
    _append_missed_targets,
    _classify_hits_and_false_alarms,
)

ON_TARGET_DVA = 1.75


def raw_idents(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    """Build the frame `_classify_hits_and_false_alarms` consumes, from (nearest_target, time, distance_dva)."""
    return pd.DataFrame(
        [{cnst.SUBJECT_STR: 1, cnst.TRIAL_STR: 1,
          cnst.TARGET_STR: tgt, cnst.TIME_STR: t, cnst.DISTANCE_DVA_STR: d} for tgt, t, d in rows]
    )


def target_ids(names: list[str]) -> np.ndarray:
    return np.array(names)


class TestClassification:
    def test_hit_within_threshold(self):
        out = _classify_hits_and_false_alarms(raw_idents([("target0", 1000.0, 0.5)]), ON_TARGET_DVA)
        assert out[cnst.IDENTIFICATION_CATEGORY_STR].tolist() == [SDT.HIT]
        assert out[cnst.TARGET_STR].tolist() == ["target0"]

    def test_repeated_hit_on_the_same_target(self):
        out = _classify_hits_and_false_alarms(
            raw_idents([("target0", 1000.0, 0.5), ("target0", 3000.0, 0.4)]), ON_TARGET_DVA
        )
        assert out[cnst.IDENTIFICATION_CATEGORY_STR].tolist() == [SDT.HIT, SDT.REPEATED_HIT]

    def test_false_alarm_carries_no_target(self):
        """C4: the nearest target is not an identification of it, so the label must be cleared."""
        out = _classify_hits_and_false_alarms(raw_idents([("target0", 500.0, 9.0)]), ON_TARGET_DVA)
        assert out[cnst.IDENTIFICATION_CATEGORY_STR].tolist() == [SDT.FALSE_ALARM]
        assert out[cnst.TARGET_STR].isna().all(), "a false alarm must not be attributed to a target"

    def test_false_alarm_keeps_its_distance(self):
        """How near the false alarm came is still worth recording."""
        out = _classify_hits_and_false_alarms(raw_idents([("target0", 500.0, 9.0)]), ON_TARGET_DVA)
        assert out[cnst.DISTANCE_DVA_STR].tolist() == [9.0]

    def test_false_alarm_before_a_hit_on_the_same_target(self):
        """The exact shape that produced the 12 mis-timed targets in the real data."""
        out = _classify_hits_and_false_alarms(
            raw_idents([("target0", 500.0, 9.0), ("target0", 4000.0, 0.5)]), ON_TARGET_DVA
        )
        labelled = out[out[cnst.TARGET_STR].notna()]
        assert labelled[cnst.TIME_STR].tolist() == [4000.0], "only the genuine hit may keep the target label"


class TestMissedTargets:
    def test_unidentified_targets_appended_with_infinite_time(self):
        idents = _classify_hits_and_false_alarms(raw_idents([("target0", 1000.0, 0.5)]), ON_TARGET_DVA)
        out = _append_missed_targets(idents, target_ids(["target0", "target1"]))
        missed = out[out[cnst.IDENTIFICATION_CATEGORY_STR] == SDT.MISS]
        assert missed[cnst.TARGET_STR].tolist() == ["target1"]
        assert missed[cnst.TIME_STR].tolist() == [np.inf]

    def test_false_alarm_does_not_suppress_a_miss(self):
        """C4 knock-on: nulling the FA target must not make target0 look identified."""
        idents = _classify_hits_and_false_alarms(raw_idents([("target0", 500.0, 9.0)]), ON_TARGET_DVA)
        out = _append_missed_targets(idents, target_ids(["target0"]))
        missed = out[out[cnst.IDENTIFICATION_CATEGORY_STR] == SDT.MISS]
        assert missed[cnst.TARGET_STR].tolist() == ["target0"], "an un-hit target must still be recorded as a miss"

    def test_every_target_is_represented(self):
        """The invariant `_map_ident_time` now relies on: no target may be missing from the identification table.

        A target may hold more than one row when it was hit repeatedly - `_identification_time_lookup` collapses
        those to the first hit - so this asserts coverage, not uniqueness.
        """
        idents = _classify_hits_and_false_alarms(
            raw_idents([
                ("target0", 500.0, 9.0),     # false alarm, target label cleared
                ("target0", 4000.0, 0.5),    # genuine hit
                ("target1", 6000.0, 0.3),    # hit
                ("target1", 7000.0, 0.3),    # repeated hit -> a second row for target1
            ]),
            ON_TARGET_DVA,
        )
        out = _append_missed_targets(idents, target_ids(["target0", "target1", "target2"]))
        labelled = out[out[cnst.TARGET_STR].notna()]
        assert set(labelled[cnst.TARGET_STR]) == {"target0", "target1", "target2"}, (
            "every target must appear as a hit or a miss"
        )
