import numpy as np
import pandas as pd
import pytest

import constants as cnst
from pipeline.align.build_visits import build_visits


def _fixations(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _dists(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


THRESHOLD_DVA = 1.75
MERGE_MS = 100.0


class TestBuildVisits:

    def _make_simple(self):
        fixations = _fixations([
            {cnst.SUBJECT_STR: 1, cnst.TRIAL_STR: 1, cnst.EYE_STR: "left", cnst.EVENT_STR: 0,
             cnst.START_TIME_STR: 0.0, cnst.END_TIME_STR: 50.0, "duration": 50.0,
             "to_trial_end": 950.0, cnst.X: 100.0, cnst.Y: 100.0, "num_fixs_to_strip": np.inf},
            {cnst.SUBJECT_STR: 1, cnst.TRIAL_STR: 1, cnst.EYE_STR: "left", cnst.EVENT_STR: 2,
             cnst.START_TIME_STR: 60.0, cnst.END_TIME_STR: 110.0, "duration": 50.0,
             "to_trial_end": 890.0, cnst.X: 105.0, cnst.Y: 102.0, "num_fixs_to_strip": np.inf},
            {cnst.SUBJECT_STR: 1, cnst.TRIAL_STR: 1, cnst.EYE_STR: "left", cnst.EVENT_STR: 4,
             cnst.START_TIME_STR: 500.0, cnst.END_TIME_STR: 550.0, "duration": 50.0,
             "to_trial_end": 450.0, cnst.X: 800.0, cnst.Y: 800.0, "num_fixs_to_strip": np.inf},
        ])
        dists = _dists([
            {cnst.SUBJECT_STR: 1, cnst.TRIAL_STR: 1, cnst.EYE_STR: "left", cnst.EVENT_STR: 0,
             cnst.TARGET_STR: "icon7", "distance_px": 10.0, cnst.DISTANCE_DVA_STR: 0.5},
            {cnst.SUBJECT_STR: 1, cnst.TRIAL_STR: 1, cnst.EYE_STR: "left", cnst.EVENT_STR: 2,
             cnst.TARGET_STR: "icon7", "distance_px": 12.0, cnst.DISTANCE_DVA_STR: 0.6},
            {cnst.SUBJECT_STR: 1, cnst.TRIAL_STR: 1, cnst.EYE_STR: "left", cnst.EVENT_STR: 4,
             cnst.TARGET_STR: "icon7", "distance_px": 200.0, cnst.DISTANCE_DVA_STR: 10.0},
        ])
        return fixations, dists

    def test_groups_consecutive_on_target_fixations(self):
        fixations, dists = self._make_simple()
        visits = build_visits(fixations, dists, THRESHOLD_DVA, MERGE_MS)
        assert len(visits) == 1
        assert visits.iloc[0][cnst.TARGET_STR] == "icon7"
        assert visits.iloc[0]["num_fixations"] == 2

    def test_visit_timing(self):
        fixations, dists = self._make_simple()
        visits = build_visits(fixations, dists, THRESHOLD_DVA, MERGE_MS)
        v = visits.iloc[0]
        assert v[cnst.START_TIME_STR] == 0.0
        assert v[cnst.END_TIME_STR] == 110.0
        assert v["duration"] == 110.0

    def test_temporal_gap_splits_visits(self):
        fixations = _fixations([
            {cnst.SUBJECT_STR: 1, cnst.TRIAL_STR: 1, cnst.EYE_STR: "left", cnst.EVENT_STR: 0,
             cnst.START_TIME_STR: 0.0, cnst.END_TIME_STR: 50.0, "duration": 50.0,
             "to_trial_end": 950.0, cnst.X: 100.0, cnst.Y: 100.0, "num_fixs_to_strip": np.inf},
            {cnst.SUBJECT_STR: 1, cnst.TRIAL_STR: 1, cnst.EYE_STR: "left", cnst.EVENT_STR: 2,
             cnst.START_TIME_STR: 200.0, cnst.END_TIME_STR: 250.0, "duration": 50.0,
             "to_trial_end": 750.0, cnst.X: 105.0, cnst.Y: 102.0, "num_fixs_to_strip": np.inf},
        ])
        dists = _dists([
            {cnst.SUBJECT_STR: 1, cnst.TRIAL_STR: 1, cnst.EYE_STR: "left", cnst.EVENT_STR: 0,
             cnst.TARGET_STR: "icon7", "distance_px": 10.0, cnst.DISTANCE_DVA_STR: 0.5},
            {cnst.SUBJECT_STR: 1, cnst.TRIAL_STR: 1, cnst.EYE_STR: "left", cnst.EVENT_STR: 2,
             cnst.TARGET_STR: "icon7", "distance_px": 12.0, cnst.DISTANCE_DVA_STR: 0.6},
        ])
        visits = build_visits(fixations, dists, THRESHOLD_DVA, MERGE_MS)
        assert len(visits) == 2

    def test_no_on_target_fixations(self):
        fixations = _fixations([
            {cnst.SUBJECT_STR: 1, cnst.TRIAL_STR: 1, cnst.EYE_STR: "left", cnst.EVENT_STR: 0,
             cnst.START_TIME_STR: 0.0, cnst.END_TIME_STR: 50.0, "duration": 50.0,
             "to_trial_end": 950.0, cnst.X: 800.0, cnst.Y: 800.0, "num_fixs_to_strip": np.inf},
        ])
        dists = _dists([
            {cnst.SUBJECT_STR: 1, cnst.TRIAL_STR: 1, cnst.EYE_STR: "left", cnst.EVENT_STR: 0,
             cnst.TARGET_STR: "icon7", "distance_px": 500.0, cnst.DISTANCE_DVA_STR: 25.0},
        ])
        visits = build_visits(fixations, dists, THRESHOLD_DVA, MERGE_MS)
        assert len(visits) == 0

    def test_weighted_center(self):
        fixations = _fixations([
            {cnst.SUBJECT_STR: 1, cnst.TRIAL_STR: 1, cnst.EYE_STR: "left", cnst.EVENT_STR: 0,
             cnst.START_TIME_STR: 0.0, cnst.END_TIME_STR: 50.0, "duration": 100.0,
             "to_trial_end": 950.0, cnst.X: 100.0, cnst.Y: 200.0, "num_fixs_to_strip": np.inf},
            {cnst.SUBJECT_STR: 1, cnst.TRIAL_STR: 1, cnst.EYE_STR: "left", cnst.EVENT_STR: 2,
             cnst.START_TIME_STR: 60.0, cnst.END_TIME_STR: 110.0, "duration": 100.0,
             "to_trial_end": 890.0, cnst.X: 200.0, cnst.Y: 400.0, "num_fixs_to_strip": np.inf},
        ])
        dists = _dists([
            {cnst.SUBJECT_STR: 1, cnst.TRIAL_STR: 1, cnst.EYE_STR: "left", cnst.EVENT_STR: 0,
             cnst.TARGET_STR: "icon7", "distance_px": 5.0, cnst.DISTANCE_DVA_STR: 0.25},
            {cnst.SUBJECT_STR: 1, cnst.TRIAL_STR: 1, cnst.EYE_STR: "left", cnst.EVENT_STR: 2,
             cnst.TARGET_STR: "icon7", "distance_px": 5.0, cnst.DISTANCE_DVA_STR: 0.25},
        ])
        visits = build_visits(fixations, dists, THRESHOLD_DVA, MERGE_MS)
        v = visits.iloc[0]
        assert v[cnst.X] == pytest.approx(150.0)
        assert v[cnst.Y] == pytest.approx(300.0)
