import numpy as np
import pandas as pd
import pytest

import constants as cnst
from pipeline.align.fixations_to_targets import fixations_to_targets


def _make_fixations(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _make_icons(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _make_metadata(subjects_distances: dict[int, float]) -> pd.DataFrame:
    rows = [
        {cnst.SUBJECT_STR: s, cnst.TRIAL_STR: 1, "screen_distance": d}
        for s, d in subjects_distances.items()
    ]
    return pd.DataFrame(rows)


class TestFixationsToTargets:

    @pytest.fixture
    def simple_data(self):
        fixations = _make_fixations([
            {cnst.SUBJECT_STR: 1, cnst.TRIAL_STR: 1, cnst.EYE_STR: "left", cnst.EVENT_STR: 0, cnst.X: 100.0, cnst.Y: 100.0},
            {cnst.SUBJECT_STR: 1, cnst.TRIAL_STR: 1, cnst.EYE_STR: "left", cnst.EVENT_STR: 2, cnst.X: 900.0, cnst.Y: 100.0},
        ])
        icons = _make_icons([
            {cnst.SUBJECT_STR: 1, cnst.TRIAL_STR: 1, cnst.ICON_STR: "icon7", cnst.X: 103.0, cnst.Y: 104.0, "is_target": True},
            {cnst.SUBJECT_STR: 1, cnst.TRIAL_STR: 1, cnst.ICON_STR: "icon92", cnst.X: 500.0, cnst.Y: 500.0, "is_target": True},
            {cnst.SUBJECT_STR: 1, cnst.TRIAL_STR: 1, cnst.ICON_STR: "icon50", cnst.X: 600.0, cnst.Y: 600.0, "is_target": False},
        ])
        metadata = _make_metadata({1: 61.4})
        return fixations, icons, metadata

    def test_output_shape(self, simple_data):
        fix, icons, meta = simple_data
        result = fixations_to_targets(fix, icons, meta)
        assert len(result) == 4  # 2 fixations x 2 targets

    def test_only_targets_appear(self, simple_data):
        fix, icons, meta = simple_data
        result = fixations_to_targets(fix, icons, meta)
        assert set(result[cnst.TARGET_STR]) == {"icon7", "icon92"}

    def test_distance_values(self, simple_data):
        fix, icons, meta = simple_data
        result = fixations_to_targets(fix, icons, meta)
        row = result.loc[
            (result[cnst.EVENT_STR] == 0) & (result[cnst.TARGET_STR] == "icon7")
        ].iloc[0]
        expected_px = np.sqrt(3**2 + 4**2)
        assert row[f"{cnst.DISTANCE_STR}_px"] == pytest.approx(expected_px)
        assert row[cnst.DISTANCE_DVA_STR] > 0

    def test_nan_fixations_excluded(self):
        fixations = _make_fixations([
            {cnst.SUBJECT_STR: 1, cnst.TRIAL_STR: 1, cnst.EYE_STR: "left", cnst.EVENT_STR: 0, cnst.X: np.nan, cnst.Y: np.nan},
        ])
        icons = _make_icons([
            {cnst.SUBJECT_STR: 1, cnst.TRIAL_STR: 1, cnst.ICON_STR: "icon7", cnst.X: 100.0, cnst.Y: 100.0, "is_target": True},
        ])
        metadata = _make_metadata({1: 61.4})
        result = fixations_to_targets(fixations, icons, metadata)
        assert len(result) == 0

    def test_empty_fixations(self):
        fixations = _make_fixations([])
        icons = _make_icons([
            {cnst.SUBJECT_STR: 1, cnst.TRIAL_STR: 1, cnst.ICON_STR: "icon7", cnst.X: 100.0, cnst.Y: 100.0, "is_target": True},
        ])
        metadata = _make_metadata({1: 61.4})
        result = fixations_to_targets(fixations, icons, metadata)
        assert len(result) == 0

    def test_dva_uses_per_subject_screen_distance(self):
        fixations = _make_fixations([
            {cnst.SUBJECT_STR: 1, cnst.TRIAL_STR: 1, cnst.EYE_STR: "left", cnst.EVENT_STR: 0, cnst.X: 0.0, cnst.Y: 0.0},
            {cnst.SUBJECT_STR: 2, cnst.TRIAL_STR: 1, cnst.EYE_STR: "left", cnst.EVENT_STR: 0, cnst.X: 0.0, cnst.Y: 0.0},
        ])
        icons = _make_icons([
            {cnst.SUBJECT_STR: 1, cnst.TRIAL_STR: 1, cnst.ICON_STR: "icon0", cnst.X: 100.0, cnst.Y: 0.0, "is_target": True},
            {cnst.SUBJECT_STR: 2, cnst.TRIAL_STR: 1, cnst.ICON_STR: "icon0", cnst.X: 100.0, cnst.Y: 0.0, "is_target": True},
        ])
        metadata = _make_metadata({1: 50.0, 2: 70.0})
        result = fixations_to_targets(fixations, icons, metadata)
        dva_s1 = result.loc[result[cnst.SUBJECT_STR] == 1, cnst.DISTANCE_DVA_STR].iloc[0]
        dva_s2 = result.loc[result[cnst.SUBJECT_STR] == 2, cnst.DISTANCE_DVA_STR].iloc[0]
        assert dva_s1 > dva_s2  # closer screen = larger DVA per pixel
