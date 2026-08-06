"""Eye-movement detection and event feature computation.

Covers CODE_REVIEW findings M15 (`pixel_size` receives the viewer distance) and M16 (screen geometry comes from the
`peyes` defaults rather than the project's monitor).
"""

import warnings

import numpy as np
import pandas as pd
import peyes
import pytest

import constants as cnst
from data_models.LWSEnums import DominantEyeEnum
from data_models.parse.eye_movements import detect_eye_movements

VIEWER_DISTANCE_CM = 60.0
PIXEL_SIZE_CM = cnst.PIXEL_SIZE_CM
SACCADE_PX = 300.0


def synthetic_saccade_trace(amplitude_px: float = SACCADE_PX) -> pd.DataFrame:
    """500 ms fixation, a 30 ms horizontal saccade of `amplitude_px`, then a 500 ms fixation."""
    t = np.arange(0, 1030, 2.0)
    x = np.where(t < 500, 400.0, np.where(t < 530, 400.0 + (t - 500) * (amplitude_px / 30), 400.0 + amplitude_px))
    y = np.full_like(t, 400.0)
    return pd.DataFrame(
        {"time": t, "left_x": x, "left_y": y, "left_pupil": 3.0, "right_x": x, "right_y": y, "right_pupil": 3.0}
    )


def expected_amplitude_deg(amplitude_px: float, pixel_size_cm: float, viewer_distance_cm: float) -> float:
    return 2 * np.degrees(np.arctan(amplitude_px * pixel_size_cm / 2 / viewer_distance_cm))


def summarize(gaze: pd.DataFrame) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _labels, events = detect_eye_movements(
            gaze, DominantEyeEnum.LEFT, VIEWER_DISTANCE_CM, pixel_size_cm=PIXEL_SIZE_CM, only_labels=False
        )
        return peyes.summarize_events(events)


def test_segmentation_is_correct():
    """Segmentation comes from `detect()`, which always received the correct pixel size."""
    summary = summarize(synthetic_saccade_trace())
    assert summary["event_type"].tolist() == ["FIXATION", "SACCADE", "FIXATION"]


def test_saccade_amplitude_in_degrees():
    summary = summarize(synthetic_saccade_trace())
    saccade = summary[summary["event_type"] == "SACCADE"].iloc[0]
    expected = expected_amplitude_deg(SACCADE_PX, PIXEL_SIZE_CM, VIEWER_DISTANCE_CM)
    assert saccade["amplitude"] == pytest.approx(expected, rel=0.05), (
        f"expected ~{expected:.2f} deg, got {saccade['amplitude']:.2f} deg"
    )


def test_outlier_reasons_are_independent_of_pixel_size():
    """M15 is inert for current results because outlier_reasons is duration- and pixel-based only.

    Guards the claim: if peyes ever implements its velocity/dispersion TODO, this test starts failing and M15 stops
    being latent.
    """
    gaze = synthetic_saccade_trace()
    summary = summarize(gaze)
    assert summary["outlier_reasons"].map(len).sum() == 0, (
        "a clean synthetic trace produced outlier reasons; the criteria may no longer be purely duration/pixel based"
    )


class TestScreenGeometry:
    """M16: peyes' screen geometry must come from cnst.TOBII_MONITOR, not from peyes' own defaults."""

    def test_peyes_monitor_dimensions_come_from_the_project(self):
        from peyes._DataModels import config as peyes_cnfg

        monitor = peyes_cnfg.SCREEN_MONITOR
        assert monitor["width"] == pytest.approx(cnst.TOBII_MONITOR.width_mm / 10)
        assert monitor["height"] == pytest.approx(cnst.TOBII_MONITOR.height_mm / 10)

    def test_peyes_resolution_gates_the_outlier_check(self):
        """The one SCREEN_MONITOR field `get_outlier_reasons` reads, via its pixel_outside_screen criterion."""
        from peyes._DataModels import config as peyes_cnfg

        assert peyes_cnfg.SCREEN_MONITOR["resolution"] == (
            cnst.TOBII_MONITOR.width,
            cnst.TOBII_MONITOR.height,
        )

    def test_peyes_derives_pixel_size_by_a_different_convention(self):
        """Documents a real, deliberate divergence rather than asserting a match that cannot hold.

        `peyes` derives pixel size from the screen diagonal; `cnst.PIXEL_SIZE_CM` averages the two dimensions. They
        differ by ~0.16%. This does not affect the pipeline: every `detect` / `create_events` call passes
        `pixel_size_cm` explicitly, so peyes' own value is never consulted. The guard is here so that if a caller
        ever starts relying on the peyes default, the discrepancy is a documented fact rather than a surprise.
        """
        from peyes._DataModels import config as peyes_cnfg

        peyes_value = peyes_cnfg.SCREEN_MONITOR["pixel_size"]
        assert peyes_value != pytest.approx(PIXEL_SIZE_CM, rel=1e-6), "conventions unexpectedly converged"
        assert peyes_value == pytest.approx(PIXEL_SIZE_CM, rel=0.01), "divergence larger than the known ~0.16%"


class TestEventDurationConfig:
    """H7 / T3: the duration bounds must be the project's, not inherited peyes defaults."""

    @pytest.mark.parametrize(
        "event_type, min_key, max_key",
        [
            ("fixation", "FIXATION_MIN_DURATION_MS", "FIXATION_MAX_DURATION_MS"),
            ("saccade", "SACCADE_MIN_DURATION_MS", "SACCADE_MAX_DURATION_MS"),
        ],
    )
    def test_duration_bounds_come_from_config(self, event_type, min_key, max_key):
        import config as cnfg
        import peyes._utils.constants as peyes_cnst
        from peyes._DataModels import config as peyes_cnfg
        from peyes._DataModels.EventLabelEnum import EventLabelEnum

        mapping = peyes_cnfg.EVENT_MAPPING[EventLabelEnum[event_type.upper()]]
        assert mapping[peyes_cnst.MIN_DURATION_STR] == getattr(cnfg, min_key)
        assert mapping[peyes_cnst.MAX_DURATION_STR] == getattr(cnfg, max_key)

    def test_configure_peyes_is_idempotent(self):
        from peyes._DataModels import config as peyes_cnfg
        from data_models.parse.eye_movements import configure_peyes

        before = dict(peyes_cnfg.SCREEN_MONITOR)
        configure_peyes()
        configure_peyes()
        assert dict(peyes_cnfg.SCREEN_MONITOR) == before
