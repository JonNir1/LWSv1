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
PIXEL_SIZE_CM = cnst.PIXEL_SIZE_MM / 10
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
    """M15 does not touch `detect()`, so the fixation/saccade structure must be right regardless."""
    summary = summarize(synthetic_saccade_trace())
    assert summary["event_type"].tolist() == ["FIXATION", "SACCADE", "FIXATION"]


@pytest.mark.xfail(
    strict=True,
    reason="M15: create_events is passed pixel_size=viewer_distance_cm, inflating every visual-angle feature. "
    "A 300 px saccade reports ~179 deg instead of ~7.9 deg",
)
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
    @pytest.mark.xfail(
        strict=True,
        reason="M16: the project never calls peyes.set_screen_monitor, so peyes keeps its own default monitor "
        "(53.1 cm wide) rather than cnst.TOBII_MONITOR (53.0 cm)",
    )
    def test_peyes_monitor_matches_the_project_monitor(self):
        from peyes._DataModels import config as peyes_cnfg

        monitor = peyes_cnfg.SCREEN_MONITOR
        assert monitor["width"] == pytest.approx(cnst.TOBII_MONITOR.width_mm / 10)
        assert monitor["height"] == pytest.approx(cnst.TOBII_MONITOR.height_mm / 10)
        assert monitor["pixel_size"] == pytest.approx(PIXEL_SIZE_CM)

    def test_peyes_resolution_matches_and_gates_the_outlier_check(self):
        """The one SCREEN_MONITOR field the outlier check reads. Currently correct by coincidence of defaults."""
        from peyes._DataModels import config as peyes_cnfg

        assert peyes_cnfg.SCREEN_MONITOR["resolution"] == (
            cnst.TOBII_MONITOR.width,
            cnst.TOBII_MONITOR.height,
        )


class TestEventDurationConfig:
    @pytest.mark.xfail(
        strict=True,
        reason="H7: the project sets only min_duration, so fixation max_duration keeps the peyes default of "
        "2500 ms and silently flags longer fixations as outliers",
    )
    def test_fixation_max_duration_is_set_by_the_project(self):
        from peyes._DataModels import config as peyes_cnfg
        from peyes._DataModels.EventLabelEnum import EventLabelEnum
        import peyes._utils.constants as peyes_cnst

        fixation = peyes_cnfg.EVENT_MAPPING[EventLabelEnum.FIXATION]
        assert fixation[peyes_cnst.MAX_DURATION_STR] != 2500, (
            "fixation max_duration is still the peyes default; it has never been chosen for this paradigm (T3)"
        )
