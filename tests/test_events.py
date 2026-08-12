"""Event preprocessing: the persisted eye-movement table.

Two things are under test. First, CODE_REVIEW finding H1 - `_num_fixations_to_strip` scans a frame holding both
eyes' events back-to-back, so the count can run off the end of one eye's sequence and into the other's. Second,
the properties the events table must hold now that it is a *superset* of the old fixations table: the fixation
subset has to be exactly what `fixations.pkl` used to be, and non-fixations must not carry a fabricated position.
"""

import numpy as np
import pandas as pd
import pytest

import constants as cnst
from data_models.SearchArray import SearchArray
from data_models.parse.eye_movements import (
    _extract_event_features,
    _num_fixations_to_strip,
    process_trial_events,
)

# a point inside / outside the exemplar strip rectangle, per SearchArray._BOTTOM_STRIP_*
IN_STRIP = (960.0, 1000.0)
OFF_STRIP = (300.0, 300.0)

FIXATION, SACCADE = "FIXATION", "SACCADE"


def test_strip_reference_points_are_what_the_tests_assume():
    """Guard the coordinates the rest of this module relies on."""
    assert SearchArray.is_in_bottom_strip(IN_STRIP)
    assert not SearchArray.is_in_bottom_strip(OFF_STRIP)


def event_frame(spec: list[tuple[str, bool]], event_type: str = FIXATION) -> pd.DataFrame:
    """Build an events frame from (eye, is_in_strip) pairs, ordered as `get_raw_eye_movements` emits them.

    That ordering is all of the left eye's events, then all of the right eye's - the concatenation in
    `Trial.get_raw_eye_movements`.
    """
    rows = []
    for i, (eye, in_strip) in enumerate(spec):
        x, y = IN_STRIP if in_strip else OFF_STRIP
        rows.append({"eye": eye, "event": i, "event_type": event_type, "x": x, "y": y})
    return pd.DataFrame(rows)


def interleaved_frame(spec: list[tuple[str, str, bool]]) -> pd.DataFrame:
    """Build an events frame from (eye, event_type, is_in_strip) triples; non-fixations get NaN coordinates."""
    rows = []
    for i, (eye, event_type, in_strip) in enumerate(spec):
        x, y = (IN_STRIP if in_strip else OFF_STRIP) if event_type == FIXATION else (np.nan, np.nan)
        rows.append({"eye": eye, "event": i, "event_type": event_type, "x": x, "y": y})
    return pd.DataFrame(rows)


class TestNumFixationsToStrip:
    def test_counts_forward_to_the_next_strip_fixation(self):
        frame = event_frame([("left", False), ("left", False), ("left", True)])
        assert _num_fixations_to_strip(frame).tolist() == [2, 1, 0]

    def test_infinite_when_the_strip_is_never_visited(self):
        frame = event_frame([("left", False), ("left", False)])
        assert _num_fixations_to_strip(frame).tolist() == [np.inf, np.inf]

    def test_does_not_count_across_the_eye_boundary(self):
        """Left eye never enters the strip; the right eye's first fixation does.

        The last left-eye fixation must be inf, not 1.
        """
        frame = event_frame(
            [("left", False), ("left", False), ("right", True), ("right", False)]
        )
        result = _num_fixations_to_strip(frame).tolist()
        assert result[:2] == [np.inf, np.inf], f"left-eye counts leaked into the right eye: {result[:2]}"

    def test_leak_can_wrongly_reject_an_lws_candidate(self):
        """A left-eye fixation far from any strip visit gets count 1 purely because the right eye starts in the strip.

        With the default threshold of 3 fixations, that flips the event from LWS candidate to rejected.
        """
        frame = event_frame([("left", False), ("right", True)])
        last_left = _num_fixations_to_strip(frame).iloc[0]
        assert last_left >= 3, f"expected inf (never returns to strip), got {last_left}"

    def test_saccades_do_not_inflate_the_count(self):
        """The count is of *fixations* to the strip, not events.

        With a saccade between every pair of fixations, scanning the frame positionally would return 4 for the
        first fixation where the answer is 2. This is the regression the events table introduces if the filter is
        forgotten - and it feeds the `not_before_exemplar_visit` LWS criterion directly.
        """
        frame = interleaved_frame([
            ("left", FIXATION, False), ("left", SACCADE, False),
            ("left", FIXATION, False), ("left", SACCADE, False),
            ("left", FIXATION, True),
        ])
        result = _num_fixations_to_strip(frame)
        assert result.loc[[0, 2, 4]].tolist() == [2, 1, 0]

    def test_non_fixations_get_nan(self):
        frame = interleaved_frame([
            ("left", FIXATION, False), ("left", SACCADE, False), ("left", FIXATION, True),
        ])
        assert np.isnan(_num_fixations_to_strip(frame).iloc[1])

    def test_returns_all_nan_when_there_are_no_fixations(self):
        frame = interleaved_frame([("left", SACCADE, False), ("left", SACCADE, False)])
        assert _num_fixations_to_strip(frame).isna().all()


def raw_summary(spec: list[tuple[str, int, tuple, tuple]]) -> pd.DataFrame:
    """Build a frame shaped like `Trial.get_raw_eye_movements()` output from
    (event_type, label, center_pixel, pixel_std) tuples."""
    rows = []
    for i, (event_type, label, center, std) in enumerate(spec):
        rows.append({
            "event_type": event_type, "label": label,
            "start_time": 100.0 * i, "end_time": 100.0 * i + 50.0, "duration": 50.0,
            "center_pixel": center, "pixel_std": std,
            "is_outlier": False, "outlier_reasons": [],
        })
    index = pd.MultiIndex.from_arrays(
        [["left"] * len(rows), list(range(len(rows)))], names=[cnst.EYE_STR, cnst.EVENT_STR]
    )
    return pd.DataFrame(rows, index=index)


class TestEventFeatures:
    """`x`/`y` mean 'the position the eye was holding', which only a fixation has."""

    @pytest.fixture
    def features(self) -> pd.DataFrame:
        # label 1 = FIXATION, 2 = SACCADE, per peyes' EventLabelEnum
        return _extract_event_features(
            raw_summary([
                (FIXATION, 1, (100.0, 200.0), (3.0, 4.0)),
                (SACCADE, 2, (500.0, 600.0), (30.0, 40.0)),
            ]),
            trial_end_time=10_000.0,
        )

    def test_fixation_keeps_its_centre(self, features):
        assert (features.loc[0, cnst.X], features.loc[0, cnst.Y]) == (100.0, 200.0)

    def test_saccade_centre_is_not_written_into_x_y(self, features):
        """A saccade's `center_pixel` is the midpoint of a trajectory crossed at speed, never a held position."""
        assert np.isnan(features.loc[1, cnst.X]) and np.isnan(features.loc[1, cnst.Y])

    def test_pixel_std_is_split_into_float_columns(self, features):
        assert features["std_x"].tolist() == [3.0, 30.0]
        assert features["std_y"].dtype == float

    def test_spread_is_kept_for_every_event(self, features):
        """Unlike `x`/`y`, spread is a measurement rather than a fabricated location, so it is not nulled."""
        assert features["std_x"].notna().all()

    def test_tuple_columns_are_dropped(self, features):
        assert "center_pixel" not in features.columns and "pixel_std" not in features.columns

    def test_label_is_dropped_in_favour_of_event_type(self, features):
        assert cnst.LABEL_STR not in features.columns
        assert features["event_type"].tolist() == [FIXATION, SACCADE]

    def test_event_keys_are_restored_as_columns(self, features):
        assert features[cnst.EYE_STR].tolist() == ["left", "left"]
        assert features[cnst.EVENT_STR].tolist() == [0, 1]

    def test_to_trial_end_is_time_remaining(self, features):
        assert features["to_trial_end"].tolist() == [9950.0, 9850.0]


class TestProcessTrialEvents:
    @pytest.fixture
    def events(self) -> pd.DataFrame:
        raw = raw_summary([
            (FIXATION, 1, (100.0, 100.0), (1.0, 1.0)),
            (SACCADE, 2, (500.0, 500.0), (9.0, 9.0)),
            (FIXATION, 1, IN_STRIP, (1.0, 1.0)),
        ])
        return process_trial_events(raw, end_time=10_000.0)

    def test_every_event_is_kept(self, events):
        assert len(events) == 3
        assert events["event_type"].tolist() == [FIXATION, SACCADE, FIXATION]

    def test_event_ids_are_the_rank_among_all_events(self, events):
        """The property that makes the fixation subset bit-identical to the old table: `event` never renumbers."""
        assert events[cnst.EVENT_STR].tolist() == [0, 1, 2]

    def test_fixation_only_columns_are_null_for_the_saccade(self, events):
        saccade = events.loc[events["event_type"] == SACCADE].iloc[0]
        for col in [cnst.X, cnst.Y, "num_fixs_to_strip"]:
            assert pd.isna(saccade[col]), f"{col} should be null for a saccade"

    def test_strip_count_skips_the_saccade(self, events):
        fixations = events.loc[events["event_type"] == FIXATION]
        assert fixations["num_fixs_to_strip"].tolist() == [1, 0]

    def test_rejects_a_non_positive_trial_end(self):
        with pytest.raises(AssertionError):
            process_trial_events(
                raw_summary([(FIXATION, 1, (1.0, 1.0), (1.0, 1.0))]),
                end_time=0.0,
            )
