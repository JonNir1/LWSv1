"""Fixation preprocessing: distance to the exemplar strip.

Covers CODE_REVIEW finding H1 - `_num_fixations_to_strip` scans a frame holding both eyes' fixations
back-to-back, so the count can run off the end of one eye's sequence and into the other's.
"""

import numpy as np
import pandas as pd
import pytest

from data_models.SearchArray import SearchArray
from data_models.preprocess.fixations import _num_fixations_to_strip

# a point inside / outside the exemplar strip rectangle, per SearchArray._BOTTOM_STRIP_*
IN_STRIP = (960.0, 1000.0)
OFF_STRIP = (300.0, 300.0)


def test_strip_reference_points_are_what_the_tests_assume():
    """Guard the coordinates the rest of this module relies on."""
    assert SearchArray.is_in_bottom_strip(IN_STRIP)
    assert not SearchArray.is_in_bottom_strip(OFF_STRIP)


def fixation_frame(spec: list[tuple[str, bool]]) -> pd.DataFrame:
    """Build a fixation frame from (eye, is_in_strip) pairs, ordered as `get_raw_eye_movements` emits them.

    That ordering is all of the left eye's fixations, then all of the right eye's - the concatenation in
    `Trial.get_raw_eye_movements`.
    """
    rows = []
    for i, (eye, in_strip) in enumerate(spec):
        x, y = IN_STRIP if in_strip else OFF_STRIP
        rows.append({"eye": eye, "event": i, "x": x, "y": y})
    return pd.DataFrame(rows)


class TestNumFixationsToStrip:
    def test_counts_forward_to_the_next_strip_fixation(self):
        frame = fixation_frame([("left", False), ("left", False), ("left", True)])
        assert _num_fixations_to_strip(frame).tolist() == [2, 1, 0]

    def test_infinite_when_the_strip_is_never_visited(self):
        frame = fixation_frame([("left", False), ("left", False)])
        assert _num_fixations_to_strip(frame).tolist() == [np.inf, np.inf]

    def test_does_not_count_across_the_eye_boundary(self):
        """Left eye never enters the strip; the right eye's first fixation does.

        The last left-eye fixation must be inf, not 1.
        """
        frame = fixation_frame(
            [("left", False), ("left", False), ("right", True), ("right", False)]
        )
        result = _num_fixations_to_strip(frame).tolist()
        assert result[:2] == [np.inf, np.inf], f"left-eye counts leaked into the right eye: {result[:2]}"

    def test_leak_can_wrongly_reject_an_lws_candidate(self):
        """A left-eye fixation far from any strip visit gets count 1 purely because the right eye starts in the strip.

        With the default threshold of 3 fixations, that flips the event from LWS candidate to rejected.
        """
        frame = fixation_frame([("left", False), ("right", True)])
        last_left = _num_fixations_to_strip(frame).iloc[0]
        assert last_left >= 3, f"expected inf (never returns to strip), got {last_left}"
