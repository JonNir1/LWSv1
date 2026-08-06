"""LWS / target-return event classification.

Covers CODE_REVIEW finding C4 (identification-time lookup) and the semantics settled in the resolved design
decisions: misses carry `time = inf`, identification time is the first hit, and false alarms identify nothing.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import make_fixation_row, make_idents

from analysis.helpers.funnels.event_classification import (
    _identification_time_lookup,
    is_after_identification,
    is_before_identification,
)


def events(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def event_on(target: str, start_time: float, end_time: float, subject: int = 1, trial: int = 1) -> dict:
    """A minimal event row as the funnel sees it: keyed by (subject, trial, target) with a time span."""
    return {
        "subject": subject,
        "trial": trial,
        "target": target,
        "start_time": start_time,
        "end_time": end_time,
        "to_trial_end": 10_000.0 - end_time,
        "num_fixs_to_strip": np.inf,
    }


class TestIdentificationTimeLookup:
    def test_single_hit(self):
        lookup = _identification_time_lookup(make_idents([("target0", "hit", 4000.0)]))
        assert lookup.loc[(1, 1, "target0")] == 4000.0

    def test_false_alarm_does_not_shadow_the_real_hit(self):
        """FA near target0 at t=500, genuine hit on target0 at t=4000. Identification time must be 4000."""
        idents = make_idents([("target0", "false_alarm", 500.0), ("target0", "hit", 4000.0)])
        assert _identification_time_lookup(idents).loc[(1, 1, "target0")] == 4000.0

    def test_repeated_hit_does_not_move_identification_time(self):
        """Rows deliberately supplied out of order: the earliest hit must still win."""
        idents = make_idents([("target1", "repeated_hit", 3000.0), ("target1", "hit", 1000.0)])
        assert _identification_time_lookup(idents).loc[(1, 1, "target1")] == 1000.0

    def test_miss_is_infinite(self):
        """Decision 1: a never-identified target has time = inf, and `dropna` must not remove it."""
        lookup = _identification_time_lookup(make_idents([("target2", "miss", np.inf)]))
        assert lookup.loc[(1, 1, "target2")] == np.inf


class TestBeforeAfterIdentification:
    def test_before_identification(self):
        ident = _identification_time_lookup(make_idents([("target0", "hit", 4000.0)]))
        data = events(event_on("target0", 1000.0, 1500.0), event_on("target0", 5000.0, 5500.0))
        assert is_before_identification(data, ident).tolist() == [True, False]

    def test_after_identification(self):
        ident = _identification_time_lookup(make_idents([("target0", "hit", 4000.0)]))
        data = events(event_on("target0", 1000.0, 1500.0), event_on("target0", 5000.0, 5500.0))
        assert is_after_identification(data, ident).tolist() == [False, True]

    def test_missed_target_makes_every_event_pre_identification(self):
        """Decision 1: with time = inf, every on-target event on a missed target is an LWS candidate."""
        ident = _identification_time_lookup(make_idents([("target2", "miss", np.inf)]))
        data = events(event_on("target2", 1000.0, 1500.0), event_on("target2", 9000.0, 9500.0))
        assert is_before_identification(data, ident).tolist() == [True, True]
        assert is_after_identification(data, ident).tolist() == [False, False]

    def test_unknown_target_raises(self):
        """A target with no identification row at all is a data error, not a 'not LWS' verdict."""
        ident = _identification_time_lookup(make_idents([("target0", "hit", 4000.0)]))
        data = events(event_on("target_unknown", 1000.0, 1500.0))
        with pytest.raises((KeyError, ValueError, AssertionError)):
            is_before_identification(data, ident)
