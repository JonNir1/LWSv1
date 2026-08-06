"""Trigger-log parsing: action classification and trial-boundary detection.

Covers CODE_REVIEW findings C2, C3, M4 and H5. Tests that encode a known-unfixed bug are `xfail(strict=True)`.
"""

import numpy as np
import pandas as pd
import pytest

from data_models.LWSEnums import SubjectActionCategoryEnum
from data_models.parse.triggers_and_gaze import (
    _ExperimentTriggerEnum as Trg,
    _align_triggers_and_gaze,
    _read_triggers,
)

# NOTE: `_is_between_triggers` is a closure defined inside `_align_triggers_and_gaze`, so it cannot be imported and
# tested directly. Trial-boundary behaviour is therefore exercised through `_align_triggers_and_gaze`. Extracting it
# to module level is part of the H5 fix.


def write_trigger_log(tmp_path, trigger_codes: list[int], times: list[float] | None = None) -> str:
    """Write a tab-separated trigger log in the E-Prime export format `_read_triggers` expects."""
    times = times if times is not None else [100.0 * (i + 1) for i in range(len(trigger_codes))]
    path = tmp_path / "triggers.txt"
    pd.DataFrame({"ClockTime": times, "BioSemiCode": trigger_codes}).to_csv(path, sep="\t", index=False)
    return str(path)


def actions_of(triggers: pd.DataFrame) -> list[SubjectActionCategoryEnum]:
    """Non-NO_ACTION actions, in time order."""
    acts = triggers["action"].dropna()
    return [SubjectActionCategoryEnum(a) for a in acts if a != SubjectActionCategoryEnum.NO_ACTION]


class TestActionClassification:
    def test_mark_and_confirm(self, tmp_path):
        path = write_trigger_log(tmp_path, [Trg.START_RECORD, Trg.SPACE_ACT, Trg.CONFIRM_ACT, Trg.STOP_RECORD])
        assert actions_of(_read_triggers(path)) == [SubjectActionCategoryEnum.MARK_AND_CONFIRM]

    def test_mark_and_reject(self, tmp_path):
        path = write_trigger_log(tmp_path, [Trg.START_RECORD, Trg.SPACE_ACT, Trg.NOT_CONFIRM_ACT, Trg.STOP_RECORD])
        assert actions_of(_read_triggers(path)) == [SubjectActionCategoryEnum.MARK_AND_REJECT]

    @pytest.mark.xfail(
        strict=True,
        reason="C2: MARK_ONLY is written to a stray 'subj_action' column instead of cnst.ACTION_STR, "
        "so it never reaches the actions table",
    )
    def test_mark_only_is_recorded(self, tmp_path):
        """Subject marks a target but the trial ends before they confirm."""
        path = write_trigger_log(tmp_path, [Trg.START_RECORD, Trg.SPACE_ACT, Trg.STIMULUS_OFF, Trg.STOP_RECORD])
        assert actions_of(_read_triggers(path)) == [SubjectActionCategoryEnum.MARK_ONLY]

    @pytest.mark.xfail(
        strict=True,
        reason="C3: with no pending mark, start_identify_idx is None and `.loc[None, ...]` raises KeyError",
    )
    def test_attempted_mark_without_pending_mark(self, tmp_path):
        """A rejected space press with no preceding successful mark is an event in its own right."""
        path = write_trigger_log(tmp_path, [Trg.START_RECORD, Trg.SPACE_NO_ACT, Trg.STOP_RECORD])
        assert actions_of(_read_triggers(path)) == [SubjectActionCategoryEnum.ATTEMPTED_MARK]

    @pytest.mark.xfail(
        strict=True,
        reason="C3: ATTEMPTED_MARK overwrites the pending mark's row, then CONFIRM_ACT overwrites it again, "
        "so the attempted mark is lost",
    )
    def test_attempted_mark_does_not_clobber_pending_mark(self, tmp_path):
        """Space pressed twice before confirming: both the rejected press and the confirmed mark should survive."""
        path = write_trigger_log(
            tmp_path, [Trg.START_RECORD, Trg.SPACE_ACT, Trg.SPACE_NO_ACT, Trg.CONFIRM_ACT, Trg.STOP_RECORD]
        )
        assert sorted(actions_of(_read_triggers(path))) == sorted(
            [SubjectActionCategoryEnum.MARK_AND_CONFIRM, SubjectActionCategoryEnum.ATTEMPTED_MARK]
        )

    @pytest.mark.xfail(
        strict=True,
        reason="M4: index label 0 is falsy, so `assert not start_identify_idx` and `if start_identify_idx and ...` "
        "misread a pending mark held at row 0",
    )
    def test_mark_at_row_zero(self, tmp_path):
        """A trigger log whose very first row is a key press."""
        path = write_trigger_log(tmp_path, [Trg.SPACE_ACT, Trg.SPACE_ACT, Trg.CONFIRM_ACT])
        # two SPACE_ACTs in a row must trip the "follows previous" assertion, whichever row they sit on
        with pytest.raises(AssertionError):
            _read_triggers(path)


def align(trigger_codes: list[int], with_block: bool = True) -> pd.DataFrame:
    """Run `_align_triggers_and_gaze` over a trigger sequence with one interleaved gaze sample per trigger.

    A BLOCK_1 trigger is prepended by default, matching real logs - without one, `_align_triggers_and_gaze` raises
    (see `test_no_block_trigger_raises`).

    Returns the gaze frame, whose `trial` column carries the derived trial boundaries.
    """
    if with_block:
        trigger_codes = [Trg.BLOCK_1, *trigger_codes]
    times = [100.0 * (i + 1) for i in range(len(trigger_codes))]
    triggers = pd.DataFrame(
        {
            "time": times,
            "trigger": [Trg(c) for c in trigger_codes],
            "action": pd.array([SubjectActionCategoryEnum.NO_ACTION] * len(times), dtype="Int64"),
        }
    )
    # one gaze sample 1 ms after each trigger, so gaze rows fall strictly inside the trigger-defined segments
    gaze = pd.DataFrame(
        {
            "time": [t + 1 for t in times],
            "left_x": 500.0, "left_y": 500.0, "left_pupil": 3.0,
            "right_x": 505.0, "right_y": 500.0, "right_pupil": 3.0,
            "image_num": 1, "condition": "color",
        }
    )
    _, aligned_gaze = _align_triggers_and_gaze(triggers, gaze)
    return aligned_gaze


class TestTrialBoundaries:
    @pytest.mark.xfail(
        strict=True,
        reason="M17: `del ... start_idx` at triggers_and_gaze.py:102 runs unconditionally, but start_idx is only "
        "bound when a BLOCK_* trigger is present -> UnboundLocalError",
    )
    def test_no_block_trigger_raises(self):
        """A trigger log with no BLOCK trigger should still align, not crash on a cleanup `del`."""
        gaze = align([Trg.STIMULUS_ON, Trg.STIMULUS_OFF], with_block=False)
        assert gaze["trial"].dropna().unique().tolist() == [1]

    def test_balanced(self):
        gaze = align([Trg.NULL, Trg.STIMULUS_ON, Trg.NULL, Trg.STIMULUS_OFF, Trg.NULL])
        assert gaze["trial"].dropna().unique().tolist() == [1]

    def test_two_balanced_trials(self):
        gaze = align([Trg.STIMULUS_ON, Trg.STIMULUS_OFF, Trg.NULL, Trg.STIMULUS_ON, Trg.STIMULUS_OFF])
        assert gaze["trial"].dropna().unique().tolist() == [1, 2]

    @pytest.mark.xfail(
        strict=True,
        reason="H5: np.vstack requires equal start/end counts, so a truncated final trial raises "
        "instead of being ignored",
    )
    def test_unclosed_final_trial(self):
        """Recording stops mid-trial: the last STIMULUS_ON never gets its STIMULUS_OFF."""
        gaze = align([Trg.STIMULUS_ON, Trg.STIMULUS_OFF, Trg.NULL, Trg.STIMULUS_ON])
        assert gaze["trial"].dropna().unique().tolist() == [1]

    @pytest.mark.xfail(
        strict=True,
        reason="H5: pairing is positional, so a dropped STIMULUS_OFF pairs trial 1's start with trial 2's end "
        "and swallows the gap between them",
    )
    def test_dropped_end_trigger_does_not_merge_trials(self):
        """A dropped STIMULUS_OFF must not merge two trials into one long one."""
        gaze = align([Trg.STIMULUS_ON, Trg.NULL, Trg.STIMULUS_ON, Trg.STIMULUS_OFF])
        in_trial = gaze["trial"].notna().tolist()
        assert in_trial[1] is False, "the gap between the two trials must not be marked as in-trial"
