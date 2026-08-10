import warnings
from enum import IntEnum as _IntEnum
from typing import Optional

import numpy as np
import pandas as pd

import constants as cnst
from data_models.LWSEnums import SubjectActionCategoryEnum

# parsing constants
_TRIGGER_FIELD_MAP = {"ClockTime": cnst.TIME_STR, "BioSemiCode": cnst.TRIGGER_STR}
_TOBII_FIELD_MAP = {
    "RTTime": cnst.TIME_STR,
    "GazePointPositionDisplayXLeftEye": cnst.LEFT_X_STR,
    "GazePointPositionDisplayYLeftEye": cnst.LEFT_Y_STR,
    "PupilDiameterLeftEye": cnst.LEFT_PUPIL_STR,
    "GazePointPositionDisplayXRightEye": cnst.RIGHT_X_STR,
    "GazePointPositionDisplayYRightEye": cnst.RIGHT_Y_STR,
    "PupilDiameterRightEye": cnst.RIGHT_PUPIL_STR,
    "ImageNum": f"{cnst.IMAGE_STR}_num",
    "ConditionName": cnst.CONDITION_STR,
    # "BlockNum": cnst.BLOCK_STR,                           # block number as recorded by Tobii - NOT USING THIS
    # "RunningSample": cnst.TRIAL_STR,                      # trial number as recorded by Tobii - NOT USING THIS
    # "TrialNum": f"{cnst.TRIAL_STR}_in_{cnst.BLOCK_STR}",  # trial-in-block number as recorded by Tobii - NOT USING THIS
}
_MUTUAL_COLUMNS = [cnst.TIME_STR, cnst.BLOCK_STR, cnst.TRIAL_STR, cnst.IS_RECORDING_STR]
_TRIGGER_COLUMNS = [cnst.TRIGGER_STR, cnst.ACTION_STR]
_GAZE_COLUMNS = [col for col in _TOBII_FIELD_MAP.values() if col != cnst.TIME_STR]


class _ExperimentTriggerEnum(_IntEnum):
    """
    Mapping between trigger names and their numeric codes used in the experiment.
    Manually adapted from E-Prime's `.prm` files to define the triggers.
    """
    NULL = 0
    START_RECORD = 254
    STOP_RECORD = 255

    # Block Triggers
    BLOCK_1 = 101
    BLOCK_2 = 102
    BLOCK_3 = 103
    BLOCK_4 = 104
    BLOCK_5 = 105
    BLOCK_6 = 106
    BLOCK_7 = 107
    BLOCK_8 = 108
    BLOCK_9 = 109

    # Trial Triggers
    TRIAL_START = 11
    TRIAL_END = 12
    TARGETS_ON = 13             # targets screen
    TARGETS_OFF = 14
    STIMULUS_ON = 15            # search-array screen
    STIMULUS_OFF = 16

    # Key Presses
    SPACE_ACT = 211             # marks current gaze location as the target
    SPACE_NO_ACT = 212          # unable to mark current gaze location as the target
    CONFIRM_ACT = 221           # confirms choice of the target
    CONFIRM_NO_ACT = 222        # unable to confirm choice of the target
    NOT_CONFIRM_ACT = 231       # undo the choice of the target
    NOT_CONFIRM_NO_ACT = 232    # unable to undo the choice of the target
    OTHER_KEY = 241             # any other key pressed
    ABORT_TRIAL = 242           # user request to abort the trial


def parse_triggers_and_gaze(triggers_path, gaze_path) -> (pd.DataFrame, pd.DataFrame):
    """
    Parses gaze and triggers data:
    1. Reads the Tobii gaze and trigger log files
    2. Merges the two dataframes based on timestamp, to align the data
    3. Add columns to indicate block number, trial number, and whether data was recorded
    4. Splits the merged dataframe back into gaze and trigger dataframes
    5. Returns the processed triggers and gaze dataframes
    """
    # read triggers & gaze
    triggers = _read_triggers(triggers_path)
    gaze = _read_gaze(gaze_path)
    triggers, gaze = _align_triggers_and_gaze(triggers, gaze)
    return triggers, gaze


def _assign_block_numbers(trigs: pd.Series) -> pd.Series:
    """
    Assign a block number to each sample, starting at the first trigger of each block and running to the end of the
    recording (a later block trigger overwrites it). Samples before the first block trigger are NA.
    """
    blocks = pd.Series(np.nan, index=trigs.index)
    for trg in _ExperimentTriggerEnum:
        if not trg.name.startswith("BLOCK_"):
            continue
        is_block_start = trigs.eq(trg)
        if not is_block_start.any():
            continue
        first_pos = int(np.flatnonzero(is_block_start.to_numpy())[0])
        blocks.iloc[first_pos:] = int(trg.name.split("_")[-1])
    return blocks.astype('Int64')


def _is_between_triggers(
        trigs: pd.Series, start: int, end: int, close_trailing: bool = True,
) -> pd.Series:
    """
    Mark every sample from each `start` trigger through the first `end` trigger that follows it.

    Pairs by scanning in order rather than positionally, so an unbalanced log degrades gracefully instead of raising
    from `np.vstack` or silently pairing one segment's start with another's end.

    A `start` still open at the end of the log is closed at the last row when `close_trailing` (the default), and
    dropped otherwise. Closing is right for this dataset: measured across all 27 raw subject directories
    (2026-08-06), subjects 38, 42, 43 and 44 each have 60 `STIMULUS_ON` and 59 `STIMULUS_OFF`, and the unclosed
    final segment spans 99.9-103.3% of that subject's median trial duration with ~12,500 gaze samples. The trial ran
    to completion; only the closing trigger was never written. Dropping it would discard a full trial of real data
    from each of those subjects.

    The residual cost is that the trial's end time becomes the last recorded sample rather than the true stimulus
    offset, so `to_trial_end` may be overstated by however long recording continued past offset - bounded by the
    ~210 ms `STIMULUS_OFF` -> `TRIAL_END` gap seen elsewhere in these logs. That is small against the 1000 ms
    `not_close_to_trial_end` threshold, but it is why the warning reports the span: a *genuinely* truncated trial
    would show up there as a short segment and should be excluded.

    A `start` arriving while another is still open (a dropped `end` mid-log) does not occur in this dataset; that
    segment is dropped, since its extent is unknowable.
    """
    res = pd.Series(False, index=range(len(trigs)))
    codes = trigs.to_numpy()
    open_at: Optional[int] = None
    for pos, code in enumerate(codes):
        if code == start:
            if open_at is not None:
                warnings.warn(
                    f"trigger {start} at position {pos} while the one at {open_at} is still open; "
                    f"the {end} trigger appears to be missing, so that segment is dropped.",
                    RuntimeWarning,
                )
            open_at = pos
        elif code == end and open_at is not None:
            res.iloc[open_at:pos + 1] = True
            open_at = None
    if open_at is not None:
        if close_trailing:
            res.iloc[open_at:] = True
            warnings.warn(
                f"trigger {start} at position {open_at} has no matching {end}; closing the segment at the last "
                f"recorded sample ({len(trigs) - open_at} rows). Check that span against a typical segment - a "
                f"much shorter one means the recording really was cut short.",
                RuntimeWarning,
            )
        else:
            warnings.warn(
                f"trigger {start} at position {open_at} has no matching {end}; the trailing segment is dropped.",
                RuntimeWarning,
            )
    return res


def _align_triggers_and_gaze(triggers, gaze) -> (pd.DataFrame, pd.DataFrame):
    merged = pd.merge(gaze, triggers, how='outer', on=[cnst.TIME_STR])  # merge on time

    # add block column
    merged[cnst.BLOCK_STR] = _assign_block_numbers(merged[cnst.TRIGGER_STR])

    # add trial column
    is_trial = _is_between_triggers(
        # NOTE: can also use _ExperimentTriggerEnum.TRIAL_START/TRIAL_END, but will contain unnecessary data
        merged[cnst.TRIGGER_STR], _ExperimentTriggerEnum.STIMULUS_ON, _ExperimentTriggerEnum.STIMULUS_OFF
    )
    is_trial_start = is_trial.ne(is_trial.shift()) & is_trial  # find the start of each trial
    trial_num = is_trial_start.cumsum()  # assign trial numbers
    trial_num.loc[~is_trial] = np.nan  # set non-trial rows to NaN
    merged[cnst.TRIAL_STR] = trial_num.astype('Int64')

    # add `is_recording` columns
    merged[cnst.IS_RECORDING_STR] = _is_between_triggers(
        merged[cnst.TRIGGER_STR], _ExperimentTriggerEnum.START_RECORD, _ExperimentTriggerEnum.STOP_RECORD
    )
    merged[cnst.IS_RECORDING_STR] = merged[cnst.IS_RECORDING_STR].astype(bool)

    # reorder columns
    cols_ord = [cnst.TIME_STR, cnst.TRIGGER_STR]
    cols_ord += [col for col in merged.columns if col not in cols_ord]
    merged = merged[cols_ord]

    # split out the triggers
    triggers = merged.loc[merged[cnst.TRIGGER_STR].notna(), _MUTUAL_COLUMNS + _TRIGGER_COLUMNS].copy()
    for col in _TRIGGER_COLUMNS:
        # assign column-wise: a frame-wide `.loc[:, cols] = <Int64 frame>` over the mixed float64/Int64 pair the
        # outer merge leaves behind raises `AttributeError: '_hasna'` on pandas 3.
        triggers[col] = triggers[col].fillna(0).astype('Int64')
    triggers[cnst.ACTION_STR] = triggers[cnst.ACTION_STR].map(lambda act: SubjectActionCategoryEnum(act))

    # split out the gaze data
    is_gaze = merged[_GAZE_COLUMNS].notna().any(axis=1)
    gaze = merged.loc[is_gaze, _MUTUAL_COLUMNS + _GAZE_COLUMNS]
    return triggers, gaze


def _read_gaze(gaze_path: str) -> pd.DataFrame:
    gaze = pd.read_csv(gaze_path, sep="\t")
    gaze.rename(columns=_TOBII_FIELD_MAP, inplace=True)
    # replace missing/invalid values to NaN
    et_cols = [
        cnst.LEFT_X_STR, cnst.LEFT_Y_STR, cnst.RIGHT_X_STR, cnst.RIGHT_Y_STR, cnst.LEFT_PUPIL_STR, cnst.RIGHT_PUPIL_STR
    ]
    gaze[et_cols] = gaze[et_cols].replace(cnst.TOBII_MISSING_VALUES, cnst.MISSING_VALUE, inplace=False)
    gaze.loc[gaze["GazePointValidityLeftEye"] == 0, [cnst.LEFT_X_STR, cnst.LEFT_Y_STR]] = cnst.MISSING_VALUE
    gaze.loc[gaze["GazePointValidityRightEye"] == 0, [cnst.RIGHT_X_STR, cnst.RIGHT_Y_STR]] = cnst.MISSING_VALUE
    gaze.loc[gaze["PupilValidityLeftEye"] == 0, cnst.LEFT_PUPIL_STR] = cnst.MISSING_VALUE
    gaze.loc[gaze["PupilValidityRightEye"] == 0, cnst.RIGHT_PUPIL_STR] = cnst.MISSING_VALUE
    gaze = gaze.astype({col: float for col in et_cols})

    # correct to tobii's resolution
    gaze[cnst.LEFT_X_STR] *= cnst.TOBII_MONITOR.width
    gaze[cnst.LEFT_Y_STR] *= cnst.TOBII_MONITOR.height
    gaze[cnst.RIGHT_X_STR] *= cnst.TOBII_MONITOR.width
    gaze[cnst.RIGHT_Y_STR] *= cnst.TOBII_MONITOR.height
    return gaze


def _drop_empty_trigger_rows(triggers: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows whose trigger code is missing, so `_ExperimentTriggerEnum(nan)` cannot raise.

    E-Prime's export ends several logs with one incomplete line: 7 of the 27 raw subject directories (33, 34, 35, 38,
    39, 42, 44) carry exactly one NaN code, always the final row. Dropping it is safe - it holds no event. An
    *interior* NaN would mean a genuinely lost trigger rather than a truncated write, so warn about that case
    instead of passing over it silently.
    """
    is_missing = triggers[cnst.TRIGGER_STR].isna()
    if not is_missing.any():
        return triggers
    positions = np.flatnonzero(is_missing.to_numpy())
    trailing = set(positions) <= set(range(len(triggers) - len(positions), len(triggers)))
    if not trailing:
        warnings.warn(
            f"trigger log has {len(positions)} missing code(s) away from the end of the file "
            f"(positions {positions[:5].tolist()}); a trigger was lost mid-recording.",
            RuntimeWarning,
        )
    return triggers.loc[~is_missing].reset_index(drop=True)


def _read_triggers(triggers_path: str) -> pd.DataFrame:
    triggers = pd.read_csv(triggers_path, sep="\t")
    triggers.rename(columns=_TRIGGER_FIELD_MAP, inplace=True)
    triggers = _drop_empty_trigger_rows(triggers)
    triggers[cnst.TRIGGER_STR] = triggers[cnst.TRIGGER_STR].map(lambda trgr: _ExperimentTriggerEnum(trgr))

    # add `action` column
    triggers[cnst.ACTION_STR] = SubjectActionCategoryEnum.NO_ACTION
    start_identify_idx = None
    # iterate the trigger column directly: `iterrows()` coerces each row to a common dtype, which turns the trigger
    # into a float and makes `trg.name` unavailable in the assertion messages below.
    for idx, code in triggers[cnst.TRIGGER_STR].items():
        trg = _ExperimentTriggerEnum(code)

        if trg == _ExperimentTriggerEnum.SPACE_NO_ACT:
            # subject pressed space but E-Prime rejected the mark; this is an event in its own right and does not
            # belong to any pending mark, so record it on its own row
            triggers.loc[idx, cnst.ACTION_STR] = SubjectActionCategoryEnum.ATTEMPTED_MARK
            continue

        if trg == _ExperimentTriggerEnum.SPACE_ACT:
            # subject marks target
            assert start_identify_idx is None, f"{trg.name} follows previous {trg.name} (idx: {idx})"
            start_identify_idx = idx
            continue

        if trg in [
            _ExperimentTriggerEnum.CONFIRM_ACT,
            _ExperimentTriggerEnum.NOT_CONFIRM_ACT,
        ]:
            # subject performed an action after marking target
            assert start_identify_idx is not None and start_identify_idx < idx,\
                f"{trg.name} not follows a previous {_ExperimentTriggerEnum.SPACE_ACT.name} (idx: {idx})"
            if trg == _ExperimentTriggerEnum.CONFIRM_ACT:
                # subject confirms the identified target
                triggers.loc[start_identify_idx, cnst.ACTION_STR] = SubjectActionCategoryEnum.MARK_AND_CONFIRM
            else:
                # subject rejects previously identified (non-target) item
                triggers.loc[start_identify_idx, cnst.ACTION_STR] = SubjectActionCategoryEnum.MARK_AND_REJECT
            start_identify_idx = None
            continue

        if start_identify_idx is not None and trg in [
            _ExperimentTriggerEnum.ABORT_TRIAL,
            _ExperimentTriggerEnum.STIMULUS_OFF,
            _ExperimentTriggerEnum.TRIAL_END,
        ]:
            # subject ran out of time before confirming target
            assert start_identify_idx < idx, f"{trg.name} not follows a previous {_ExperimentTriggerEnum.SPACE_ACT.name} (idx: {idx})"
            triggers.loc[start_identify_idx, cnst.ACTION_STR] = SubjectActionCategoryEnum.MARK_ONLY
            start_identify_idx = None
            continue
    triggers[cnst.ACTION_STR] = triggers[cnst.ACTION_STR].astype('Int64')
    return triggers
