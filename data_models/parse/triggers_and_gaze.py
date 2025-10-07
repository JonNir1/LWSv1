from enum import IntEnum as _IntEnum

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


def _align_triggers_and_gaze(triggers, gaze) -> (pd.DataFrame, pd.DataFrame):
    merged = pd.merge(gaze, triggers, how='outer', on=[cnst.TIME_STR])  # merge on time

    # add block column
    merged[cnst.BLOCK_STR] = np.nan
    for trg in _ExperimentTriggerEnum:
        name = trg.name
        if not name.startswith("BLOCK_"):
            continue
        block_num = int(name.split("_")[-1])
        is_block_start = merged[cnst.TRIGGER_STR].eq(trg)
        if not is_block_start.any():
            # no such block trigger - skip
            continue
        start_idx = is_block_start.idxmax()  # find the first occurrence of the block trigger
        merged.loc[merged.index[start_idx:], cnst.BLOCK_STR] = block_num
    merged[cnst.BLOCK_STR] = merged[cnst.BLOCK_STR].astype('Int64')
    del trg, name, block_num, is_block_start, start_idx

    def _is_between_triggers(trigs: pd.Series, start: int, end: int) -> pd.Series:
        """
        Returns a boolean series indicating whether the values in the 'trigs' series occur after 'start' and before 'end'.
        """
        start_idxs = np.nonzero(trigs == start)[0]
        end_idxs = np.nonzero(trigs == end)[0]
        start_end_idxs = np.vstack([start_idxs, end_idxs]).T
        res = pd.Series(np.full_like(trigs, False, dtype=bool))
        for (start, end) in start_end_idxs:
            res.iloc[start:end + 1] = True
        return res

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
    triggers = merged.loc[merged[cnst.TRIGGER_STR].notna(), _MUTUAL_COLUMNS + _TRIGGER_COLUMNS]
    triggers.loc[:, _TRIGGER_COLUMNS] = triggers[_TRIGGER_COLUMNS].fillna(0).astype('Int64')
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


def _read_triggers(triggers_path: str) -> pd.DataFrame:
    triggers = pd.read_csv(triggers_path, sep="\t")
    triggers.rename(columns=_TRIGGER_FIELD_MAP, inplace=True)
    triggers[cnst.TRIGGER_STR] = triggers[cnst.TRIGGER_STR].map(lambda trgr: _ExperimentTriggerEnum(trgr))

    # add `action` column
    triggers[cnst.ACTION_STR] = SubjectActionCategoryEnum.NO_ACTION
    start_identify_idx = None
    for idx, series in triggers.iterrows():
        trg = series[cnst.TRIGGER_STR]

        if trg == _ExperimentTriggerEnum.SPACE_NO_ACT:
            # subject attempted to mark target but failed
            triggers.loc[start_identify_idx, cnst.ACTION_STR] = SubjectActionCategoryEnum.ATTEMPTED_MARK
            continue

        if trg == _ExperimentTriggerEnum.SPACE_ACT:
            # subject marks target
            assert not start_identify_idx, f"{trg.name} follows previous {trg.name} (idx: {idx})"
            start_identify_idx = idx
            continue

        if trg in [
            _ExperimentTriggerEnum.CONFIRM_ACT,
            _ExperimentTriggerEnum.NOT_CONFIRM_ACT,
        ]:
            # subject performed an action after marking target
            assert start_identify_idx and start_identify_idx < idx,\
                f"{trg.name} not follows a previous {_ExperimentTriggerEnum.SPACE_ACT.name} (idx: {idx})"
            if trg == _ExperimentTriggerEnum.CONFIRM_ACT:
                # subject confirms the identified target
                triggers.loc[start_identify_idx, cnst.ACTION_STR] = SubjectActionCategoryEnum.MARK_AND_CONFIRM
            else:
                # subject rejects previously identified (non-target) item
                triggers.loc[start_identify_idx, cnst.ACTION_STR] = SubjectActionCategoryEnum.MARK_AND_REJECT
            start_identify_idx = None
            continue

        if start_identify_idx and trg in [
            _ExperimentTriggerEnum.ABORT_TRIAL,
            _ExperimentTriggerEnum.STIMULUS_OFF,
            _ExperimentTriggerEnum.TRIAL_END,
        ]:
            # subject ran out of time before confirming target
            assert start_identify_idx < idx, f"{trg.name} not follows a previous {_ExperimentTriggerEnum.SPACE_ACT.name} (idx: {idx})"
            triggers.loc[start_identify_idx, 'subj_action'] = SubjectActionCategoryEnum.MARK_ONLY
            start_identify_idx = None
            continue
    triggers[cnst.ACTION_STR] = triggers[cnst.ACTION_STR].astype('Int64')
    return triggers
