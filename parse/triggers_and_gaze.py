import numpy as np
import pandas as pd

import config as cnfg
from data_models.LWSEnums import SearchActionTypesEnum

MUTUAL_COLUMNS = [cnfg.TIME_STR, cnfg.BLOCK_STR, cnfg.TRIAL_STR, cnfg.IS_RECORDING_STR]
TRIGGER_COLUMNS = [cnfg.TRIGGER_STR, cnfg.ACTION_STR]
GAZE_COLUMNS = [col for col in cnfg.TOBII_FIELD_MAP.values() if col != cnfg.TIME_STR]


def parse_triggers_and_gaze(triggers_path, gaze_path) -> (pd.DataFrame, pd.DataFrame):
    """
    Parses gaze and triggers data:
    1. Reads the Tobii gaze and trigger log files
    2. Merges the two dataframes based on timestamp, to align the data
    3. Add columns to indicate block number, trial number, and whether data was recorded
    4. Splits the merged dataframe back into gaze and trigger dataframes and returns them
    """
    # read triggers & gaze
    triggers = _read_triggers(triggers_path)
    gaze = _read_gaze(gaze_path)
    triggers, gaze = _align_triggers_and_gaze(triggers, gaze)
    return triggers, gaze


def _align_triggers_and_gaze(triggers, gaze) -> (pd.DataFrame, pd.DataFrame):
    merged = pd.merge(gaze, triggers, how='outer', on=[cnfg.TIME_STR])  # merge on time

    # add block column
    merged[cnfg.BLOCK_STR] = np.nan
    for trg in cnfg.ExperimentTriggerEnum:
        name = trg.name
        if not name.startswith("BLOCK_"):
            continue
        block_num = int(name.split("_")[-1])
        is_block_start = merged[cnfg.TRIGGER_STR].eq(trg)
        if not is_block_start.any():
            # no such block trigger - skip
            continue
        start_idx = is_block_start.idxmax()  # find the first occurrence of the block trigger
        merged.loc[merged.index[start_idx:], cnfg.BLOCK_STR] = block_num
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
        merged[cnfg.TRIGGER_STR], cnfg.ExperimentTriggerEnum.TRIAL_START, cnfg.ExperimentTriggerEnum.TRIAL_END
    )
    is_trial_start = is_trial.ne(is_trial.shift()) & is_trial  # find the start of each trial
    trial_num = is_trial_start.cumsum()  # assign trial numbers
    trial_num.loc[~is_trial] = np.nan  # set non-trial rows to NaN
    merged[cnfg.TRIAL_STR] = trial_num

    # add `is_recording` columns
    merged[cnfg.IS_RECORDING_STR] = _is_between_triggers(
        merged[cnfg.TRIGGER_STR], cnfg.ExperimentTriggerEnum.START_RECORD, cnfg.ExperimentTriggerEnum.STOP_RECORD
    )

    # reorder columns
    cols_ord = [cnfg.TIME_STR, cnfg.TRIGGER_STR]
    cols_ord += [col for col in merged.columns if col not in cols_ord]
    merged = merged[cols_ord]

    # split back to gaze and triggers
    triggers = merged.loc[
        merged[cnfg.TRIGGER_STR].notna(), MUTUAL_COLUMNS + TRIGGER_COLUMNS]
    is_gaze = merged[GAZE_COLUMNS].notna().any(axis=1)
    gaze = merged.loc[is_gaze, MUTUAL_COLUMNS + GAZE_COLUMNS]
    return triggers, gaze


def _read_gaze(gaze_path: str) -> pd.DataFrame:
    gaze = pd.read_csv(gaze_path, sep="\t")
    gaze.rename(columns=cnfg.TOBII_FIELD_MAP, inplace=True)
    # replace missing/invalid values to NaN
    et_cols = [
        cnfg.LEFT_X_STR, cnfg.LEFT_Y_STR, cnfg.RIGHT_X_STR, cnfg.RIGHT_Y_STR, cnfg.LEFT_PUPIL_STR, cnfg.RIGHT_PUPIL_STR
    ]
    gaze[et_cols] = gaze[et_cols].replace(cnfg.TOBII_MISSING_VALUES, cnfg.MISSING_VALUE, inplace=False)
    gaze.loc[gaze["GazePointValidityLeftEye"] == 0, [cnfg.LEFT_X_STR, cnfg.LEFT_Y_STR]] = cnfg.MISSING_VALUE
    gaze.loc[gaze["GazePointValidityRightEye"] == 0, [cnfg.RIGHT_X_STR, cnfg.RIGHT_Y_STR]] = cnfg.MISSING_VALUE
    gaze.loc[gaze["PupilValidityLeftEye"] == 0, cnfg.LEFT_PUPIL_STR] = cnfg.MISSING_VALUE
    gaze.loc[gaze["PupilValidityRightEye"] == 0, cnfg.RIGHT_PUPIL_STR] = cnfg.MISSING_VALUE
    gaze = gaze.astype({col: float for col in et_cols})

    # correct to tobii's resolution
    gaze[cnfg.LEFT_X_STR] *= cnfg.TOBII_MONITOR.width
    gaze[cnfg.LEFT_Y_STR] *= cnfg.TOBII_MONITOR.height
    gaze[cnfg.RIGHT_X_STR] *= cnfg.TOBII_MONITOR.width
    gaze[cnfg.RIGHT_Y_STR] *= cnfg.TOBII_MONITOR.height
    return gaze


def _read_triggers(triggers_path: str) -> pd.DataFrame:
    triggers = pd.read_csv(triggers_path, sep="\t")
    triggers.rename(columns=cnfg.TRIGGER_FIELD_MAP, inplace=True)
    triggers[cnfg.TRIGGER_STR] = triggers[cnfg.TRIGGER_STR].map(lambda trgr: cnfg.ExperimentTriggerEnum(trgr))

    # add `action` column
    triggers[cnfg.ACTION_STR] = cnfg.MISSING_VALUE
    start_identify_idx = None
    for idx, series in triggers.iterrows():
        trg = series[cnfg.TRIGGER_STR]

        if trg == cnfg.ExperimentTriggerEnum.SPACE_NO_ACT:
            # subject attempted to mark target but failed
            triggers.loc[start_identify_idx, cnfg.ACTION_STR] = SearchActionTypesEnum.ATTEMPTED_MARK
            continue

        if trg == cnfg.ExperimentTriggerEnum.SPACE_ACT:
            # subject marks target
            assert not start_identify_idx, f"{trg.name} follows previous {trg.name} (idx: {idx})"
            start_identify_idx = idx
            continue

        if trg in [
            cnfg.ExperimentTriggerEnum.CONFIRM_ACT,
            cnfg.ExperimentTriggerEnum.NOT_CONFIRM_ACT,
        ]:
            # subject performed an action after marking target
            assert start_identify_idx and start_identify_idx < idx,\
                f"{trg.name} not follows a previous {cnfg.ExperimentTriggerEnum.SPACE_ACT.name} (idx: {idx})"
            if trg == cnfg.ExperimentTriggerEnum.CONFIRM_ACT:
                # subject confirms the identified target
                triggers.loc[start_identify_idx, cnfg.ACTION_STR] = SearchActionTypesEnum.MARK_AND_CONFIRM
            else:
                # subject rejects previously identified (non-target) item
                triggers.loc[start_identify_idx, cnfg.ACTION_STR] = SearchActionTypesEnum.MARK_AND_REJECT
            start_identify_idx = None
            continue

        if start_identify_idx and trg in [
            cnfg.ExperimentTriggerEnum.ABORT_TRIAL,
            cnfg.ExperimentTriggerEnum.STIMULUS_OFF,
            cnfg.ExperimentTriggerEnum.TRIAL_END,
        ]:
            # subject ran out of time before confirming target
            assert start_identify_idx < idx, f"{trg.name} not follows a previous {cnfg.ExperimentTriggerEnum.SPACE_ACT.name} (idx: {idx})"
            triggers.loc[start_identify_idx, 'subj_action'] = SearchActionTypesEnum.MARK_ONLY
            start_identify_idx = None
            continue

    return triggers

