from typing import Literal

import numpy as np
import pandas as pd
import scipy.stats as stats

import config as cnfg


def calc_bad_actions_rate(metadata: pd.DataFrame) -> pd.DataFrame:
    """ Calculate the rate of bad actions (e.g. MARK_AND_REJECT) for each subject and trial type. """
    return _calc_mean_by_trial_type(metadata, "bad_actions")

def calc_dprime(
        metadata: pd.DataFrame,
        idents: pd.DataFrame,
        on_target_threshold_dva: float,
) -> pd.DataFrame:
    """
    Calculate d' (d-prime) for each subject and trial type, based on the hit and false alarm rates.
    Returns a DataFrame with mean and SEM of d' averaged across subjects and trial types, as well as an overall values.
    """
    metadata_with_dprime = _calc_dprime_per_trial(metadata, idents, on_target_threshold_dva)
    dprime = _calc_mean_by_trial_type(metadata_with_dprime, "d_prime")
    return dprime


def calc_sdt_class_rate(
        metadata: pd.DataFrame,
        idents: pd.DataFrame,
        classification: Literal["hit", "miss", "false_alarm"],
        on_target_threshold_dva: float,
) -> pd.DataFrame:
    metadata_with_count = _calc_sdt_class_rate_per_trial(metadata, idents, classification, on_target_threshold_dva)
    rates = _calc_mean_by_trial_type(metadata_with_count, "rate")
    return rates


def _calc_mean_by_trial_type(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Calculate the mean and SEM of the specified column values, with a breakdown by subject and trial type and across
    each of them, as well as an overall mean and SEM.
    """
    assert column in data.columns, f"Column '{column}' not found in the data."
    assert cnfg.SUBJECT_STR in data.columns, f"Column '{cnfg.SUBJECT_STR}' not found in data."
    assert cnfg.TRIAL_STR in data.columns, f"Column '{cnfg.TRIAL_STR}' not found in data."
    subj_trial_type_rate = pd.concat([
        data.groupby([cnfg.SUBJECT_STR, cnfg.TRIAL_TYPE_STR])[column].mean().rename("mean"),
        data.groupby([cnfg.SUBJECT_STR, cnfg.TRIAL_TYPE_STR])[column].sem().rename("sem"),
    ], axis=1).reset_index(drop=False)
    subject_rate = pd.concat([
        data.groupby(cnfg.SUBJECT_STR)[column].mean().rename("mean"),
        data.groupby(cnfg.SUBJECT_STR)[column].sem().rename("sem"),
    ], axis=1).reset_index(drop=False)
    trial_type_rate = pd.concat([
        data.groupby(cnfg.TRIAL_TYPE_STR)[column].mean().rename("mean"),
        data.groupby(cnfg.TRIAL_TYPE_STR)[column].sem().rename("sem"),
    ], axis=1).reset_index(drop=False)
    overall_rate = pd.DataFrame({
        cnfg.SUBJECT_STR: [cnfg.ALL_STR.upper()],
        cnfg.TRIAL_TYPE_STR: [cnfg.ALL_STR.upper()],
        "mean": [data[column].mean()],
        "sem": [data[column].sem()]
    })
    mean = (
        pd.concat([subj_trial_type_rate, subject_rate, trial_type_rate, overall_rate], axis=0)
        .fillna({cnfg.TRIAL_TYPE_STR: cnfg.ALL_STR.upper(), cnfg.SUBJECT_STR: cnfg.ALL_STR.upper()})
        .sort_values(by=[cnfg.SUBJECT_STR, cnfg.TRIAL_TYPE_STR])
        .reset_index(drop=True)
    )
    return mean


def _calc_sdt_class_rate_per_trial(
        metadata: pd.DataFrame,
        idents: pd.DataFrame,
        classification: Literal["hit", "miss", "false_alarm", "correct_reject"],
        on_target_threshold_dva: float,
) -> pd.DataFrame:
    """
    Calculates the rate of a specific SDT classification (hit, miss, false alarm, or correct reject) for each trial
    based on the trial's metadata (number of targets/distractors) and the subject's identification data (performing
    hits, misses, or false alarms).
    Returns a DataFrame with the subject, trial type, count of the classification, and the rate of the classification.
    """
    is_hit = idents[f"{cnfg.DISTANCE_STR}_dva"] <= on_target_threshold_dva
    is_miss = np.isinf(idents[f"{cnfg.DISTANCE_STR}_dva"])
    is_fa = ~is_hit & ~is_miss
    if classification == "hit":
        selector = is_hit
    elif classification == "miss":
        selector = is_miss
    elif classification == "false_alarm":
        selector = is_fa
    elif classification == "correct_reject":
        selector = is_fa
    else:
        raise NameError(f"Unknown SDT class '{classification}'.")
    count = (
        idents
        .loc[selector]
        .groupby([cnfg.SUBJECT_STR, cnfg.TRIAL_STR])
        .size()
        .rename("count")
    )
    metadata_with_count = (
        metadata
        .copy()
        .merge(count, on=[cnfg.SUBJECT_STR, cnfg.TRIAL_STR], how="left")
        .fillna({"count": 0})
    )
    if classification == "hit" or classification == "miss":
        # hits & misses are divided by the number of targets
        metadata_with_count["rate"] = metadata_with_count["count"] / metadata_with_count["num_targets"]
    elif classification == "false_alarm":
        # FA is divided by the number of distractors
        metadata_with_count["rate"] = metadata_with_count["count"] / metadata_with_count["num_distractors"]
    elif classification == "correct_reject":
        # CR is the number of distractors minus the number of false alarms; divide by the number of distractors
        metadata_with_count["count"] = metadata_with_count["num_distractors"] - metadata_with_count["count"]
        metadata_with_count["rate"] = metadata_with_count["count"] / metadata_with_count["num_distractors"]
    else:
        raise NameError(f"Unknown SDT class '{classification}'.")
    return metadata_with_count


def _calc_dprime_per_trial(
        metadata: pd.DataFrame, idents: pd.DataFrame, on_target_threshold_dva: float
) -> pd.DataFrame:
    """ Calculate d' (d-prime) for each trial. """
    hit_rate_per_trial = _calc_sdt_class_rate_per_trial(
        metadata, idents, "hit", on_target_threshold_dva
    )["rate"]
    hit_rate_z = hit_rate_per_trial.map(lambda hr: stats.norm.ppf(hr))
    fa_rate_per_trial = _calc_sdt_class_rate_per_trial(
        metadata, idents, "false_alarm", on_target_threshold_dva
    )["rate"]
    fa_rate_z = fa_rate_per_trial.map(lambda far: stats.norm.ppf(far))
    d_prime = (hit_rate_z - fa_rate_z).rename("d_prime")
    metadata_with_dprime = pd.concat([metadata, d_prime], axis=1)
    return metadata_with_dprime
