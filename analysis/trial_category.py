from typing import Literal, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats

import config as cnfg
from analysis.helpers.sdt import D_PRIME_CORRECTIONS
from analysis.helpers.sdt import calc_dprime_per_trial, calc_sdt_class_per_trial


def calc_bad_actions_rate(metadata: pd.DataFrame) -> pd.DataFrame:
    """ Calculate the rate of bad actions (e.g. MARK_AND_REJECT) for each subject and trial category. """
    return _calc_mean_by_trial_category(metadata, "bad_actions")

def calc_sdt_class_rate(
        metadata: pd.DataFrame,
        idents: pd.DataFrame,
        classification: Literal["hit", "miss", "false_alarm"],
) -> pd.DataFrame:
    metadata_with_count = calc_sdt_class_per_trial(metadata, idents, classification)
    rates = _calc_mean_by_trial_category(metadata_with_count, "rate")
    return rates

def calc_dprime(
        metadata: pd.DataFrame, idents: pd.DataFrame, correction: Optional[D_PRIME_CORRECTIONS]
) -> pd.DataFrame:
    """
    Calculate d' (d-prime) for each subject and trial category, based on the hit and false alarm rates.
    Returns a DataFrame with mean and SEM of d' averaged across subjects and trial categories, as well as an overall values.
    """
    metadata_with_dprime = calc_dprime_per_trial(metadata, idents, correction)
    dprime = _calc_mean_by_trial_category(metadata_with_dprime, "d_prime")
    return dprime


def _calc_mean_by_trial_category(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Calculate the mean and SEM of the specified column values, with a breakdown by subject and trial category and across
    each of them, as well as an overall mean and SEM.
    """
    assert column in data.columns, f"Column '{column}' not found in the data."
    assert cnfg.SUBJECT_STR in data.columns, f"Column '{cnfg.SUBJECT_STR}' not found in data."
    assert cnfg.TRIAL_STR in data.columns, f"Column '{cnfg.TRIAL_STR}' not found in data."
    subj_trial_cat_rate = pd.concat([
        data.groupby([cnfg.SUBJECT_STR, cnfg.TRIAL_CATEGORY_STR])[column].mean().rename("mean"),
        data.groupby([cnfg.SUBJECT_STR, cnfg.TRIAL_CATEGORY_STR])[column].sem().rename("sem"),
    ], axis=1).reset_index(drop=False)
    subject_rate = pd.concat([
        data.groupby(cnfg.SUBJECT_STR)[column].mean().rename("mean"),
        data.groupby(cnfg.SUBJECT_STR)[column].sem().rename("sem"),
    ], axis=1).reset_index(drop=False)
    trial_cat_rate = pd.concat([
        data.groupby(cnfg.TRIAL_CATEGORY_STR)[column].mean().rename("mean"),
        data.groupby(cnfg.TRIAL_CATEGORY_STR)[column].sem().rename("sem"),
    ], axis=1).reset_index(drop=False)
    overall_rate = pd.DataFrame({
        cnfg.SUBJECT_STR: [cnfg.ALL_STR.upper()],
        cnfg.TRIAL_CATEGORY_STR: [cnfg.ALL_STR.upper()],
        "mean": [data[column].mean()],
        "sem": [data[column].sem()]
    })
    mean = (
        pd.concat([subj_trial_cat_rate, subject_rate, trial_cat_rate, overall_rate], axis=0)
        .fillna({cnfg.TRIAL_CATEGORY_STR: cnfg.ALL_STR.upper(), cnfg.SUBJECT_STR: cnfg.ALL_STR.upper()})
        .sort_values(by=[cnfg.SUBJECT_STR, cnfg.TRIAL_CATEGORY_STR])
        .reset_index(drop=True)
    )
    return mean

