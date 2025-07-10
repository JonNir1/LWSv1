from typing import Literal, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats

import config as cnfg
from data_models.LWSEnums import SignalDetectionCategoryEnum

D_PRIME_CORRECTIONS = Literal["loglinear", "hautus", "macmillan", "mk", "none"]


def calc_dprime_per_trial(
        metadata: pd.DataFrame, idents: pd.DataFrame, correction: Optional[D_PRIME_CORRECTIONS]
) -> pd.DataFrame:
    """ Calculate d' (d-prime) for each subject-trial pair. """
    hit_counts = calc_sdt_class_per_trial(metadata, idents, "hit")["count"].rename("hits")
    fa_counts = calc_sdt_class_per_trial(metadata, idents, "false_alarm")["count"].rename("false_alarms")
    metadata_with_counts = pd.concat(
        [metadata.drop(columns=["duration", "bad_actions"]), hit_counts, fa_counts], axis=1
    )
    d_prime = metadata_with_counts.apply(
        lambda row: calc_dprime(
            hits=row["hits"],
            false_alarms=row["false_alarms"],
            num_targets=row["num_targets"],
            num_distractors=row["num_distractors"],
            correction=correction,
        ), axis=1
    ).rename("d_prime")
    metadata_with_dprime = pd.concat([metadata_with_counts, d_prime], axis=1)
    return metadata_with_dprime


def calc_sdt_class_per_trial(
        metadata: pd.DataFrame,
        idents: pd.DataFrame,
        sdt_class: Literal["hit", "miss", "false_alarm", "correct_reject"],
) -> pd.DataFrame:
    """
    Calculates the rate of a specific SDT classification (hit, miss, false alarm, or correct reject) for each pair of
    subject-trial values, based on the trial's metadata (number of targets/distractors) and the subject's
    identification data (performing hits, misses, or false alarms).
    Returns a DataFrame with the subject, trial category, count of the classification, and the rate of the classification.
    """
    # identify identifications from the provided SDT class
    if sdt_class == "correct_reject":
        # for CRs we calculate the FA count and subtract it from the number of distractors
        selector = SignalDetectionCategoryEnum.FALSE_ALARM
    else:
        # for all other classes we use the same sdt-class
        selector = SignalDetectionCategoryEnum(sdt_class.lower())
    assert cnfg.IDENTIFICATION_CATEGORY_STR in idents.columns, f"Column '{cnfg.IDENTIFICATION_CATEGORY_STR}' not found in `idents` DF."
    selector = idents[cnfg.IDENTIFICATION_CATEGORY_STR] == selector
    # append the count of the selected SDT class to the metadata
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
        .drop(columns=["duration", "bad_actions"])
        .merge(count, on=[cnfg.SUBJECT_STR, cnfg.TRIAL_STR], how="left")
        .fillna({"count": 0})
    )
    # calculate the rate
    if sdt_class == "hit" or sdt_class == "miss":
        # hits & misses are divided by the number of targets
        metadata_with_count["rate"] = metadata_with_count["count"] / metadata_with_count["num_targets"]
    elif sdt_class == "false_alarm":
        # FA is divided by the number of distractors
        metadata_with_count["rate"] = metadata_with_count["count"] / metadata_with_count["num_distractors"]
    elif sdt_class == "correct_reject":
        # CR is the number of distractors minus the number of false alarms; divide by the number of distractors
        metadata_with_count["count"] = metadata_with_count["num_distractors"] - metadata_with_count["count"]
        metadata_with_count["rate"] = metadata_with_count["count"] / metadata_with_count["num_distractors"]
    else:
        raise NameError(f"Unknown SDT class '{sdt_class}'.")
    return metadata_with_count


def calc_dprime(
        hits: int, false_alarms: int, num_targets: int, num_distractors: int, correction: Optional[D_PRIME_CORRECTIONS]
) -> float:
    """
    Calculate d' (d-prime) from the number of hits and false alarms, and the number of targets (P) and distractors (N).
    If the hit-rate or false-alarm rate is at the floor (0) or ceiling (1), a correction can be applied to avoid infinite
    d-prime values: "mcmillan" (Macmillan & Kaplan, 1985) or "loglinear" (Hautus, 1995).
    Returns the d-prime value.
    """
    assert np.isfinite(num_targets) and num_targets > 0, f"Number of targets must be finite and positive, got {num_targets}."
    assert np.isfinite(num_distractors) and num_distractors > 0, f"Number of distractors must be finite and positive, got {num_distractors}."
    assert 0 <= hits <= num_targets, f"Hits must be between 0 and num_targets ({num_targets}), got {hits}."
    assert 0 <= false_alarms <= num_distractors, f"False alarms must be between 0 and num_distractors ({num_distractors}), got {false_alarms}."
    hr = hits / num_targets
    far = false_alarms / num_distractors
    if 0 < hr < 1 and 0 < far < 1:
        # no correction needed
        return stats.norm.ppf(hr) - stats.norm.ppf(far)
    # need to apply correction for floor/ceiling effects
    if not correction:
        correction = "none"
    correction = correction.lower().strip().replace("_", "").replace("-", "").replace("&", "")
    if correction == "none" or not correction:
        return stats.norm.ppf(hr) - stats.norm.ppf(far)
    if correction in {"mk", "macmillan", "macmillankaplan"}:
        # apply Macmillan & Kaplan (1985) correction
        if hr == 0:
            hr = 0.5 / num_targets
        if hr == 1:
            hr = 1 - 0.5 / num_targets
        if far == 0:
            far = 0.5 / num_distractors
        if far == 1:
            far = 1 - 0.5 / num_distractors
        return stats.norm.ppf(hr) - stats.norm.ppf(far)
    if correction in {"loglinear", "hautus"}:
        # apply Hautus (1995) correction
        prevalence = num_targets / (num_targets + num_distractors)
        new_hits = hits + prevalence
        new_false_alarms = false_alarms + 1 - prevalence
        new_num_targets = num_targets + 2 * prevalence
        new_num_distractors = num_distractors + 2 * (1 - prevalence)
        hr = new_hits / new_num_targets
        far = new_false_alarms / new_num_distractors
        return stats.norm.ppf(hr) - stats.norm.ppf(far)
    raise ValueError(f"Unknown correction method '{correction}' for d-prime calculation.")


