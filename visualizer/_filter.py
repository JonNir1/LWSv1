import warnings

import pandas as pd

from constants import SUBJECT_STR, TRIAL_STR


def filter_to_trial(df: pd.DataFrame, trial: "Trial") -> pd.DataFrame:
    """
    Filter a DataFrame to rows matching the given Trial's subject and trial number.
    If the DataFrame lacks subject/trial columns, returns it as-is (assumes pre-filtered).
    """
    has_subject = SUBJECT_STR in df.columns
    has_trial = TRIAL_STR in df.columns

    if not has_subject and not has_trial:
        return df

    if has_subject != has_trial:
        present = SUBJECT_STR if has_subject else TRIAL_STR
        missing = TRIAL_STR if has_subject else SUBJECT_STR
        warnings.warn(
            f"DataFrame has '{present}' column but not '{missing}'. "
            f"Filtering on '{present}' only."
        )

    mask = pd.Series(True, index=df.index)
    if has_subject:
        mask &= df[SUBJECT_STR] == trial._subject.id
    if has_trial:
        mask &= df[TRIAL_STR] == trial.trial_num

    return df.loc[mask]
