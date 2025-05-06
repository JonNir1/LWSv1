import numpy as np
import pandas as pd


#TODO:

def is_between_triggers(triggers: pd.Series, start: int, end: int) -> pd.Series:
    """
    Returns a boolean series indicating whether the values in the 'triggers' series occur after 'start' and before 'end'.
    """
    start_idxs = np.nonzero(triggers == start)[0]
    end_idxs = np.nonzero(triggers == end)[0]
    start_end_idxs = np.vstack([start_idxs, end_idxs]).T
    res = pd.Series(np.full_like(triggers, False, dtype=bool))
    for (start, end) in start_end_idxs:
        res.iloc[start:end + 1] = True
    return res

