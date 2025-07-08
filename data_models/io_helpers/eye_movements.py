from typing import Union, Tuple

import pandas as pd
import peyes

import config as cnfg
from data_models.LWSEnums import DominantEyeEnum

def detect_eye_movements(
        gaze: pd.DataFrame,
        eye: DominantEyeEnum,
        viewer_distance_cm: float,
        detector = cnfg.DETECTOR,
        pixel_size_cm: float = cnfg.PIXEL_SIZE_MM,
        only_labels: bool = True,
) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    t = gaze[cnfg.TIME_STR].values
    x = gaze[cnfg.RIGHT_X_STR if eye == DominantEyeEnum.RIGHT else cnfg.LEFT_X_STR].values
    y = gaze[cnfg.RIGHT_Y_STR if eye == DominantEyeEnum.RIGHT else cnfg.LEFT_Y_STR].values
    labels, _ = detector.detect(t=t, x=x, y=y, viewer_distance_cm=viewer_distance_cm, pixel_size_cm=pixel_size_cm)
    labels = pd.Series(
        labels,
        index=gaze.index,
        name=cnfg.RIGHT_LABEL_STR if eye == DominantEyeEnum.RIGHT else cnfg.LEFT_LABEL_STR
    )
    if only_labels:
        return labels
    pupil = gaze[cnfg.RIGHT_PUPIL_STR if eye == DominantEyeEnum.RIGHT else cnfg.LEFT_PUPIL_STR].values
    events = peyes.create_events(
        labels=labels, t=t, x=x, y=y, pupil=pupil,
        viewer_distance=viewer_distance_cm, pixel_size=viewer_distance_cm
    )
    events = pd.Series(events, name=cnfg.RIGHT_EVENT_STR if eye == DominantEyeEnum.RIGHT else cnfg.LEFT_EVENT_STR)
    return labels, events
