from typing import Union, Tuple

import pandas as pd
import peyes

import constants as cnst
from data_models.LWSEnums import DominantEyeEnum


## Eye-Movement Detection Configurations ##
_MIN_EVENT_DURATION = 5     # ms
peyes.set_event_configurations("fixation", min_duration=50)
peyes.set_event_configurations("saccade", min_duration=_MIN_EVENT_DURATION)
_DETECTOR = peyes.create_detector(
    algorithm="Engbert",
    missing_value=cnst.MISSING_VALUE,
    min_event_duration=_MIN_EVENT_DURATION,
    pad_blinks_time=0,      # ms
)


def detect_eye_movements(
        gaze: pd.DataFrame,
        eye: DominantEyeEnum,
        viewer_distance_cm: float,
        detector=_DETECTOR,
        pixel_size_cm: float = cnst.PIXEL_SIZE_MM,
        only_labels: bool = True,
) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    t = gaze[cnst.TIME_STR].values
    x = gaze[cnst.RIGHT_X_STR if eye == DominantEyeEnum.RIGHT else cnst.LEFT_X_STR].values
    y = gaze[cnst.RIGHT_Y_STR if eye == DominantEyeEnum.RIGHT else cnst.LEFT_Y_STR].values
    labels, _ = detector.detect(t=t, x=x, y=y, viewer_distance_cm=viewer_distance_cm, pixel_size_cm=pixel_size_cm)
    labels = pd.Series(
        labels,
        index=gaze.index,
        name=cnst.RIGHT_LABEL_STR if eye == DominantEyeEnum.RIGHT else cnst.LEFT_LABEL_STR
    )
    if only_labels:
        return labels
    pupil = gaze[cnst.RIGHT_PUPIL_STR if eye == DominantEyeEnum.RIGHT else cnst.LEFT_PUPIL_STR].values
    events = peyes.create_events(
        labels=labels, t=t, x=x, y=y, pupil=pupil,
        viewer_distance=viewer_distance_cm, pixel_size=viewer_distance_cm
    )
    events = pd.Series(events, name=cnst.RIGHT_EVENT_STR if eye == DominantEyeEnum.RIGHT else cnst.LEFT_EVENT_STR)
    return labels, events
