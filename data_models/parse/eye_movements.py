from typing import Union, Tuple

import pandas as pd
import peyes

import config as cnfg
import constants as cnst
from data_models.LWSEnums import DominantEyeEnum


def configure_peyes() -> None:
    """
    Push this project's screen geometry and event-duration bounds into `peyes`' module-level configuration.

    `peyes` keeps these as global state. Left unset it falls back to its own defaults, which are *a* Tobii rig but not
    this one (53.1 cm wide vs `TOBII_MONITOR`'s 53.0), and to duration bounds never chosen for this paradigm. Both
    feed `get_outlier_reasons`, and therefore decide which fixations survive `read_data(drop_outliers=True)` - so
    they are set explicitly here rather than inherited.

    Idempotent; safe to call more than once.
    """
    peyes.set_screen_monitor(
        width_cm=cnst.TOBII_MONITOR.width_mm / 10,
        height_cm=cnst.TOBII_MONITOR.height_mm / 10,
        width_px=cnst.TOBII_MONITOR.width,
        height_px=cnst.TOBII_MONITOR.height,
    )
    peyes.set_event_configurations(
        "fixation", min_duration=cnfg.FIXATION_MIN_DURATION_MS, max_duration=cnfg.FIXATION_MAX_DURATION_MS,
    )
    peyes.set_event_configurations(
        "saccade", min_duration=cnfg.SACCADE_MIN_DURATION_MS, max_duration=cnfg.SACCADE_MAX_DURATION_MS,
    )


## Eye-Movement Detection Configurations ##
# NOTE: applied at import because `_DETECTOR` below is built at import. `run_pipeline` calls `configure_peyes()`
# again explicitly, so the configuration is visible at the entry point rather than only as an import side effect.
configure_peyes()
_DETECTOR = peyes.create_detector(
    algorithm="Engbert",
    missing_value=cnst.MISSING_VALUE,
    min_event_duration=cnfg.MIN_EVENT_DURATION_MS,
    pad_blinks_time=0,      # ms
)


def detect_eye_movements(
        gaze: pd.DataFrame,
        eye: DominantEyeEnum,
        viewer_distance_cm: float,
        detector=_DETECTOR,
        pixel_size_cm: float = cnst.PIXEL_SIZE_CM,
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
        viewer_distance=viewer_distance_cm, pixel_size=pixel_size_cm
    )
    events = pd.Series(events, name=cnst.RIGHT_EVENT_STR if eye == DominantEyeEnum.RIGHT else cnst.LEFT_EVENT_STR)
    return labels, events
