import os
import copy

import numpy as np
import pandas as pd
import peyes

import config as cnfg
from data_models.LWSEnums import DominantEyeEnum
from data_models.Subject import Subject

SUBJ_PREFIX = "v4-1-1"
SUBJ_PATH = os.path.join(cnfg.RAW_DATA_PATH, f"{SUBJ_PREFIX} GalChen Demo")


# read subject data
subj = Subject.from_raw(
    exp_name=cnfg.EXPERIMENT_NAME, subject_id=1, session=1, dirname="v4-1-1 GalChen Demo", verbose=True
)

# extract specific trial
trial = subj.get_trials()[33]

target_identification_data = trial.extract_target_identification()      # identify targets' identification time
# TODO: ignore if distance is above predefined threshold (replace `time>threshold` with inf)

# extract fixations
left_em = trial.get_eye_movements(eye=DominantEyeEnum.Left)
left_fixs = list(filter(lambda e: e.label == 1, left_em))
left_fixs_df = peyes.summarize_events(left_fixs)
right_em = trial.get_eye_movements(eye=DominantEyeEnum.Right)
right_fixs = list(filter(lambda e: e.label == 1, right_em))
right_fixs_df = peyes.summarize_events(right_fixs)
fixs_df = pd.concat([left_fixs_df, right_fixs_df], keys=[cnfg.LEFT_STR, cnfg.RIGHT_STR], axis=0)
fixs_df.index.names = ["eye", "fixation"]
del left_em, right_em, left_fixs, right_fixs, left_fixs_df, right_fixs_df

# calc distance to targets
targets = trial.get_targets()
center_pixels = np.vstack(fixs_df['center_pixel'].values)
fix_dists = trial.calculate_target_distances(center_pixels[:, 0], center_pixels[:, 1]) * subj.px2deg
# TODO: determine if fixation close to target, number proximal fixations
# TODO: determine if fixation is prior to target identification
# TODO: determine if fixation within bottom strip

# LWS instance:
#   fixation on target
#   before identification
#   not coinciding with trial end
#   succeeding fixation is not in bottom strip
#   succeeding fixation is not same-target's identification fixation
