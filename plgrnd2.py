from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

import config as cnfg
from pre_process.pipeline import read_subject, process_subject

pio.renderers.default = "browser"



## Read Subject Data
_SUBJECTS = [(1, "v4-1-1 GalChen Demo"), (2, "v4-2-1 Netta Demo"), (3, "v4-3-1 Rotem Demo")]
PROCESSED_SUBJECTS = dict()
for i, (subject_id, data_dir) in enumerate(_SUBJECTS):
    subj = read_subject(
        subject_id=subject_id, exp_name=cnfg.EXPERIMENT_NAME, session=1, data_dir=data_dir, verbose=True
    )
    targets, metadata, idents, fixations, visits = process_subject(subj, verbose=True)
    results = {
        cnfg.SUBJECT_STR: subj,
        cnfg.TARGET_STR: targets,
        cnfg.METADATA_STR: metadata,
        "identification": idents,
        cnfg.FIXATION_STR: fixations,
        cnfg.VISIT_STR: visits,
    }
    PROCESSED_SUBJECTS[subject_id] = results
del _SUBJECTS, i, subject_id, data_dir, subj, targets, metadata, idents, fixations, visits, results

def concat_subject_results(to_concat: Literal["identification", "fixation", "visit"]) -> pd.DataFrame:
    conatenated = (
        pd.concat([subj_res[to_concat] for subj_res in PROCESSED_SUBJECTS.values()], keys=PROCESSED_SUBJECTS.keys(), axis=0)
        .reset_index(drop=False)
        .rename(columns={"level_0": cnfg.SUBJECT_STR})
        .drop(columns=["level_1"])
    )
    return conatenated
all_idents = concat_subject_results("identification")
all_fixations = concat_subject_results(cnfg.FIXATION_STR)
all_visits = concat_subject_results(cnfg.VISIT_STR)


## LWS Funnel - fixations
from analysis.figures.funnel_fig import create_funnel_figure
create_funnel_figure(all_fixations, "fixations").show()
