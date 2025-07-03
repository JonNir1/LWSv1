from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

import config as cnfg
from pre_process.pipeline import parse_subject, process_subject

pio.renderers.default = "browser"



## Read Subject Data
_SUBJECTS = [(1, "v4-1-1 GalChen Demo"), (2, "v4-2-1 Netta Demo"), (3, "v4-3-1 Rotem Demo")]
PROCESSED_SUBJECTS = dict()
for i, (subject_id, data_dir) in enumerate(_SUBJECTS):
    subj = parse_subject(
        subject_id=subject_id, exp_name=cnfg.EXPERIMENT_NAME, data_dir=data_dir, session=1, verbose=True
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


## LWS Funnel - fixations
from analysis.lws_funnel import fixation_funnel
fixs = get_fixations(SUBJECT[0], save=True, verbose=True)
fixs_funnel = fixation_funnel(fixations=fixs)
fix_funnel_sizes = {k: len(v) for (k, v) in fixs_funnel.items()}
fix_funnel_fig = go.Figure(go.Funnelarea(
    labels=list(fix_funnel_sizes.keys()),
    values=list(fix_funnel_sizes.values()),
    textinfo="label+value",
    title=dict(text="<b>LWS Funnel - Fixations</b>", font=dict(size=20)),
))
fix_funnel_fig.show()
lws_fixations = fixs_funnel["not_end_with_trial"]

## LWS Funnel - visits
from analysis.lws_funnel import visit_funnel
visits = get_visits(SUBJECT[0], save=True, verbose=True)
visits_funnel = visit_funnel(visits=visits, distance_col="weighted_distance")
vis_funnel_sizes = {k: len(v) for (k, v) in visits_funnel.items()}
vis_funnel_fig = go.Figure(go.Funnelarea(
    labels=list(vis_funnel_sizes.keys()),
    values=list(vis_funnel_sizes.values()),
    textinfo="label+value",
    title=dict(text="<b>LWS Funnel - Visits</b>", font=dict(size=20)),
))
vis_funnel_fig.show()
lws_visits = visits_funnel["not_end_with_trial"]
