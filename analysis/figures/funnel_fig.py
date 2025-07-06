from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import config as cnfg
from pre_process.lws_funnel import LWS_FUNNEL_STEPS


def create_funnel_figure(data: pd.DataFrame, funnel_type: Literal["fixations", "visits"]) -> go.Figure:
    """ Creates a funnel figure for the provided multi-subject fixations/visits data. """
    funnel_sizes = _calc_funnel_sizes(data)
    fig = go.Figure()
    for subj_id, subj_funnel in funnel_sizes.groupby(cnfg.SUBJECT_STR):
        fig.add_trace(
            go.Funnel(
                name=f"Subject {subj_id}", legendgroup=f"Subject {subj_id}",
                y=subj_funnel["step"], x=subj_funnel["size"],
                textinfo="value+percent initial",
                marker=dict(color=px.colors.qualitative.Pastel[int(subj_id)]),
                connector=dict(visible=False),
            )
        )
    fig.update_layout(
        width=800, height=600,
        title=dict(text=f"LWS-{funnel_type.capitalize()} Funnel", font=cnfg.TITLE_FONT),
    )
    return fig


def _calc_funnel_sizes(data: pd.DataFrame,) -> pd.DataFrame:
    funnel_steps = [step for step in LWS_FUNNEL_STEPS if step in data.columns]
    funnel_sizes = dict()
    for subj_id, subj_fixations in data.groupby(cnfg.SUBJECT_STR):
        for i in range(len(funnel_steps)):
            curr_step = funnel_steps[i]
            curr_and_prev_steps = funnel_steps[:i + 1]
            step_size = subj_fixations[curr_and_prev_steps].all(axis=1).sum()
            funnel_sizes[(subj_id, curr_step)] = step_size
    funnel_sizes = (
        pd.Series(funnel_sizes)
        .reset_index(drop=False)
        .rename(columns={"level_0": cnfg.SUBJECT_STR, "level_1": "step", 0: "size"})
    )
    return funnel_sizes
