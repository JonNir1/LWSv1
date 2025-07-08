from typing import Literal

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import config as cnfg
from analysis.pipeline.lws_funnel import LWS_FUNNEL_STEPS

_COLOR_SCHEME = px.colors.qualitative.Dark24


def create_funnel_figure(
        data: pd.DataFrame, funnel_type: Literal["fixations", "visits"], show_individuals: bool = False,
) -> go.Figure:
    """ Creates a funnel figure for the provided multi-subject fixations/visits data. """
    num_colors = len(_COLOR_SCHEME)
    fig = go.Figure()
    if show_individuals:
        for i, (subj_id, subj_data) in enumerate(data.groupby(cnfg.SUBJECT_STR)):
            name = f"Subject {subj_id}"
            subj_funnel_sizes = _calc_funnel_sizes(subj_data, name)
            fig.add_trace(
                go.Funnel(
                    name=name, legendgroup=name,
                    y=subj_funnel_sizes["step"], x=subj_funnel_sizes["size"],
                    textinfo="value+percent initial",
                    marker=dict(color=_COLOR_SCHEME[i % num_colors]),
                    connector=dict(visible=False),
                    showlegend=True,
                )
            )
    else:
        name = cnfg.ALL_STR
        funnel_sizes = _calc_funnel_sizes(data, name)
        fig.add_trace(
            go.Funnel(
                name=name, legendgroup=name,
                y=funnel_sizes["step"], x=funnel_sizes["size"],
                textinfo="value+percent initial",
                marker=dict(color=_COLOR_SCHEME[0]),
                connector=dict(visible=False),
                showlegend=False,
            )
        )
    fig.update_layout(
        width=800, height=600,
        title=dict(text=f"LWS-{funnel_type.capitalize()} Funnel", font=cnfg.TITLE_FONT),
    )
    return fig


def _calc_funnel_sizes(subset: pd.DataFrame, name: str) -> pd.DataFrame:
    funnel_steps = [step for step in LWS_FUNNEL_STEPS if step in subset.columns]
    funnel_sizes = dict()
    for i in range(len(funnel_steps)):
        curr_step = funnel_steps[i]
        curr_and_prev_steps = funnel_steps[:i + 1]
        step_size = subset[curr_and_prev_steps].all(axis=1).sum()
        funnel_sizes[(name, curr_step)] = step_size
    funnel_sizes = (
        pd.Series(funnel_sizes)
        .reset_index(drop=False)
        .rename(columns={"level_0": cnfg.SUBJECT_STR, "level_1": "step", 0: "size"})
    )
    return funnel_sizes
