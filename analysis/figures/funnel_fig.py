from typing import Literal

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import config as cnfg
from analysis.lws_funnel import LWS_FUNNEL_STEPS

_COLOR_SCHEME = px.colors.qualitative.Dark24


def create_funnel_figure(
        step_sizes: pd.DataFrame, funnel_type: Literal["fixations", "visits"], show_individuals: bool = False,
) -> go.Figure:
    """ Creates a funnel figure for the provided multi-subject LWS funnel sizes. """
    assert "step" in step_sizes.columns, f"Data must contain `step` column."
    fig = go.Figure()
    if show_individuals:
        assert cnfg.SUBJECT_STR in step_sizes.columns, f"Data must contain `{cnfg.SUBJECT_STR}` column."
        for i, (subj_id, subj_data) in enumerate(step_sizes.groupby(cnfg.SUBJECT_STR)):
            fig = _add_funnel_trace(
                fig,
                subj_data,
                trace_name=f"Subject {subj_id}",
                trace_color=_COLOR_SCHEME[i % len(_COLOR_SCHEME)],
            )
    else:
        fig = _add_funnel_trace(
            fig,
            step_sizes,
            trace_name=f"{cnfg.ALL_STR.capitalize()} {funnel_type.capitalize()}s",
            trace_color=_COLOR_SCHEME[0],
        )
    fig.update_layout(
        width=800, height=600,
        title=dict(text=f"LWS-{funnel_type.capitalize()} Funnel", font=cnfg.TITLE_FONT),
        showlegend=show_individuals,
    )
    return fig


def _add_funnel_trace(
        fig: go.Figure, trace_data: pd.DataFrame, trace_name: str, trace_color: str,
) -> go.Figure:
    step_sizes = (
        trace_data
        .groupby("step")["size"]
        .sum()
        .reset_index(drop=False)
        .sort_values(by="step", key=lambda steps_series: steps_series.map(lambda step: LWS_FUNNEL_STEPS.index(step)))
    )
    fig.add_trace(go.Funnel(
        name=trace_name, legendgroup=trace_name,
        y=step_sizes["step"], x=step_sizes["size"],
        textinfo="value+percent initial",
        marker=dict(color=trace_color),
        connector=dict(visible=False),
        showlegend=True,
    ))
    return fig
