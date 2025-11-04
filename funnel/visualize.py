from typing import Literal

import pandas as pd
import plotly.graph_objects as go

import config as cnfg


def create_funnel_figure(
        step_sizes: pd.DataFrame,
        funnel_type: Literal["lws", "target_return"],
        event_type: Literal["fixations", "visits"],
        show_individuals: bool = False,
) -> go.Figure:
    """ Creates a funnel figure for the provided multi-subject funnel sizes. """
    assert "step" in step_sizes.columns, f"Data must contain `step` column."
    fig = go.Figure()
    if show_individuals:
        assert cnfg.SUBJECT_STR in step_sizes.columns, f"Data must contain `{cnfg.SUBJECT_STR}` column."
        for i, (subj_id, subj_data) in enumerate(step_sizes.groupby(cnfg.SUBJECT_STR)):
            fig = _add_funnel_trace(
                fig,
                subj_data,
                trace_name=f"Subject {subj_id}",
                trace_color=cnfg.get_discrete_color(i, loop=True),
            )
    else:
        fig = _add_funnel_trace(
            fig,
            step_sizes,
            trace_name=f"{cnfg.ALL_STR.capitalize()} {event_type.capitalize()}s",
            trace_color=cnfg.get_discrete_color("all"),
        )
    funnel_name = "LWS" if funnel_type == "lws" else "Target-Return"
    fig.update_layout(
        width=800, height=600,
        title=dict(text=f"{funnel_name} {event_type.capitalize()} Funnel", font=cnfg.TITLE_FONT),
        showlegend=show_individuals,
    )
    return fig


def _add_funnel_trace(
        fig: go.Figure, trace_data: pd.DataFrame, trace_name: str, trace_color: str,
) -> go.Figure:
    step_order = cnfg.LWS_FUNNEL_STEPS + [step for step in cnfg.TARGET_RETURN_FUNNEL_STEPS if step not in cnfg.LWS_FUNNEL_STEPS]
    step_sizes = (
        trace_data
        .groupby("step")["size"]
        .sum()
        .reset_index(drop=False)
        .sort_values(by="step", key=lambda steps_series: steps_series.map(lambda step: step_order.index(step)))
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
