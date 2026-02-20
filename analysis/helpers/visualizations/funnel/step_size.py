from typing import List

import pandas as pd
import plotly.graph_objects as go

import config as cnfg


def step_sizes_figure(
        funnel_sizes: pd.DataFrame,
        initial_step: str = "all",
        final_step: str = "final",
        title: str = "Funnel Step Sizes",
        show_individuals: bool = False,
) -> go.Figure:
    if initial_step not in funnel_sizes.columns:
        raise ValueError(f"Initial step `{initial_step}` not found in `funnel_sizes` data.")
    if final_step not in funnel_sizes.columns:
        raise ValueError(f"Final step `{final_step}` not found in `funnel_sizes` data.")
    if show_individuals and "subject" not in funnel_sizes.columns:
        raise ValueError(f"`funnel_sizes` data must contain `subject` column to show individual subject's funnels.")
    col_list = funnel_sizes.columns.tolist()
    initial_index = col_list.index(initial_step)
    final_index = col_list.index(final_step)
    step_order = col_list[initial_index:final_index + 1]
    sizes_long = funnel_sizes.melt(
        id_vars=["subject"] if show_individuals else [],
        value_vars=step_order,
        var_name="step",
        value_name="size",
    )
    fig = go.Figure()
    if show_individuals:
        for i, (subj_id, subj_data) in enumerate(sizes_long.groupby("subject")):
            fig = _add_funnel_trace(
                fig,
                subj_data,
                step_order,
                trace_name=f"Subject {subj_id}",
                trace_color=cnfg.get_discrete_color(i, loop=True),
            )
    else:
        fig = _add_funnel_trace(
            fig,
            sizes_long,
            step_order,
            trace_name="All Subjects",
            trace_color=cnfg.get_discrete_color("all"),
        )
    fig.update_layout(
        width=800, height=600,
        title=dict(text=title, font=cnfg.TITLE_FONT),
        showlegend=show_individuals,
    )
    return fig


def _add_funnel_trace(
        fig: go.Figure, trace_data: pd.DataFrame, step_order: List[str], trace_name: str, trace_color: str,
) -> go.Figure:
    step_sizes = (
        trace_data
        .groupby("step")["size"]
        .sum()
        .reset_index(drop=False)
        .sort_values(by=["step"], key=lambda steps_series: steps_series.map(lambda s: step_order.index(s)))
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
