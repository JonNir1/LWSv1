from typing import List, Optional, Union, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config as cnfg
import analysis.statistics as stat
from data_models.LWSEnums import ImageCategoryEnum, SearchArrayTypeEnum, DominantEyeEnum


def percent_identified_figure(ident_data: pd.DataFrame, drop_bads: bool = True) -> go.Figure:
    fig = _create_figure(
        ident_data,
        y_col=cnfg.IDENTIFIED_STR,
        scale=100,  # scale to percentage
        title="Percentage of Identified Targets",
        yaxes_title=f"% {cnfg.IDENTIFIED_STR.title()}",
        show_individual_trials=False,
        drop_bads=drop_bads,
    )
    fig.for_each_yaxis(lambda yax: yax.update(range=[0, 105]))
    return fig


def time_to_identification_figure(ident_data: pd.DataFrame, drop_bads: bool = True) -> go.Figure:
    fig = _create_figure(
        ident_data,
        y_col=cnfg.TIME_STR,
        scale=1.0 / cnfg.MILLISECONDS_IN_SECOND,  # scale to seconds
        title="Time of Target Identification",
        yaxes_title="Time (s)",
        show_individual_trials=True,
        drop_bads=drop_bads,
    )
    return fig


def identification_fixation_start_time_figure(
        ident_with_fixs_data: pd.DataFrame,
        dominant_eye: Optional[Union[DominantEyeEnum, Literal["left", "right"]]] = None,
        drop_bads: bool = True
) -> go.Figure:
    identifications = ident_with_fixs_data.copy()
    colname_format = f"%s_{cnfg.FIXATION_STR}_{cnfg.START_TIME_STR}"
    if dominant_eye is None:
        identifications[cnfg.START_TIME_STR] = ident_with_fixs_data[
            [colname_format % eye for eye in DominantEyeEnum]
        ].min(axis=1)
    else:
        dominant_eye = DominantEyeEnum(dominant_eye.lower()) if isinstance(dominant_eye, str) else dominant_eye
        identifications[cnfg.START_TIME_STR] = ident_with_fixs_data[colname_format % dominant_eye.name.lower()]
    fig = _create_figure(
        identifications,
        y_col=cnfg.START_TIME_STR,
        scale=1.0 / cnfg.MILLISECONDS_IN_SECOND,  # scale to seconds
        title="Identification-Fixation's Start-Time",
        yaxes_title="Time (s)",
        show_individual_trials=True,
        drop_bads=drop_bads,
    )
    return fig


def _create_figure(
        ident_data: pd.DataFrame,
        y_col: str,
        scale: float = 1.0,
        title: str = "",
        xaxes_title: str = "",
        yaxes_title: str = "",
        show_individual_trials: bool = True,
        drop_bads: bool = True,
) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2, shared_xaxes=False, shared_yaxes=False, specs=[[{"colspan": 2}, None], [{}, {}]],
        vertical_spacing=0.1, horizontal_spacing=0.05,
        subplot_titles=[
            cnfg.TRIAL_STR.title(),
            stat.TRIAL_TYPE_STR.replace("_", " ").title(),
            stat.TARGET_CATEGORY_STR.replace("_", " ").title(),
        ],
    )
    # Top Row: per trial (line plot)
    fig.add_trace(
        row=1, col=1,
        trace=_create_trial_summary_trace(ident_data, y_col, scale=scale)
    )
    if show_individual_trials:
        # add individual points for each trial in the top subplot
        category_traces = _create_target_category_traces(ident_data, y_col, scale=scale)
        for trace in category_traces:
            fig.add_trace(row=1, col=1, trace=trace)
    if drop_bads:
        ident_data = ident_data[~ident_data["is_bad"].astype(bool)]
    # Bottom Left: by trial type (bar plot)
    fig.add_trace(
        row=2, col=1,
        trace=_create_categorical_trace(
            ident_data, stat.TRIAL_TYPE_STR, y_col, SearchArrayTypeEnum, scale=scale
        )
    )
    # Bottom Right: by target category (bar plot)
    fig.add_trace(
        row=2, col=2,
        trace=_create_categorical_trace(
            ident_data, stat.TARGET_CATEGORY_STR, y_col, ImageCategoryEnum, scale=scale
        )
    )
    # update layout
    fig = _apply_layout_format(
        fig,
        title_text=title,
        xaxes_title=xaxes_title,
        yaxes_title=yaxes_title,
    )
    return fig


def _create_trial_summary_trace(
        data: pd.DataFrame, col_name: str, scale: float = 1.0
) -> go.Scatter:
    summary = data.groupby([cnfg.TRIAL_STR, "is_bad"])[col_name].mean().reset_index()
    scatter = go.Scatter(
        x=summary[cnfg.TRIAL_STR], y=summary[col_name] * scale, name='mean',
        mode='markers+lines',
        marker=dict(
            size=10, opacity=1.0,
            color=summary["is_bad"].map(lambda bad: 'red' if bad else 'black'),
            symbol=summary["is_bad"].map(lambda bad: 'x' if bad else 'circle'),
        ),
        line=dict(width=2, color='black'), connectgaps=False,
    )
    return scatter


def _create_target_category_traces(
        data: pd.DataFrame, col_name: str, scale: float = 1.0
) -> List[go.Scatter]:
    traces = []
    for cat in data[stat.TARGET_CATEGORY_STR].unique():
        cat_data = data[data[stat.TARGET_CATEGORY_STR] == cat]
        if not cat_data.empty:
            trace = go.Scatter(
                x=cat_data[cnfg.TRIAL_STR], y=cat_data[col_name] * scale,
                name=ImageCategoryEnum(cat).name.replace("_", " ").title(),
                mode='markers',
                marker=dict(
                    size=7.5, opacity=0.5,
                    color=cnfg.get_discrete_color(ImageCategoryEnum(cat).value),
                    symbol=cat_data["is_bad"].map(lambda bad: 'x' if bad else 'circle'),
                ),
            )
            traces.append(trace)
    return traces


def _create_categorical_trace(
        data: pd.DataFrame, x_col, y_col, enum_class, scale=1.0
) -> go.Bar:
    stats = stat.calculate_stats_by_enum(data[[x_col, y_col]], x_col, enum_class)
    bars = go.Bar(
        x=stats[x_col], y=stats['mean'] * scale,
        error_y=dict(type='data', array=stats['sem'] * scale, visible=True),
        name=x_col,
        marker_color=stats['color'],
    )
    return bars


def _apply_layout_format(
        fig: go.Figure,
        title_text: str = "",
        xaxes_title: str = "",
        yaxes_title: str = "",
) -> go.Figure:
    """ Helper function to format the layout of a figure according to the standard """
    fig.for_each_xaxis(lambda xax: xax.update(
        title=dict(text=xaxes_title, standoff=cnfg.AXIS_LABEL_STANDOFF, font=cnfg.AXIS_LABEL_FONT, ),
        showline=False,
        showgrid=False, gridcolor=cnfg.GRID_LINE_COLOR, gridwidth=cnfg.GRID_LINE_WIDTH,
        zeroline=False, zerolinecolor=cnfg.GRID_LINE_COLOR, zerolinewidth=cnfg.ZERO_LINE_WIDTH,
        tickfont=cnfg.AXIS_TICK_FONT,
    ))
    fig.for_each_yaxis(lambda yax: yax.update(
        title=dict(text=yaxes_title, standoff=cnfg.AXIS_LABEL_STANDOFF, font=cnfg.AXIS_LABEL_FONT, ),
        showline=True,
        showgrid=True, gridcolor=cnfg.GRID_LINE_COLOR, gridwidth=cnfg.GRID_LINE_WIDTH,
        zeroline=True, zerolinecolor=cnfg.GRID_LINE_COLOR, zerolinewidth=cnfg.ZERO_LINE_WIDTH,
    ))
    fig.update_layout(
        title=dict(text=title_text, font=cnfg.TITLE_FONT),
        # paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        showlegend=False,
    )
    return fig

