from typing import List, Optional, Union, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config as cnfg
import analysis.figures.within_subject_figures.statistics as stat
from data_models.LWSEnums import ImageCategoryEnum, SearchArrayCategoryEnum, DominantEyeEnum


def identification_rate_figure(ident_data: pd.DataFrame, drop_bads: bool = True) -> go.Figure:
    col_name = "is_identified"
    is_identified = pd.DataFrame(
        np.isfinite(ident_data[cnfg.TIME_STR].values), index=ident_data.index, columns=[col_name]
    )
    fig = _create_figure(
        is_identified,
        y_col=col_name,
        scale=100,  # scale to percentage
        title="Percentage of Identified Targets",
        yaxes_title=f"% identified",
        show_individual_trials=False,
        drop_bads=drop_bads,
    )
    fig.for_each_yaxis(lambda yax: yax.update(range=[0, 105]))
    return fig


def identification_time_figure(ident_data: pd.DataFrame, drop_bads: bool = True) -> go.Figure:
    ident_data2 = ident_data.copy()
    ident_data2.loc[
        # set not-identified distances to NaN to avoid inf-points in the plot
        ~np.isfinite(ident_data[cnfg.TIME_STR].values), cnfg.TIME_STR
    ] = np.nan
    fig = _create_figure(
        ident_data2,
        y_col=cnfg.TIME_STR,
        scale=1.0 / cnfg.MILLISECONDS_IN_SECOND,  # scale to seconds
        title="Time of Target Identification",
        yaxes_title="Time (s)",
        show_individual_trials=True,
        drop_bads=drop_bads,
    )
    return fig


def identification_distance_figure(
        ident_data: pd.DataFrame, px2deg: float, drop_bads: bool = True
) -> go.Figure:
    assert px2deg >= 0, f"`px2deg` must be a non-negative float, got {px2deg}."
    dist_col = cnfg.DISTANCE_PX_STR
    ident_data2 = ident_data.copy()
    ident_data2.loc[
        # set not-identified distances to NaN to avoid inf-points in the plot
        ~np.isfinite(ident_data[cnfg.TIME_STR].values), dist_col
    ] = np.nan
    fig = _create_figure(
        ident_data2,
        y_col=dist_col,
        scale=px2deg,   # scale to DVA
        title="Distance to Target at Identification",
        yaxes_title="Distance (DVA)",
        show_individual_trials=True,
        drop_bads=drop_bads,
    )
    fig.add_hline(  # mark the on-target threshold
        row=1, col=1, y=cnfg.ON_TARGET_THRESHOLD_DVA,
        line=dict(dash="dash", color="gray", width=1.5),
        annotation=dict(text="On-Target Threshold", font=cnfg.AXIS_LABEL_FONT, showarrow=False,),
        annotation_position="top left",
    )
    return fig


def identification_event_start_time_figure(
        ident_data: pd.DataFrame,
        event_data: pd.DataFrame,
        event_type: Literal['fixation', 'visit'],
        temporal_window: float = cnfg.VISIT_MERGING_TIME_THRESHOLD,
        dominant_eye: Optional[Union[DominantEyeEnum, Literal["left", "right"]]] = None,
        drop_bads: bool = True
) -> go.Figure:
    # prepare the data
    identifications_with_events = _append_to_identifications(ident_data, event_data, event_type, temporal_window)
    colname_format = f"%s_{event_type.lower()}_{cnfg.START_TIME_STR}"
    if dominant_eye is None:
        identifications_with_events[cnfg.START_TIME_STR] = identifications_with_events[
            [colname_format % eye for eye in DominantEyeEnum]
        ].min(axis=1)
    else:
        dominant_eye = DominantEyeEnum(dominant_eye.lower()) if isinstance(dominant_eye, str) else dominant_eye
        identifications_with_events[cnfg.START_TIME_STR] = identifications_with_events[
            colname_format % dominant_eye.name.lower()
        ]

    # create the figure
    fig = _create_figure(
        identifications_with_events,
        y_col=cnfg.START_TIME_STR,
        scale=1.0 / cnfg.MILLISECONDS_IN_SECOND,  # scale to seconds
        title=f"Identification's {event_type.title()} Start-Time",
        yaxes_title="Time (s)",
        show_individual_trials=True,
        drop_bads=drop_bads,
    )
    return fig


def identification_event_distance_figure(
        ident_data: pd.DataFrame,
        event_data: pd.DataFrame,
        event_type: Literal['fixation', 'visit'],
        px2deg: float,
        temporal_window: float = cnfg.VISIT_MERGING_TIME_THRESHOLD,
        dominant_eye: Optional[Union[DominantEyeEnum, Literal["left", "right"]]] = None,
        drop_bads: bool = True
) -> go.Figure:
    assert px2deg >= 0, f"`px2deg` must be a non-negative float, got {px2deg}."
    # prepare the data
    identifications_with_events = _append_to_identifications(ident_data, event_data, event_type, temporal_window)
    colname_format = f"%s_{event_type.lower()}_{cnfg.TARGET_DISTANCE_STR}"
    if dominant_eye is None:
        identifications_with_events[cnfg.TARGET_DISTANCE_STR] = identifications_with_events[
            [colname_format % eye for eye in DominantEyeEnum]
        ].min(axis=1)
    else:
        dominant_eye = DominantEyeEnum(dominant_eye.lower()) if isinstance(dominant_eye, str) else dominant_eye
        identifications_with_events[cnfg.TARGET_DISTANCE_STR] = identifications_with_events[
            colname_format % dominant_eye.name.lower()
        ]
    identifications_with_events.loc[
        # set not-identified distances to NaN to avoid inf-points in the plot
        ~np.isfinite(ident_data[cnfg.TIME_STR].values), cnfg.TARGET_DISTANCE_STR
    ] = np.nan

    # create the figure
    fig = _create_figure(
        identifications_with_events,
        y_col=cnfg.TARGET_DISTANCE_STR,
        scale=px2deg,   # scale to DVA
        title=f"Identification's {event_type.title()} Distance to Target",
        yaxes_title="Distance (DVA)",
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
            cnfg.TRIAL_CATEGORY_STR.replace("_", " ").title(),
            cnfg.TARGET_CATEGORY_STR.replace("_", " ").title(),
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
        ident_data = ident_data[~ident_data[f"bad_{cnfg.TRIAL_STR}"].astype(bool)]
    # Bottom Left: by trial type (bar plot)
    fig.add_trace(
        row=2, col=1,
        trace=_create_categorical_trace(
            ident_data, cnfg.TRIAL_CATEGORY_STR, y_col, SearchArrayCategoryEnum, scale=scale
        )
    )
    # Bottom Right: by target category (bar plot)
    fig.add_trace(
        row=2, col=2,
        trace=_create_categorical_trace(
            ident_data, cnfg.TARGET_CATEGORY_STR, y_col, ImageCategoryEnum, scale=scale
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
    summary = data.groupby([cnfg.TRIAL_STR, f"bad_{cnfg.TRIAL_STR}"])[col_name].mean().reset_index()
    scatter = go.Scatter(
        x=summary[cnfg.TRIAL_STR], y=summary[col_name] * scale, name='mean',
        mode='markers+lines',
        marker=dict(
            size=10, opacity=1.0,
            color=summary[f"bad_{cnfg.TRIAL_STR}"].map(lambda bad: 'red' if bad else 'black'),
            symbol=summary[f"bad_{cnfg.TRIAL_STR}"].map(lambda bad: 'x' if bad else 'circle'),
        ),
        line=dict(width=2, color='black'), connectgaps=False,
    )
    return scatter


def _create_target_category_traces(
        data: pd.DataFrame, col_name: str, scale: float = 1.0
) -> List[go.Scatter]:
    traces = []
    for cat in data[cnfg.TARGET_CATEGORY_STR].unique():
        cat_data = data[data[cnfg.TARGET_CATEGORY_STR] == cat]
        if not cat_data.empty:
            trace = go.Scatter(
                x=cat_data[cnfg.TRIAL_STR], y=cat_data[col_name] * scale,
                name=ImageCategoryEnum(cat).name.replace("_", " ").title(),
                mode='markers',
                marker=dict(
                    size=7.5, opacity=0.5,
                    color=cnfg.get_discrete_color(ImageCategoryEnum(cat).value),
                    symbol=cat_data[f"bad_{cnfg.TRIAL_STR}"].map(lambda bad: 'x' if bad else 'circle'),
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


def _append_to_identifications(
        identifications: pd.DataFrame,
        events: pd.DataFrame,
        event_type: Literal['fixation', 'visit'],
        temporal_window: float,
) -> pd.DataFrame:
    """
    Match events (fixations or visits) to identifications based on the identification time. If an identification
    occurs during an event, this event is matched to the identification. If not, the closest preceding event within a
    temporal-window is matched. The matched event's data - start and end times, event ID, and distance to target -
    are added to the identifications DataFrame.

    :param identifications: DataFrame; identifications data.
    :param events: DataFrame; events data (fixations or visits).
    :param event_type: str; either 'fixation' or 'visit'.
    :param temporal_window: float; maximum allowed time-difference (in ms) between identification and event end time.

    :return: pd.DataFrame; updated identifications DataFrame with additional columns for matched events.
    """
    event_type = event_type
    if event_type not in [cnfg.FIXATION_STR, cnfg.VISIT_STR]:
        raise ValueError(
            f"Invalid event_type: {event_type}. Must be `{cnfg.FIXATION_STR}` or `{cnfg.VISIT_STR}`."
        )
    # add columns to identifications DataFrame
    new_identifications = identifications.copy()
    for eye in DominantEyeEnum:
        for col in ["", f"{cnfg.START_TIME_STR}", f"{cnfg.END_TIME_STR}", cnfg.TARGET_DISTANCE_STR]:
            col_name = f"{eye}_{event_type}_{col}".strip("_")
            new_identifications[col_name] = np.nan

    # match event data to identifications
    for i, row in new_identifications.iterrows():
        trial, target, ident_time = row[cnfg.TRIAL_STR], row[cnfg.TARGET_STR], row[cnfg.TIME_STR]
        if not np.isfinite(ident_time):
            # no identification time, skip
            continue
        for eye in DominantEyeEnum:
            try:
                trial_events = events.xs((trial, eye), level=[cnfg.TRIAL_STR, cnfg.EYE_STR])
            except KeyError as _e:
                # no events in this trial for this eye, skip
                continue
            # check if identified during an event
            during = trial_events[
                (trial_events[cnfg.START_TIME_STR] <= ident_time) & (ident_time <= trial_events[cnfg.END_TIME_STR])
            ]
            if not during.empty:
                # identified during an event, take the only one
                assert len(during) == 1
                chosen = during.iloc[0]
            else:
                # not identified during an event, find the closest preceding event
                delta_t = ident_time - trial_events[cnfg.START_TIME_STR]
                trial_events_before = trial_events[
                    # only consider visits that ended before identification and within the allowed time-difference
                    (0 <= delta_t) & (delta_t <= temporal_window)
                ].copy()
                if trial_events_before.empty:
                    # no events before identification, skip
                    continue
                trial_events_before["delta_t"] = ident_time - trial_events_before[cnfg.END_TIME_STR]
                chosen = trial_events_before.loc[trial_events_before["delta_t"].idxmin()]

            # write chosen event's data to identifications DataFrame
            event_id = chosen.name if event_type == cnfg.FIXATION_STR else chosen.name[1]  # fixation ID or visit ID
            new_identifications.at[i, f"{eye}_{event_type}"] = event_id
            new_identifications.at[i, f"{eye}_{event_type}_{cnfg.START_TIME_STR}"] = chosen[cnfg.START_TIME_STR]
            new_identifications.at[i, f"{eye}_{event_type}_{cnfg.END_TIME_STR}"] = chosen[cnfg.END_TIME_STR]
            distance_col = target if event_type == cnfg.FIXATION_STR else f"min_{cnfg.DISTANCE_STR}"
            new_identifications.at[i, f"{eye}_{event_type}_{cnfg.TARGET_STR}_{cnfg.DISTANCE_STR}"] = chosen[distance_col]
    return new_identifications
