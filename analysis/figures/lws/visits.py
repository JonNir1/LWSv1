from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config as cnfg
from data_models.LWSEnums import SignalDetectionCategoryEnum


# TODO


def _visits_figure(
        visits: pd.DataFrame, metadata: pd.DataFrame, identifications: pd.DataFrame, targets: pd.DataFrame,
) -> go.Figure:
    column_mapping = {
        "Start-Time (ms)" : cnfg.START_TIME_STR,
        "Duration (ms)" : "duration",
        "Num Fixations" : cnfg.NUM_FIXATIONS_STR,
    }
    fig = make_subplots(
        rows=2, cols=len(column_mapping), column_titles=column_mapping,
        shared_yaxes=False, shared_xaxes=True,
        vertical_spacing=0.08, horizontal_spacing=0.02,
    )
    for c, (col_title, col_name) in enumerate(column_titles.values()):
        col_name = cnfg.TIME_STR if c == 0 else cnfg.DISTANCE_DVA_STR
        all_data = data[[cnfg.SUBJECT_STR, cnfg.TRIAL_STR, cnfg.TRIAL_CATEGORY_STR, col_name]]

        # top row: all data points
        texts = all_data.apply(
            lambda row:
            f"{cnfg.SUBJECT_STR.capitalize()} {row[cnfg.SUBJECT_STR]:02d} "
            f"{cnfg.TRIAL_STR.capitalize()} {row[cnfg.TRIAL_STR]:02d} ({row[cnfg.TRIAL_CATEGORY_STR]})",
            axis=1
        )
        fig.add_trace(
            row=1, col=c + 1, trace=go.Violin(
                y0=col_name, x=all_data[col_name], text=texts,
                name=cnfg.ALL_STR.upper(), legendgroup=cnfg.ALL_STR.upper(), width=1.75,
                hoverinfo="x+y+text", orientation="h", side="positive", spanmode='hard',
                marker=dict(color=cnfg.get_discrete_color("all")),
                box=dict(visible=False),
                meanline=dict(visible=True),
                points="all", pointpos=-0.5,
                showlegend=c==0,
            )
        )

        # bottom row: per-subject distribution
        for subj_id in all_data[cnfg.SUBJECT_STR].unique():
            subj_string = f"{cnfg.SUBJECT_STR.capitalize()} {subj_id:02d}"
            subj_data = all_data[all_data[cnfg.SUBJECT_STR] == subj_id]
            texts = subj_data.apply(
                lambda row:
                f"{subj_string} {cnfg.TRIAL_STR.capitalize()} {row[cnfg.TRIAL_STR]:02d} ({row[cnfg.TRIAL_CATEGORY_STR]})",
                axis=1
            )
            fig.add_trace(
                row=2, col=c + 1, trace=go.Violin(
                    y0=col_name, x=subj_data[col_name], text=texts,
                    name=subj_string, legendgroup=subj_string, width=1.75,
                    hoverinfo="all", orientation="h", side="positive", spanmode='hard',
                    marker=dict(color=cnfg.get_discrete_color(subj_id, loop=True), opacity=0.5),
                    box=dict(visible=False),
                    meanline=dict(visible=True),
                    points="all", pointpos=-0.5,
                    showlegend=c==0,
                )
            )

    # Update annotations, axes, and layout
    fig.update_annotations(font=cnfg.AXIS_LABEL_FONT)
    fig.update_yaxes(showticklabels=False)  # Hide y-axis labels
    fig.update_xaxes(
        title=None, showline=False,
        showgrid=True, gridcolor=cnfg.GRID_LINE_COLOR, gridwidth=cnfg.GRID_LINE_WIDTH,
        zeroline=False, zerolinecolor=cnfg.GRID_LINE_COLOR, zerolinewidth=cnfg.ZERO_LINE_WIDTH,
        tickfont=cnfg.AXIS_TICK_FONT,
    )
    fig.update_layout(
        width=1200, height=675,
        title=dict(text="Target-Detection Comparison", font=cnfg.TITLE_FONT),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        # plot_bgcolor='rgba(0, 0, 0, 0)',
        showlegend=True,
    )
    return fig


def _extract_data(
        visits: pd.DataFrame, metadata: pd.DataFrame, identifications: pd.DataFrame, targets: pd.DataFrame,
        visit_category: Literal["all", "first", "identification", "non-identification"]
) -> pd.DataFrame:
    data = (
        visits
        .merge(
            metadata[[cnfg.SUBJECT_STR, cnfg.TRIAL_STR, cnfg.TRIAL_CATEGORY_STR]],
            on=[cnfg.SUBJECT_STR, cnfg.TRIAL_STR], how="left"
        )
        .merge(
            identifications.loc[
                np.isin(
                    identifications[cnfg.IDENTIFICATION_CATEGORY_STR],
                    [SignalDetectionCategoryEnum.HIT, SignalDetectionCategoryEnum.MISS]
                ),
                [cnfg.SUBJECT_STR, cnfg.TRIAL_STR, cnfg.TARGET_STR, cnfg.TIME_STR]
            ],
            on=[cnfg.SUBJECT_STR, cnfg.TRIAL_STR, cnfg.TARGET_STR], how="left"
        )
        .merge(
            targets[[cnfg.SUBJECT_STR, cnfg.TRIAL_STR, cnfg.TARGET_STR, cnfg.CATEGORY_STR]],
            on=[cnfg.SUBJECT_STR, cnfg.TRIAL_STR, cnfg.TARGET_STR], how="left"
        )
        .rename(columns={cnfg.CATEGORY_STR: cnfg.TARGET_CATEGORY_STR, cnfg.TIME_STR: cnfg.TARGET_TIME_STR})
    )
    data["num_fixations"] = data[cnfg.EVENT_STR].map(len)
    data["is_ident_visit"] = (
            (data[cnfg.START_TIME_STR] <= data[cnfg.TARGET_TIME_STR]) &
            (data[cnfg.END_TIME_STR] >= data[cnfg.TARGET_TIME_STR])
    )
    return None
