import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config as cnfg


def action_times_figure(actions: pd.DataFrame) -> go.Figure:
    grouped = (
        actions
        .sort_values(by=cnfg.TIME_STR)
        .groupby([cnfg.SUBJECT_STR, cnfg.TRIAL_STR])
    )
    data = (
        pd.concat([
            grouped[cnfg.TIME_STR].first().rename("start_to_first"),
            grouped["to_trial_end"].last().rename("last_to_end"),
        ], axis=1)
        .groupby(cnfg.SUBJECT_STR)
        .agg(
            min_time=("start_to_first", "min"),
            max_time=("start_to_first", "max"),
            mean_time=("start_to_first", "mean"),
            sem_time=("start_to_first", "sem"),
            mean_to_trial_end=("last_to_end", "mean"),
            sem_to_trial_end=("last_to_end", "sem"),
        )
    )
    subplot_titles = [
        "Time to First Action (min)", "Time to First Action (max)",
        "Time to First Action (mean)", "Time to First Action (sem)",
        "Last Action to Trial End (mean)", "Last Action to Trial End (sem)"
    ]
    fig = make_subplots(
        rows=3, cols=2, subplot_titles=subplot_titles,
        shared_xaxes=False, shared_yaxes=False,
        vertical_spacing=0.08, horizontal_spacing=0.02,
    )
    for r in range(3):
        for c in range(2):
            col_name = data.columns[2 * r + c]
            y0 = col_name.replace("_", " ").title()
            texts = data.index.map(lambda subj_id: f"{cnfg.SUBJECT_STR.capitalize()} {subj_id:02d}")
            fig.add_trace(
                row=r + 1, col=c + 1, trace=go.Violin(
                    y0=y0, x=data[col_name], text=texts, hoverinfo="x+y+text",
                    orientation="h", side="positive", spanmode='hard',
                    box=dict(visible=False),
                    meanline=dict(visible=True),
                    points="all", pointpos=-0.5,
                    showlegend=False,
                )
            )
    fig.for_each_annotation(lambda ann: ann.update(font=cnfg.AXIS_LABEL_FONT, ))
    fig.update_yaxes(showticklabels=False)  # Hide y-axis labels
    fig.update_xaxes(
        title=None, showline=False,
        showgrid=False, gridcolor=cnfg.GRID_LINE_COLOR, gridwidth=cnfg.GRID_LINE_WIDTH,
        zeroline=False, zerolinecolor=cnfg.GRID_LINE_COLOR, zerolinewidth=cnfg.ZERO_LINE_WIDTH,
        tickfont=cnfg.AXIS_TICK_FONT,
    )
    fig.update_layout(
        width=1200, height=675,
        title=dict(text="Subjects' Action-Times Comparison", font=cnfg.TITLE_FONT),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        # plot_bgcolor='rgba(0, 0, 0, 0)',
        showlegend=False,
    )
    return fig


def action_counts_figure(actions: pd.DataFrame, metadata: pd.DataFrame) -> go.Figure:
    subplot_titles = [
        "Number of Actions per Trial (mean)", "Number of Actions per Trial (within-subject SEM)",
        "Number of Trials with Bad Actions", "Number of Trials without Actions"
    ]
    fig = make_subplots(
        rows=2, cols=2, subplot_titles=subplot_titles,
        shared_xaxes=False, shared_yaxes=False,
        vertical_spacing=0.15, horizontal_spacing=0.02,
    )
    fig_dict = {
        (0, 0): dict(data=_actions_per_trial(actions), column="mean_count", y0="average actions per trial"),
        (0, 1): dict(data=_actions_per_trial(actions), column="sem_count", y0="SEM actions per trial"),
        (1, 0): dict(data=_trials_with_bad_actions(metadata), column="bad_actions", y0="bad action trials"),
        (1, 1): dict(data=_trials_without_actions(actions), column="missing_actions", y0="missing actions trials"),
    }
    for r in range(2):
        for c in range(2):
            data, col_name, y0 = fig_dict[(r, c)]["data"], fig_dict[(r, c)]["column"], fig_dict[(r, c)]["y0"]
            texts = data[cnfg.SUBJECT_STR].map(
                lambda subj_id: f"{cnfg.SUBJECT_STR.capitalize()}_{subj_id:02d}"
            )
            fig.add_trace(
                row=r + 1, col=c + 1, trace=go.Violin(
                    y0=y0, x=data[col_name], text=texts, hoverinfo="x+y+text",
                    orientation="h", side="positive", spanmode='hard',
                    box=dict(visible=False),
                    meanline=dict(visible=True),
                    points="all", pointpos=-0.5,
                    showlegend=False,
                )
            )
    fig.for_each_annotation(lambda ann: ann.update(font=cnfg.AXIS_LABEL_FONT, ))
    fig.update_yaxes(showticklabels=False)  # Hide y-axis labels
    fig.update_xaxes(
        title=None, showline=False,
        showgrid=False, gridcolor=cnfg.GRID_LINE_COLOR, gridwidth=cnfg.GRID_LINE_WIDTH,
        zeroline=False, zerolinecolor=cnfg.GRID_LINE_COLOR, zerolinewidth=cnfg.ZERO_LINE_WIDTH,
        tickfont=cnfg.AXIS_TICK_FONT,
    )
    fig.update_layout(
        width=1400, height=600,
        title=dict(text="Subjects' Action-Count Comparison", font=cnfg.TITLE_FONT),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        # plot_bgcolor='rgba(0, 0, 0, 0)',
        showlegend=False,
    )
    return fig


def _actions_per_trial(actions: pd.DataFrame) -> pd.DataFrame:
    counts = (
        actions
        .groupby([cnfg.SUBJECT_STR, cnfg.TRIAL_STR])
        .size()
        .reset_index(name="action_counts")
    )
    grouped = counts.groupby(cnfg.SUBJECT_STR)["action_counts"]
    res = pd.concat([
        grouped.mean().rename("mean_count"), grouped.sem().rename("sem_count"),
    ], axis=1, ignore_index=False).reset_index()
    return res


def _trials_without_actions(actions: pd.DataFrame) -> pd.DataFrame:
    subjects = actions[cnfg.SUBJECT_STR].unique()
    trials = np.arange(1, 61)  # assuming trials are numbered from 1 to 60 for all subjects
    expected = pd.MultiIndex.from_product([subjects, trials], names=[cnfg.SUBJECT_STR, cnfg.TRIAL_STR])
    observed = actions.set_index([cnfg.SUBJECT_STR, cnfg.TRIAL_STR]).index.drop_duplicates()
    missing = expected.difference(observed).to_frame(index=False)
    missing_actions_count = missing.groupby(cnfg.SUBJECT_STR).size().rename("missing_actions").reset_index()
    return missing_actions_count


def _trials_with_bad_actions(metadata: pd.DataFrame) -> pd.DataFrame:
    bad_actions_count = metadata.groupby(cnfg.SUBJECT_STR)["bad_actions"].sum().reset_index()
    return bad_actions_count

