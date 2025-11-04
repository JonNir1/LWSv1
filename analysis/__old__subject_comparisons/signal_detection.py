import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config as cnfg


def signal_detection_figure(sdt_metrics: pd.DataFrame) -> go.Figure:
    row_titles = ["Subject Mean", "Subject SEM"]
    column_titles = ["<i>Hit-Rate</i>", "<i>d'</i>", "<i>A'</i>", "<i>f1</i>"]
    subplot_comments = ["(Better=<i>Higher</i>)", "(Better=<i>Lower</i>)"]
    data_columns = ["hit_rate", "d_prime", "a_prime", "f1_score"]
    fig = make_subplots(
        rows=len(row_titles), cols=len(column_titles),
        row_titles=row_titles, column_titles=column_titles,
        subplot_titles=subplot_comments,
        shared_xaxes=False, shared_yaxes=False,
        vertical_spacing=0.08, horizontal_spacing=0.02,
    )
    for c, col in enumerate(column_titles):
        data_col = data_columns[c]
        grouped = (
            sdt_metrics
            .loc[:, [cnfg.SUBJECT_STR, data_col]]
            .groupby(cnfg.SUBJECT_STR)
        )[data_col]
        data = pd.concat([grouped.mean().rename("mean"), grouped.sem().rename("sem")], axis=1).reset_index(drop=False)
        texts = data[cnfg.SUBJECT_STR].map(lambda subj_id: f"{cnfg.SUBJECT_STR.capitalize()} {subj_id:02d}")
        for r, _row in enumerate(row_titles):
            row_name = "mean" if r == 0 else "sem"
            xs = data[row_name]
            fig.add_trace(
                row=r + 1, col=c + 1,
                trace=go.Violin(
                    y0=data_col, x=xs, text=texts, name=f"{data_col} ({row_name})",
                    orientation="h", side="positive", spanmode='hard',
                    box=dict(visible=False),
                    meanline=dict(visible=True),
                    points="all", pointpos=-0.5,
                    showlegend=False,
                )
            )
    # Update annotations
    for ann in fig.layout.annotations:
        if ann.text in column_titles:
            ann.font = cnfg.SUBTITLE_FONT
            continue
        if ann.text in row_titles:
            ann.font = cnfg.AXIS_LABEL_FONT
            ann.textangle = -90
            ann.x = -0.02
            ann.xanchor = "right"
            ann.xref = "paper"
            continue
        if ann.text in subplot_comments:
            ann.font = cnfg.COMMENT_FONT
            ann.textangle = -90
            ann.x = -0.015
            comment_index = subplot_comments.index(ann.text)
            corresponding_row_ann = [
                row_ann for row_ann in fig.layout.annotations if row_ann.text == row_titles[comment_index]
            ][0]
            ann.y = corresponding_row_ann.y
            ann.yanchor = corresponding_row_ann.yanchor
    # Update axes and layout
    fig.for_each_xaxis(lambda xax: xax.update(
        title=None, showline=False,
        showgrid=True, gridcolor=cnfg.GRID_LINE_COLOR, gridwidth=cnfg.GRID_LINE_WIDTH,
        zeroline=False, zerolinecolor=cnfg.GRID_LINE_COLOR, zerolinewidth=cnfg.ZERO_LINE_WIDTH,
        tickfont=cnfg.AXIS_TICK_FONT,
    ))
    fig.for_each_yaxis(lambda yax: yax.update(
        showticklabels=False
    ))
    fig.update_layout(
        width=1600, height=550,
        title=dict(text="Subject Validity by Trial Category", font=cnfg.TITLE_FONT),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        # plot_bgcolor='rgba(0, 0, 0, 0)',
        showlegend=False,
    )
    return fig
