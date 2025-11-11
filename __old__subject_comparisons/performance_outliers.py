import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config as cnfg
from __old__subject_comparisons.trial_category import calc_bad_actions_rate, calc_sdt_class_rate, calc_dprime


def create_subject_comparison_figure(metadata: pd.DataFrame, idents: pd.DataFrame) -> go.Figure:
    bad_actions_rate = calc_bad_actions_rate(metadata)
    hit_rate = calc_sdt_class_rate(metadata, idents, "hit")
    d_prime = calc_dprime(metadata, idents, "loglinear")

    data_sources = [bad_actions_rate, hit_rate, d_prime]
    row_titles = ["Between-Subject<br>Comparison (mean)", "Within-Subject<br>Variability (sem)"]
    column_titles = ["Bad Actions Rate", "Hit Rate", "d'"]
    subplot_titles = [
        "(Better=<i>Lower</i>)", "(Better=<i>Higher</i>)", "(Better=<i>Higher</i>)",
        "(Better=<i>Lower</i>)", "(Better=<i>Lower</i>)", "(Better=<i>Lower</i>)"
    ]
    x_title = cnfg.TRIAL_CATEGORY_STR.replace("_", " ").title()
    fig = make_subplots(
        rows=len(row_titles), cols=len(column_titles),
        row_titles=row_titles, column_titles=column_titles,
        subplot_titles=subplot_titles, x_title=x_title,
        shared_xaxes=True, shared_yaxes=False,
        vertical_spacing=0.03, horizontal_spacing=0.05,
    )

    for c in range(len(column_titles)):
        data = data_sources[c]
        data = data.loc[data[cnfg.SUBJECT_STR] != cnfg.ALL_STR.upper()]
        xs = data[cnfg.TRIAL_CATEGORY_STR]
        texts = data[cnfg.SUBJECT_STR].map(lambda subj_id: f"{cnfg.SUBJECT_STR.capitalize()}_{subj_id:02d}")
        for r in range(len(row_titles)):
            ys = data["mean"] if r == 0 else data["sem"]
            fig.add_trace(
                row=r + 1, col=c + 1,
                trace=go.Violin(
                    x=xs, y=ys, text=texts,
                    side="positive", spanmode='hard',
                    box=dict(visible=False),
                    meanline=dict(visible=True),
                    points="all", pointpos=-1,
                    showlegend=False,
                )
            )

    for ann in fig.layout.annotations:
        if ann.text in column_titles or ann.text == x_title:
            ann.font = cnfg.SUBTITLE_FONT
            continue
        if ann.text in row_titles:
            ann.font = cnfg.AXIS_LABEL_FONT
            ann.textangle = -90
            ann.x = -0.02
            ann.xanchor = "right"
            ann.xref = "paper"
            continue
        if ann.text in subplot_titles:
            ann.font = cnfg.COMMENT_FONT
            ann.y = ann.y - 0.05
            ann.x = ann.x - 0.1
    fig.for_each_xaxis(lambda xax: xax.update(
        title=None, showline=False,
        showgrid=False, gridcolor=cnfg.GRID_LINE_COLOR, gridwidth=cnfg.GRID_LINE_WIDTH,
        zeroline=False, zerolinecolor=cnfg.GRID_LINE_COLOR, zerolinewidth=cnfg.ZERO_LINE_WIDTH,
        tickfont=cnfg.AXIS_TICK_FONT,
    ))
    fig.for_each_yaxis(lambda yax: yax.update(
        title=None, showline=False,
        showgrid=True, gridcolor=cnfg.GRID_LINE_COLOR, gridwidth=cnfg.GRID_LINE_WIDTH,
        zeroline=False, zerolinecolor=cnfg.GRID_LINE_COLOR, zerolinewidth=cnfg.ZERO_LINE_WIDTH,
        tickfont=cnfg.AXIS_TICK_FONT,
    ))
    fig.update_layout(
        width=1500, height=700,
        title=dict(text="Subject Validity by Trial Category", font=cnfg.TITLE_FONT),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        # plot_bgcolor='rgba(0, 0, 0, 0)',
        showlegend=False,
    )
    return fig

