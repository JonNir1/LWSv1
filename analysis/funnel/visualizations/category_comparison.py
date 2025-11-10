import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config as cnfg

_DEFAULT_DISTRIBUTION_OPACITY = 0.75
_DEFAULT_INDIVIDUAL_ERROR_BAR = False
_DEFAULT_INDIVIDUAL_MARKER_SIZE = 6
_DEFAULT_INDIVIDUAL_LINE_WIDTH = 2
_DEFAULT_INDIVIDUAL_LINE_OPACITY = 0.5


def category_comparison_figure(
        data: pd.DataFrame,
        categ_col: str,
        show_distributions: bool = True,
        show_individuals: bool = True,
        **kwargs
) -> go.Figure:
    if categ_col not in data.columns:
        raise ValueError(f"`data` must contain `{categ_col}` column for category comparison.")
    if not show_distributions and not show_individuals:
        raise ValueError("At least one of `show_distributions` or `show_individuals` must be True.")
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.7, 0.3],)
    if show_distributions:
        fig = _add_distribution_traces(
            fig,
            data,
            categ_col,
            opacity=kwargs.get("distribution_opacity", _DEFAULT_DISTRIBUTION_OPACITY) if show_individuals else 1.0,
            show_box=kwargs.get("show_box", not show_individuals),
            show_mean=kwargs.get("show_mean", not show_individuals),
            show_legend=not show_individuals,
        )
    if show_individuals:
        if "subject" not in data.columns:
            raise ValueError(f"`data` must contain `subject` column to add individual traces.")
        fig = _add_individual_traces(
            fig,
            data,
            categ_col,
            show_error=kwargs.get("show_error", _DEFAULT_INDIVIDUAL_ERROR_BAR),
            marker_size=kwargs.get("marker_size", _DEFAULT_INDIVIDUAL_MARKER_SIZE),
            line_width=kwargs.get("line_width", _DEFAULT_INDIVIDUAL_LINE_WIDTH),
            line_opacity=kwargs.get("line_opacity", _DEFAULT_INDIVIDUAL_LINE_OPACITY),
        )
    default_title_text = categ_col.split("_")[0].capitalize() + " Category Comparison"
    fig.update_layout(
        width=kwargs.get("width", 800), height=kwargs.get("height", 600),
        title=dict(text=kwargs.get("title", default_title_text), font=cnfg.TITLE_FONT),
        yaxis=dict(
            range=(-5, min(100, 100 * max(data["mean"] + 0.5 * data["sem"]))),
            title=dict(text="Proportion (%)", font=cnfg.AXIS_LABEL_FONT),
        ),
        xaxis=dict(title=dict(text="Category", font=cnfg.AXIS_LABEL_FONT),),
        showlegend=True,
    )
    return fig


def _add_distribution_traces(
        fig: go.Figure,
        data: pd.DataFrame,
        categ_col: str,
        opacity: float,
        show_box: bool,
        show_mean: bool,
        show_legend: bool,
) -> go.Figure:
    assert categ_col in data.columns
    data_copy = data.copy()
    data_copy[categ_col] = data_copy[categ_col].map(lambda cat: str(cat).upper())   # categories as uppercase
    for cat in data_copy[categ_col].unique():
        cat_data = data_copy[data_copy[categ_col] == cat]
        col = 1 if cat != "ALL" else 2
        fig.add_trace(
            row=1, col=col, trace=go.Violin(
                x=cat_data[categ_col],
                y=100 * cat_data["mean"],
                marker=dict(color="gray", line=dict(color="black", width=1)),
                name=cat,
                spanmode="hard",
                opacity=opacity,
                box=dict(visible=show_box,),
                meanline=dict(visible=show_mean),
                showlegend=show_legend,
            )
        )
    return fig


def _add_individual_traces(
        fig: go.Figure,
        data: pd.DataFrame,
        categ_col: str,
        show_error: bool,
        marker_size: int,
        line_width: int,
        line_opacity: float,
) -> go.Figure:
    assert categ_col in data.columns
    assert "subject" in data.columns
    data_copy = data.copy()
    data_copy[categ_col] = data_copy[categ_col].map(lambda cat: str(cat).upper())   # categories as uppercase
    for i, subj in enumerate(data_copy["subject"].unique()):
        subj_name = f"Subject {subj}"
        subj_color = cnfg.get_discrete_color(i, loop=True)
        subj_color = (int(subj_color[1:3], 16), int(subj_color[3:5], 16), int(subj_color[5:7], 16))
        subj_data = data_copy[data_copy["subject"] == subj]
        for is_per_category in [True, False]:
            if is_per_category:
                subset_data = subj_data[subj_data[categ_col] != "ALL"]
                show_err = show_error
            else:
                subset_data = subj_data[subj_data[categ_col] == "ALL"]
                show_err = True     # always show error for "ALL" category
            fig.add_trace(
                row=1, col=1 if is_per_category else 2, trace=go.Scatter(
                    x=subset_data[categ_col],
                    y=100 * subset_data["mean"],
                    error_y=dict(
                        type="data", array=100 * subset_data["sem"],
                        color=f"rgba{subj_color + (1,)}", thickness=1.5, width=3,
                        visible=show_err,
                    ),
                    name=subj_name, legendgroup=subj_name, showlegend=is_per_category,
                    marker=dict(size=marker_size, color=f"rgba{subj_color + (1,)}"),
                    line=dict(width=line_width, color=f"rgba{subj_color + (line_opacity,)}"),
                    text=f"<b>Subject</b>:\t{subj}",
                    hoverinfo="text+y", mode="markers+lines",
                )
            )
    return fig
