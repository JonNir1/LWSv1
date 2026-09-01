"""Shared plotly/matplotlib figure-building helpers for the poster notebooks."""
import copy
import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import arviz as az

import config as cnfg
from _publications_._modeling import AnalysisContext, ContrastGroupType

LABEL_FONT = {**cnfg.AXIS_LABEL_FONT, **dict(size=40)}
TICK_FONT = {**cnfg.AXIS_TICK_FONT, **dict(size=36)}


def save_figure(fig: go.Figure, name: str, width_cm: float, height_cm: float, output_dir: str, dpi: int = 300):
    PLOTLY_DEFAULT_DPI = 96
    CM_PER_INCH = 2.54
    width_in = width_cm / CM_PER_INCH
    width_px = int(width_in * PLOTLY_DEFAULT_DPI)
    height_in = height_cm / CM_PER_INCH
    height_px = int(height_in * PLOTLY_DEFAULT_DPI)

    fig_copy = copy.deepcopy(fig)
    fig_copy.update_xaxes(title=None)
    fig_copy.update_yaxes(title=dict(standoff=40))
    fig_copy.update_layout(
        width=width_px, height=height_px,
        title=None, paper_bgcolor="rgba(0, 0, 0, 0)", plot_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(t=0, b=0, l=40, r=5),
    )
    fig_copy.write_image(
        os.path.join(output_dir, f"{name}.png"),
        scale=dpi / PLOTLY_DEFAULT_DPI
    )


def plot_posterior_distributions(groups: List[ContrastGroupType], context: AnalysisContext, title: str = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.get_cmap("tab10")
    for i, grp in enumerate(groups):
        color = cmap(i)
        average_distribution = grp.extract_posteriors_probabilities_per_subject(context).mean(dim="subject")
        az.plot_kde(
            average_distribution.stack(sample=("chain", "draw")).values,
            ax=ax,
            label=grp.name,
            plot_kwargs={"color": color},
            fill_kwargs={"color": color, "alpha": 0.3},
        )
    title = title or f"Posterior Distributions: {[grp.name for grp in groups]}"
    ax.set_title(title)
    ax.set_xlabel("Predicted LWS probability")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.show()


def get_annotation_text(p: float) -> str:
    assert 0 <= p <= 1, f"p value must be between 0 and 1, got {p}"
    if p > 0.999:
        return "P[Δ > 0] > 0.999"
    if p < 0.001:
        return "P[Δ > 0] < 0.001"
    return f"P[Δ > 0] = {p:.3f}"


def add_posterior_bar_trace(
        fig: go.Figure, group: ContrastGroupType, context: AnalysisContext, xpos: float, color: str, pattern: str = ""
) -> float:
    """
    Adds a bar plot to the figure representing the mean posterior probability for the given group, with error bars representing the SEM across subjects.
    Returns the height of the bar (mean + SEM) for potential use in annotation placement.
    """
    overall_mean, overall_sem = group.aggregate_posterior(context)
    fig.add_trace(go.Bar(
        name=group.name, showlegend=False,
            x=[xpos], y=[overall_mean],
            error_y=dict(
                type="data",
                array=[overall_sem],
                color="rgba(0,0,0,1.0)",
                thickness=2,
                width=8,
                visible=True
            ),
            marker=dict(color=color, pattern=dict(shape=pattern)),
            hovertemplate=(
                f"<b>{group.name}</b><br>"
                f"Posterior mean: {overall_mean:.3f}<br>"
                f"Posterior SEM (across subjects): {overall_sem:.3f}<extra></extra>"
            ),
            width=0.75,
    ))
    bar_height = overall_mean + overall_sem
    return bar_height


def add_posterior_scatter_trace(
    fig: go.Figure, group: ContrastGroupType, context: AnalysisContext, xpos: float, color: str, symbol: str, size: int,
) -> float:
    """
    Adds a scatter dot to the figure representing the mean posterior probability for the given group, with error bars representing the SEM across subjects.
    Returns the height of the dot (mean + SEM) for potential use in annotation placement.
    """
    overall_mean, overall_sem = group.aggregate_posterior(context)
    fig.add_trace(go.Scatter(
        name=group.name, showlegend=False,
            x=[xpos], y=[overall_mean],
            error_y=dict(
                type="data",
                array=[overall_sem],
                color="rgba(0,0,0,1.0)",
                thickness=2,
                width=8,
                visible=True
            ),
            marker=dict(color=color, symbol=symbol, size=size), mode="markers",
            hovertemplate=(
                f"<b>{group.name}</b><br>"
                f"Posterior mean: {overall_mean:.3f}<br>"
                f"Posterior SEM (across subjects): {overall_sem:.3f}<extra></extra>"
            ),
    ))
    scatter_height = overall_mean + overall_sem
    return scatter_height


def add_empirical_scatter_trace(
        fig: go.Figure, group: ContrastGroupType, context: AnalysisContext, xpos: float, rng: np.random.Generator,
) -> float:
    """
    Adds scatter points to the figure representing the empirical probabilities of each subject for the given group, with error representing the SEM across trials.
    Returns the maximum height of the scatter dots (mean + SEM) for potential use in annotation placement.
    """
    empirical_probs_df = group.extract_empirical_probabilities_per_subject(context)
    jitter = rng.uniform(-0.25, 0.25, size=len(empirical_probs_df))
    fig.add_trace(go.Scatter(
        name=f"Empirical Data", showlegend=False,
        x=xpos + jitter, y=empirical_probs_df["mean"],
        error_y=dict(
            type="data",
            array=empirical_probs_df["sem"],
            color="rgba(0,0,0,0.25)",
            thickness=1,
            visible=True
        ),
        mode="markers",
        marker=dict(color="black", size=7, opacity=0.35, symbol="circle"),
        hovertemplate=(
            f"<b>Empirical Data</b><br>"
            f"group={group.name}<br>"
            "mean=%{y:.3f}<extra></extra>"
        ),
    ))
    scatter_height = (empirical_probs_df["mean"] + empirical_probs_df["sem"]).max()
    return scatter_height


def add_statistical_annotations(fig: go.Figure, configs: List[Tuple[str, float, float]], y_pos):
    for ann_cnfg in configs:
        fig.add_shape(
            type="line", x0=ann_cnfg[1], x1=ann_cnfg[2], y0=y_pos, y1=y_pos,
            line=dict(color="black", width=2),
        )
        fig.add_annotation(
            x=(ann_cnfg[2] + ann_cnfg[1]) / 2, xanchor="center",
            y=y_pos, yanchor="bottom",
            text=ann_cnfg[0], font=TICK_FONT,
            showarrow=False,
        )


def finalize_figure(
        fig: go.Figure, title: str, x_label: str, x_tickvals: List[float], x_ticklabels: List[str],
):
    assert len(x_ticklabels) == len(x_tickvals)
    fig.update_xaxes(
        title=dict(text=x_label, font=LABEL_FONT, standoff=10),
        tickmode="array", tickvals=x_tickvals, ticktext=x_ticklabels,
        tickfont=TICK_FONT,
        showgrid=False,
    )
    fig.update_yaxes(
        title=dict(text="P[recog. error | target visit]", font=LABEL_FONT, standoff=10),
        tickfont=TICK_FONT,
        showgrid=True, gridcolor=cnfg.GRID_LINE_COLOR, gridwidth=cnfg.GRID_LINE_WIDTH,
        zeroline=True, zerolinecolor="black", zerolinewidth=2,
    )
    fig.update_layout(
        title=dict(
            text=title, font=cnfg.TITLE_FONT,
            x=0.5, xanchor="center", y=0.95, yanchor="middle",
        ),
        plot_bgcolor="white",
        margin=dict(l=40, r=15, t=60, b=20),
    )
