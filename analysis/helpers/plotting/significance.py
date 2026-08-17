import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config as cnfg


def contrast_bar_chart(
        plot_df: pd.DataFrame,
        title: str,
        y_title: str = "P[LWS | on-target]",
        x_title: str = "Target Category",
        width: int = 1000,
        height: int = 500,
) -> go.Figure:
    """
    Bar chart of estimated marginal means with significance annotation lines.

    ``plot_df`` must have columns:
        contrast_group, prob, prob_sem_approx, x_pos, contrast, p_value,
        color, pattern
    """
    fig = px.bar(
        plot_df,
        x="x_pos", y="prob", error_y="prob_sem_approx",
        color="contrast", pattern_shape="pattern",
        hover_name="contrast_group",
        hover_data={"prob": ":.4f", "contrast": True, "pattern": False, "x_pos": False},
    )

    y_line = max(
        round(plot_df["prob"] + 1.5 * plot_df["prob_sem_approx"], 3)
    )
    for cntrs in plot_df["contrast"].unique():
        subset = plot_df[plot_df["contrast"] == cntrs]
        x_min, x_max = subset["x_pos"].min(), subset["x_pos"].max()
        fig.add_shape(
            type="line",
            x0=x_min, x1=x_max,
            y0=y_line, y1=y_line,
            line=dict(color="black", width=1),
        )
        pval = subset["p_value"].iloc[0]
        if pval < 0.001:
            annotation_text = "p < 0.001"
        elif pval <= 0.05:
            annotation_text = f"p={pval:.3f}"
        else:
            annotation_text = "n.s."
        annotation_text = "<i>" + annotation_text + "</i>"
        fig.add_annotation(
            x=(x_min + x_max) / 2, xanchor="center",
            y=y_line * 1.0, yanchor="bottom",
            text=annotation_text,
            font=cnfg.AXIS_TICK_FONT,
            showarrow=False,
        )

    fig.update_xaxes(
        title=dict(text=x_title, font=cnfg.AXIS_LABEL_FONT),
        ticktext=plot_df["contrast_group"].map(lambda cg: cg.replace(" ", "<br>")),
        tickvals=plot_df["x_pos"],
        tickfont=cnfg.AXIS_TICK_FONT,
    )
    fig.update_yaxes(
        title=dict(text=y_title, font=cnfg.AXIS_LABEL_FONT),
        tickfont=cnfg.AXIS_TICK_FONT,
    )
    fig.update_layout(
        width=width, height=height,
        title=dict(text=title, font=cnfg.TITLE_FONT),
        showlegend=False,
    )
    return fig
