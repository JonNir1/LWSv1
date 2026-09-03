import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config as cnfg


def plot_gam_predictions_ribbon(
        csv_path: str,
        x_col: str = "trial",
        prob_col: str = "prob",
        title: str = "GAM-Estimated LWS Dynamics",
        x_title: str = "Trial Number",
        y_title: str = "Predicted P[LWS]",
        y_range: tuple[float, float] = (-0.01, 1.01),
        x_tickvals: list[int] | None = None,
) -> go.Figure:
    """
    Read GAM predictions from a CSV, aggregate across subjects,
    and plot the population mean with a 95% CI ribbon.
    """
    df = pd.read_csv(csv_path, index_col=None)
    subj_agg = df.groupby(["subject", x_col])[prob_col].mean().reset_index()
    pop_stats = (
        subj_agg
        .groupby(x_col)
        .agg(mean_prob=(prob_col, "mean"), sem_prob=(prob_col, "sem"))
        .reset_index()
    )
    pop_stats["upper_ci"] = pop_stats["mean_prob"] + (1.96 * pop_stats["sem_prob"])
    pop_stats["lower_ci"] = pop_stats["mean_prob"] - (1.96 * pop_stats["sem_prob"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.concat([pop_stats[x_col], pop_stats[x_col].iloc[::-1]]),
        y=pd.concat([pop_stats["upper_ci"], pop_stats["lower_ci"].iloc[::-1]]),
        name="95% CI (Across Subjects)", showlegend=False,
        fill="toself",
        fillcolor="rgba(0,0,0, 0.2)",
        line=dict(color="rgba(0,0,0,0.25)"),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=pop_stats[x_col],
        y=pop_stats["mean_prob"],
        name="Predicted Trend", showlegend=True,
        mode="lines",
        line=dict(color="black", width=3),
    ))

    fig.update_xaxes(
        title=dict(text=x_title, font=cnfg.AXIS_LABEL_FONT),
        tickfont=cnfg.AXIS_TICK_FONT,
        showgrid=False, zeroline=False,
    )
    if x_tickvals is not None:
        fig.update_xaxes(tickvals=x_tickvals)
    fig.update_yaxes(
        title=dict(text=y_title, font=cnfg.AXIS_LABEL_FONT),
        range=list(y_range),
        tickvals=np.arange(0, 1.01, 0.25), tickfont=cnfg.AXIS_TICK_FONT,
        showgrid=True, zeroline=True,
    )
    fig.update_layout(
        title=dict(text=title, font=cnfg.TITLE_FONT),
        legend=dict(orientation="h", yanchor="middle", y=1, xanchor="right", x=0.95),
        margin=dict(t=40, b=40, l=40, r=10),
        template="plotly_white",
    )
    return fig


def plot_lws_time_dynamics(
        empirical_df: pd.DataFrame,
        estimate_df: pd.DataFrame | None = None,
        time_bin_size: int = 500,
        title: str = "Within-Trial LWS Dynamics: Empirical vs. GAM Estimates",
        num_cols: int = 3,
) -> go.Figure:
    """
    Binned empirical LWS data with optional GAM smooth overlay.
    Population panel on top, individual subject panels below.
    """
    data = empirical_df[["subject", "trial_category", "start_time", "is_lws"]].copy()
    data["time_bin"] = (data["start_time"] // time_bin_size) * time_bin_size

    sub_ovr = (
        data.groupby(["subject", "time_bin"], observed=True)["is_lws"]
        .mean().reset_index().assign(trial_category="ALL")
    )
    sub_cat = (
        data.groupby(["subject", "trial_category", "time_bin"], observed=True)["is_lws"]
        .mean().reset_index()
    )
    subj_stats = pd.concat([sub_ovr, sub_cat], ignore_index=True)
    pop_stats = (
        subj_stats.groupby(["trial_category", "time_bin"], observed=True)["is_lws"]
        .agg(["mean", "sem"]).reset_index()
    )

    subjects = sorted(subj_stats["subject"].unique())
    num_rows = (len(subjects) // num_cols) + 2
    specs = (
        [[{"colspan": num_cols}] + [None] * (num_cols - 1)]
        + [[{}] * num_cols for _ in range(num_rows - 1)]
    )

    fig = make_subplots(
        rows=num_rows, cols=num_cols, specs=specs,
        subplot_titles=["Population Average"] + [f"S{s}" for s in subjects],
        shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.05,
    )

    categories = pop_stats["trial_category"].unique()

    def _add_subplot_data(target_df, row, col, is_pop=False, est_data=None):
        for i, cat in enumerate(categories):
            if not is_pop and cat != "ALL":
                continue
            color = cnfg.get_discrete_color("all") if cat == "ALL" else cnfg.get_discrete_color(i)
            subset = target_df[target_df["trial_category"] == cat]

            fig.add_trace(go.Scatter(
                x=subset["time_bin"],
                y=subset["mean" if is_pop else "is_lws"],
                name=cat, mode="markers", showlegend=is_pop,
                marker=dict(color=color, size=6 if is_pop else 4, opacity=0.4),
                error_y=dict(type="data", array=subset["sem"], visible=True) if is_pop else None,
            ), row=row, col=col)

            if is_pop and est_data is not None:
                est_subset = est_data[est_data["trial_category"] == cat]
                est_pop = est_subset.groupby("start_time")["prob"].mean().reset_index()
                fig.add_trace(go.Scatter(
                    x=est_pop["start_time"], y=est_pop["prob"],
                    mode="lines", line=dict(color=color, width=3),
                    showlegend=False, hoverinfo="skip",
                ), row=row, col=col)

    _add_subplot_data(pop_stats, row=1, col=1, is_pop=True, est_data=estimate_df)
    for i, s in enumerate(subjects):
        r, c = (i // num_cols) + 2, (i % num_cols) + 1
        _add_subplot_data(subj_stats[subj_stats["subject"] == s], row=r, col=c)

    fig.update_layout(
        height=300 * num_rows, width=1000,
        template="plotly_white",
        title_text=title,
    )
    return fig


def plot_gam_spatial_predictions(
        csv_path: str,
        screen_width: int = 1920,
        screen_height: int = 1080,
        title: str = "GAM-Predicted LWS Probability Surface",
) -> go.Figure:
    """
    Read spatial GAM predictions and visualize as heatmap subplots
    (overall + per trial_category).
    """
    df = pd.read_csv(csv_path, index_col=None)
    x_coords = sorted(df["x"].unique())
    y_coords = sorted(df["y"].unique())

    categories = sorted(df["trial_category"].unique())
    k = len(categories)
    cols = 2 if k in [2, 4] else 3
    sub_rows = int(np.ceil(k / cols))
    total_rows = sub_rows + 1
    fig = make_subplots(
        rows=total_rows, cols=cols,
        specs=[[{"colspan": cols}] + [None] * (cols - 1)] + [[{}] * cols for _ in range(sub_rows)],
        subplot_titles=["Overall (Population Mean)"] + [str(cat) for cat in categories],
        vertical_spacing=0.1,
    )

    def _get_z_matrix(data: pd.DataFrame) -> np.ndarray:
        # spatial_gam.R flags grid cells with too few nearby observations as extrapolation (is_supported=False);
        # mask them to NaN so they render blank instead of as if they were real estimates.
        subj_agg = data.groupby(["subject", "x", "y"])[["prob", "is_supported"]].agg(
            {"prob": "mean", "is_supported": "all"}
        ).reset_index()
        overall_agg = subj_agg.groupby(["x", "y"])[["prob", "is_supported"]].agg(
            {"prob": "mean", "is_supported": "all"}
        ).reset_index()
        overall_agg.loc[~overall_agg["is_supported"], "prob"] = np.nan
        return overall_agg.pivot(index="y", columns="x", values="prob").values

    z_overall = _get_z_matrix(df)
    fig.add_trace(
        go.Heatmap(
            z=z_overall, x=x_coords, y=y_coords,
            colorscale="jet", zsmooth="best", showscale=True,
        ),
        row=1, col=1,
    )

    for i, cat in enumerate(categories):
        cat_data = df[df["trial_category"] == cat]
        z_cat = _get_z_matrix(cat_data)
        fig.add_trace(
            go.Heatmap(
                z=z_cat, x=x_coords, y=y_coords,
                colorscale="jet", zsmooth="best", showscale=False,
            ),
            row=(i // cols) + 2, col=(i % cols) + 1,
        )

    fig.update_xaxes(
        range=[0, screen_width],
        showline=True, mirror=True, linecolor="black",
        showticklabels=False,
    )
    fig.update_yaxes(
        range=[screen_height, 0],
        showline=True, mirror=True, linecolor="black",
        showticklabels=False,
    )
    fig.update_layout(
        title_text=title,
        width=1000, height=800,
        template="plotly_white",
    )
    return fig


def plot_gam_eccentricity_predictions(
        csv_path: str,
        title: str = "GAM-Estimated LWS Probability by Eccentricity and Angle",
) -> go.Figure:
    """
    Read the polar/eccentricity GAM predictions (r-sweep and theta-sweep rows, distinguished by the
    `sweep` column - see spatial_gam.R) and plot each as a population mean + 95% CI ribbon, in the same
    style as `plot_gam_predictions_ribbon`.
    """
    df = pd.read_csv(csv_path, index_col=None)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["P[LWS] vs. Eccentricity (r)", "P[LWS] vs. Angle (θ)"])

    def _add_ribbon(col: int, x_col: str, x_title: str):
        subset = df[df["sweep"] == x_col]
        subj_agg = subset.groupby(["subject", x_col])["prob"].mean().reset_index()
        pop_stats = (
            subj_agg.groupby(x_col).agg(mean_prob=("prob", "mean"), sem_prob=("prob", "sem")).reset_index()
        )
        pop_stats["upper_ci"] = pop_stats["mean_prob"] + (1.96 * pop_stats["sem_prob"])
        pop_stats["lower_ci"] = pop_stats["mean_prob"] - (1.96 * pop_stats["sem_prob"])

        fig.add_trace(go.Scatter(
            x=pd.concat([pop_stats[x_col], pop_stats[x_col].iloc[::-1]]),
            y=pd.concat([pop_stats["upper_ci"], pop_stats["lower_ci"].iloc[::-1]]),
            name="95% CI", showlegend=False, fill="toself",
            fillcolor="rgba(0,0,0, 0.2)", line=dict(color="rgba(0,0,0,0.25)"), hoverinfo="skip",
        ), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=pop_stats[x_col], y=pop_stats["mean_prob"],
            name="Predicted Trend", showlegend=False, mode="lines", line=dict(color="black", width=3),
        ), row=1, col=col)
        fig.update_xaxes(title=dict(text=x_title, font=cnfg.AXIS_LABEL_FONT), row=1, col=col)
        fig.update_yaxes(
            title=dict(text="Predicted P[LWS]", font=cnfg.AXIS_LABEL_FONT),
            range=[-0.01, 1.01], tickvals=np.arange(0, 1.01, 0.25), row=1, col=col,
        )

    _add_ribbon(1, "r", "Distance from Screen Center (px)")
    _add_ribbon(2, "theta", "Angle (deg, 0=right, 90=up)")

    fig.update_layout(
        title=dict(text=title, font=cnfg.TITLE_FONT),
        width=1100, height=500,
        margin=dict(t=60, b=40, l=40, r=10),
        template="plotly_white",
    )
    return fig
