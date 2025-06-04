from enum import EnumType

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config as cnfg

_TRIAL_TYPE_STR = f"{cnfg.TRIAL_STR}_type"
_BAD_TRIAL_STR = f"bad_{cnfg.TRIAL_STR}"
_TARGET_CATEGORY_STR = f"{cnfg.TARGET_STR}_{cnfg.CATEGORY_STR}"


def percent_bad_trials_figure(ident_data: pd.DataFrame,) -> go.Figure:
    """ Creates a figure showing the percentage of bad trials. """
    sub_data = ident_data[[cnfg.TRIAL_STR, _TRIAL_TYPE_STR, _BAD_TRIAL_STR]].drop_duplicates().set_index(cnfg.TRIAL_STR)
    sub_data[_BAD_TRIAL_STR] = sub_data[_BAD_TRIAL_STR].astype(int)     # convert to int for calculating percentage
    bad_trials = _calculate_rate_per_trial_type(sub_data)
    fig = px.bar(
        bad_trials, x=_TRIAL_TYPE_STR, y='mean', color=_TRIAL_TYPE_STR, error_y='sem',
        labels={_TRIAL_TYPE_STR: 'Trial Type', 'mean': '% Bad Trials'},
        color_discrete_map={typ: color for typ, color in bad_trials[[_TRIAL_TYPE_STR, 'color']].values}
    )
    fig.update_layout(title='Percentage of Trials with "Bad" Subject-Action by Trial Type')
    return fig


def percent_identified_figure(ident_data: pd.DataFrame, drop_bads: bool = True) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2, shared_xaxes=False, shared_yaxes=False, specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=["Trial", "Trial Type", "Target Category",],
        vertical_spacing=0.1, horizontal_spacing=0.05,
    )

    # Top Row: percent identified per trial
    per_trial = ident_data.groupby([cnfg.TRIAL_STR, _BAD_TRIAL_STR])['identified'].mean().rename("% identified") * 100
    per_trial = per_trial.reset_index(inplace=False)
    fig.add_trace(
        row=1, col=1,
        trace=go.Scatter(
            x=per_trial[cnfg.TRIAL_STR], y=per_trial['% identified'],
            mode='markers+lines', name='trial',
            line=dict(width=2, color='black'),
            marker=dict(
                size=10,
                symbol=per_trial[_BAD_TRIAL_STR].map(lambda bad: 'x' if bad else 'circle'),
                opacity=per_trial[_BAD_TRIAL_STR].map(lambda bad: 0.75 if bad else 1.0),
            ),
        )
    )

    if drop_bads:
        ident_data = ident_data[~ident_data[_BAD_TRIAL_STR].astype(bool)]

    # Bottom Left: percent identified by trial type
    per_trial_type = _calculate_rate_per_trial_type(ident_data[[_TRIAL_TYPE_STR, "identified"]])
    fig.add_trace(
        row=2, col=1,
        trace=go.Bar(
            x=per_trial_type[_TRIAL_TYPE_STR], y=per_trial_type['mean'],
            error_y=dict(type='data', array=per_trial_type['sem'], visible=True),
            name='trial type',
            marker_color=per_trial_type['color'],
        )
    )

    # Bottom Right: percent identified by target category
    per_target_category = _calculate_rate_per_target_category(ident_data[[_TARGET_CATEGORY_STR, "identified"]])
    fig.add_trace(
        row=2, col=2,
        trace=go.Bar(
            x=per_target_category[_TARGET_CATEGORY_STR], y=per_target_category['mean'],
            error_y=dict(type='data', array=per_target_category['sem'], visible=True),
            name='target category',
            marker_color=per_target_category['color'],
        )
    )

    # update layout
    fig.update_layout(
        title="Percentage of Targets Identified",
        showlegend=False,
    )
    return fig


def _target_identification_data(targets_df: pd.DataFrame, metadata_df: pd.DataFrame):
    ident_data = pd.merge(
        targets_df[[cnfg.TIME_STR, f"{cnfg.TARGET_STR}_{cnfg.CATEGORY_STR}"]],
        metadata_df[[f"{cnfg.TRIAL_STR}_type", "is_bad"]],
        left_index=True, right_index=True, how='left'
    ).reset_index(drop=False)
    ident_data.rename(columns={"level_1": cnfg.TARGET_STR, "is_bad": _BAD_TRIAL_STR}, inplace=True)
    ident_data['identified'] = np.isfinite(ident_data[cnfg.TIME_STR].values)
    return ident_data


def _calculate_rate_per_trial_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to calculate the percentage of something (e.g., identified) per trial type.
    :param df: a 2-column DataFrame with one column being "trial_type" and the other being the value to average.
    :return: a DataFrame with the average and sem for each trial type.
    """
    from data_models.LWSEnums import SearchArrayTypeEnum
    per_trial_type = _calculate_rate_per_nominal_col(df, _TRIAL_TYPE_STR)
    per_trial_type[_TRIAL_TYPE_STR] = per_trial_type[_TRIAL_TYPE_STR].map(
        lambda typ: SearchArrayTypeEnum(typ).name.lower() if typ in SearchArrayTypeEnum else typ
    )
    per_trial_type['color'] = per_trial_type[_TRIAL_TYPE_STR].map(
        lambda typ: cnfg.get_discrete_color(typ if typ==cnfg.ALL_STR else SearchArrayTypeEnum[typ.upper()].value)
    )
    return per_trial_type


def _calculate_rate_per_target_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to calculate the percentage of something (e.g., target_distance) per target category.
    :param df: a 2-column DataFrame with one column being "target_category" and the other being the value to average.
    :return: a DataFrame with the average and sem for each target category.
    """
    from data_models.LWSEnums import ImageCategoryEnum
    per_target_category = _calculate_rate_per_nominal_col(df, _TARGET_CATEGORY_STR)
    per_target_category[_TARGET_CATEGORY_STR] = per_target_category[_TARGET_CATEGORY_STR].map(
        lambda typ: ImageCategoryEnum(typ).name.lower() if typ in ImageCategoryEnum else typ
    )
    per_target_category['color'] = per_target_category[_TARGET_CATEGORY_STR].map(
        lambda typ: cnfg.get_discrete_color(typ if typ == cnfg.ALL_STR else ImageCategoryEnum[typ.upper()].value)
    )
    return per_target_category


def _calculate_rate_per_nominal_col(df: pd.DataFrame, nominal_col: str) -> pd.DataFrame:
    """
    Helper function to calculate the percentage of something (e.g., identified) per target category.
    :param df: a 2-column DataFrame with one column being `nominal_col` and the other being the value to average.
    :param nominal_col: the name of the column that contains the nominal categories (e.g., target category).
    :return: a DataFrame with the average and sem for each target category.
    """
    assert nominal_col in df.columns, f"Expected column `{nominal_col}` in DataFrame."
    assert len(df.columns) == 2, "DataFrame should only contain 2 columns."
    other_col = [col for col in df.columns if col != nominal_col][0]
    per_category = df.groupby(nominal_col)[other_col]
    per_category = pd.concat([per_category.mean().rename("mean"), per_category.sem().rename("sem")], axis=1)
    per_category.loc[cnfg.ALL_STR] = (df[other_col].mean(), df[other_col].sem())
    per_category *= 100   # convert to percentage
    per_category = per_category.reset_index(drop=False, inplace=False)
    return per_category

