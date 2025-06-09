from enum import EnumType

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config as cnfg

_TRIAL_TYPE_STR = f"{cnfg.TRIAL_STR}_type"
_TARGET_CATEGORY_STR = f"{cnfg.TARGET_STR}_{cnfg.CATEGORY_STR}"


def percent_bad_trials_figure(ident_data: pd.DataFrame,) -> go.Figure:
    # TODO: this is only interesting as a comparison across subjects, not per subject.
    """ Creates a figure showing the percentage of bad trials. """
    sub_data = ident_data[[cnfg.TRIAL_STR, _TRIAL_TYPE_STR, "is_bad"]].drop_duplicates().set_index(cnfg.TRIAL_STR)
    sub_data["is_bad"] = sub_data["is_bad"].astype(int)     # convert to int for calculating percentage
    bad_trials = _calculate_rate_per_trial_type(sub_data)
    fig = px.bar(
        bad_trials, x=_TRIAL_TYPE_STR, y='mean', color=_TRIAL_TYPE_STR, error_y='sem',
        labels={_TRIAL_TYPE_STR: 'Trial Type', 'mean': '% Bad Trials'},
        color_discrete_map={typ: color for typ, color in bad_trials[[_TRIAL_TYPE_STR, 'color']].values}
    )
    fig.update_layout(
        title='Percentage of Trials with "Bad" Subject-Action by Trial Type',
        # paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
    )
    return fig


def _target_identification_data(targets_df: pd.DataFrame, metadata_df: pd.DataFrame):
    # TODO: add fixation start-time and time from trial start, for the identification fixation
    ident_data = pd.merge(
        targets_df[[cnfg.TIME_STR, f"{cnfg.TARGET_STR}_{cnfg.CATEGORY_STR}"]],
        metadata_df[[f"{cnfg.TRIAL_STR}_type", "is_bad"]],
        left_index=True, right_index=True, how='left'
    ).reset_index(drop=False)
    ident_data.rename(columns={"level_1": cnfg.TARGET_STR}, inplace=True)
    ident_data[cnfg.IDENTIFIED_STR] = np.isfinite(ident_data[cnfg.TIME_STR].values)
    ident_data.loc[~ident_data[cnfg.IDENTIFIED_STR].values, cnfg.TIME_STR] = np.nan    # set non-identified times to NaN
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

