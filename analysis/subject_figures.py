import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config as cnfg


def percent_identified_figure(ident_data: pd.DataFrame, drop_bads: bool = True) -> go.Figure:
    BAD_TRIAL_STR = f"bad_{cnfg.TRIAL_STR}"
    fig = make_subplots(
        rows=2, cols=2, shared_xaxes=False, shared_yaxes=False, specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=["Trial", "Trial Type", "Target Category",],
        vertical_spacing=0.1, horizontal_spacing=0.05,
    )

    # Top Row: percent identified per trial
    per_trial = ident_data.groupby([cnfg.TRIAL_STR, BAD_TRIAL_STR])['identified'].mean().rename("% identified") * 100
    per_trial = per_trial.reset_index(inplace=False)
    fig.add_trace(
        row=1, col=1,
        trace=go.Scatter(
            x=per_trial[cnfg.TRIAL_STR], y=per_trial['% identified'],
            mode='markers+lines', name='trial',
            line=dict(width=2, color='black'),
            marker=dict(
                size=15,
                symbol=per_trial[BAD_TRIAL_STR].map(lambda bad: 'x' if bad else 'circle'),
                opacity=per_trial[BAD_TRIAL_STR].map(lambda bad: 0.75 if bad else 1.0),
            ),
        )
    )

    if drop_bads:
        ident_data = ident_data[~ident_data[BAD_TRIAL_STR]]

    # Bottom Left: percent identified by trial type
    from data_models.LWSEnums import SearchArrayTypeEnum
    TRIAL_TYPE_STR = f"{cnfg.TRIAL_STR}_type"
    per_trial_type = ident_data.groupby(TRIAL_TYPE_STR)['identified']
    per_trial_type = pd.concat([per_trial_type.mean().rename("mean"), per_trial_type.sem().rename("sem")], axis=1)
    per_trial_type.loc[cnfg.ALL_STR] = (ident_data['identified'].mean(), ident_data['identified'].sem())
    per_trial_type *= 100
    per_trial_type = per_trial_type.reset_index(drop=False, inplace=False)
    per_trial_type[TRIAL_TYPE_STR] = per_trial_type[TRIAL_TYPE_STR].map(
        lambda typ: SearchArrayTypeEnum(typ).name.lower() if typ in SearchArrayTypeEnum else typ
    )
    per_trial_type['color'] = per_trial_type[TRIAL_TYPE_STR].map(
        lambda typ: cnfg.get_discrete_color(typ if typ==cnfg.ALL_STR else SearchArrayTypeEnum[typ.upper()].value)
    )
    fig.add_trace(
        row=2, col=1,
        trace=go.Bar(
            x=per_trial_type[TRIAL_TYPE_STR], y=per_trial_type['mean'],
            error_y=dict(type='data', array=per_trial_type['sem'], visible=True),
            name='trial type',
            marker_color=per_trial_type['color'],
        )
    )

    # Bottom Right: percent identified by target category
    from data_models.LWSEnums import ImageCategoryEnum
    TARGET_CATEGORY_STR = f"{cnfg.TARGET_STR}_category"
    per_target_category = ident_data.groupby(TARGET_CATEGORY_STR)['identified']
    per_target_category = pd.concat([
        per_target_category.mean().rename("mean"), per_target_category.sem().rename("sem")
    ], axis=1)
    per_target_category.loc[cnfg.ALL_STR] = (ident_data['identified'].mean(), ident_data['identified'].sem())
    per_target_category *= 100
    per_target_category = per_target_category.reset_index(drop=False, inplace=False)
    per_target_category[TARGET_CATEGORY_STR] = per_target_category[TARGET_CATEGORY_STR].map(
        lambda typ: ImageCategoryEnum(typ).name.lower() if typ in ImageCategoryEnum else typ
    )
    per_target_category['color'] = per_target_category[TARGET_CATEGORY_STR].map(
        lambda typ: cnfg.get_discrete_color(typ if typ==cnfg.ALL_STR else ImageCategoryEnum[typ.upper()].value)
    )
    fig.add_trace(
        row=2, col=2,
        trace=go.Bar(
            x=per_target_category[TARGET_CATEGORY_STR], y=per_target_category['mean'],
            error_y=dict(type='data', array=per_target_category['sem'], visible=True),
            name='target category',
            marker_color=per_target_category['color'],
        )
    )
    return fig


def _target_identification_data(targets_df: pd.DataFrame, metadata_df: pd.DataFrame):
    ident_data = pd.merge(
        targets_df[[cnfg.TIME_STR, f"{cnfg.TARGET_STR}_{cnfg.CATEGORY_STR}"]],
        metadata_df[[f"{cnfg.TRIAL_STR}_type", "bad_actions"]],
        left_index=True, right_index=True, how='left'
    ).reset_index(drop=False)
    ident_data.rename(columns={"level_1": cnfg.TARGET_STR, "bad_actions": BAD_TRIAL_STR}, inplace=True)
    ident_data['identified'] = np.isfinite(ident_data[cnfg.TIME_STR].values)
    return ident_data

