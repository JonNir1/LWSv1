from enum import Enum
from typing import Type

import pandas as pd

import config as cnfg


def calculate_stats_by_enum(df: pd.DataFrame, nominal_col: str, enum_class: Type[Enum],) -> pd.DataFrame:
    """
    Helper function to calculate statistics of some value (e.g., target_distance) per nominal column, where the nominal
    values are defined by an Enum class.
    The functions also adds a "color" column to the DataFrame, which contains the color associated with each nominal value.
    """
    stats = _calculate_stats_per_nominal_column(df, nominal_col)
    stats[nominal_col] = stats[nominal_col].map(lambda v: enum_class(v).name.lower() if v in enum_class else v)
    stats["color"] = stats[nominal_col].map(
        lambda typ: cnfg.get_discrete_color(typ if typ == cnfg.ALL_STR else enum_class[typ.upper()].value)
    )
    return stats


def _calculate_stats_per_nominal_column(df: pd.DataFrame, nominal_col: str) -> pd.DataFrame:
    """
    Helper function to calculate the median, mean and SEM for each nominal category.
    :param df: a 2-column DataFrame with one column being `nominal_col` and the other being the value to average.
    :param nominal_col: the name of the column that contains the nominal categories (e.g., target category).
    :return: a DataFrame with the median, mean and SEM for each nominal category.
    """
    assert nominal_col in df.columns, f"Expected column `{nominal_col}` in DataFrame."
    assert len(df.columns) == 2, "DataFrame should only contain 2 columns."
    other_col = [col for col in df.columns if col != nominal_col][0]
    per_category = df.groupby(nominal_col)[other_col]
    per_category = pd.concat([
        per_category.median().rename("median"),
        per_category.mean().rename("mean"),
        per_category.sem().rename("sem")
    ], axis=1)
    per_category.loc[cnfg.ALL_STR] = (df[other_col].median(), df[other_col].mean(), df[other_col].sem())
    per_category = per_category.reset_index(drop=False, inplace=False)
    return per_category

