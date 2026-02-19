
import pandas as pd


def convert_criteria_to_funnel(criteria_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a DataFrame of boolean criteria columns into a funnel DataFrame where a value is True only if all criteria
    up to and including that column are True for that row.
    We assume that the columns are ordered in the sequence of the funnel steps.

    Returns a boolean DataFrame with the same index and columns as the input.
    """
    funnel_df = pd.DataFrame(index=criteria_df.index)
    cumulative_criteria = pd.Series(True, index=criteria_df.index)
    for column in criteria_df.columns:
        cumulative_criteria &= criteria_df[column]
        funnel_df[column] = cumulative_criteria
    return funnel_df.sort_index()
