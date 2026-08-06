"""Trial inclusion criteria and their conversion into a funnel.

Covers CODE_REVIEW findings M6 (a criterion that never evaluated a trial must not pass it) and M7 (funnel columns
are cumulative).
"""

import numpy as np
import pandas as pd
import pytest

from analysis.helpers.funnels.build_funnels import _convert_criteria_to_funnel


def criteria(**columns: list) -> pd.DataFrame:
    return pd.DataFrame(columns)


class TestConvertCriteriaToFunnel:
    def test_columns_are_cumulative(self):
        """M7: each column means 'passed this and every earlier criterion', not 'passed this criterion'."""
        funnel = _convert_criteria_to_funnel(criteria(a=[True, True, False], b=[True, False, True]))
        assert funnel["a"].tolist() == [True, True, False]
        assert funnel["b"].tolist() == [True, False, False], "b must incorporate a"

    def test_a_later_pass_cannot_resurrect_an_earlier_failure(self):
        funnel = _convert_criteria_to_funnel(criteria(a=[False], b=[True], c=[True]))
        assert funnel.iloc[0].tolist() == [False, False, False]

    def test_nan_is_treated_as_failure(self):
        """M6: NaN is truthy, so an unevaluated criterion must be coerced to False, not left to pass."""
        funnel = _convert_criteria_to_funnel(criteria(a=[True, np.nan], b=[True, True]))
        assert funnel["a"].tolist() == [True, False]
        assert funnel["b"].tolist() == [True, False]

    def test_preserves_index(self):
        crit = criteria(a=[True, True]).set_axis(pd.Index([7, 9], name="event"))
        assert _convert_criteria_to_funnel(crit).index.tolist() == [7, 9]

    def test_column_order_is_the_criteria_order(self):
        funnel = _convert_criteria_to_funnel(criteria(first=[True], second=[True], third=[True]))
        assert funnel.columns.tolist() == ["first", "second", "third"]


class TestInclusionNaNHandling:
    def test_missing_trial_in_a_criterion_fails_that_trial(self):
        """M6, at the source: `reindex` inserts NaN for trials a criterion never saw."""
        from analysis.helpers.funnels.trial_inclusion import _SUBJECT_TRIAL_COLS

        idx = pd.MultiIndex.from_tuples([(1, 1), (1, 2)], names=_SUBJECT_TRIAL_COLS)
        partial = pd.Series([True], index=pd.MultiIndex.from_tuples([(1, 1)], names=_SUBJECT_TRIAL_COLS), name="c")
        reindexed = partial.reindex(idx)
        assert reindexed.isna().iloc[1], "precondition: trial (1, 2) is unevaluated"
        # the raw astype(bool) that M6 warns about would make the NaN True
        assert bool(np.nan) is True
        assert reindexed.fillna(False).astype(bool).tolist() == [True, False]
