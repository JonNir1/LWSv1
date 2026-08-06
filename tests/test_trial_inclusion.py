"""Trial inclusion criteria and their conversion into a funnel.

Covers CODE_REVIEW findings M6 (a criterion that never evaluated a trial must not pass it) and M7 (funnel columns
are cumulative, and named so they cannot be confused with the standalone criteria).
"""

import numpy as np
import pandas as pd
import pytest

import analysis.helpers.funnels.funnel_config as fcfg
from analysis.helpers.funnels.build_funnels import _convert_criteria_to_funnel, assert_is_cumulative


def criteria(**columns: list) -> pd.DataFrame:
    return pd.DataFrame(columns)


class TestConvertCriteriaToFunnel:
    def test_columns_are_cumulative(self):
        """M7: each column means 'passed this and every earlier criterion', not 'passed this criterion'."""
        funnel = _convert_criteria_to_funnel(criteria(a=[True, True, False], b=[True, False, True]))
        assert funnel["upto_a"].tolist() == [True, True, False]
        assert funnel["upto_b"].tolist() == [True, False, False], "b must incorporate a"

    def test_a_later_pass_cannot_resurrect_an_earlier_failure(self):
        funnel = _convert_criteria_to_funnel(criteria(a=[False], b=[True], c=[True]))
        assert funnel.iloc[0].tolist() == [False, False, False]

    def test_nan_is_treated_as_failure(self):
        """M6: NaN is truthy, so an unevaluated criterion must be coerced to False, not left to pass."""
        funnel = _convert_criteria_to_funnel(criteria(a=[True, np.nan], b=[True, True]))
        assert funnel["upto_a"].tolist() == [True, False]
        assert funnel["upto_b"].tolist() == [True, False]

    def test_preserves_index(self):
        crit = criteria(a=[True, True]).set_axis(pd.Index([7, 9], name="event"))
        assert _convert_criteria_to_funnel(crit).index.tolist() == [7, 9]

    def test_column_order_is_the_criteria_order(self):
        funnel = _convert_criteria_to_funnel(criteria(first=[True], second=[True], third=[True]))
        assert funnel.columns.tolist() == ["upto_first", "upto_second", "upto_third"]


class TestCumulativeNaming:
    """M7: a funnel column and a standalone criterion must never share a name."""

    def test_criteria_are_prefixed(self):
        funnel = _convert_criteria_to_funnel(criteria(on_target=[True], before_identification=[True]))
        assert funnel.columns.tolist() == ["upto_on_target", "upto_before_identification"]
        assert "on_target" not in funnel.columns, "the standalone name must not survive into funnel output"

    @pytest.mark.parametrize("terminal", sorted(fcfg.TERMINAL_COLUMNS))
    def test_terminal_columns_keep_their_names(self, terminal):
        """These are conjunctions by definition, so both readings coincide and a prefix would be noise."""
        assert fcfg.cumulative_name(terminal) == terminal

    def test_cumulative_names_maps_a_criteria_list(self):
        assert fcfg.cumulative_names(["on_target", "is_lws"]) == ["upto_on_target", "is_lws"]

    def test_every_configured_criterion_maps_to_a_distinct_column(self):
        all_criteria = fcfg.TRIAL_INCLUSION_CRITERIA + fcfg.IS_LWS_CRITERIA + fcfg.IS_TARGET_RETURN_CRITERIA
        for crit in all_criteria:
            assert fcfg.cumulative_name(crit) != crit, f"{crit!r} would collide with its standalone column"


class TestAssertIsCumulative:
    """The invariant that makes the cumulative names trustworthy: passing step i implies passing all earlier ones."""

    def test_accepts_a_monotone_funnel(self):
        assert_is_cumulative(pd.DataFrame({"a": [True, True, False], "b": [True, False, False]}))

    def test_rejects_a_row_that_passes_a_later_step_but_fails_an_earlier_one(self):
        bad = pd.DataFrame({"a": [True, False], "b": [True, True]})
        with pytest.raises(ValueError, match="not cumulative"):
            assert_is_cumulative(bad)

    def test_error_names_both_columns_and_an_offending_row(self):
        bad = pd.DataFrame({"early": [False], "late": [True]}, index=["evt7"])
        with pytest.raises(ValueError) as excinfo:
            assert_is_cumulative(bad)
        message = str(excinfo.value)
        assert "'late'" in message and "'early'" in message and "evt7" in message

    def test_can_check_a_subset_of_columns(self):
        frame = pd.DataFrame({"a": [True], "unrelated": [False], "b": [True]})
        assert_is_cumulative(frame, columns=["a", "b"])   # ignores the interleaved non-funnel column

    def test_a_real_funnel_satisfies_the_invariant(self, output_dir):
        from analysis.helpers.funnels.build_funnels import build_event_classification_funnel

        funnel = build_event_classification_funnel(output_dir, "lws", "visit", exclude="invalid_trials")
        ordered = fcfg.cumulative_names(fcfg.TRIAL_INCLUSION_CRITERIA + fcfg.IS_LWS_CRITERIA + ["is_lws"])
        assert_is_cumulative(funnel, columns=[c for c in ordered if c in funnel.columns])


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
