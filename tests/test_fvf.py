"""Functional Visual Field estimation.

Both estimators are checked by recovery: synthesise fixation-target distances from a *known* field radius and
confirm the estimator returns it. That is the only way to test an estimator whose real-data answer is unknown.
"""

import numpy as np
import pandas as pd
import pytest

import config as cnfg
from analysis.fvf.fvf import (
    estimate_by_encircling,
    estimate_by_foveation_falloff,
    estimate_by_launch_distance,
    estimate_by_selection_hazard,
    estimate_fvf,
    selection_opportunities,
    with_on_target,
)

TRUE_FVF = 4.0
THRESHOLD = cnfg.ON_TARGET_THRESHOLD_DVA


def synthetic_fixations(true_fvf: float = TRUE_FVF, n_subjects: int = 5, n_trials: int = 120, seed: int = 1):
    """One target per trial, approached from a random peripheral distance and foveated iff within `true_fvf`.

    Returned already in the long format `fixations_to_icons()`/`DataStore.fixation_target_dists` produces
    (subject, trial, eye, event, target, distance_dva), built only from fixation events - the shape every
    estimator now consumes directly, with no reshaping.

    The approach is drawn strictly beyond the on-target threshold so it is never itself an on-target fixation -
    otherwise the "launching" fixation would resolve to an earlier, unrelated one.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for subject in range(1, n_subjects + 1):
        for trial in range(1, n_trials + 1):
            approach = rng.uniform(THRESHOLD + 0.25, 12.0)
            event = 0
            for far in rng.uniform(10, 18, size=2):
                rows.append(dict(subject=subject, trial=trial, eye="right", event=event,
                                 target="icon10", distance_dva=float(far)))
                event += 1
            rows.append(dict(subject=subject, trial=trial, eye="right", event=event,
                             target="icon10", distance_dva=float(approach)))
            event += 1
            if approach <= true_fvf:
                rows.append(dict(subject=subject, trial=trial, eye="right", event=event,
                                 target="icon10", distance_dva=0.4))
    return pd.DataFrame(rows)


class TestOnTarget:
    def test_marks_on_target(self):
        long = with_on_target(synthetic_fixations(n_trials=5), THRESHOLD)
        assert long["on_target"].equals(long["distance_dva"] <= THRESHOLD)
        assert set(long["target"]) == {"icon10"}


class TestLaunchFixationOrdering:
    """CODE_REVIEW.md L161-163: the launch-fixation lookup must never resolve to a saccade or blink.

    `fixation_target_dists` is built only from `event_type == FIXATION` rows (`fixations_to_icons()` receives the
    fixation subset, never saccades/blinks), so `event` values within a (subject, trial, eye, target) group are
    already a contiguous fixation-only sequence - stepping back one *position* is safe even though `event` itself
    is a rank over all events (fixations, saccades, blinks) in the trial, i.e. not contiguous integers.
    """

    def test_launch_resolves_to_the_correct_fixation_despite_nonconsecutive_event_numbers(self):
        # event numbers 0, 3, 7 stand in for a fixation-only table where saccades/blinks (events 1-2, 4-6) were
        # already filtered out upstream - the gaps must not affect which row is "the previous fixation".
        long = pd.DataFrame([
            dict(subject=1, trial=1, eye="right", event=0, target="icon10", distance_dva=15.0),
            dict(subject=1, trial=1, eye="right", event=3, target="icon10", distance_dva=3.0),  # true launch
            dict(subject=1, trial=1, eye="right", event=7, target="icon10", distance_dva=0.4),  # on target
        ])
        per_subject, pooled, launches = estimate_by_launch_distance(long, on_target_threshold_dva=THRESHOLD)
        assert pooled == pytest.approx(3.0)
        assert launches["launch_dva"].iloc[0] == pytest.approx(3.0)


class TestFoveationFalloff:
    """Estimator A - P(foveated | peripheral approach distance)."""

    def test_recovers_a_known_radius(self):
        _per_subject, pooled, _curve = estimate_by_foveation_falloff(synthetic_fixations())
        assert pooled == pytest.approx(TRUE_FVF, abs=0.5)

    def test_recovers_per_subject(self):
        per_subject, _pooled, _curve = estimate_by_foveation_falloff(synthetic_fixations())
        assert len(per_subject) == 5
        assert np.allclose(per_subject.to_numpy(), TRUE_FVF, atol=0.6)

    def test_tracks_a_larger_field(self):
        """A wider true field must produce a larger estimate - the estimator is not returning a constant."""
        _p, narrow, _c = estimate_by_foveation_falloff(synthetic_fixations(true_fvf=3.0))
        _p, wide, _c = estimate_by_foveation_falloff(synthetic_fixations(true_fvf=7.0))
        assert wide > narrow + 2.0

    def test_curve_decreases_with_distance(self):
        _p, _pooled, curve = estimate_by_foveation_falloff(synthetic_fixations())
        assert curve["rate"].iloc[0] > curve["rate"].iloc[-1]
        assert curve["centre"].is_monotonic_increasing

    def test_censored_curve_returns_nan_and_warns(self):
        """A curve that never falls to half its asymptote has no falloff point.

        Returning the last bin centre would look like an estimate while being the edge of the data - which is
        exactly what happens on the real fixation tables, where the predictor is saturated.
        """
        # every target foveated regardless of distance -> the rate never descends
        rng = np.random.default_rng(3)
        rows = []
        for trial in range(1, 200):
            approach = rng.uniform(THRESHOLD + 0.25, 12.0)
            rows.append(dict(subject=1, trial=trial, eye="right", event=0,
                             target="icon10", distance_dva=float(approach)))
            rows.append(dict(subject=1, trial=trial, eye="right", event=1,
                             target="icon10", distance_dva=0.4))
        with pytest.warns(RuntimeWarning, match="never drops"):
            _per_subject, pooled, _curve = estimate_by_foveation_falloff(pd.DataFrame(rows))
        assert np.isnan(pooled)


class TestLaunchDistance:
    """Estimator B - the distance the target was selected from."""

    def test_recovers_a_known_radius(self):
        _per_subject, pooled, _launches = estimate_by_launch_distance(synthetic_fixations())
        assert pooled == pytest.approx(TRUE_FVF, abs=0.5)

    def test_never_exceeds_the_true_field(self):
        """Every launch is by construction an approach that led to foveation, so all are within the field."""
        _p, _pooled, launches = estimate_by_launch_distance(synthetic_fixations())
        assert launches["launch_dva"].max() <= TRUE_FVF

    def test_launches_are_peripheral(self):
        """A launch fixation is not itself on target - otherwise it would not be a launch."""
        _p, _pooled, launches = estimate_by_launch_distance(synthetic_fixations())
        assert (launches["launch_dva"] > THRESHOLD).all()

    def test_requires_an_event_column(self):
        fixation_target_dists = synthetic_fixations(n_trials=5).drop(columns=["event"])
        with pytest.raises(ValueError, match="event"):
            estimate_by_launch_distance(fixation_target_dists)


class TestSelectionHazard:
    """Estimator C - P(next saccade lands on the target | current distance), per fixation."""

    def test_recovers_a_known_radius(self):
        _per_subject, pooled, _curve = estimate_by_selection_hazard(synthetic_fixations())
        assert pooled == pytest.approx(TRUE_FVF, abs=0.5)

    def test_tracks_a_larger_field(self):
        _p, narrow, _c = estimate_by_selection_hazard(synthetic_fixations(true_fvf=3.0))
        _p, wide, _c = estimate_by_selection_hazard(synthetic_fixations(true_fvf=7.0))
        assert wide > narrow + 2.0

    def test_hazard_decreases_with_distance(self):
        _p, _pooled, curve = estimate_by_selection_hazard(synthetic_fixations())
        assert curve["rate"].iloc[0] > curve["rate"].iloc[-1]

    def test_opportunities_exclude_post_foveation_fixations(self):
        """Only fixations before a target's first on-target fixation are opportunities to select it."""
        opportunities = selection_opportunities(synthetic_fixations(n_trials=20))
        assert (opportunities["distance_dva"] > THRESHOLD).all(), "an opportunity cannot already be on target"

    def test_exactly_one_selection_per_foveated_target(self):
        """The launching fixation is unique: one selection per (trial, target) that ended up foveated."""
        opportunities = selection_opportunities(synthetic_fixations(n_trials=50))
        per_target = opportunities.groupby(["subject", "trial", "target"], observed=True)["selected"].sum()
        assert set(per_target.unique()) <= {0, 1}

    def test_does_not_saturate(self):
        """C's defining advantage over A: most opportunities are declined, so the curve has room to fall."""
        opportunities = selection_opportunities(synthetic_fixations())
        assert opportunities["selected"].mean() < 0.5


class TestEncirclingCriterion:
    """Estimator D - Young & Hulleman (2013) / Papesh et al. (2021)'s encircling criterion.

    Unlike A-C, this operates over every item in the display (not just targets) and doesn't use
    ON_TARGET_THRESHOLD_DVA at all, so its synthetic fixtures are built directly in the fixations_to_icons()
    shape (subject, trial, icon, distance_dva) rather than reusing `synthetic_fixations()`.
    """

    @staticmethod
    def _single_fixation_trial(subject: int, trial: int, set_size: int) -> pd.DataFrame:
        """One fixation; icon k sits at distance k + 0.5 DVA, so a 1-DVA-step sweep gives an exact, known answer."""
        return pd.DataFrame({
            "subject": subject, "trial": trial, "eye": "right", "event": 0,
            "icon": [f"icon{k}" for k in range(set_size)],
            "distance_dva": [k + 0.5 for k in range(set_size)],
        })

    def test_recovers_the_known_critical_radius(self):
        """32 items, 1 target -> critical count ceil(33/2) = 17 (the exact example from Papesh et al., 2021)."""
        dists = self._single_fixation_trial(subject=1, trial=1, set_size=32)
        metadata = pd.DataFrame({"subject": [1], "trial": [1], "num_targets": [1]})
        per_subject, pooled, per_trial = estimate_by_encircling(dists, metadata, max_radius_dva=20.0)
        assert per_trial["fvf_dva"].iloc[0] == pytest.approx(17.0)
        assert pooled == pytest.approx(17.0)
        assert per_subject.loc[1] == pytest.approx(17.0)

    def test_more_targets_lowers_the_critical_radius(self):
        """More targets -> lower critical count (the searcher can quit sooner) -> smaller FVF, same display."""
        one_target = self._single_fixation_trial(subject=1, trial=1, set_size=32)
        three_targets = self._single_fixation_trial(subject=1, trial=2, set_size=32)
        dists = pd.concat([one_target, three_targets], ignore_index=True)
        metadata = pd.DataFrame({"subject": [1, 1], "trial": [1, 2], "num_targets": [1, 3]})
        _per_subject, _pooled, per_trial = estimate_by_encircling(dists, metadata, max_radius_dva=20.0)
        fvf_by_trial = per_trial.set_index("trial")["fvf_dva"]
        assert fvf_by_trial[2] < fvf_by_trial[1]

    def test_censored_trial_returns_nan(self):
        """A display too small to ever reach the critical count within the swept range returns NaN rather than the
        edge of the sweep. critical_count = ceil(4/2) = 2, but only one icon is ever within max_radius_dva."""
        dists = pd.DataFrame({
            "subject": 1, "trial": 1, "eye": "right", "event": 0,
            "icon": ["icon0", "icon1", "icon2"],
            "distance_dva": [0.5, 100.0, 200.0],
        })
        metadata = pd.DataFrame({"subject": [1], "trial": [1], "num_targets": [1]})
        _per_subject, pooled, per_trial = estimate_by_encircling(dists, metadata, max_radius_dva=5.0)
        assert np.isnan(per_trial["fvf_dva"].iloc[0])
        assert np.isnan(pooled)

    def test_does_not_depend_on_on_target_threshold(self):
        """D takes no on_target_threshold_dva argument at all - the whole point is independence from it."""
        import inspect
        assert "on_target_threshold_dva" not in inspect.signature(estimate_by_encircling).parameters


class TestCombinedTable:
    def test_reports_both_estimators_and_the_threshold(self):
        table = estimate_fvf(synthetic_fixations())
        assert list(table.columns) == [
            "foveation_falloff", "launch_distance", "selection_hazard", "on_target_threshold"
        ]
        assert "all" in table.index, "the pooled row must be present"
        assert (table["on_target_threshold"] == THRESHOLD).all()

    def test_both_estimators_agree_on_synthetic_data(self):
        """They measure the same thing by different routes; on clean data they should land close together."""
        table = estimate_fvf(synthetic_fixations())
        pooled = table.loc["all"]
        assert pooled["foveation_falloff"] == pytest.approx(pooled["launch_distance"], abs=0.75)

    def test_estimates_exceed_the_on_target_threshold(self):
        """A field smaller than the on-target radius would mean targets are only ever found by landing on them."""
        pooled = estimate_fvf(synthetic_fixations()).loc["all"]
        assert pooled["foveation_falloff"] > THRESHOLD
        assert pooled["launch_distance"] > THRESHOLD
        assert pooled["selection_hazard"] > THRESHOLD

    def test_all_three_agree_on_synthetic_data(self):
        """On clean data all three measure the same thing; they diverge only on the real scanpaths."""
        pooled = estimate_fvf(synthetic_fixations()).loc["all"]
        for name in ("foveation_falloff", "launch_distance", "selection_hazard"):
            assert pooled[name] == pytest.approx(TRUE_FVF, abs=0.6), f"{name} missed the true radius"
