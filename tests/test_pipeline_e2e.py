"""End-to-end pipeline test with synthetic data (stages 2+3).

Exercises the full compose path: stage-1 DataFrames -> stage-2 align -> stage-3 classify,
verifying that all stages produce a coherent DataStore without crashing. Does not require
raw data or pickles on disk.
"""

import numpy as np
import pandas as pd
import pytest

import constants as cnst
import pipeline.config as pcfg
from analysis.helpers.read_data import DataStore
from data_models.LWSEnums import SubjectActionCategoryEnum, SignalDetectionCategoryEnum
from pipeline.stage2_align.fixations_to_targets import fixations_to_targets
from pipeline.stage2_align.build_visits import build_visits
from pipeline.stage2_align.target_identifications import build_identifications
from pipeline.stage3_classify.run_stage3 import run_stage3


def _make_icons(n_subjects: int = 2, n_trials: int = 3, grid_rows: int = 2, grid_cols: int = 3) -> pd.DataFrame:
    rows = []
    spacing = 80
    for subj in range(1, n_subjects + 1):
        for trial in range(1, n_trials + 1):
            for r in range(grid_rows):
                for c in range(grid_cols):
                    idx = r * grid_cols + c
                    is_target = idx < 2
                    rows.append({
                        "subject": subj,
                        "trial": trial,
                        "icon": f"icon{idx}",
                        "x": 100.0 + c * spacing,
                        "y": 100.0 + r * spacing,
                        "angle": float(idx * 5),
                        "is_target": is_target,
                        "category": "HUMAN_FACE" if is_target else "OBJECT_HANDMADE",
                        "sub_path": f"stim/img_{idx}.png",
                    })
    return pd.DataFrame(rows)


def _make_eye_movements(icons: pd.DataFrame) -> pd.DataFrame:
    targets = icons[icons["is_target"]]
    rows = []
    for (subj, trial), trial_targets in targets.groupby(["subject", "trial"]):
        t = 0.0
        event_idx = 0
        tgt_list = trial_targets.to_dict("records")

        for i, tgt in enumerate(tgt_list):
            rows.append(_fixation(subj, trial, event_idx, t, 200.0, tgt["x"], tgt["y"]))
            event_idx += 1
            t += 250.0

        rows.append(_fixation(subj, trial, event_idx, t, 150.0, 500.0, 500.0))
        event_idx += 1
        t += 200.0

        rows.append(_fixation(subj, trial, event_idx, t, 100.0, 400.0, 700.0, in_strip=True))
        event_idx += 1
        t += 150.0

        for i, tgt in enumerate(tgt_list[:1]):
            rows.append(_fixation(subj, trial, event_idx, t, 200.0, tgt["x"] + 5.0, tgt["y"] + 5.0))
            event_idx += 1
            t += 250.0

    df = pd.DataFrame(rows)
    df["event_type"] = pd.Categorical(df["event_type"])
    return df


def _fixation(subj, trial, event, start, duration, x, y, in_strip=False):
    return {
        "subject": subj,
        "trial": trial,
        "eye": "right",
        "event": event,
        "event_type": "FIXATION",
        "start_time": start,
        "end_time": start + duration,
        "duration": duration,
        "to_trial_end": 10_000.0 - (start + duration),
        "x": x,
        "y": y,
        "start_x": x - 10.0,
        "start_y": y,
        "end_x": x + 10.0,
        "end_y": y,
        "std_x": 2.0,
        "std_y": 2.0,
        "dispersion": 5.0,
        "ellipse_area": 12.0,
        "distance": 20.0,
        "amplitude": 0.5,
        "azimuth": 0.0,
        "cumulative_distance": 100.0,
        "cumulative_amplitude": 2.5,
        "peak_velocity": 300.0,
        "median_velocity": 100.0,
        "min_velocity": 10.0,
        "is_outlier": False,
        "outlier_reasons": [],
        "num_fixs_to_strip": 1.0 if not in_strip else 0.0,
    }


def _make_actions(icons: pd.DataFrame) -> pd.DataFrame:
    targets = icons[icons["is_target"]]
    rows = []
    for (subj, trial), _ in targets.groupby(["subject", "trial"]):
        rows.append({
            "subject": subj,
            "trial": trial,
            "time": 150.0,
            "action": SubjectActionCategoryEnum.MARK_AND_CONFIRM.value,
            "to_trial_end": 9850.0,
        })
    return pd.DataFrame(rows)


def _make_metadata(icons: pd.DataFrame, px2deg: float = 0.025) -> pd.DataFrame:
    rows = []
    for (subj, trial), grp in icons.groupby(["subject", "trial"]):
        n_targets = grp["is_target"].sum()
        rows.append({
            "subject": subj,
            "trial": trial,
            "block": 1,
            "trial_category": "COLOR",
            "duration": 10_000.0,
            "gaze_coverage": 95.0,
            "dominant_eye": "right",
            "hand": "right",
            "sex": "male",
            "num_targets": int(n_targets),
            "num_distractors": int(len(grp) - n_targets),
            "num_actions": 1,
            "bad_actions": False,
            "px2deg": px2deg,
        })
    return pd.DataFrame(rows)


class TestPipelineE2E:
    """Verify stages 2+3 compose into a coherent DataStore on synthetic data."""

    @pytest.fixture()
    def stage1_data(self):
        icons = _make_icons()
        return icons, _make_actions(icons), _make_metadata(icons), _make_eye_movements(icons)

    def test_stages_compose(self, stage1_data):
        icons, actions, metadata, eye_movements = stage1_data
        fixations = eye_movements[eye_movements["event_type"] == "FIXATION"]

        on_target_dva = pcfg.ON_TARGET_THRESHOLD_DVA
        visit_merge_ms = pcfg.VISIT_MERGING_TIME_THRESHOLD

        dists = fixations_to_targets(fixations, icons, metadata)
        visits = build_visits(fixations, dists, on_target_dva, visit_merge_ms)
        idents = build_identifications(
            fixations, icons, actions, metadata,
            pcfg.IDENTIFICATION_ACTIONS, on_target_dva,
        )

        partial = DataStore(
            icons=icons, actions=actions, metadata=metadata, eye_movements=eye_movements,
            fixation_target_dists=dists, visits=visits, identifications=idents,
            trial_funnel=pd.DataFrame(), event_funnels={},
            on_target_threshold_dva=on_target_dva, visit_merging_time_threshold=visit_merge_ms,
            min_gaze_coverage=pcfg.DEFAULT_GAZE_COVERAGE_PERCENT_THRESHOLD,
            min_fixation_rate=pcfg.DEFAULT_FIXATION_RATE_THRESHOLD,
        )

        trial_funnel, event_funnels = run_stage3(partial)

        assert trial_funnel is not None and len(trial_funnel) > 0
        assert isinstance(event_funnels, dict)
        assert len(event_funnels) == 4

        n_subjects = icons["subject"].nunique()
        n_trials = metadata.shape[0]
        assert trial_funnel.shape[0] == n_trials
        assert trial_funnel["subject"].nunique() == n_subjects

        for key in ("lws_fixation", "lws_visit", "target_return_fixation", "target_return_visit"):
            assert key in event_funnels, f"Missing funnel: {key}"
            ef = event_funnels[key]
            assert len(ef) > 0, f"Empty funnel: {key}"

    def test_datastore_fields_populated(self, stage1_data):
        icons, actions, metadata, eye_movements = stage1_data
        fixations = eye_movements[eye_movements["event_type"] == "FIXATION"]

        on_target_dva = pcfg.ON_TARGET_THRESHOLD_DVA
        visit_merge_ms = pcfg.VISIT_MERGING_TIME_THRESHOLD

        dists = fixations_to_targets(fixations, icons, metadata)
        visits = build_visits(fixations, dists, on_target_dva, visit_merge_ms)
        idents = build_identifications(
            fixations, icons, actions, metadata,
            pcfg.IDENTIFICATION_ACTIONS, on_target_dva,
        )
        trial_funnel, event_funnels = run_stage3(DataStore(
            icons=icons, actions=actions, metadata=metadata, eye_movements=eye_movements,
            fixation_target_dists=dists, visits=visits, identifications=idents,
            trial_funnel=pd.DataFrame(), event_funnels={},
            on_target_threshold_dva=on_target_dva, visit_merging_time_threshold=visit_merge_ms,
            min_gaze_coverage=pcfg.DEFAULT_GAZE_COVERAGE_PERCENT_THRESHOLD,
            min_fixation_rate=pcfg.DEFAULT_FIXATION_RATE_THRESHOLD,
        ))

        data = DataStore(
            icons=icons, actions=actions, metadata=metadata, eye_movements=eye_movements,
            fixation_target_dists=dists, visits=visits, identifications=idents,
            trial_funnel=trial_funnel, event_funnels=event_funnels,
            on_target_threshold_dva=on_target_dva, visit_merging_time_threshold=visit_merge_ms,
            min_gaze_coverage=pcfg.DEFAULT_GAZE_COVERAGE_PERCENT_THRESHOLD,
            min_fixation_rate=pcfg.DEFAULT_FIXATION_RATE_THRESHOLD,
        )

        assert data.fixations is not None and len(data.fixations) > 0
        assert data.targets is not None and len(data.targets) > 0
        assert data.trial_funnel is not None and len(data.trial_funnel) > 0
        assert len(data.event_funnels) == 4

    def test_on_target_fixations_exist(self, stage1_data):
        icons, actions, metadata, eye_movements = stage1_data
        fixations = eye_movements[eye_movements["event_type"] == "FIXATION"]
        dists = fixations_to_targets(fixations, icons, metadata)
        within = dists[dists["distance_dva"] <= pcfg.ON_TARGET_THRESHOLD_DVA]
        assert len(within) > 0, "Synthetic data should have at least one on-target fixation"

    def test_identifications_exist(self, stage1_data):
        icons, actions, metadata, eye_movements = stage1_data
        fixations = eye_movements[eye_movements["event_type"] == "FIXATION"]
        idents = build_identifications(
            fixations, icons, actions, metadata,
            pcfg.IDENTIFICATION_ACTIONS, pcfg.ON_TARGET_THRESHOLD_DVA,
        )
        hits = idents[idents["identification_category"] == SignalDetectionCategoryEnum.HIT]
        assert len(hits) > 0, "Synthetic data should produce at least one hit"
