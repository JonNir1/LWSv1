from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import pandas as pd

import pipeline.config as pcfg
from pipeline.stage3_classify.build_funnels import build_trial_inclusion_funnel, build_event_classification_funnel

if TYPE_CHECKING:
    from analysis.helpers.read_data import DataStore


def run_stage3(
    data: DataStore,
    min_gaze_coverage: int | float = pcfg.DEFAULT_GAZE_COVERAGE_PERCENT_THRESHOLD,
    min_fixation_rate: float = pcfg.DEFAULT_FIXATION_RATE_THRESHOLD,
    bad_actions: Optional[pcfg.BAD_ACTIONS_TYPE] = None,
    require_actions: bool = False,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Run stage-3 classification: trial inclusion + event funnels for all combinations.

    Returns (trial_funnel, event_funnels) where event_funnels is keyed by
    "{funnel_type}_{event_type}", e.g. "lws_visit", "target_return_fixation".
    """
    trial_funnel = build_trial_inclusion_funnel(
        data,
        min_gaze_coverage=min_gaze_coverage,
        min_fixation_rate=min_fixation_rate,
        bad_actions=bad_actions,
        require_actions=require_actions,
    )
    event_funnels: dict[str, pd.DataFrame] = {}
    for funnel_type in ("lws", "target_return"):
        for event_type in ("fixation", "visit"):
            key = f"{funnel_type}_{event_type}"
            event_funnels[key] = build_event_classification_funnel(
                data,
                funnel_type=funnel_type,
                event_type=event_type,
                min_gaze_coverage=min_gaze_coverage,
                min_fixation_rate=min_fixation_rate,
                bad_actions=bad_actions,
                require_actions=require_actions,
                exclude="none",
            )
    return trial_funnel, event_funnels
