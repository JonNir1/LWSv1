"""Top-level pipeline entry point: stage 1 (parse) -> stage 2 (align) -> stage 3 (classify)."""

from __future__ import annotations

from time import perf_counter
from typing import Optional

import config as cnfg
import pipeline.config as pcfg
from analysis.helpers.read_data import DataStore, load_data
from pipeline.stage1_parse.run_stage1 import run_stage1


def run_pipeline(
    raw_data_path: str = cnfg.RAW_DATA_PATH,
    output_path: str = cnfg.OUTPUT_PATH,
    save: bool = True,
    verbose: bool = True,
    force_reparse: bool = False,
    # stage-2 thresholds
    on_target_threshold_dva: float = pcfg.ON_TARGET_THRESHOLD_DVA,
    visit_merging_time_threshold: float = pcfg.VISIT_MERGING_TIME_THRESHOLD,
    identification_actions: Optional[list] = None,
    # stage-3 thresholds
    min_gaze_coverage: float = pcfg.DEFAULT_GAZE_COVERAGE_PERCENT_THRESHOLD,
    min_fixation_rate: float = pcfg.DEFAULT_FIXATION_RATE_THRESHOLD,
) -> DataStore:
    """
    Run the full pipeline and return a DataStore with all three stages computed.

    Stage 1 parses raw data to pickles (optionally saving them).
    Stages 2 and 3 are computed on-the-fly by load_data().
    """
    if identification_actions is None:
        identification_actions = pcfg.IDENTIFICATION_ACTIONS

    # stage 1: parse raw data to pickles
    t0 = perf_counter()
    run_stage1(
        raw_data_path=raw_data_path,
        identification_actions=identification_actions,
        save=save,
        verbose=verbose,
        force_reparse=force_reparse,
    )
    if verbose:
        print(f"Stage 1 (parse): {perf_counter() - t0:.1f}s")

    # stages 2 + 3: align + classify (on-the-fly via load_data)
    t1 = perf_counter()
    data = load_data(
        output_path,
        drop_bad_eye=True,
        drop_outliers=True,
        missing="raise",
        on_target_threshold_dva=on_target_threshold_dva,
        visit_merging_time_threshold=visit_merging_time_threshold,
        identification_actions=identification_actions,
        min_gaze_coverage=min_gaze_coverage,
        min_fixation_rate=min_fixation_rate,
    )
    if verbose:
        print(f"Stages 2+3 (align + classify): {perf_counter() - t1:.1f}s")
        print(f"Total pipeline: {perf_counter() - t0:.1f}s")

    return data
