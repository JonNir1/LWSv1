"""Shared fixtures for the LWSv1 test suite.

Tests are written against the findings in `CODE_REVIEW.md`. Each test that encodes a known bug is marked
`xfail(strict=True)` so that the suite is green *now* (documenting the bug) and turns red the moment the bug is
fixed without the marker being removed - at which point the marker comes off and the test becomes a regression guard.
"""

import os

import numpy as np
import pandas as pd
import pytest

import config as cnfg
from data_models.LWSEnums import SignalDetectionCategoryEnum


@pytest.fixture(scope="session")
def output_dir() -> str:
    """Path to the built pickles, or skip the test if they are not on this machine."""
    if not os.path.isdir(cnfg.OUTPUT_PATH):
        pytest.skip(f"built pickles not found at {cnfg.OUTPUT_PATH}")
    return cnfg.OUTPUT_PATH


@pytest.fixture(scope="session")
def loaded(output_dir: str):
    """The six pickles, unfiltered - no eye dropping, no outlier dropping."""
    from analysis.helpers.read_data import read_data

    return read_data(output_dir, drop_bad_eye=False, drop_outliers=False, missing="raise")


def make_fixation_row(
    eye: str,
    event: int,
    start_time: float,
    duration: float,
    x: float,
    y: float,
    target_distances_dva: dict[str, float] | None = None,
    trial: int = 1,
    subject: int = 1,
) -> dict:
    """Build one row shaped like `Subject.get_fixations()` output."""
    row = {
        "subject": subject,
        "trial": trial,
        "eye": eye,
        "event": event,
        "start_time": start_time,
        "end_time": start_time + duration,
        "duration": duration,
        "to_trial_end": 10_000.0 - (start_time + duration),
        "x": x,
        "y": y,
        "outlier_reasons": [],
    }
    for target, dva in (target_distances_dva or {}).items():
        row[f"{target}_distance_dva"] = dva
        row[f"{target}_distance_px"] = dva / 0.0266
    return row


def make_idents(rows: list[tuple[str, str, float]], subject: int = 1, trial: int = 1) -> pd.DataFrame:
    """Build an `idents`-shaped frame from (target, category, time) triples.

    `category` is the *name* of a `SignalDetectionCategoryEnum` member, e.g. "hit" / "false_alarm" / "miss".
    """
    return pd.DataFrame(
        [
            {
                "subject": subject,
                "trial": trial,
                "target": target,
                cnfg.IDENTIFICATION_CATEGORY_STR: SignalDetectionCategoryEnum(category),
                "time": time,
                "distance_dva": 0.5 if category in {"hit", "repeated_hit"} else np.inf,
            }
            for target, category, time in rows
        ]
    )
