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
    """Path to the built pickles, or skip if they are absent or unreadable in this environment."""
    if not os.path.isdir(cnfg.OUTPUT_PATH):
        pytest.skip(f"built pickles not found at {cnfg.OUTPUT_PATH}")
    try:
        pd.read_pickle(os.path.join(cnfg.OUTPUT_PATH, "metadata.pkl"))
    except FileNotFoundError:
        pytest.skip(f"built pickles not found at {cnfg.OUTPUT_PATH}")
    except ModuleNotFoundError as exc:
        # H8: the pickles were written under numpy>=2 but `peyes` pins numpy~=1.2, so the environment that can run
        # the pipeline cannot read its own output.
        pytest.skip(
            f"pickles at {cnfg.OUTPUT_PATH} are unreadable in this environment ({exc}); "
            f"numpy is {np.__version__} - see CODE_REVIEW.md H8"
        )
    for name, refactor in [("icons.pkl", "icon"), ("eye_movements.pkl", "eye-movements")]:
        # a build predating either refactor cannot be checked against: the first replaced positional `target{j}`
        # ids with stable `icon{i}` ones, the second replaced `fixations.pkl` with its event-table superset
        if not os.path.isfile(os.path.join(cnfg.OUTPUT_PATH, name)):
            pytest.skip(
                f"build at {cnfg.OUTPUT_PATH} predates the {refactor} refactor (no {name}); re-run the pipeline"
            )
    return cnfg.OUTPUT_PATH


@pytest.fixture(scope="session")
def stimuli_dir() -> str:
    """Path to the stimulus `.mat` files, or skip if they are not on this machine."""
    path = os.path.join(cnfg.SEARCH_ARRAY_PATH, f"generated_stim{cnfg.STIMULI_VERSION}")
    if not os.path.isdir(path):
        pytest.skip(f"stimuli not found at {path}")
    return path


@pytest.fixture(scope="session")
def array_info(stimuli_dir: str) -> dict:
    """`ArrayInfo.mat` - the stimulus-generation config the search arrays were built from."""
    from pymatreader import read_mat

    path = os.path.join(stimuli_dir, "ArrayInfo.mat")
    if not os.path.isfile(path):
        pytest.skip(f"ArrayInfo.mat not found in {stimuli_dir}")
    return read_mat(path)["ArrayInfo"]


@pytest.fixture(scope="session")
def loaded(output_dir: str):
    """All tables, unfiltered: no eye dropping, no outlier dropping."""
    from analysis.helpers.read_data import load_data

    return load_data(output_dir, drop_bad_eye=False, drop_outliers=False, missing="raise")


@pytest.fixture(scope="session")
def data_store(output_dir: str):
    """DataStore with default filtering (bad eye dropped, outliers dropped)."""
    from analysis.helpers.read_data import load_data

    return load_data(output_dir, drop_bad_eye=True, drop_outliers=True, missing="raise")


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
    """Build one row shaped like a fixation row of `Subject.get_events()` output."""
    row = {
        "subject": subject,
        "trial": trial,
        "eye": eye,
        "event": event,
        "event_type": "FIXATION",
        "start_time": start_time,
        "end_time": start_time + duration,
        "duration": duration,
        "to_trial_end": 10_000.0 - (start_time + duration),
        "x": x,
        "y": y,
        "is_outlier": False,
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
