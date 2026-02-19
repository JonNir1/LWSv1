import os

import pandas as pd


def read_data(dir_path: str, drop_bad_eye: bool = True):
    try:
        targets = pd.read_pickle(os.path.join(dir_path, 'targets.pkl'))
    except FileNotFoundError:
        print("Targets data not found.")
        targets = None
    try:
        actions = pd.read_pickle(os.path.join(dir_path, 'actions.pkl'))
    except FileNotFoundError:
        print("Actions data not found.")
        actions = None
    try:
        metadata = pd.read_pickle(os.path.join(dir_path, 'metadata.pkl'))
    except FileNotFoundError:
        print("Metadata data not found.")
        metadata = None
    try:
        idents = pd.read_pickle(os.path.join(dir_path, 'idents.pkl'))
    except FileNotFoundError:
        print("Identifications data not found.")
        idents = None
    try:
        fixations = pd.read_pickle(os.path.join(dir_path, 'fixations.pkl'))
    except FileNotFoundError:
        print("Fixations data not found.")
        fixations = None
    try:
        visits = pd.read_pickle(os.path.join(dir_path, 'visits.pkl'))
    except FileNotFoundError:
        print("Visits data not found.")
        visits = None
    if drop_bad_eye:
        if fixations is not None and metadata is not None:
            fixations = _drop_bad_eye(fixations, metadata)
        if visits is not None and metadata is not None:
            visits = _drop_bad_eye(visits, metadata)
    return targets, actions, metadata, idents, fixations, visits


def _drop_bad_eye(events: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    events = (
        events
        .copy()  # avoid modifying original data
        .merge(  # append dominant eye from metadata
            metadata[["subject", "trial", "dominant_eye"]],
            on=["subject", "trial"],
            how="left"
        )
    )
    events = events.loc[events["eye"] == events["dominant_eye"]].drop(columns=["dominant_eye"])
    return events
