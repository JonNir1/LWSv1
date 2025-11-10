import os

import pandas as pd


def read_saved_data(dir_path: str):
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
    return targets, actions, metadata, idents, fixations, visits
