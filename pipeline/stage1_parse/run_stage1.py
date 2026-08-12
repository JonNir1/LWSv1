import json
import os
from time import time
from typing import List, Union

import pandas as pd

import config as cnfg
from data_models.LWSEnums import SubjectActionCategoryEnum
from data_models.parse.eye_movements import configure_peyes

from pipeline.stage1_parse.parse_raw_data import parse_all_subjects
from pipeline.stage1_parse.build_dataframes import build_dataframes



def run_stage1(
        raw_data_path: str = cnfg.RAW_DATA_PATH,
        identification_actions: Union[SubjectActionCategoryEnum, List[SubjectActionCategoryEnum]] = cnfg.IDENTIFICATION_ACTIONS,
        save: bool = True,
        verbose: bool = True,
        force_reparse: bool = False,
) -> (
        pd.DataFrame,   # icons
        pd.DataFrame,   # actions
        pd.DataFrame,   # metadata
        pd.DataFrame,   # eye movements
):
    start_time = time()
    if isinstance(identification_actions, SubjectActionCategoryEnum):
        identification_actions = [identification_actions]
    if not identification_actions:
        raise ValueError(f"Must specify actions for argument `identification_actions`.")
    configure_peyes()
    subjects, bad_subjects = parse_all_subjects(raw_data_path, verbose, force_reparse=force_reparse)
    if not subjects:
        raise RuntimeError(f"No subjects could be parsed from {raw_data_path!r}. Failures: {bad_subjects}")
    icons, actions, metadata, eye_movements = build_dataframes(
        subjects,
        identification_actions=identification_actions,
        verbose=False,
    )
    if save:
        save_to = cnfg.OUTPUT_PATH
        if verbose:
            print("Saving data to output path:", save_to)
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        icons.to_pickle(os.path.join(save_to, 'icons.pkl'))
        actions.to_pickle(os.path.join(save_to, 'actions.pkl'))
        metadata.to_pickle(os.path.join(save_to, 'metadata.pkl'))
        eye_movements.to_pickle(os.path.join(save_to, 'eye_movements.pkl'))
        with open(os.path.join(save_to, 'parse_failures.json'), 'w', encoding='utf-8') as f:
            json.dump({"n_subjects": len(subjects), "failures": bad_subjects}, f, indent=2)
    if verbose:
        print(f"Stage 1 (parse) completed in {time() - start_time:.2f} seconds.")
    return icons, actions, metadata, eye_movements



