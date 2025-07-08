import os
from typing import Optional

from tqdm import tqdm

import config as cnfg
from data_models.Subject import Subject


def preprocess_all_subjects(
        raw_data_path: str = cnfg.RAW_DATA_PATH,
        verbose: bool = True,
):
    subject_dirs = [
        subdir for subdir in os.listdir(raw_data_path)
        if os.path.isdir(os.path.join(raw_data_path, subdir)) and not "do not use" in subdir.lower()
    ]
    subjects, bad_subjects = [], dict()
    for subj_dir in tqdm(subject_dirs, desc="Preprocessing Subjects", disable=not verbose):
        exp_name, subj_id, _extra = subj_dir.split("-")
        try:
            subj = preprocess_single_subject(int(subj_id), exp_name, session=1, data_dir=subj_dir, verbose=verbose)
            subjects.append(subj)
        except Exception as e:
            bad_subjects[subj_id] = str(e)
            if verbose:
                print(f"Failed to process subject {subj_id}: {e}")
    if verbose:
        if bad_subjects:
            print(f"Some subjects could not be processed: {bad_subjects.keys()}")
        else:
            print(f"All {len(subjects)} subjects processed successfully.")
    return subjects


def preprocess_single_subject(
        subject_id: int,
        exp_name: str,
        session: int = 1,
        data_dir: Optional[str] = None,
        verbose: bool = False,
) -> Subject:
    try:
        subj = Subject.from_pickle(exp_name=exp_name, subject_id=subject_id,)
        _fixs = subj.get_fixations(save=True, verbose=False)
    except FileNotFoundError:
        subj = Subject.from_raw(
            exp_name=exp_name, subject_id=subject_id, session=session, data_dir=data_dir, verbose=verbose
        )
        subj.to_pickle(overwrite=False)
        _fixs = subj.get_fixations(save=True, verbose=verbose)
    return subj
