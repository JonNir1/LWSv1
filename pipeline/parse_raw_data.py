import os
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from data_models.Subject import Subject

# Errors that mean "this subject's raw data is unusable" and so justify skipping the subject. Anything else is a bug
# in the pipeline and must propagate rather than silently reduce N.
RECOVERABLE_PARSE_ERRORS = (FileNotFoundError, ValueError, AssertionError, KeyError)


def parse_all_subjects(
        raw_data_path: str, verbose: bool = True, strict: bool = False,
) -> Tuple[List[Subject], Dict[str, str]]:
    """
    Parse every subject directory under `raw_data_path`.

    :param raw_data_path: directory holding one `<exp>-<id>-<session>` directory per subject.
    :param verbose: print progress and a per-subject failure summary.
    :param strict: if True, re-raise the first parse failure instead of skipping that subject.

    :return: (subjects, bad_subjects), where `bad_subjects` maps subject directory -> error message. Callers should
        record `bad_subjects`: a silently reduced N is the failure mode this return value exists to prevent.
    """
    subject_dirs = [
        subdir for subdir in os.listdir(raw_data_path)
        if os.path.isdir(os.path.join(raw_data_path, subdir)) and not "do not use" in subdir.lower()
    ]
    subjects, bad_subjects = [], dict()
    for subj_dir in tqdm(subject_dirs, desc="Preprocessing Subjects", disable=not verbose):
        try:
            # the split is inside the try: a directory not matching `<exp>-<id>-<session>` is bad input for this one
            # subject, not a reason to abort the whole run
            exp_name, subj_id, _extra = subj_dir.split("-")
            subj = parse_single_subject(int(subj_id), exp_name, session=1, data_dir=subj_dir, verbose=verbose)
            subjects.append(subj)
        except RECOVERABLE_PARSE_ERRORS as e:
            if strict:
                raise
            bad_subjects[subj_dir] = f"{type(e).__name__}: {e}"
            if verbose:
                print(f"Failed to process subject directory {subj_dir!r}: {type(e).__name__}: {e}")
    if verbose:
        if bad_subjects:
            print(
                f"Processed {len(subjects)} subjects; {len(bad_subjects)} could not be processed: "
                f"{sorted(bad_subjects.keys())}"
            )
        else:
            print(f"All {len(subjects)} subjects processed successfully.")
    return subjects, bad_subjects


def parse_single_subject(
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
