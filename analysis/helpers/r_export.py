import os

import pandas as pd

DEFAULT_FUNNEL_CSV_PATH = os.path.join("R", "funnel_results.csv")  # relative to analysis/, the notebooks' own cwd
DEFAULT_KEY_COLUMNS = ("subject", "trial", "target", "visit")


def ensure_funnel_csv_for_r(
        funnel_results: pd.DataFrame,
        required_columns: list[str],
        path: str = DEFAULT_FUNNEL_CSV_PATH,
        key_columns: tuple[str, ...] = DEFAULT_KEY_COLUMNS,
) -> None:
    """
    Make sure `path` (the shared CSV every analysis/R/*_gam.R script reads) has every column in
    `required_columns`, without clobbering columns other notebooks may have already contributed.

    If the file already has all required columns, it is left untouched. Otherwise it is (re)built by
    outer-merging `funnel_results` into whatever is already on disk, on `key_columns` - every notebook
    building this funnel shares the same (subject, trial, target, visit) row grain, so this only adds
    columns, it never duplicates or drops rows.
    """
    columns_to_write = list(key_columns) + [c for c in required_columns if c not in key_columns]
    if os.path.exists(path):
        existing_header = pd.read_csv(path, nrows=0).columns
        missing = [c for c in required_columns if c not in existing_header]
        if not missing:
            print(f"{path} already has the required columns; leaving as-is.")
            return
        existing = pd.read_csv(path)
        # Only bring in the columns that are actually missing (plus the join key) - a column both frames
        # already share (e.g. two notebooks both needing trial_category) must not be merged in again, or
        # pandas suffixes both copies (_x/_y) instead of recognizing them as the same column.
        merged = existing.merge(funnel_results[list(key_columns) + missing], on=list(key_columns), how="outer")
        merged.to_csv(path, index=False)
        print(f"{path} was missing columns {sorted(missing)}; merged and overwrote.")
        return
    funnel_results[columns_to_write].to_csv(path, index=False)
    print(f"{path} did not exist; created.")
