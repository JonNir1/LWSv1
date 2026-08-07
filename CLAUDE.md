# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git workflow (applies to every Claude session)

**`main` is off limits.** Never commit to it, merge into it, rebase it, reset it, or check it out. The only exception
is an explicit, specific instruction from the user in the current session. `dev` is the integration branch.

The flow is worktree -> `dev` -> `origin/dev`, and each arrow needs the user to ask for it:

1. **Work on the session's worktree branch.** Small, atomic commits, each standalone where possible. Commit as you
   go rather than batching.
2. **Rebase onto local `dev`** to pick up the user's changes, not onto `main`.
3. **Merge into local `dev` only when the user says so**, and use a merge commit (`--no-ff`) so the branch's shape
   stays visible in the history.
4. **Push to `origin/dev` only when the user says so**, and only after (3). Never push a worktree branch straight to
   the remote. Never push to `origin/main`.

### Testing cadence

Match the cost of the check to the size of the change:

- **Per commit:** run only the tests covering what you touched - the test file for that module, or the tests you
  just wrote or edited (`pytest tests/test_<module>.py`, or `-k` on the relevant names). Do not run the full suite
  for every commit.
- **Before merging into `dev`:** run the **full** suite, and say plainly if anything fails.

### Backup tags

Tag a backup ref (`backup/<something>`) at meaningful milestones - before a large rebase that will replay many
commits, or after completing a body of work worth returning to. Not on every merge, and not for small incremental
changes; a tag per commit is noise that makes the real checkpoints harder to find.

## What this project is

Analysis code for the LWS ("Looking Without Seeing") v1 experiment: a visual-search task recorded with a Tobii
eye-tracker and BioSemi triggers (E-Prime). Subjects search a grid of icons ("search array") for targets shown as
exemplars in a bottom strip of the screen, and mark a target by pressing space and then confirming.

The central research construct is an **LWS instance**: a fixation/visit that lands on a target *before* the subject
identified it (and is not explainable by trial-end truncation or by a recent glance at the exemplar strip).
The mirror construct is a **target-return**: an on-target event *after* identification.

## Environment

- Python: use the venv at the **main checkout** root, `C:\Users\nirjo\Documents\University\PhD\Projects\LWSv1\.venv\`.
  Worktrees do not get their own `.venv`. Never `pip install` globally.
- Key third-party deps: `peyes` (eye-movement detection), `pymatreader` (stimulus `.mat` files), `pandas`, `numpy`,
  `plotly`, `screeninfo`, `tqdm`, `scipy`, `bambi`/`pymc` (used in some notebooks).
- R: `mgcv` for the GAM scripts under `analysis/R/`.
- There are no `__init__.py` files and no packaging metadata: **all code must be run with the repo root as CWD /
  sys.path root** (`import config as cnfg`, `from analysis.helpers... import ...`).
- There is no test suite, no linter config, and no build step.

## Commands

Run the full preprocessing pipeline (from repo root, in the venv). `run_pipeline.py` has no `__main__` block, so
invoke it from a REPL/notebook or `-c`:

```bash
python -c "from pipeline.run_pipeline import run_pipeline; run_pipeline(save=True, verbose=True)"
```

Fit a GAM (from repo root; the scripts use `file.path(\"analysis\", \"R\", ...)` relative paths):

```bash
Rscript analysis/R/time_on_task_gam.R
```

The R scripts read `analysis/R/funnel_results.csv`, which is **not** in git (`*.csv` is gitignored) and is exported by
hand from a notebook after building a funnel. They write `*_predictions.csv` back into `analysis/R/`, which the
notebooks then read to overlay model estimates on plotly figures.

## Architecture

Two stages, separated by a set of pickled DataFrames on disk.

### Stage 1: raw data -> tidy DataFrames (`pipeline/`, `data_models/`)

`run_pipeline()` = `parse_all_subjects()` then `build_dataframes()`, saving six pickles to `cnfg.OUTPUT_PATH`:
`targets.pkl`, `actions.pkl`, `metadata.pkl`, `idents.pkl`, `fixations.pkl`, `visits.pkl`.

Object model (`Subject` -> list of `Trial` -> one `SearchArray` each):

- `data_models/parse/triggers_and_gaze.py` reads the E-Prime trigger log and Tobii gaze file, merges them on time,
  and derives `block` / `trial` / `is_recording` columns from trigger codes (`_ExperimentTriggerEnum`). Trial
  boundaries come from `STIMULUS_ON`/`STIMULUS_OFF`, not `TRIAL_START`/`TRIAL_END`. Key-press trigger sequences are
  collapsed into a `SubjectActionCategoryEnum` per action (mark+confirm, mark-only, attempted-mark, mark+reject).
- `Trial.__init__` does the heavy preprocessing eagerly: loads the `SearchArray` from its `.mat` file, computes
  per-sample distances to every target, and runs `peyes` Engbert detection **separately for each eye**.
- `data_models/preprocess/fixations.py` turns detected events into the fixation table, adding per-target distances in
  px and DVA, the closest target, and `num_fixs_to_strip` (how many fixations until the next one lands in the
  exemplar strip; `inf` if never).
- `data_models/preprocess/visits.py` groups consecutive on-target fixations into *visits*. A fixation can belong to at
  most one visit per target but may be part of visits to several targets simultaneously.
- `data_models/preprocess/target_identifications.py` matches each identification action to the nearest gaze sample in
  time, finds the closest target, and labels it hit / repeated_hit / false_alarm; unidentified targets are appended as
  misses with `time = inf`.

Caching is layered: `Subject.pkl` and `fixation_df.pkl` are written per subject under
`OUTPUT_PATH/subjects/<exp>_Subject_NN/`, and `parse_single_subject` prefers the pickle over re-parsing raw data.
**Changing preprocessing code has no effect until those per-subject pickles are deleted** (see `CODE_REVIEW.md` H3).

### Stage 2: funnels and analysis (`analysis/`)

Everything downstream starts from `read_data(dir_path)` (`analysis/helpers/read_data.py`), which loads the six
pickles into a `LoadedData` dataclass and optionally drops non-dominant-eye rows and outlier fixations.

The "funnel" is the core abstraction: an ordered list of boolean criteria, converted to **cumulative** pass columns
(`_convert_criteria_to_funnel`), so each column means "passed this and every earlier criterion". Two entry points in
`analysis/helpers/funnels/build_funnels.py`:

- `build_trial_inclusion_funnel(...)` -> per (subject, trial), criteria from `TRIAL_INCLUSION_CRITERIA` plus
  `is_valid_trial`.
- `build_event_classification_funnel(data_dir, funnel_type, event_type, ...)` -> per fixation or per visit,
  trial-level criteria joined onto event-level criteria (`IS_LWS_CRITERIA` or `IS_TARGET_RETURN_CRITERIA`), plus
  `is_lws` / `is_target_return`, enriched with `trial_category`, `target_category`, `target_angle`.

Criteria ordering lives in `analysis/helpers/funnels/funnel_config.py` (the lists), the predicates in
`trial_inclusion.py` and `event_classification.py` (each returns a named boolean Series). To add a criterion: write
the predicate, register it in the `criteria_functions` dict, and insert its name into the relevant list.

Two things to keep in mind when reading funnel output:

- **Columns are cumulative, but keep the raw criterion name.** `on_target` means "passed every earlier criterion
  *and* is on target". This matters when computing proportions from the exported CSV (`CODE_REVIEW.md` M7).
- **`event_type` changes target attribution.** A fixation row carries only its *closest* target; a visit row exists
  per (target, visit), so one fixation can contribute to several. Fixation- and visit-level counts are not
  comparable denominators.

Every target has an identification time by construction: the first hit, or `inf` if it was never identified (so
every on-target event on a missed target is pre-identification). A missing time is a data error and raises.

`analysis/helpers/sdt.py` computes hit/miss/FA/CR counts and rates, d' (with Macmillan & Kaplan or log-linear
corrections), A', and F1 per subject-trial.

Analysis lives in notebooks at `analysis/*.ipynb` (`hit_rate`, `time_on_task`, `time_in_trial`, `spatial_effects`,
`stimulus_features`, `trial_exclusion`, `gaze_behavior`, `ssm_and_ab`), each of which builds a funnel and plots it.
`analysis/helpers/default_value_selection/_*.ipynb` are the notebooks that justify the hyperparameter defaults.
`__old__subject_comparisons/` is superseded, `_publications_/` holds figure notebooks, `plgrnd2.py` is a scratchpad.

## Conventions

- Column and key names are centralized in `constants.py` as `*_STR` constants and referenced as `cnst.X` /
  `cnfg.X` rather than string literals; `config.py` does `from constants import *`, so `cnfg.TRIAL_STR` also works.
- `funnel_config.py` is the source of truth for funnel behavior, not `config.py`.
- Times are ms relative to trial onset; distances exist in both px and DVA (`px2deg` derived per subject from screen
  distance and `TOBII_MONITOR`). `to_trial_end` is time remaining, not elapsed.
- Both eyes are detected and kept through stage 1; the non-dominant eye is dropped at read time via
  `read_data(drop_bad_eye=True)`.
- Figures use plotly, with fonts/colors and `get_discrete_color()` defined in `config.py`.

## Data locations

Paths in `config.py` are absolute and machine-specific. On this machine the working copies live under
`C:\Users\nirjo\Desktop\HCNL\LWS\`:

| What | Path | `config.py` constant |
| --- | --- | --- |
| Raw data (28 subject dirs) | `<base>\RawData` | `RAW_DATA_PATH` |
| Built pickles | `<base>\Results` | `OUTPUT_PATH` |
| Stimuli (`.mat` search arrays) | `<base>\Stimuli` | `SEARCH_ARRAY_PATH` |
| Publications | `<base>\Publications` | `PUBLICATIONS_PATH` |
| Icon images | **not on this machine** | `IMAGE_DIR_PATH` -> `S:\Lab-Shared\...` |

`config.py` sets `_BASE_PATH` twice: the lab share first, then a local override marked `# TODO: remove me!`. All four
paths above derive from it, so pointing the pipeline at a different machine is a one-line change. `IMAGE_DIR_PATH` is
separate and still points at `S:`; it is only used for `_SearchArrayImage.path`, not by the pipeline.

Stage 2 needs only `OUTPUT_PATH` and works fully offline.

## Known issues

**`CODE_REVIEW.md` is the register of known bugs, latent assumptions, and open analysis questions.** Read it before
changing preprocessing or interpreting funnel output; several Critical items currently affect the numbers. Add to it
rather than to this file when you find something new.

Data quirks from the experiment itself are in `README.md` (interleaved record/trial triggers, ~55 ms offset between the
BioSemi trial-start trigger and the first Tobii sample); open analyses are in `plans.md`.
