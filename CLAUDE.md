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
- Tests: `pytest tests/ -q` from repo root. `pyproject.toml` sets `pythonpath = ["."]` and `testpaths = ["tests"]`.
  No linter config and no build step.

## Commands

Run the full pipeline (parse + align + classify, from repo root, in the venv):

```bash
python -c "from pipeline.run import run_pipeline; run_pipeline(save=True, verbose=True)"
```

Run stage 1 only (parse raw data to pickles):

```bash
python -c "from pipeline.stage1_parse.run_stage1 import run_stage1; run_stage1(save=True, verbose=True)"
```

Run tests:

```bash
pytest tests/ -q
```

Fit a GAM (from repo root; the scripts use `file.path(\"analysis\", \"R\", ...)` relative paths):

```bash
Rscript analysis/R/time_on_task_gam.R
```

The R scripts read `analysis/R/funnel_results.csv`, which is **not** in git (`*.csv` is gitignored) and is exported by
hand from a notebook after building a funnel. They write `*_predictions.csv` back into `analysis/R/`, which the
notebooks then read to overlay model estimates on plotly figures.

## Architecture

Three stages: parse (raw data to tables), align (join fixations to targets, build visits and identifications),
and classify (funnels, LWS/target-return). Stage 1 persists pickles; stages 2 and 3 compute on-the-fly.

`pipeline/run.py:run_pipeline()` composes all three stages and returns a `DataStore` with every table populated.
`analysis/helpers/read_data.py:load_data()` does stages 2+3 from pre-built pickles (the common notebook entry point).

All pipeline thresholds, criteria lists, and naming helpers live in `pipeline/config.py`. The root-level `config.py`
re-exports pipeline thresholds via `from pipeline.config import *` for backward compatibility (TODO: drop the
re-export once all consumers import from `pipeline.config` directly).

### Stage 1: parse (`pipeline/stage1_parse/`, `data_models/`)

`run_stage1()` = `parse_all_subjects()` then `build_dataframes()`, saving four pickles to `cnfg.OUTPUT_PATH`:
`icons.pkl`, `actions.pkl`, `metadata.pkl`, `eye_movements.pkl`.

Shared distance math lives in `utils/distances.py`: `pixel_distance(x1, y1, x2, y2)` and
`px2deg(screen_distance_cm)`.

### `eye_movements.pkl` and why fixations are a view

One row per **(subject, trial, eye, event)**. `event` is the event's positional rank among *all* events of that
eye in the trial. `DataStore.fixations` is a derived view (`event_type == "FIXATION"`), not a separate file.

| group | columns | populated for |
| --- | --- | --- |
| keys | `subject`, `trial`, `eye`, `event` | all |
| type | `event_type` (categorical; FIXATION / SACCADE / BLINK) | all |
| timing | `start_time`, `end_time`, `duration`, `to_trial_end` | all |
| location | `x`, `y` | **fixations only** |
| saccade geometry | `start_x`, `start_y`, `end_x`, `end_y` | all (only meaningful for saccades) |
| spread | `std_x`, `std_y`, `dispersion`, `ellipse_area` | all, interpretable for fixations |
| kinematics | `distance`, `amplitude`, `azimuth`, `cumulative_distance`, `cumulative_amplitude`, `peak_velocity`, `median_velocity`, `min_velocity` | all |
| quality | `is_outlier`, `outlier_reasons` | all |
| strip distance | `num_fixs_to_strip` | **fixations only** |

**`x`/`y` are NaN for non-fixations, deliberately.** A saccade's `center_pixel` is the midpoint of a trajectory
crossed at speed, not a held position. The spread features are kept for every event (measurements, not location).

**Anything that counts fixations must filter on `event_type` first.** The table is ~2.1x the rows it used to be
(saccades and blinks are included), so a bare `len()` roughly doubles.

### `icons.pkl` and the stable icon identifier

One row per **(subject, trial, icon)** with all 180 icons. `icon{i}` is the stable identifier (flat row-major
index over the 10x18 grid). `DataStore.targets` is a derived view (`is_target` subset, identifier renamed to
`target`). Stored as categoricals (~7 MB for 27 subjects).

Target distances are not in the events table. They are computed on-the-fly in stage 2 by
`fixations_to_targets()` in long format.

Object model (`Subject` -> list of `Trial` -> one `SearchArray` each):

- `data_models/parse/triggers_and_gaze.py` reads the E-Prime trigger log and Tobii gaze file, merges them on time,
  and derives `block` / `trial` / `is_recording` columns from trigger codes (`_ExperimentTriggerEnum`). Trial
  boundaries come from `STIMULUS_ON`/`STIMULUS_OFF`, not `TRIAL_START`/`TRIAL_END`. Key-press trigger sequences are
  collapsed into a `SubjectActionCategoryEnum` per action (mark+confirm, mark-only, attempted-mark, mark+reject).
- `Trial.__init__` does the heavy preprocessing eagerly: loads the `SearchArray` from its `.mat` file and runs
  `peyes` Engbert detection **separately for each eye**.
- `data_models/parse/eye_movements.py` handles both detection (Engbert via `peyes`) and tabulation of every
  detected event, adding `num_fixs_to_strip` (how many **fixations** until one lands in the exemplar strip;
  `inf` if never, NaN for non-fixations).

**Why there are no per-icon visits.** The on-target radius is 68 px against 76 px icon spacing (ratio 0.90), so
an all-within-threshold rule over 180 icons would claim ~2-3 icons per fixation with no principled tiebreak.
Target-visits remain the primary construct.

Caching is layered: `Subject.pkl` and `eye_movements_df.pkl` are written per subject under
`OUTPUT_PATH/subjects/<exp>_Subject_NN/`, and `parse_single_subject` prefers the pickle over re-parsing raw data.
Both carry a `<name>.cache.json` sidecar keyed on the stage-1 source files (`pipeline/stage1_parse/cache_key.py`).

### Stage 2: align (`pipeline/stage2_align/`)

Stage 2 computes on-the-fly from stage-1 pickles, producing three tables stored on `DataStore`:

- `fixation_target_dists`: long-format (subject, trial, eye, event, target, distance_px, distance_dva), from
  `pipeline/stage2_align/fixations_to_targets.py`
- `visits`: from `pipeline/stage2_align/build_visits.py`, groups consecutive on-target fixations
- `identifications`: from `pipeline/stage2_align/target_identifications.py`, matches identification actions to fixation
  positions and classifies as hit / repeated_hit / false_alarm / miss

Nothing from stage 2 is persisted as pickles. All outputs depend on researcher-chosen thresholds
(`on_target_threshold_dva`, `visit_merging_time_threshold`) stored on `DataStore`.

### Stage 3: classify (`pipeline/stage3_classify/`)

The "funnel" is the core abstraction: an ordered list of boolean criteria, converted to **cumulative** pass columns
(`_convert_criteria_to_funnel`), so each column means "passed this and every earlier criterion".

`run_stage3(data)` in `pipeline/stage3_classify/run_stage3.py` builds all funnels and returns
`(trial_funnel, event_funnels)`. `event_funnels` is a dict keyed by `"{funnel_type}_{event_type}"` (e.g.
`"lws_visit"`, `"target_return_fixation"`). Both are stored on `DataStore`.

Two entry points in `pipeline/stage3_classify/build_funnels.py`:

- `build_trial_inclusion_funnel(data: DataStore, ...)` -> per (subject, trial), criteria from
  `TRIAL_INCLUSION_CRITERIA` plus `is_valid_trial`.
- `build_event_classification_funnel(data: DataStore, funnel_type, event_type, ...)` -> per fixation or per visit,
  trial-level criteria joined onto event-level criteria (`IS_LWS_CRITERIA` or `IS_TARGET_RETURN_CRITERIA`), plus
  `is_lws` / `is_target_return`, enriched with `trial_category`, `target_category`, `target_angle`.

Criteria ordering and naming helpers live in `pipeline/config.py` (the lists, `CUMULATIVE_PREFIX`, `cumulative_name()`,
`cumulative_names()`). The predicates live in `trial_inclusion.py` and `event_classification.py` (each returns a named
boolean Series). To add a criterion: write the predicate, register it in the `criteria_functions` dict, and insert its
name into the relevant list in `pipeline/config.py`.

For fixation-level funnels, `assign_fixation_targets()` in `event_classification.py` uses the long-format
`fixation_target_dists` from stage 2 to assign the closest within-threshold target to each fixation. The visit path
uses `weighted_distance_dva` as before.

Two things to keep in mind when reading funnel output:

- **Columns are cumulative with `upto_` prefix.** `upto_on_target` means "passed every earlier criterion *and* is on
  target". Terminal columns (`is_valid_trial`, `is_lws`, `is_target_return`) keep their names since both readings
  coincide. `cumulative_name()` / `cumulative_names()` in `pipeline/config.py` map criteria to column names.
  Concretely: `is_lws`/`is_target_return` are cumulative through the **trial-level** criteria too (they're prepended
  before the event-level ones), so both silently already require `is_valid_trial` even though neither
  `IS_LWS_CRITERIA` nor `IS_TARGET_RETURN_CRITERIA` mentions it. See `CODE_REVIEW.md` L8.
- **`event_type` changes target attribution.** A fixation row carries only its *closest* target; a visit row exists
  per (target, visit), so one fixation can contribute to several. Fixation- and visit-level counts are not
  comparable denominators.
- **Fixation- and visit-level `is_lws`/`is_target_return` can disagree for fixations inside a visit.** Each grain
  evaluates `before_identification`/`after_identification` against its own `start_time`/`end_time`, so a visit
  classified `identification` (its span contains `ident_time`) can still have its first fixation independently
  classified `is_lws` at the fixation level, since that one fixation ends before `ident_time`. Not a bug — each
  grain is internally consistent — but a trap when the two are combined or compared. See `CODE_REVIEW.md` M18.

Every target has an identification time by construction: the first hit, or `inf` if it was never identified (so
every on-target event on a missed target is pre-identification). A missing time is a data error and raises.

### `DataStore` (`analysis/helpers/read_data.py`)

Frozen dataclass holding all tables from all three stages:

- **Stage 1:** `icons`, `actions`, `metadata`, `eye_movements` (all `Optional[pd.DataFrame]`)
- **Stage 2:** `fixation_target_dists`, `visits`, `identifications`
- **Stage 3:** `trial_funnel` (DataFrame), `event_funnels` (dict[str, DataFrame])
- **Thresholds:** `on_target_threshold_dva`, `visit_merging_time_threshold`, `min_gaze_coverage`, `min_fixation_rate`
- **Derived properties:** `fixations` (event_type == FIXATION view), `targets` (is_target subset of icons)

### Shared utilities and analysis helpers

`utils/sdt.py` computes hit/miss/FA/CR counts and rates, d' (with Macmillan & Kaplan or log-linear
corrections), A', and F1 per subject-trial.

`analysis/helpers/visualizations/funnel/size_and_proportion.py` computes step sizes for funnel visualization.

Analysis lives in notebooks at `analysis/*.ipynb` (`hit_rate`, `time_on_task`, `time_in_trial`, `spatial_effects`,
`stimulus_features`, `trial_exclusion`, `gaze_behavior`, `ssm_and_ab`), each of which builds a funnel and plots it.
`analysis/helpers/default_value_selection/_determine_fvf.ipynb` justifies the FVF hyperparameter.
Stage-3 threshold notebooks (`_determine_fixation_rate`, `_determine_fixs_to_strip`, `_determine_time_to_trial_end`)
live in `pipeline/stage3_classify/`.
`__old__subject_comparisons/` is superseded, `_publications_/` holds figure notebooks, `plgrnd2.py` is a scratchpad.

## Conventions

- Column and key names are centralized in `constants.py` as `*_STR` constants and referenced as `cnst.X` /
  `cnfg.X` rather than string literals; `config.py` does `from constants import *`, so `cnfg.TRIAL_STR` also works.
- `pipeline/config.py` is the source of truth for all pipeline thresholds and funnel behavior.
- Times are ms relative to trial onset; distances exist in both px and DVA (`px2deg` derived per subject from screen
  distance and `TOBII_MONITOR`). `to_trial_end` is time remaining, not elapsed.
- Both eyes are detected and kept through stage 1; the non-dominant eye is dropped at read time via
  `read_data(drop_bad_eye=True)`. Anything grouped per eye must **preserve the group's index** if it goes through
  `groupby.transform` - pandas aligns the returned Series by label, so a callable that rebuilds a fresh
  `RangeIndex` silently corrupts every group but the first (`CODE_REVIEW.md` H1a).
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
