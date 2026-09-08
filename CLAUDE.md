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

### `dev` -> `main` (PR workflow)

Only when the user explicitly asks. Requires `gh` CLI, authenticated (`gh auth login`; PATH may need a manual
refresh in a fresh shell after install - `$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")`
in PowerShell).

1. `gh pr create --base main --head dev ...`.
2. `gh pr merge <n> --merge` (merge commit, not squash/rebase, to match this repo's `--no-ff` convention).
3. `git fetch origin`, then fast-forward local `main` to `origin/main`.
4. Sync local `dev` to the same commit and `git push origin dev`, so `dev` doesn't lag behind the merge.
5. Tag the merged commit last, once `main` and `dev` (local and remote) all agree - ask the user for a tag name
   rather than inventing one.
6. Return to the session's worktree branch and rebase it onto `dev`.

**Pitfall: check which branch is actually checked out in the main checkout before fast-forwarding.** The main
checkout (the non-worktree clone at the repo root) does not necessarily have `main` checked out - it commonly has
`dev` checked out instead, since that's the branch sessions rebase onto. Running `git merge --ff-only origin/main`
there fast-forwards whatever branch is currently checked out, silently moving `dev` instead of `main` if `dev` is
checked out. Always confirm with `git branch --show-current` (or check `git rev-parse main dev` before and after)
rather than assuming the merge landed on the intended branch. If it lands on the wrong one, update the other ref
directly with `git branch -f <branch> origin/<branch>` (safe when that branch isn't checked out anywhere - check
`git worktree list` first).

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
- Key deps: `peyes` (eye-movement detection), `pymatreader` (stimulus `.mat` files), `pandas`, `numpy`, `plotly`,
  `screeninfo`, `tqdm`, `scipy`, `bambi`/`pymc` (some notebooks); R's `mgcv` for `analysis/R/`.
- No `__init__.py` files or packaging metadata: **all code runs with the repo root as CWD / sys.path root**
  (`import config as cnfg`, `from analysis.helpers... import ...`). No linter config, no build step.
- Tests: `pytest tests/ -q` from repo root (`pyproject.toml` sets `pythonpath = ["."]` and `testpaths = ["tests"]`).
- **R/Python bridge**: notebooks that fit R models (`mgcv`, `lme4`) call `analysis/helpers/r_bridge.py::setup_rpy2()`
  before any `rpy2`/`pymer4` import. Rtools45 is installed on this machine, so `rpy2` uses its normal init
  path; `setup_rpy2()` still prepends R's DLL directory to `PATH` (needed for Windows' `LoadLibrary` to find
  R's compiled package DLLs) and points `.libPaths()` at this machine's package library.

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

Most `analysis/R/*_gam.R` / `*_glmm.R` scripts (`time_on_task_gam.R`, `time_in_trial_gam.R`,
`spatial_cartesian_gam.R`, `spatial_polar_gam.R`, `spatial_eccentricity_glmm.R`) are not run standalone -
their notebook calls `setup_rpy2()`, hands the funnel data over in-memory via
`r_bridge.py::to_r_dataframe()`, then `source_r()`s the script, which expects `dat` to already exist in the
R global environment. Re-run the notebook to refit; there's no CSV to inspect.

The `fvf_*_gam.R` scripts are the exception and still work the traditional way (from repo root; they use
`file.path(\"analysis\", \"R\", ...)` relative paths):

```bash
Rscript analysis/R/fvf_over_trials_gam.R
```

They read a CSV exported by hand from their companion notebook (e.g. `fvf_radius_by_trial.csv`) and write a
`*_predictions.csv` back, which the notebook then reads to overlay model estimates on plotly figures.

## Architecture

Three stages: parse (raw data to tables), align (join fixations to targets, build visits and identifications),
classify (funnels, LWS/target-return). Stage 1 persists pickles; stages 2 and 3 compute on-the-fly.

`pipeline/run.py:run_pipeline()` composes all three stages, returning a `DataStore` with every table populated.
`analysis/helpers/read_data.py:load_data()` does stages 2+3 from pre-built pickles (the common notebook entry
point). All pipeline thresholds, criteria lists, and naming helpers live in `pipeline/config.py`; the root-level
`config.py` re-exports them via `from pipeline.config import *` for backward compatibility (TODO: drop the
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

**`x`/`y` are NaN for non-fixations, deliberately**: a saccade's `center_pixel` is a trajectory midpoint, not a
held position. **Anything that counts fixations must filter on `event_type` first** - the table is ~2.1x the
rows it used to be (saccades and blinks now included), so a bare `len()` roughly doubles.

### `icons.pkl` and the stable icon identifier

One row per **(subject, trial, icon)** with all 180 icons. `icon{i}` is the stable identifier (flat row-major
index over the 10x18 grid). `DataStore.targets` is a derived view (`is_target` subset, identifier renamed to
`target`). Stored as categoricals (~7 MB for 27 subjects).

Target distances are not in the events table. They are computed on-the-fly in stage 2 by
`fixations_to_icons()` in long format.

Object model (`Subject` -> list of `Trial` -> one `SearchArray` each):

- `data_models/parse/triggers_and_gaze.py` merges the E-Prime trigger log with the Tobii gaze file on time and
  derives `block`/`trial`/`is_recording` from trigger codes. Trial boundaries come from
  `STIMULUS_ON`/`STIMULUS_OFF`, not `TRIAL_START`/`TRIAL_END`. Key-press sequences collapse into a
  `SubjectActionCategoryEnum` per action (mark+confirm, mark-only, attempted-mark, mark+reject).
- `Trial.__init__` loads the `SearchArray` from its `.mat` file and runs `peyes` Engbert detection **separately
  for each eye**.
- `data_models/parse/eye_movements.py` detects and tabulates every event, adding `num_fixs_to_strip` (fixations
  until one lands in the exemplar strip; `inf` if never, NaN for non-fixations).

**No per-icon visits:** the on-target radius (68px) against icon spacing (76px) is too tight a ratio (0.90) for
an all-within-threshold rule to pick one icon per fixation without an arbitrary tiebreak. Target-visits remain
the primary construct.

Caching is layered: `Subject.pkl` and `eye_movements_df.pkl` are written per subject under
`OUTPUT_PATH/subjects/<exp>_Subject_NN/`, and `parse_single_subject` prefers the pickle over re-parsing raw data.
Both carry a `<name>.cache.json` sidecar keyed on the stage-1 source files (`pipeline/stage1_parse/cache_key.py`).

### Stage 2: align (`pipeline/stage2_align/`)

Stage 2 computes on-the-fly from stage-1 pickles, producing three tables stored on `DataStore`:

- `fixation_target_dists`: long-format (subject, trial, eye, event, target, distance_px, distance_dva), from
  `pipeline/stage2_align/fixations_to_icons.py` (called with `icons.loc[icons["is_target"]]`; the function itself
  is generic over which icons it's given, e.g. the full icon set for array-coverage work under `analysis/fvf/`)
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
- `build_event_classification_funnel(data: DataStore, funnel_type, event_type)` -> per fixation or per visit,
  **only** event-level criteria (`IS_LWS_CRITERIA` or `IS_TARGET_RETURN_CRITERIA`), plus `is_lws` /
  `is_target_return`, enriched with `trial_category`, `target_category`, `target_angle`. Trial validity is not
  part of the event funnel; consumers filter via `data.trial_funnel["is_valid_trial"]`.

Criteria ordering and naming helpers live in `pipeline/config.py` (`CUMULATIVE_PREFIX`, `cumulative_name()`,
`cumulative_names()`); predicates live in `trial_inclusion.py` and `event_classification.py`. To add a criterion:
write the predicate, register it in `criteria_functions`, and insert its name into the relevant `pipeline/config.py`
list. For fixation-level funnels, `assign_fixation_targets()` picks the closest within-threshold target per
fixation from stage 2's `fixation_target_dists`; the visit path uses `weighted_distance_dva`.

Two things to keep in mind when reading funnel output:

- **`event_type` changes target attribution.** A fixation row carries only its *closest* target; a visit row exists
  per (target, visit), so one fixation can contribute to several. Fixation- and visit-level counts are not
  comparable denominators.
- **Fixation- and visit-level `is_lws`/`is_target_return` can disagree for fixations inside a visit**, since each
  grain evaluates `before_identification`/`after_identification` against its own `start_time`/`end_time`. Not a
  bug, each grain is internally consistent, but a trap when the two are combined. See `CODE_REVIEW.md` M18.

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
`analysis/fvf/` holds the functional-visual-field work: `fvf.py` (four FVF estimators; the comparison and which
one to use live in `compare_fvf_types.ipynb`, not the module docstring), `threshold_sweep.ipynb` (FVF vs.
`ON_TARGET_THRESHOLD_DVA`), `array_coverage.ipynb` (% of a trial's icons within a subject's FVF),
`fvf_over_trials.ipynb` (FVF radius itself over trials, estimator D only, GAM in
`analysis/R/fvf_over_trials_gam.R`), `fvf_coverage_over_trials.ipynb` (array coverage % over trials, GAM in
`analysis/R/fvf_coverage_over_trials_gam.R`), and `fvf_based_sdt.ipynb` (notebook-only prototype of an
FVF-conditioned hit/FA-rate denominator, `CODE_REVIEW.md` T1).
Stage-3 threshold notebooks (`_determine_fixation_rate`, `_determine_fixs_to_strip`, `_determine_time_to_trial_end`)
live in `pipeline/stage3_classify/`.
`__old__subject_comparisons/` is superseded, `_publications_/` holds figure notebooks, `plgrnd2.py` is a scratchpad.

`visualizer/` renders a trial's raw/processed data as images or video for inspection, not analysis: `_colors.py`
and `_filter.py` are shared helpers; `_stimulus.py` draws the stimulus with target markings colored by
identification category; `heatmap.py` and `scanpath.py` return matplotlib figures (gaze-density heatmap, scanpath
plot); `gaze_video.py` renders an animated video of gaze + fixations + actions over the trial stimulus. No
notebook demonstrates usage yet; entry points are each module's `create_*`/`plot_*` function.

## Conventions

- Column/key names live in `constants.py` as `*_STR` constants, referenced as `cnst.X` / `cnfg.X` (`config.py`
  does `from constants import *`, so `cnfg.TRIAL_STR` also works). `pipeline/config.py` is the source of truth
  for pipeline thresholds and funnel behavior.
- Times are ms relative to trial onset; distances exist in both px and DVA (`px2deg` derived per subject from
  screen distance and `TOBII_MONITOR`). `to_trial_end` is time remaining, not elapsed.
- Both eyes are detected through stage 1; the non-dominant eye is dropped at read time via
  `read_data(drop_bad_eye=True)`. Per-eye `groupby.transform` callables must **preserve the group's index** -
  a fresh `RangeIndex` silently corrupts every group but the first (`CODE_REVIEW.md` H1a).
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

`config.py` sets `_BASE_PATH` twice (lab share, then a local override marked `# TODO: remove me!`); all four paths
above derive from it, so retargeting the pipeline to a different machine is a one-line change. `IMAGE_DIR_PATH` is
separate, still points at `S:`, and is only used for `_SearchArrayImage.path`.

Stage 2 needs only `OUTPUT_PATH` and works fully offline.

## Known issues

**`CODE_REVIEW.md` is the register of known bugs, latent assumptions, and open analysis questions.** Read it before
changing preprocessing or interpreting funnel output; several Critical items currently affect the numbers. Add to it
rather than to this file when you find something new.

Data quirks from the experiment itself are in `README.md` (interleaved record/trial triggers, ~55 ms offset between the
BioSemi trial-start trigger and the first Tobii sample); open analyses are in `plans.md`.
