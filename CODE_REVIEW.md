# LWSv1 Code Review

Review of the preprocessing pipeline (`pipeline/`, `data_models/`), the analysis layer (`analysis/`), and the R GAM
scripts. Focus: correctness, latent assumptions about the data and the theory, structure, and documentation.

Every claim below was checked against the source; behaviours that depend on library semantics (pandas 2.3.3 /
numpy 2.3.4 / the installed `peyes`) were verified by running them. Design questions raised by the review were settled
on 2026-08-05 and are recorded under **[Resolved design decisions](#resolved-design-decisions)**; the research
questions they surfaced are under **[Open tasks](#open-tasks-not-bugs)**.

Numbering note: IDs are kept stable so earlier references still resolve, even where severity changed.
**C1 → M15** (verified inert for current output). **H7 → Low** (measured: 6 of 116,947 fixations). **H8 resolved** by
the numpy/pandas upgrade. New findings surfaced while building the test suite: **H7**, **H8**, **H9**, **M17**.

### Verification status

`tests/` encodes the findings as executable claims. Each test for an unfixed bug is `xfail(strict=True)`, so the suite
is green now and turns red the moment a bug is fixed without its marker being removed.

Suite status: **61 passed, 5 xfailed** on numpy 2.5.1 / pandas 3.0.5 / peyes 0.0.9.6.

Of the 5 remaining `xfail`s, three are "fixed in code, but the built pickles predate the fix" (C4 frequency, H2
magnitude, M1 dtypes) and clear on the next `run_pipeline()`. Two are H5, deferred by decision.

| Finding | Test | Status |
| --- | --- | --- |
| C2 `MARK_ONLY` never recorded | `test_mark_only_is_recorded` | **confirmed → fixed** |
| C3 `KeyError: None`; attempted mark clobbered | `test_attempted_mark_*` | **confirmed → fixed** |
| C4 FA shadows the real hit | `test_false_alarm_does_not_shadow_the_real_hit` | **confirmed** — lookup returns 500.0, not 4000.0 |
| C4 first-hit guarantee is incidental | `test_repeated_hit_does_not_move_identification_time` | **confirmed → fixed** |
| C4 **frequency** | `test_c4_false_alarms_shadowing_hits` | **confirmed** — 12 targets affected; LWS window truncated by a median **3684 ms** (max 12904 ms) |
| H1 cross-eye strip count | `test_does_not_count_across_the_eye_boundary` | **confirmed** — `[2.0, 1.0]` where `[inf, inf]` is correct |
| H1 wrongly rejects an LWS candidate | `test_leak_can_wrongly_reject_an_lws_candidate` | **confirmed** — count 1 vs threshold 3 |
| H2 **magnitude** | `test_h2_outlier_exclusion_is_a_noop_for_visits` | **confirmed** — 11,830/116,947 fixations dropped (10.1%), **0** of 5,720 visits |
| H5 unbalanced triggers | `test_unclosed_final_trial`, `test_dropped_end_trigger_*` | **confirmed** — `np.vstack` `ValueError` |
| H7 **magnitude** | `test_h7_long_fixation_cap_is_immaterial_in_this_dataset` | **downgraded** — only 6/116,947 (0.005%) exceed the cap |
| H9 pandas 3 breaks trigger alignment | `TestTrialBoundaries::test_balanced` | **confirmed** — `AttributeError: '_hasna'` |
| M1 metadata dtypes | `test_m1_metadata_is_object_dtype` | **confirmed** — all numeric columns are `object` |
| M4 falsy-zero guard | `test_mark_at_row_zero` | **confirmed** — guard assertion does not fire |
| M15 wrong `pixel_size` | `test_saccade_amplitude_in_degrees` | **confirmed** — 300 px saccade reports **179.24°**, correct is 7.92° |
| M15 is inert for current output | `test_outlier_reasons_are_independent_of_pixel_size` | **holds** — clean trace yields no outlier reasons |
| M16 monitor mismatch | `test_peyes_monitor_matches_the_project_monitor` | **confirmed** — 53.1 cm vs 53.0 cm |
| M17 `del` on unbound name | `test_no_block_trigger_raises` | **confirmed → fixed** |
| C3 real-world frequency | — | **not measured** — raw data (`S:`) not mounted |

Decision 1 semantics (miss → `inf`, every on-target event pre-identification) are pinned as *passing* tests, so a
future change cannot alter them silently.

**Severity:**

| Level | Meaning |
| --- | --- |
| **Critical** | Produces wrong numbers in the headline result, or silently drops data. Fix before running any analysis. |
| **High** | Wrong or misleading results under plausible data conditions; or makes results irreproducible. |
| **Medium** | Latent bug, fragile assumption, or a real statistical concern that changes interpretation. |
| **Low** | Style, dead code, tooling, documentation. |

Environment these results were produced on: numpy 2.5.1, pandas 3.0.5, peyes 0.0.9.6, Python 3.12.2.
Reproduce with `pytest tests/ -q`; add `-s` to see the measured magnitudes.

---

## Resolved design decisions

Settled 2026-08-05. These are the intended semantics; the fixes below implement them.

1. **Every target has an identification time**: a positive number (hit) or `inf` (miss). A *missing* time must never
   occur — so `event_classification.py:71-77`'s "missing -> False (conservative)" describes a state that should be
   impossible, and must become an assertion rather than a silent fallback. `inf` correctly makes every on-target event
   on a never-identified target pass `before_identification`.
2. **A false alarm is not an identification of any target.** Its `target` must be a filler (`None` / not-a-target), not
   the nearest target. This is the root fix for C4.
3. **Identification time is the first hit** for that target; `repeated_hit` does not move it.
4. **Only the dominant eye is used throughout the pipeline.** The current non-dominant-eye fallback (H6) is intended
   behaviour that is nonetheless wrong, and should be removed.

## Open tasks (not bugs)

Recorded here so they are not re-litigated as defects. These need a research decision, not a fix.

- **T1. A better-engineered d'.** The FA-rate denominator is currently every non-target icon (`sdt.py:149`), which makes
  FA rate tiny and d' large. d' is known to be a poor fit for this paradigm — A' was added for that reason. A
  better-specified version (e.g. denominator = number of *fixated* items, or items within the functional visual field)
  is worth designing. Until then, prefer A' when reporting sensitivity.
- **T2. What makes a *visit* an outlier?** Does one outlier fixation (or saccade) inside a visit contaminate the whole
  visit, or only a visit whose fixations are all outliers? Needed before visit-level outlier exclusion can be
  implemented (see H2, which for now must refuse the request rather than ignore it).
- **T3. What is the longest plausible fixation in this paradigm?** Currently 2500 ms, inherited unexamined from a
  `peyes` default, and it silently removes longer fixations from every analysis (see H7). Long dwells on
  not-yet-identified targets are theoretically the strongest LWS candidates, so this threshold needs the same
  justification treatment as `TIME_TO_TRIAL_END_THRESHOLD` and `FIXATIONS_TO_STRIP_THRESHOLD` in
  `analysis/helpers/default_value_selection/`.

---

## Critical

### C2. `MARK_ONLY` is written to the wrong column and is never recorded

**STATUS: FIXED** (`9de4780`).

**Where:** `data_models/parse/triggers_and_gaze.py:215`

```python
triggers.loc[start_identify_idx, 'subj_action'] = SubjectActionCategoryEnum.MARK_ONLY
```

**Description.** Every other branch writes to `cnst.ACTION_STR` (`"action"`). This one writes the string literal
`'subj_action'`, creating a stray column that is then discarded (only `_TRIGGER_COLUMNS` survive line 138).

**Outcome.** A "marked but ran out of time before confirming" event is never labelled. It stays `NO_ACTION`, so:
- `SubjectActionCategoryEnum.MARK_ONLY` in `cnfg.IDENTIFICATION_ACTIONS` (the commented-out option in `config.py:31`)
  would silently match nothing.
- `MARK_ONLY` is in `DEFAULT_BAD_ACTIONS`, so the `no_bad_action` trial criterion never excludes these trials — trials
  where the subject was mid-identification at trial end are treated as clean.
- `metadata["num_actions"]` undercounts.

**Fix.** `triggers.loc[start_identify_idx, cnst.ACTION_STR] = SubjectActionCategoryEnum.MARK_ONLY`. Eliminate the class
of bug by never using string literals for column names in this module (M12).

**Validate.** Parse one subject and assert `MARK_ONLY` appears in `actions["action"].unique()` for at least one trial
where a `SPACE_ACT` trigger is followed by `STIMULUS_OFF` with no intervening `CONFIRM_ACT`. Add a unit test on a
hand-built trigger table covering all four `SubjectActionCategoryEnum` sequences.

---

### C3. `ATTEMPTED_MARK` branch raises `KeyError` (or clobbers a pending mark)

**STATUS: FIXED** (`9de4780`).

**Where:** `data_models/parse/triggers_and_gaze.py:181-184`

```python
if trg == _ExperimentTriggerEnum.SPACE_NO_ACT:
    triggers.loc[start_identify_idx, cnst.ACTION_STR] = SubjectActionCategoryEnum.ATTEMPTED_MARK
    continue
```

**Description.** `start_identify_idx` is `None` unless a `SPACE_ACT` is currently pending. Verified on pandas 2.3.3:
`df.loc[None, "action"] = value` raises `KeyError: None` — it does *not* enlarge. Two distinct failure modes:

- **No pending mark** (the semantically expected case for "unable to mark"): `KeyError`, which propagates out of
  `Subject.from_raw` and is swallowed by the blanket `except Exception` in `parse_all_subjects` (H4). **The entire
  subject is dropped from the dataset**, with only a `print` if `verbose=True`.
- **A mark is pending:** the assignment overwrites the pending row's action with `ATTEMPTED_MARK`, and
  `start_identify_idx` is *not* cleared, so a later `CONFIRM_ACT` overwrites it again with `MARK_AND_CONFIRM`. The
  attempted mark is lost either way.

**Outcome.** Silent, whole-subject data loss, or a lost/overwritten action. Because the failure surfaces only as a
smaller `len(subjects)`, it is easy to miss.

**Fix.** Decide the semantics first: `SPACE_NO_ACT` means the key press was rejected by E-Prime, which is an event in
its own right and does not belong to any pending mark. Record it on **its own row**:

```python
if trg == _ExperimentTriggerEnum.SPACE_NO_ACT:
    triggers.loc[idx, cnst.ACTION_STR] = SubjectActionCategoryEnum.ATTEMPTED_MARK
    continue
```

This also removes the `None`-index path entirely.

**Validate.**
1. Unit test: trigger table containing a `SPACE_NO_ACT` with no preceding `SPACE_ACT` parses without raising and yields
   exactly one `ATTEMPTED_MARK`.
2. Unit test: `SPACE_ACT` → `SPACE_NO_ACT` → `CONFIRM_ACT` yields one `MARK_AND_CONFIRM` *and* one `ATTEMPTED_MARK`.
3. After fixing, re-run `parse_all_subjects` and confirm `len(subjects)` matches the expected N and `bad_subjects` is
   empty.

---

### C4. A false alarm can be used as the identification time of a real target

**STATUS: FIXED** (`c25b527`).

**Where:** `analysis/helpers/funnels/event_classification.py:122-134`

```python
s = idents.dropna(subset=["time"]).set_index(["subject", "trial", "target"])["time"]
if s.index.has_duplicates:
    s = s[~s.index.duplicated(keep="first")]
```

**Description.** `idents` contains one row per identification *category* — `hit`, `repeated_hit`, `false_alarm`, and
`miss` — and **all of them carry a `target` label**, because `_find_closest_target` assigns the nearest target to every
identification action regardless of distance (`target_identifications.py:105`). Rows are sorted by time
(`target_identifications.py:125`) with misses appended last, so `keep="first"` selects the earliest row for that target
*of any category*.

Consequences, verified:
- If the subject false-alarms near target *k* at t=500 and then correctly identifies target *k* at t=4000, the lookup
  returns **500**. Visits between 500 and 4000 are misclassified as target-returns instead of LWS candidates, and the
  true LWS window is truncated.
- `dropna` does not remove `inf`, so a missed target keeps `time = inf`. That part is correct (decision 1), but it
  means the docstring's "missing -> False (conservative)" describes a branch that should never be reachable.

**Outcome — measured.** Against the built pickles (2026-08-06): of 1,639 identification rows, 47 are false alarms and
**all 47 carry a target label**. 19 targets have both a false alarm and a hit, and in **12** of them the false alarm
comes first — so those 12 take their identification time from the false alarm. The LWS window is truncated by a
median of **3,684 ms** (max 12,904 ms): visits in that window are misclassified as target-returns instead of LWS
candidates.

Direct corruption of the two headline constructs. The `no_miss_with_false_alarm` trial criterion masks only the subset
of trials that have *both* a miss and a FA; the hit-preceded-by-FA case is not covered.

**Fix — at the source (decision 2).** A false alarm is not an identification of any target, so it must not carry a
target label. In `target_identifications.py:_classify_hits_and_false_alarms`, null out the target once the row is
classified as a false alarm:

```python
is_false_alarm = idents_copy[cnst.IDENTIFICATION_CATEGORY_STR] == SignalDetectionCategoryEnum.FALSE_ALARM
idents_copy.loc[is_false_alarm, cnst.TARGET_STR] = None
```

Keep `distance_px` / `distance_dva` on the row — they record how far the false alarm was from the nearest target, which
is worth having — but drop the target attribution itself.

Two knock-on changes this requires:

- **`_append_missed_targets`** (`target_identifications.py:141-144`) derives missed targets as "all targets minus HIT
  targets", which is unaffected by nulling FA targets. Confirm this by test rather than by eye.
- **`_identification_time_lookup`** (`event_classification.py:122-134`) must then take the **first hit** per target
  (decision 3) and assert completeness (decision 1):

```python
HIT_CATEGORIES = {SignalDetectionCategoryEnum.HIT, SignalDetectionCategoryEnum.REPEATED_HIT}

def _identification_time_lookup(idents: pd.DataFrame) -> pd.Series:
    """Map (subject, trial, target) -> time of the FIRST hit, or `inf` if the target was never identified."""
    ...
    hits = idents[idents[cnst.IDENTIFICATION_CATEGORY_STR].isin(HIT_CATEGORIES)]
    lookup = hits.groupby(["subject", "trial", "target"], observed=True)["time"].min()
    misses = idents[idents[cnst.IDENTIFICATION_CATEGORY_STR] == SignalDetectionCategoryEnum.MISS]
    lookup = pd.concat([lookup, misses.set_index(["subject", "trial", "target"])["time"]])
    return lookup[~lookup.index.duplicated(keep="first")]
```

- **`_map_ident_time`** (`event_classification.py:137-148`) must then **raise** on an unmapped
  `(subject, trial, target)` instead of silently producing NaN — per decision 1, every target has a time. Update the
  `is_before_identification` / `is_after_identification` docstrings, which currently promise the opposite.

**Validate.**
1. Unit test: `idents` = `{t=500, target0, false_alarm}`, `{t=4000, target0, hit}` → lookup `target0 -> 4000`.
2. Unit test: `{t=1000, target1, hit}`, `{t=3000, target1, repeated_hit}` → `target1 -> 1000`.
3. Unit test: `{t=inf, target2, miss}` → `target2 -> inf`, and every on-target event on target2 passes
   `before_identification`.
4. Invariant test on real data: `set(idents.dropna(subset=["target"])[["subject","trial","target"]])` covers every
   target in `targets.pkl`, and no false-alarm row has a non-null target.
5. Re-run the LWS funnel and compare `is_lws` counts before/after — expect a shift concentrated in trials containing
   false alarms.

---

## High

### H1. `num_fixs_to_strip` is computed across both eyes concatenated

**STATUS: FIXED** (`54362bf`).

**Where:** `data_models/preprocess/fixations.py:121-146`, called from `process_trial_fixations:53`

**Description.** `Trial.get_raw_eye_movements()` concatenates left-eye and right-eye events
(`Trial.py:150-155`), so the frame reaching `_num_fixations_to_strip` is *all left-eye fixations, then all right-eye
fixations*. `num_to_true` scans forward over that flat array, so for a left-eye fixation with no later left-eye strip
fixation, the "next strip fixation" is found in the **right eye's** sequence — which restarts at the beginning of the
trial.

**Outcome.** `num_fixs_to_strip` is wrong for late left-eye fixations. It should be `inf` (never returns to the strip)
but instead gets a finite count computed against the other eye. When the right eye's early fixations are in the strip,
the value can be small (e.g. 1), so the `not_before_exemplar_visit` LWS criterion **rejects genuine LWS candidates**.
Since the right-eye block always follows the left, right-eye fixations are unaffected — the bias is one-sided, which
means it does not cancel out across subjects with different dominant eyes.

**Fix.** Compute per `(eye)` group inside the trial (the function already receives a single trial):

```python
fixs_to_strip = (
    fix_features.groupby(cnst.EYE_STR, sort=False, group_keys=False)
    .apply(lambda grp: num_to_true(grp[[cnst.X, cnst.Y]].apply(...)))
)
```

Keep the index aligned to `fix_features` so the `pd.concat` on line 54 still lines up. Also note `is_in_strip` is
currently built with a fresh `RangeIndex` (line 124-127) rather than `index=fix_features.index` — that happens to work
only because `_extract_fixation_features` resets the index; make it explicit.

**Validate.** Unit test: synthetic two-eye fixation frame where the left eye never enters the strip and the right eye's
first fixation does — assert every left-eye `num_fixs_to_strip` is `inf`. Then, on real data, assert
`fixations.groupby(["subject","trial","eye"])["num_fixs_to_strip"].last()` is `inf` wherever that eye's last fixations
are outside the strip.

---

### H2. `drop_outliers` never applies to visits, so `exclude="outliers"` is a silent no-op for visit funnels

**STATUS: FIXED** (`e3d432a`).

**Where:** `analysis/helpers/read_data.py:42-46`; `analysis/helpers/funnels/build_funnels.py:63-64`

**Description.** `read_data` filters `fixations` by `outlier_reasons` but leaves `visits` untouched — `visits.pkl` was
built in stage 1 from *all* fixations, and the visit table carries no outlier column. Yet
`build_event_classification_funnel(..., event_type="visit", exclude="outliers"|"both")` accepts and appears to honour
the request; `"both"` is the default.

**Outcome — measured.** Against the built pickles (2026-08-06), `drop_outliers=True` removes **11,830 of 116,947
fixations (10.1%)** and **0 of 5,720 visits** — an exact no-op, confirmed by
`test_h2_outlier_exclusion_is_a_noop_for_visits`.

So every visit-level analysis (the primary unit in `time_on_task`, `time_in_trial`, `spatial_effects`, `ssm_and_ab`)
silently includes outlier-derived visits, while the parallel fixation-level analysis excludes a tenth of its data.
The two event levels are therefore built on materially different samples, and the discrepancy is invisible from the
call site.

**Fix — now.** What makes a visit an outlier is an open research question (T2), so do **not** invent a rule. Make the
API honest instead: in `build_event_classification_funnel`, when `event_type == "visit"` and `exclude` is `"outliers"`
or `"both"`, raise `NotImplementedError` (or warn loudly and force `exclude="invalid_trials"`) referencing T2. A caller
must not be able to believe outliers were removed when they were not.

Note this makes the current default (`exclude="both"`) fail for visit funnels, which is the point — every existing
visit-level notebook call must be revisited deliberately.

**Fix — after T2 is decided.** Propagate outlier information into the visit table in
`preprocess/visits.py:_extract_visit_features` (e.g. `n_outlier_fixations`, `n_fixations`), then apply the chosen rule
in `read_data`.

**Validate.** Now: assert `build_event_classification_funnel(..., event_type="visit", exclude="both")` raises. After
T2: assert `exclude="none"` and `exclude="outliers"` return different row counts (they are currently identical — that
equality is the bug, and is the regression test).

---

### H3. Per-subject pickle caches are silent and unversioned

**STATUS: FIXED** (`5215e36`).

**Where:** `pipeline/parse_raw_data.py:39-48`; `data_models/Subject.py:339-350`

**Description.** `parse_single_subject` prefers `Subject.pkl` over re-parsing, and `get_fixations` prefers
`fixation_df.pkl`. Neither cache records the code version or the hyperparameters used to build it
(`on_target_threshold_dva`, the `peyes` detector settings, the trigger-parsing logic).

**Outcome.** Any fix in this document — including C2, C3, C4, H1 — has **no effect** until those pickles are deleted by
hand. Results are silently a mixture of old and new code, and are not reproducible from a clean checkout. This is the
single largest reproducibility risk in the repo.

**Fix.** Add a cache key. Minimum viable version: write a sidecar `cache_meta.json` next to each pickle containing the
git commit of `data_models/` + the relevant hyperparameters, and invalidate when it differs. Add a
`force_reparse: bool = False` argument threaded from `run_pipeline`. Log (not `print`) every cache hit with the key.

**Validate.** Change `ON_TARGET_THRESHOLD_DVA`, re-run `run_pipeline`, and assert the fixation table changes without any
manual deletion.

---

### H4. `parse_all_subjects` swallows all exceptions; the dirname split sits outside the `try`

**STATUS: FIXED** (`d84dafc`).

**Where:** `pipeline/parse_raw_data.py:15-28`

```python
for subj_dir in tqdm(subject_dirs, ...):
    exp_name, subj_id, _extra = subj_dir.split("-")   # outside the try
    try:
        subj = parse_single_subject(...)
    except Exception as e:
        bad_subjects[subj_id] = str(e)
```

**Description.** Two problems. (a) `except Exception` catches genuine code bugs (C3 is exactly this) and converts them
into a quietly missing subject. (b) The `split("-")` is *outside* the `try` and unpacks to exactly 3 parts, so a
directory not matching `exp-id-session` aborts the whole run — the opposite of the intended tolerance.

**Outcome.** N changes without a hard failure. Downstream, nothing records which subjects are missing or why; the only
trace is stdout, and `verbose=False` is passed to `build_dataframes` from `run_pipeline:51`.

**Fix.** Move the split inside the `try`. Narrow the caught exception types to the ones that legitimately mean "this
subject's raw data is unusable" (`FileNotFoundError`, `AssertionError`, a project-specific `RawDataError`) and let the
rest propagate. Return `bad_subjects` from `parse_all_subjects` (currently discarded) and persist it beside the
pickles.

**Validate.** Add a malformed directory to a fixture raw-data tree; assert the run completes, the subject appears in the
returned `bad_subjects`, and the reason is recorded. Assert an injected `TypeError` propagates rather than being
swallowed.

---

### H5. `_is_between_triggers` assumes start/end triggers are equal in count and correctly interleaved

**STATUS: DEFERRED** — needs raw-data re-parsing to validate. Confirmed by
`test_unclosed_final_trial` and `test_dropped_end_trigger_does_not_merge_trials` (both `xfail(strict)`).

A scan-in-order version that dropped unclosed segments was written and then reverted (`74129aa`, reverted in
`eff16a7`): discarding a trial is safe but lossy, and a better option exists.

**Agreed approach — recover the boundary from `TRIAL_END`.** `_ExperimentTriggerEnum` carries `TRIAL_START`/
`TRIAL_END` alongside `STIMULUS_ON`/`STIMULUS_OFF`; `TRIAL_END` lands roughly 1 s after `STIMULUS_OFF`. When a
trial's `STIMULUS_OFF` is missing, close the trial at its `TRIAL_END` instead. That yields a *real* boundary, ~1 s
long rather than however long the inter-trial interval happens to be, so the trial is kept without fabricating its
extent.

Three things to settle when implementing, all of which need the raw data:

1. How often a `STIMULUS_OFF` is actually missing, and whether it correlates with subject, session position or trial
   duration. If it does, dropping trials would be a selection effect rather than a rounding error.
2. Whether `TRIAL_END` is reliably present when `STIMULUS_OFF` is not — they may well be lost together.
3. Whether the ~1 s tail should be trimmed back to an estimated stimulus offset, since `to_trial_end` feeds the
   `not_close_to_trial_end` LWS criterion at a 1000 ms threshold; an extra second of tail sits exactly on it.

Until then the original positional pairing stands, documented in the function's docstring. Note H4 changes its
failure mode for the better: the `ValueError` is now caught as a recoverable parse error, so the affected subject is
recorded in `bad_subjects` and `parse_failures.json` rather than vanishing silently.

**Where:** `data_models/parse/triggers_and_gaze.py:104-114`

```python
start_idxs = np.nonzero(trigs == start)[0]
end_idxs = np.nonzero(trigs == end)[0]
start_end_idxs = np.vstack([start_idxs, end_idxs]).T
```

**Description.** `np.vstack` requires equal lengths and pairs purely by position. `README.md` already documents that
record/trial triggers are interleaved (`start_trial → start_record → end_trial → end_record`). An aborted final trial,
a truncated recording, or a dropped trigger gives unequal counts (`ValueError`) or, worse, equal counts that pair the
wrong start with the wrong end — silently mislabelling trial boundaries.

**Outcome.** With unequal counts: the subject is dropped via H4. With mispairing: gaze samples are assigned to the wrong
trial, and every downstream time-relative measure for that subject is wrong, with no error raised.

**Fix.** Pair by scanning in order rather than by position, and validate explicitly:

```python
def _is_between_triggers(trigs: pd.Series, start: int, end: int) -> pd.Series:
    """Mark samples between each `start` trigger and the next `end` trigger that follows it."""
    res = pd.Series(False, index=range(len(trigs)))
    open_at: int | None = None
    for pos, trg in enumerate(trigs.to_numpy()):
        if trg == start and open_at is None:
            open_at = pos
        elif trg == end and open_at is not None:
            res.iloc[open_at:pos + 1] = True
            open_at = None
    if open_at is not None:
        warnings.warn(f"Unclosed {start!r} trigger at position {open_at}; ignoring trailing segment.")
    return res
```

**Validate.** Unit tests for: balanced triggers (unchanged result), a trailing unclosed `STIMULUS_ON`, and a doubled
`STIMULUS_ON`. Assert trial counts per subject match E-Prime's expected trial count.

---

### H6. Target distances silently fall back to the non-dominant eye

**STATUS: FIXED** (`f1cb289`).

**Where:** `data_models/Trial.py:195-206`

```python
main = left_dists if self._subject.eye == DominantEyeEnum.LEFT else right_dists
second = right_dists if self._subject.eye == DominantEyeEnum.LEFT else left_dists
dists = main.fillna(second)
```

**Description.** Per-sample gaze-to-target distances use the dominant eye, filling gaps from the other eye. These
distance columns are what `extract_trial_identifications` uses to pick the closest target and to classify
hit vs false alarm (`target_identifications.py:104-110`, `115-135`).

**Outcome.** During dominant-eye data loss, a hit/false-alarm decision — and therefore hit rate, A', and the
identification time feeding LWS classification — is made from the non-dominant eye. This is inconsistent with the rest
of the pipeline, where `read_data(drop_bad_eye=True)` deliberately discards the non-dominant eye. The mixing is
undocumented and invisible in the output.

**Fix (decision 4: dominant eye only, throughout).** Drop the `fillna(second)`; a sample with no dominant-eye gaze gets
NaN distances and is simply not classifiable. Then audit the rest of stage 1 for the same pattern — the eager
per-eye work in `Trial.__init__` (`Trial.py:44-49`) computes and stores both eyes' events regardless, and
`read_data(drop_bad_eye=True)` is what finally discards one. Dropping the non-dominant eye once, in `Trial`, would make
the invariant hold everywhere and roughly halve stage-1 detection cost; keeping both eyes is only worth it if you still
want the `_determine_*` calibration notebooks to compare eyes.

**Validate.** Count identifications where the dominant eye is NaN but a distance exists — must be 0 after the fix.
Report the number of identifications lost to dominant-eye dropout in the pipeline summary, so the cost of the stricter
rule is visible rather than silent.

---

### H7 (downgraded to Low; config made explicit in `1d6cb47`). Fixations longer than 2500 ms are dropped as outliers, by an inherited library default

**Where:** `data_models/parse/eye_movements.py:12-13`; effective values verified from `peyes._DataModels.config`

```python
peyes.set_event_configurations("fixation", min_duration=50)     # max_duration left at the peyes default
peyes.set_event_configurations("saccade", min_duration=_MIN_EVENT_DURATION)
```

**Description.** `get_outlier_reasons` flags an event when `duration > MAX_DURATION`. The project sets only
`min_duration`, so `max_duration` keeps the `peyes` defaults — verified live: **fixation 2500 ms**, saccade 200 ms.
Any fixation longer than 2.5 s therefore gets `outlier_reasons = ["max_duration"]` and is removed by
`read_data(drop_outliers=True)`, the default for every funnel.

**Outcome — measured, and much smaller than predicted.** Against the built pickles (2026-08-06):

| | |
| --- | --- |
| fixations total | 116,947 |
| duration > 2500 ms | **6 (0.005%)** |
| duration percentiles | p50 = 170 ms, p95 = 346 ms, p99 = 770 ms, max = 2797 ms |
| of those 6, on-target | **6 (100%)** vs a 10.0% base rate |

The predicted *direction* holds exactly — every fixation the cap removes is on-target, against a 10% base rate — but
at n = 6 it cannot move any result. **Downgraded from High: the mechanism is real, the impact is not.**

What remains is a hygiene point rather than a correctness one: this is an inherited analysis decision sitting directly
on the dependent variable, invisible because it is a library default rather than a project constant, and it appears
nowhere in `config.py`, `funnel_config.py`, or the `default_value_selection/` notebooks that justify every *other*
hyperparameter. It should be chosen rather than defaulted (T3), and it should be watched: the guard assertion in
`test_h7_long_fixation_cap_is_immaterial_in_this_dataset` fails if the affected share ever exceeds 0.1%.

Note this is unrelated to M15 — the criterion is duration-based, so it is computed correctly. The concern is that the
threshold was never chosen.

**Fix.** Set `max_duration` explicitly for both event types alongside the existing `min_duration` calls, with the value
recorded in `config.py` and justified the way the other thresholds are. Whether the right value is 2500 ms, something
longer, or `inf` is a research decision (see T3).

**Validate.**
1. Report the distribution of fixation durations and the count of fixations with
   `outlier_reasons == ["max_duration"]`, split by on-target vs off-target. If on-target fixations are
   disproportionately represented, the current default is biasing the LWS measure.
2. Re-run the LWS funnel with `max_duration = inf` and compare `is_lws` counts.

---

### H9. pandas 3.0 breaks trigger/gaze alignment: stage 1 cannot run

**STATUS: FIXED** (`12a67af`).

**Where:** `data_models/parse/triggers_and_gaze.py:139`

```python
triggers.loc[:, _TRIGGER_COLUMNS] = triggers[_TRIGGER_COLUMNS].fillna(0).astype('Int64')
```

**Description.** After the outer merge, `trigger` is `float64` (the column holds `IntEnum` members, which pandas
infers as `int64`, then the gaze-only rows introduce NaN) while `action` is `Int64`. Assigning an `Int64` frame into
`.loc[:, cols]` over that dtype combination raises under pandas 3.0.5:

```
AttributeError: 'Series' object has no attribute '_hasna'
```

Confirmed the real parser produces the breaking combination — `_read_triggers` returns
`{'time': float64, 'trigger': int64, 'action': Int64}`, and the merge turns `trigger` into `float64`. Reproduced
standalone; it is not an artefact of the test fixtures. Under pandas 2.3.3 the same line worked, so this arrived with
the upgrade taken to resolve H8.

**Outcome.** Blocking for stage 1: no subject can be parsed from raw data on the current environment. Stage 2 is
unaffected — the pickles read fine and every analysis path works.

**Fix.** Avoid the mixed-dtype `.loc` round-trip. Assign column-wise, and keep `trigger` a stable dtype rather than
letting the merge decide:

```python
for col in _TRIGGER_COLUMNS:
    triggers[col] = triggers[col].fillna(0).astype("Int64")
```

Better still, stop relying on inference: build `trigger` as `Int64` in `_read_triggers` and keep the enum conversion
at the point of use, so the merge cannot silently retype it.

**Validate.** Remove the `xfail` markers from `TestTrialBoundaries::test_balanced` and `::test_two_balanced_trials`.
Then parse one subject end-to-end and confirm trial counts match E-Prime's.

---

### H8. Version-fragile pickles: `peyes` pins numpy 1.x, the pickles were written by numpy 2.x

**Status: resolved for now** (2026-08-06) — numpy/pandas were upgraded to 2.5.1 / 3.0.5, above `peyes`'s pin. `peyes`
imports *and* runs correctly outside the pin: segmentation and `create_events` were both verified on a synthetic
trace, so the declared `numpy~=1.2` is conservative rather than a real incompatibility. The pickles now load and all
four empirical checks run. The upgrade did surface **H9**.

The underlying fragility below is unchanged and still worth fixing.

**Where:** environment; surfaced by `tests/test_realdata_findings.py`

**Description.** Found while trying to run the empirical checks. `peyes 0.0.9.6` declares `Requires-Dist: numpy~=1.2`
(i.e. `>=1.2, <2.0`), so installing it downgraded the project venv from numpy 2.3.4 to **1.26.4**. The six pickles in
`cnfg.OUTPUT_PATH` were written under numpy ≥2 — verified by inspecting the raw bytes, which reference
`numpy._core.numeric`, the 2.x module path. Under numpy 1.26.4 that path does not exist:

```
ModuleNotFoundError: No module named 'numpy._core.numeric'
```

So the environment that can *run* the pipeline cannot *read* its output, and vice versa. Current state:

| | numpy 1.26.4 (now) | numpy ≥2 |
| --- | --- | --- |
| `import peyes` / run stage 1 | works | unsupported by the pin |
| read the existing pickles | **fails** | works |

**Outcome (at the time).** Blocking: no analysis notebook and no empirical validation could run. It also means the
existing pickles were produced by an environment that no longer exists on this machine, so they are not reproducible
from the current lockfile-free setup — a concrete instance of the H3 risk rather than a separate one.

More generally, pickle is being used as the interchange format between stage 1 and stage 2 (`*.pkl`, plus the
per-subject caches). Pickle is not version-portable: it couples the stored data to the exact class and module layout of
the writing environment. Any future dependency bump can silently orphan the whole dataset the same way.

**Fix (remaining work).** Pin the environment — `pyproject.toml` dependencies plus a lockfile — so the analysis
environment is reproducible, and record that `peyes` is deliberately run above its declared `numpy~=1.2` pin, with the
smoke test as the justification. Then move the stage-1/stage-2 interchange off pickle to a version-portable columnar
format (parquet/feather). The object columns currently in these frames — `outlier_reasons` (list) and the visits'
`event` (list) — need a defined encoding before that move.

**Validate.** Done: `pytest tests/test_realdata_findings.py` runs instead of skipping. Still to add: an environment
smoke test asserting `import peyes`, a `detect_eye_movements` round-trip, and `read_data(cnfg.OUTPUT_PATH)` all
succeed in the same interpreter, so a future dependency change cannot re-open this silently.

---

## Medium

### M17. `del ... start_idx` raises `UnboundLocalError` when the log has no BLOCK trigger

**STATUS: FIXED** (`74129aa`).

**Where:** `data_models/parse/triggers_and_gaze.py:90-102`

```python
for trg in _ExperimentTriggerEnum:
    ...
    if not is_block_start.any():
        continue          # block_num / start_idx never bound
    start_idx = is_block_start.idxmax()
    ...
del trg, name, block_num, is_block_start, start_idx   # unconditional
```

**Description.** The `del` is manual scratch-variable cleanup, but it runs unconditionally while three of the five
names are bound only inside the loop body's `if` branch. A trigger log containing no `BLOCK_*` trigger at all raises
`UnboundLocalError: cannot access local variable 'start_idx'`. Confirmed by
`tests/test_triggers_and_gaze.py::TestTrialBoundaries::test_no_block_trigger_raises`.

**Outcome.** Latent in production — every real session should emit `BLOCK_1` — but it turns a benign edge case into a
crash that H4 would convert into a silently dropped subject. Its practical cost today is to testing: the function
cannot be exercised without constructing block triggers, which is why the boundary tests carry that extra setup.

**Fix.** Delete the `del` statement. It frees nothing meaningful — the names go out of scope when the function
returns — and scoping the loop body into a helper (`_assign_block_numbers(merged)`) removes the motivation for it
entirely.

**Validate.** Remove the `xfail` marker from `test_no_block_trigger_raises`.

---

### M15 (was C1; **FIXED** in `1d6cb47`). `peyes.create_events` receives `pixel_size = viewer_distance_cm`

**Where:** `data_models/parse/eye_movements.py:44-45`

```python
events = peyes.create_events(
    labels=labels, t=t, x=x, y=y, pupil=pupil,
    viewer_distance=viewer_distance_cm, pixel_size=viewer_distance_cm   # <-- pixel_size is wrong
)
```

**Description.** `pixel_size` should be `pixel_size_cm` (≈0.0277 cm); it gets the viewer distance (≈60 cm) instead —
off by a factor of ~2000. A real bug, but **verified to have no effect on any current result.**

**Provenance (checked).** Not a recent regression. The call was **correct** in the `plgrnd.py` scratchpad (`ab65dc3`,
2025-05-13: `pixel_size=cnfg.TOBII_PIXEL_SIZE_MM / 10`) and acquired the bug two days later when the code was extracted
into `parse/eye_movements.py` (`09e2b01`, 2025-05-15, "many changes"). It survived both later moves of the file
(`5837517`, `483011c`) untouched.

**Why it is inert today (verified against the installed `peyes`).**

- `detector.detect()` (line 33) gets the *correct* `pixel_size_cm`, so **segmentation is unaffected**.
- Every visual-angle feature `create_events` produces — `amplitude`, `velocity`, `dispersion`, `ellipse_area`,
  `azimuth`, `is_outlier` — is dropped by `_REDUNDANT_FIXATION_FEATURES` (`preprocess/fixations.py:12-17`). `x`/`y`
  come from `center_pixel`, which is in pixels.
- `get_raw_eye_movements()` has no caller outside `Trial.process_fixations:161`.
- **`outlier_reasons` does not depend on `pixel_size`.** `peyes._DataModels/Event.py:119-133` checks only:
  duration vs `MIN_DURATION`/`MAX_DURATION` (ms), `x`/`y` negative, and `x`/`y` beyond
  `SCREEN_MONITOR["resolution"]` (pixels). The velocity/dispersion checks are an explicit unimplemented
  `# TODO` in the library, and no subclass overrides the method.

So the surviving fixation columns are all pixel- or time-based. **No re-run is required for this fix.**

The screen geometry `peyes` uses for the surviving outlier check is a separate problem — see **M16**.

**Outcome if left unfixed.** Blocks the saccade-amplitude / micro-saccade analyses sketched in `plgrnd2.py:93-103`, and
becomes live the moment `peyes` implements its outlier TODO or any DVA feature stops being dropped.

**Fix.** Pass `pixel_size=pixel_size_cm`. Make the unit unambiguous while there (M8).

**Validate.** Unit test: synthetic trace with a known 5° saccade; assert the resulting event amplitude is 5° ±
tolerance. Optionally confirm inertness by rebuilding one subject and asserting `outlier_reasons` is **unchanged**.

---

### M1. `metadata` columns are all `object` dtype

**STATUS: FIXED** (`30fe583`).

**Where:** `data_models/Subject.py:256-267`

`pd.concat(list_of_mixed_type_Series, axis=1).T` produces an all-`object` frame — verified: `duration`,
`num_targets`, `gaze_coverage`, `trial` all come out as `dtype('O')`, and `metadata["duration"] / 1000` stays `object`.

**Outcome.** Arithmetic and comparisons still work but silently lose NaN semantics and numeric dtype guarantees;
`groupby`/`agg` results are unpredictable; a NaN in an `object` comparison can raise `TypeError` rather than propagate.
`_coerce_column_types` fixes only `subject`/`trial`/`target`/`target_angle` on the funnel output, not on `metadata`.

**Fix.** Build the metadata frame from records instead of transposing:
`pd.DataFrame.from_records([t.get_metadata(...) for t in trials])`, then `astype` an explicit schema dict.

**Validate.** `assert metadata.dtypes.to_dict() == EXPECTED_SCHEMA` in a test.

---

### M2. `is_on_target` for visits collapses to "minimum distance ≤ threshold"

**STATUS: FIXED** (`26968d6`).

**Where:** `analysis/helpers/funnels/event_classification.py:110-119`, `50-66`

For `event_type="visit"`, `_distance_columns` returns *all* columns ending in `distance_dva` — i.e.
`min_distance_dva`, `max_distance_dva`, **and** `weighted_distance_dva` — and `is_on_target` then takes `.any(axis=1)`.
Since `min ≤ weighted ≤ max`, the `any()` is mathematically identical to `min_distance_dva <= threshold`.

**Outcome.** The criterion is stricter-looking than it is. Because visits are *constructed* from fixations already below
the stage-1 threshold, the step is close to a no-op whenever the analysis threshold ≥ the pipeline threshold — so the
funnel's `on_target` step size is misleading. Anyone raising the threshold to test robustness gets no effect.

**Fix.** Choose the statistic deliberately and name it: `weighted_distance_dva <= threshold` (visit is on-target on
average) or `min_distance_dva <= threshold` (visit touches the target). Select the single column explicitly for visits
rather than globbing.

**Validate.** Test that raising `on_target_threshold_dva` above the pipeline value monotonically changes the visit
`on_target` count in the intended direction.

---

### M3. `SearchArray._get_path` disagrees with the directory layout the loader reads

**STATUS: FIXED** (`26968d6`).

**Where:** `data_models/SearchArray.py:183-191` vs `data_models/Trial.py:165-174`

The loader builds `.../generated_stim1/<color|bw|noise>/image_N.mat`; `_get_path` builds
`.../generated_stim1/array_<color|bw|noise>/image_N.<ext>`. `from_mat:100-104` parses the directory name with
`SearchArrayCategoryEnum[array_type_name.upper()]`, which would raise `KeyError` on `"ARRAY_COLOR"` — so the loader
reflects the real layout and `_get_path` is wrong.

**Outcome.** `SearchArray.mat_path` and `SearchArray.image_path` return non-existent paths. Any notebook trying to
display the stimulus image fails. `__eq__` (line 217) compares two equally-wrong paths, so it does not surface the bug.

**Fix.** Make `_get_path` the single source of truth and have `Trial._create_search_array` call it, so the two cannot
drift again.

**Validate.** `assert os.path.isfile(search_array.mat_path)` and `assert os.path.isfile(search_array.image_path)` for
one array of each category.

---

### M4. Falsy-zero bugs on `start_identify_idx`

**STATUS: FIXED** (`9de4780`).

**Where:** `data_models/parse/triggers_and_gaze.py:188`, `197`, `208`

```python
assert not start_identify_idx, ...          # line 188
assert start_identify_idx and start_identify_idx < idx, ...   # line 197
if start_identify_idx and trg in [...]:     # line 208
```

`start_identify_idx` is a DataFrame label; label `0` is falsy. If the first trigger row is a `SPACE_ACT`, line 188's
assertion wrongly passes on a *second* mark, and lines 197/208 wrongly treat a pending mark as absent.

**Outcome.** Misclassified actions for a trigger log beginning with a key press. Low probability (logs normally begin
with `START_RECORD`), but the failure is silent.

**Fix.** Compare against `None` explicitly: `assert start_identify_idx is None`, `if start_identify_idx is not None and ...`.

**Validate.** Unit test with a trigger table whose row 0 is `SPACE_ACT` followed by `CONFIRM_ACT`.

---

### M5. `pd.Categorical.from_codes` relies on enum values matching list positions

**STATUS: FIXED** (`26968d6`).

**Where:** `analysis/helpers/funnels/build_funnels.py:172-183`

```python
pd.Categorical.from_codes(
    data["trial_category"].map(lambda val: SearchArrayCategoryEnum[val]),
    categories=[cat.name for cat in SearchArrayCategoryEnum], ordered=True)
```

This works only because both enums happen to be zero-based and contiguous. Verified: an enum starting at 1 raises
`ValueError: codes need to be between -1 and len(categories)-1`.

**Outcome.** Adding a member with a non-contiguous value, or removing `UNKNOWN = 0`, breaks funnel construction at
runtime — or, if a gap is introduced, silently mislabels categories.

**Fix.** Use the values directly, which does not depend on the numeric codes:

```python
pd.Categorical(data["trial_category"], categories=[c.name for c in SearchArrayCategoryEnum], ordered=True)
```

Note `analysis/helpers/read_data.py:57` (`parse_as_categorical`, used by `hit_rate.ipynb`) already does it this way —
consolidate on that one function and delete the duplicate logic.

**Validate.** Test with a temporarily non-zero-based enum; the correct implementation is unaffected.

---

### M6. `reindex(...).astype(bool)` turns missing criteria into `True`

**STATUS: FIXED** (`30fe583`).

**Where:** `analysis/helpers/funnels/trial_inclusion.py:40-46`

```python
pd.concat(ordered_components, axis=1)
  .reindex(meta_idx)
  .assign(is_valid_trial=lambda df: df.all(axis=1))
  .astype(bool)
```

If any criterion Series is missing a `(subject, trial)` key, `reindex` inserts NaN; NaN is truthy, so `.astype(bool)`
yields `True`, and `df.all(axis=1)` also treats NaN as True. The trial **passes** a criterion that was never evaluated.

**Outcome.** Currently latent — all five predicates are built from `metadata` or the full `meta_idx`, so no gaps arise
today. It becomes a live bug the moment a criterion is added that derives its index from an event table (the natural
way to write one).

**Fix.** `.fillna(False)` before `.assign(...)` and before `.astype(bool)`. `_convert_criteria_to_funnel`
(`build_funnels.py:164`) already does this — mirror it here.

**Validate.** Test with a criterion Series deliberately missing one trial; assert that trial is `False`, not `True`.

---

### M7. Cumulative funnel columns keep the raw criterion name

**Where:** `analysis/helpers/funnels/build_funnels.py:160-166`; consumed in `analysis/R/helpers.R:27-28`

`_convert_criteria_to_funnel` overwrites each criterion column with the running AND of that criterion and all previous
ones, but keeps the original name. So in the exported `funnel_results.csv`, `on_target` does not mean "this event is on
target" — it means "this event passed every trial-level criterion **and** is on target".

**Outcome.** In `helpers.R`, `subset(dat, is_valid_trial)` is a no-op whenever `on_target_only = TRUE`, because
`on_target` already implies it. Harmless today, but the column names invite exactly the wrong reading, and any
downstream consumer computing e.g. "proportion of on-target events that are LWS" from the raw column will silently get
a different denominator than intended.

**Fix.** Either suffix cumulative columns (`passed_on_target`, …) while retaining the raw booleans alongside, or
document the semantics in the funnel docstring *and* in `helpers.R`. Prefer the former — the raw per-criterion booleans
are genuinely useful for diagnosing which step drops what.

**Validate.** Assert `funnel["on_target"].sum() <= criteria["on_target"].sum()` and that the two differ on real data.

---

### M8. `detect_eye_movements` default `pixel_size_cm` is a millimetre value

**STATUS: FIXED** (`1d6cb47`).

**Where:** `data_models/parse/eye_movements.py:27` — `pixel_size_cm: float = cnst.PIXEL_SIZE_MM`

The default is ~0.277 mm supplied to a parameter named `..._cm`. Both call sites in `Trial._detect_eye_movements` pass
`cnfg.PIXEL_SIZE_MM / 10` explicitly, so the default is never used — but it is a 10× trap for the next caller, and it is
the same class of error as M15.

**Fix.** Add `PIXEL_SIZE_CM = PIXEL_SIZE_MM / 10` to `constants.py` and use it as the default. Keep units in every
name.

**Validate.** `assert cnst.PIXEL_SIZE_CM == pytest.approx(cnst.PIXEL_SIZE_MM / 10)`; grep that no call site divides by
10 inline anymore.

---

### M9. `visits.py` error paths raise `AttributeError` instead of the intended message

**STATUS: FIXED** (`30fe583`).

**Where:** `data_models/preprocess/visits.py:92-93`, `103-105`

`trials = fixs_subset[cnst.TRIAL_STR].unique()` is a numpy array, but the error strings call `trials.iloc[0]` and
`eyes.iloc[0]`. Reaching either branch raises `AttributeError: 'numpy.ndarray' object has no attribute 'iloc'`,
masking the real diagnostic.

**Fix.** Use `trials[0]` / `eyes[0]`.

**Validate.** Unit test that passes a two-trial subset and asserts the raised `RuntimeError` message contains the trial
number.

---

### M10. Statistical: the GAMs treat correlated visits as independent Bernoulli trials

**Where:** `analysis/R/time_on_task_gam.R:23-28`, `time_in_trial_gam.R:24-29`, `spatial_gam.R:18-40`

Every model is `is_lws ~ trial_category + s(...) + s(subject, bs="re")`. The unit of observation is a *visit*, but
visits are nested within target within trial within subject: multiple visits to the same target in the same trial are
strongly dependent, and a subject-level random intercept does not absorb that.

**Outcome.** Standard errors are anti-conservative and p-values on the smooth terms are optimistic. The point estimates
of the smooths are likely fine; the inference is not.

**Fix.** Add the nested grouping, e.g. `+ s(trial_uid, bs="re")` where `trial_uid = interaction(subject, trial)`, or
move to `bam(..., discrete=TRUE)` with `rho` for within-trial autocorrelation if the visits are ordered in time.
Alternatively aggregate to one observation per (subject, trial, target) and model counts.

**Validate.** Compare `summary(model)` EDF/p-values before and after adding the nesting; report both. Simulate a null
dataset (shuffle `is_lws` within trial) and confirm the Type-I error rate of the current model exceeds nominal 5%.

---

### M11. Statistical: `anova(simple_model, interaction_model, test = "Chisq")` on REML fits is not valid

**Where:** `analysis/R/spatial_gam.R:49`

The two models differ in their smooth structure and are both fitted with `method = "REML"`. REML log-likelihoods are
not comparable across models with different fixed/smooth structures; `mgcv`'s own documentation warns against this
comparison.

**Fix.** Refit both with `method = "ML"` for the comparison, or use `summary(interaction_model)` on the `by=` smooths,
or compare with AIC on ML fits. Keep REML for the final reported fit.

**Validate.** Refit under ML and confirm the conclusion about a category-specific spatial pattern is unchanged; report
whichever test is used.

---

### M12. `px2deg` is a small-angle constant applied at large eccentricities

**Where:** `data_models/Subject.py:157-165`

`px2deg` is the angle subtended by *one* pixel at the screen centre, then applied as a linear multiplier to distances up
to ~1000 px. Because visual angle is a tangent function, the linear form overestimates: at 1000 px (≈27.7 cm) and 60 cm
viewing distance, linear gives ≈26.4° vs a true ≈24.8° — about 7%.

**Outcome.** Negligible where it matters most (the 1.75 DVA on-target threshold ≈ 66 px, error <0.1%), but any analysis
using DVA at large eccentricity — e.g. saccade amplitudes, target eccentricity effects in `spatial_effects.ipynb` — is
biased outward, and the bias grows with distance, so it is *not* a constant offset.

**Fix.** Provide `px2deg_at(distance_px)` using `degrees(arctan2(distance_px * pixel_size_cm, screen_distance_cm))` for
large-distance conversions; keep the linear constant only for near-threshold work and document the split.

**Validate.** Compare linear vs arctan conversion across the observed distance range; assert the discrepancy at the
on-target threshold is <1% and report the max discrepancy in the analysis.

---

### M13. Configuration is duplicated, partly stale, and hardcodes machine-specific paths

**STATUS: FIXED** (`2a49ed0`).

**Where:** `config.py`

- Lines 17-18 override `OUTPUT_PATH` and `PUBLICATIONS_PATH` with `C:\Users\nirjo\Desktop\...` and are marked
  `# TODO: remove me!`. Anyone else running the pipeline writes to a path that does not exist, or reads stale results.
- Lines 39-64 (`GAZE_COVERAGE_PERCENT_THRESHOLD`, `TIME_TO_TRIAL_END_THRESHOLD`, `FIXATIONS_TO_STRIP_THRESHOLD`,
  `_ANY_FUNNEL_STEPS`, `LWS_FUNNEL_STEPS`, `TARGET_RETURN_FUNNEL_STEPS`) are **not read by the current funnel code**,
  which uses `analysis/helpers/funnels/funnel_config.py`. The step names differ too (`instance_on_target` vs
  `on_target`), so the two files describe different pipelines. `cnfg.TIME_TO_TRIAL_END_THRESHOLD` *is* still referenced
  directly from `time_in_trial.ipynb` and `_determine_time_to_trial_end.ipynb`.
- `BAD_ACTIONS` (config.py:33) and `DEFAULT_BAD_ACTIONS` (funnel_config.py:12) are two definitions of the same thing.
- `config.py` does `from constants import *`, so `cnfg.X` and `cnst.X` both resolve and the two are used
  interchangeably across the codebase.

**Outcome.** Two competing sources of truth for funnel behaviour, with the notebooks straddling both. A reader cannot
tell which thresholds are live without tracing imports.

**Fix.** Delete the dead threshold/step lists from `config.py`; re-point the two notebooks at `funnel_config.py`;
delete `config.py:BAD_ACTIONS` in favour of `DEFAULT_BAD_ACTIONS`; move paths to environment variables or a
gitignored `config_local.py` with a checked-in `config_local.example.py`. Replace the star-import with an explicit
`import constants as cnst`.

**Validate.** `grep -r "cnfg\.\(GAZE_COVERAGE_PERCENT_THRESHOLD\|TIME_TO_TRIAL_END_THRESHOLD\|FIXATIONS_TO_STRIP_THRESHOLD\|.*_FUNNEL_STEPS\)"`
returns nothing. Pipeline runs on a machine without the Desktop path.

---

### M14. No package structure; `plgrnd2.py` cannot import

**Where:** repo-wide (no `__init__.py` anywhere); `plgrnd2.py:42-43`

```python
from analysis.helpers.funnels import build_trial_inclusion_funnel, build_event_classification_funnel, calculate_funnel_step_sizes
```

`analysis.helpers.funnels` is an implicit namespace package that exports nothing, and `calculate_funnel_step_sizes`
does not exist — the function is `calculate_step_sizes` in `size_and_proportion.py`. This import raises `ImportError`.

**Outcome.** The scratchpad is broken, and every module depends on the CWD being the repo root — with no
`pyproject.toml`, that contract is undocumented and unenforceable.

**Fix.** Add a minimal `pyproject.toml` with `[project]` metadata and install with `pip install -e .`. Add
`__init__.py` files (or configure a src layout) and re-export the funnel entry points from
`analysis/helpers/funnels/__init__.py`. Fix the symbol name in `plgrnd2.py`.

**Validate.** `python -c "import plgrnd2"` from a different CWD succeeds.

---

### M16. The screen geometry driving outlier detection is `peyes`'s default, not the project's monitor

**STATUS: FIXED** (`1d6cb47`).

**Where:** `data_models/parse/eye_movements.py:10-19` (module-level config block); consumed by
`peyes._DataModels/Event.py:127-131`

**Description.** The project configures `peyes` with `set_event_configurations` only. It never calls
`peyes.set_screen_monitor(...)`, so `cnfg.SCREEN_MONITOR` keeps the library defaults — which the
`pixel_outside_screen` outlier criterion reads directly:

```python
if (np.any(self._x > cnfg.SCREEN_MONITOR[cnst.RESOLUTION_STR][0]) or
        np.any(self._y > cnfg.SCREEN_MONITOR[cnst.RESOLUTION_STR][1])):
    reasons.append("pixel_outside_screen")
```

Verified values:

| | width_cm | height_cm | resolution | pixel_size_cm |
| --- | --- | --- | --- | --- |
| `peyes` default | 53.1 | 30.0 | (1920, 1080) | 0.0276855 |
| `cnst.TOBII_MONITOR` | 53.0 | 30.0 | (1920, 1080) | 0.0276910 |

The resolution — the only field this criterion uses — happens to match, so the check is currently correct. But it is
correct *by coincidence*: `peyes` defaults to a Tobii rig that is not quite this Tobii rig. The physical width already
disagrees by 1 mm (pixel size by 0.02%), which is direct evidence that the library defaults are not authoritative for
this setup; that field is simply unused today because `pixel_size` is always passed per-call.

**Outcome.** A silent, unpinned dependency on a third-party default sitting on the outlier filter that gates every
analysis. A `peyes` upgrade that changes the default monitor — or reuse of this code with a different display —
silently changes which fixations are flagged `pixel_outside_screen` and therefore which rows survive
`read_data(drop_outliers=True)`. Nothing in the project would report the change.

Two related smells in the same block: the configuration is applied as a **module-level import side effect**
(lines 12-13), so the thresholds depend on import order and are invisible to anyone reading `config.py`; and the
project's own screen geometry lives in `constants.py` while the library's lives in `peyes`, with no link between them.

**Fix.** Derive the `peyes` configuration from `cnst.TOBII_MONITOR` rather than letting it default, and move it out of
import-time side effects into an explicit, idempotent setup function called from the pipeline entry point:

```python
def configure_peyes() -> None:
    """Push the project's screen geometry and event thresholds into peyes' global config."""
    peyes.set_screen_monitor(
        width_cm=cnst.TOBII_MONITOR.width_mm / 10,
        height_cm=cnst.TOBII_MONITOR.height_mm / 10,
        width_px=cnst.TOBII_MONITOR.width,
        height_px=cnst.TOBII_MONITOR.height,
    )
    peyes.set_event_configurations("fixation", min_duration=..., max_duration=...)   # see H7
    peyes.set_event_configurations("saccade", min_duration=..., max_duration=...)
```

Call it once from `run_pipeline`. Pair it with H7, which sets the duration half of the same configuration.

**Validate.** After `configure_peyes()`, assert
`peyes._DataModels.config.SCREEN_MONITOR["resolution"] == (cnst.TOBII_MONITOR.width, cnst.TOBII_MONITOR.height)` and
that `pixel_size` agrees with `cnst.PIXEL_SIZE_MM / 10` to within rounding. Add the assertion as a unit test so a
`peyes` upgrade that moves the defaults fails loudly instead of silently.

---

## Low

### L1. No tests

There is no test suite, and the pipeline has many pure, easily-testable functions. Minimum set, ordered by value —
each of these would have caught a Critical or High issue above:

| Test | Catches |
| --- | --- |
| `test_read_triggers_action_sequences` — all four `SubjectActionCategoryEnum` sequences from a hand-built trigger table | C2, C3, M4 |
| `test_is_between_triggers_unbalanced` — unclosed / doubled triggers | H5 |
| `test_num_fixations_to_strip_per_eye` — synthetic two-eye frame | H1 |
| `test_identification_time_prefers_hit_over_false_alarm` | C4 |
| `test_visit_assignment` — on/off-target sequences with gaps around the merge threshold | visits regression |
| `test_dprime_corrections` — floor/ceiling with each correction against hand-computed values | `sdt.py` regression |
| `test_convert_criteria_to_funnel_is_cumulative` — including NaN handling | M6, M7 |

Suggested layout: `tests/` at repo root, `pytest` + `pytest-cov`, fixtures building small synthetic trigger/gaze frames
(no raw data dependency). Add `[tool.pytest.ini_options] testpaths = ["tests"]` to the new `pyproject.toml` (M14).

### L2. No linter/formatter/type-checker configuration

No `pyproject.toml`, `ruff.toml`, or mypy config. Type hints are inconsistent: `parse_subject_info(file_path) -> dict`
(untyped parameter, unparameterised return), `_extract_singleton_column(df, col_name)` (no annotations),
`_validate_inputs(...)` (no return type), `detect_eye_movements(..., detector=_DETECTOR)` (untyped). Recommended:
`ruff` with `select = ["E","W","F","I","B","C4","UP","SIM"]`, `line-length = 120`, plus `mypy` in non-strict mode
initially (`disallow_untyped_defs = true` for `data_models/` and `analysis/helpers/` only).

### L3. Resource handling and small correctness nits

- `parse_subject_info` (`subject_info.py:15-26`) opens the file without a context manager; an exception leaks the
  handle. Use `with io.open(...) as f:`.
- `Trial.__eq__` is defined without `__hash__`, making `Trial` unhashable — fine today, surprising later.
- `calc_aprime_per_trial` (`sdt.py:75`) computes `diff` and never uses it.
- `size_and_proportion.py:3` imports `numpy` unused.
- `trial_inclusion.py:50` `all_pass` is dead code.
- `Subject.get_targets` / `get_actions` / `get_metadata` wrap `tqdm(..., disable=True)` — the progress bar is
  hardcoded off; either remove it or thread `verbose` through.

### L4. Hardcoded stimulus geometry

`SearchArray._BOTTOM_STRIP_TOP_LEFT/_BOTTOM_STRIP_BOTTOM_RIGHT = (720, 910), (1200, 1080)` carries a
`# TODO: read this from stimulus generation config` (`SearchArray.py:67`). The exemplar-strip rectangle directly
determines `num_fixs_to_strip` and therefore the `not_before_exemplar_visit` LWS criterion. `_NUM_ROWS/_NUM_COLS` and
`_RESOLUTION` are likewise hardcoded and asserted against the `.mat` contents. Read from the stimulus config, or at
minimum assert the strip lies inside the screen and document the provenance of the four numbers.

### L5. Fixation-level and visit-level analyses attribute targets differently

A fixation row carries a single `target` (the closest one, `fixations.py:102-118`), whereas a fixation on-target for two
targets generates a *visit row per target* (`visits.py:20`). Running the same funnel at `event_type="fixation"` vs
`"visit"` therefore uses different denominators and different target attribution. Not a bug, but it should be stated
wherever the two are compared.

### L6. R script hygiene

- `set.seed(42)` before a deterministic REML fit does nothing — remove, or move it to whatever is actually stochastic.
- `gam.check()` and `plot()` inside a script run via `Rscript` silently write `Rplots.pdf`; direct them to a named file.
- `spatial_gam.R:22` — `k = 15` inside `te()` is **per marginal basis**, so the smooth has ~225 basis functions; with
  the `by = trial_category` term added, the model is large and the global + by-smooths are concurve. Consider
  `te(x, y, k = c(8, 8))` and check `concurvity(model)`.
- `spatial_gam.R:60-68` — the prediction grid spans the bounding box of observed `x`/`y`, which includes screen regions
  with little or no data; predictions there are extrapolation. Mask cells with no nearby observations before plotting.
- `helpers.R:15-18` — `dat[dat == "True"] <- TRUE` coerces the whole frame to character for the comparison and relies on
  `type.convert` to undo it. It works, but `dplyr::mutate(across(where(is.character), ~ .x %in% "True"))` on the known
  boolean columns is clearer and cannot mangle a genuine string column containing `"True"`.
- Style: use `<-` consistently (already done), and put spaces around `=` in argument lists
  (`valid_only=TRUE` → `valid_only = TRUE`) to match the rest of the file.

---

## Remaining work

Every Critical is fixed, and every High except H5 (deferred by decision). All are covered by tests. What is left:

| Item | Why it is still open |
| --- | --- |
| **T1** d' denominator | research decision |
| **T2** what makes a *visit* an outlier | research decision; H2 refuses the request until this is settled |
| **T3** fixation `max_duration` | research decision; the value is now explicit in `config.py` at its previous 2500 ms |
| **H5** trigger pairing | deferred by decision: fall back to `TRIAL_END`; needs raw data to validate |
| **C3 frequency** | needs `SEARCH_ARRAY_PATH` on `S:` to re-parse from raw |
| **M7** cumulative funnel column names | naming/API change; touches the R scripts and every notebook |
| **M10, M11** GAM specification | statistical, in `analysis/R/`; needs your call on the nesting structure |
| **M12** `px2deg` small-angle approximation | matters only at large eccentricity; needs a decision on where to apply the exact form |
| **M14** packaging / `plgrnd2.py` import | `pyproject.toml` now exists; `__init__.py` files and the broken scratchpad import remain |
| **L4** hardcoded strip geometry | wants the stimulus-generation config, which is on `S:` |
| **L5, L6** | documentation and R hygiene |

**Re-run required.** Every stage-1 fix (C2, C3, C4, H1, H6, M15) changes the pickles, and the caches now
invalidate themselves (H3), so the next `run_pipeline()` rebuilds from raw. Until then the built pickles in
`OUTPUT_PATH` are pre-fix, which is why the three real-data checks remain `xfail`. That run needs `S:` mounted for
`SEARCH_ARRAY_PATH`.

## Fix order

0. ~~H8~~ — resolved by the numpy/pandas upgrade. **H9 now blocks stage 1**: fix it first or no subject can be
   re-parsed from raw data.
1. **H3 (cache invalidation) next.** Until per-subject pickles invalidate on code change, you cannot tell whether any
   later fix took effect.
2. **C2, C3, C4** — these change the numbers. C4 spans the parser and the funnel, so land it as one change; expect the
   12 measured mis-timed targets to move between the LWS and target-return categories.
3. **H1, H2, H4, H5, H6.** H2 is the largest measured effect after C4 (10.1% of fixations vs 0% of visits).
4. **L1** — the tests exist; remove each `xfail` marker as its fix lands. `strict=True` means a silent fix fails the
   suite, so the markers cannot rot.
5. **M1–M17**, then **L2–L6**. M15 is a one-line fix with no re-run needed; M16 and H7 both set `peyes` global
   configuration and should land together in one `configure_peyes()`.

Then delete every per-subject pickle and re-run the pipeline once, with the C4 invariant test as the gate.
