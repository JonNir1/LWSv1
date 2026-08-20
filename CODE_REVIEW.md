# LWSv1 Code Review

Register of known bugs, latent assumptions, and open analysis questions for the preprocessing pipeline
(`pipeline/`, `data_models/`), the analysis layer (`analysis/`), and the R GAM scripts (`analysis/R/`). Full
write-ups are kept only for items that are still open; everything resolved is a one-line row in the changelog
tables below (commit hashes are the pointer into history for the reasoning and fix). Add new findings here rather
than to `CLAUDE.md`.

Severity:

| Level | Meaning |
| --- | --- |
| **Critical** | Produces wrong numbers in the headline result, or silently drops data. Fix before running any analysis. |
| **High** | Wrong or misleading results under plausible data conditions; or makes results irreproducible. |
| **Medium** | Latent bug, fragile assumption, or a real statistical concern that changes interpretation. |
| **Low** | Style, dead code, tooling, documentation. |

Verification: `tests/` encodes findings as executable claims. Reproduce with `pytest tests/ -q` from repo root
(venv at the main checkout root). Current: **185 passed**, 14 files, 0 failures/xfail (checked 2026-08-19).

---

## Resolved design decisions

Settled 2026-08-05. These are the intended semantics; C2-C4 and H6 below implement them.

1. **Every target has an identification time**: a positive number (hit) or `inf` (miss). A *missing* time must
   never occur, `event_classification.py` asserts completeness rather than falling back silently. `inf` correctly
   makes every on-target event on a never-identified target pass `before_identification`.
2. **A false alarm is not an identification of any target.** Its `target` is null, not the nearest target.
3. **Identification time is the first hit** for that target; `repeated_hit` does not move it.
4. **Only the dominant eye is used throughout the pipeline.** The non-dominant-eye fallback was removed.

## Open items

- **T2. What makes a *visit* an outlier? (reframed, 2026-08-19)** The mechanism described when T2 was first
  raised (`build_event_classification_funnel` refusing `event_type="visit"` with an `exclude` argument) no longer
  exists: the `exclude` parameter was removed entirely in the later L8 refactor. The architecture has also
  changed since: `visits` are no longer a persisted, pre-built table filtered separately at read time. `load_data()`
  (`analysis/helpers/read_data.py:86-95`) now drops `is_outlier` rows from `eye_movements` *before* deriving
  `fixations`, and `build_visits()` only ever sees that already-filtered `fixations` table. So a visit can no
  longer be constructed from a fixation that was flagged an outlier, the literal original question ("does one
  outlier fixation contaminate the whole visit") is resolved by construction. What is still genuinely open, and is
  narrower than the original question: whether a *visit itself* should have its own outlier criteria as an
  aggregate object (e.g. unusual duration or dispersion across its own, already-clean fixations), independent of
  its constituent fixations. No such visit-level outlier detection exists today, and whether it is needed has not
  been decided.
- **M10, M11, M12 (statistical, GAMs).** Deferred by decision, 2026-08-06: accepted as valid concerns, but the
  fix is a modelling choice for whoever revisits the GAMs, not a code bug. Each script carries an inline warning at
  the fit site.
  - **M10** (`time_on_task_gam.R`, `time_in_trial_gam.R`, `spatial_gam.R`): every model is
    `is_lws ~ trial_category + s(...) + s(subject, bs="re")`, but the unit of observation is a *visit*, and visits
    are nested within target within trial within subject. A subject-level random intercept does not absorb that
    nesting, so p-values on the smooth terms are optimistic. Fix: add nested grouping (e.g.
    `s(trial_uid, bs="re")`) or aggregate to one observation per (subject, trial, target).
  - **M11** (`spatial_gam.R:49`): `anova(simple_model, interaction_model, test="Chisq")` compares two REML fits
    with different smooth structure, which `mgcv` itself warns is not valid. Fix: refit both under `method="ML"`
    for the comparison, keep REML for the final reported fit.
  - **M12** (`data_models/Subject.py`): `px2deg` is a constant (angle at screen centre), but the true px->deg
    mapping depends on eccentricity. Measured against 1,582 targets: median overestimate 3.0%, p90 6.9%, worst
    9.7%, 0% exceed 10%. Small for on-target classification (radius shrinks 67.8px -> ~62px, still small against
    ~107px icon spacing) but the bias is a smooth centre->periphery gradient, exactly the shape `spatial_effects`
    models, so it is a candidate confound rather than noise. Fix (optional): compute the true subtended angle from
    both screen positions, or add target eccentricity as a covariate in the spatial model.
- **M18. Fixation-level and visit-level LWS/target-return classification can disagree** for fixations inside the
  same visit. Flagged 2026-08-12, not yet measured or fixed. `build_event_classification_funnel` computes
  `is_lws`/`is_target_return` independently per `event_type`; both apply the same
  `before_identification`/`after_identification` predicates but each row supplies its own `start_time`/`end_time`.
  A visit spanning `ident_time` (classified `identification`) necessarily has a first fixation that *ends before*
  `ident_time`, so that fixation can independently classify `is_lws = True` at the fixation level. Neither grain is
  wrong on its own; nothing currently documents or flags the disagreement. Options (research decision, not
  obviously a bug): (1) treat visit-level as canonical and broadcast to member fixations, (2) keep both independent
  but add a diagnostic count of disagreeing fixations, (3) never join `is_lws`/`is_target_return` across grains.
  See `CLAUDE.md`'s stage-3 section for the current caveat.
- **L6's `k` choice.** `spatial_gam.R`'s `te(x, y, k = c(8, 8))` (`k` is per marginal basis, so ~64 basis
  functions) is otherwise fixed and hygienic (see changelog), but whether 8 is the right value is a modelling
  choice deferred alongside M10/M11.

## Changelog (all FIXED / WITHDRAWN / DISSOLVED findings)

| ID | Title | Commit(s) |
| --- | --- | --- |
| T1 | d' denominator: FVF-conditioned prototype in `analysis/fvf/fvf_based_sdt.ipynb`; effect negligible (array coverage already ~95-98%), `sdt.py` kept as-is | n/a |
| T3 | Longest plausible fixation: measured over 116,947 fixations, smooth monotonic decay, no secondary mode; 2500 ms default kept | n/a |
| T4 | `fixations_to_targets()` long-format refactor: `pipeline/stage2_align/fixations_to_icons.py` replaces wide per-target columns; visits and identifications moved to stage 2; FVF estimators updated | `8d2ecae` (regression introduced), resolved in stage-2/3 refactor |
| T5 | Three gaps reported upstream to `peyes` (missing `start_pixel`/`end_pixel` in `summary()`, unimplemented velocity/dispersion outlier checks, `summarize_events([])` shape); workarounds stay until upstream ships fixes | n/a |
| C2 | `MARK_ONLY` written to wrong column, never recorded | `ef00fce` |
| C3 | `ATTEMPTED_MARK` branch raised `KeyError` / clobbered a pending mark | `ef00fce` |
| C4 | A false alarm could supply the identification time of a real target | `d5eedcf` |
| H1 | `num_fixs_to_strip` computed across both eyes concatenated instead of per-eye | `f3278c5` |
| H1a | The per-eye fix for H1 silently corrupted the second eye via `groupby.transform` index misalignment | `3452cd5` |
| H2 | `drop_outliers` was a silent no-op for visit funnels (0 of 5,720 visits dropped vs. 10.1% of fixations); fixed by refusing visit-level outlier exclusion until T2 was decided, then superseded when the stage-2/3 refactor made `visits` build only from already outlier-filtered `fixations`, so the raise itself was later removed as dead code alongside the `exclude` parameter (see T2) | `82296fc`, later folded into the L8 refactor |
| H3 | Per-subject pickle caches were silent and unversioned | `3d228cc` (cache-key sidecar; see M19 for a later regression) |
| H4 | `parse_all_subjects` swallowed all exceptions; dirname split sat outside the `try` | `e18864a` |
| H5 | `_is_between_triggers` assumed start/end triggers equal in count and positionally paired; rewritten to a scan-in-order loop with `close_trailing` | `863fbf0` (final rewrite; several earlier attempts and reverts precede it, see commits around `1460339`/`1ad34b2`) |
| H6 | Target distances silently fell back to the non-dominant eye | `05cccd7` |
| H7 | Fixations > 2500 ms dropped by an inherited `peyes` default, never made explicit in `config.py`; measured impact negligible (6 of 116,947 fixations), downgraded from High to Low | `8201293` (made explicit) |
| H8 | `peyes` pins `numpy~=1.2`, pickles were written under numpy 2.x; resolved by upgrading numpy/pandas above the pin (verified compatible) | n/a |
| H9 | pandas 3.0 broke trigger/gaze alignment (`AttributeError: '_hasna'`) via a mixed-dtype `.loc` assignment | `9c8a748` |
| M1 | `metadata` columns were all `object` dtype | `9556003` |
| M2 | Visit `is_on_target` collapsed to "min distance <= threshold" because `.any(axis=1)` globbed all `*_distance_dva` columns | `392f8d9` |
| M3 | `SearchArray._get_path` disagreed with the actual directory layout the loader reads | `392f8d9` |
| M4 | Falsy-zero bugs: `start_identify_idx` label `0` treated as absent | `ef00fce` |
| M5 | `pd.Categorical.from_codes` relied on enum values matching list positions | `392f8d9` |
| M6 | `reindex(...).astype(bool)` turned a missing criterion into `True` | `9556003` |
| M7 | Cumulative funnel columns kept the raw criterion name, inviting misreading | `8489119` |
| M8 | `detect_eye_movements` default `pixel_size_cm` was actually a millimetre value | `8201293` |
| M9 | `visits.py` error paths raised `AttributeError` (`.iloc` on a numpy array) instead of the intended message | `9556003` |
| M13 | Configuration duplicated across `config.py` / `funnel_config.py`, partly stale, hardcoded machine paths; consolidated into `pipeline/config.py` as single source of truth | `60a7d49` |
| M14 | "No package structure" withdrawn as not applicable (single-project analysis pipeline, run-from-repo-root is a deliberate contract); `plgrnd2.py`'s broken import fixed | `b092c10` |
| M15 | `peyes.create_events` received `pixel_size=viewer_distance_cm` (off by ~2000x); verified inert for every surviving output column, no re-run required | `8201293` |
| M16 | Outlier detection used `peyes`'s default screen geometry instead of the project's monitor (`cnst.TOBII_MONITOR`) | `8201293` |
| M17 | `del ... start_idx` raised `UnboundLocalError` when the trigger log had no `BLOCK_*` trigger | `863fbf0` |
| M19 | `_STAGE1_SOURCES` in `pipeline/stage1_parse/cache_key.py` referenced three deleted `data_models/preprocess/` files (recurrence of H3) and `_REPO_ROOT` resolved one directory too shallow, so the stage-1 cache key tracked almost no real code changes | `162ffd6` |
| L1 | No tests: now 14 files, 185 tests, all green | n/a |
| L2 | No linter/formatter/type-checker config: withdrawn, this review's own findings (wrong column name, falsy-zero guards, key collisions) are not the class of bug a linter catches | `b092c10` (reverts the ruff config added and reconsidered in `773ebe8`) |
| L3 | Resource-handling and small correctness nits (missing context manager, missing `__hash__`, unused variable, dead code, always-off `tqdm` bars) | `773ebe8`, `35bc6e6` |
| L4 | Hardcoded stimulus geometry, now validated against `ArrayInfo.mat` by `test_search_array.py::TestGeometryAgainstStimulusConfig` | n/a |
| L5 | Fixation- and visit-level analyses attribute targets differently (fixation picks closest target, visit keeps all within-threshold); partially dissolved by the stage-3 refactor, now a documented design choice rather than an artefact; residual overlap 0.2% (24/11,716 on-target fixations) | n/a |
| L7 | `_determine_time_to_trial_end.ipynb` indexed a column (`before_identification`) that did not exist on the raw `visits` table | n/a |
| L8 | Trial validity was coupled into event funnels via the cumulative chain, making `is_lws`/`is_target_return` silently depend on `is_valid_trial`; event funnels now contain only event-level criteria, consumers filter explicitly via `data.trial_funnel["is_valid_trial"]` | `9905ecf`, `0418467` |

---

## Re-run and fix-order notes

Pipeline was re-run with `force_reparse=True` on all 27 subjects (2026-08-12), reflecting every stage-1 fix above
(C2, C3, C4, H1, H1a, H6, M15). Regression comparison against the pre-refactor backup confirmed all differences
trace to intentional fixes or the corrected TOBII dimensions (527 x 296 mm). Two additional bugs found and fixed
during that run: `_map_ident_time` crashed on off-target fixations (NaN target), and `build_identifications` missed
82 trials that had targets but zero actions (no miss rows emitted).

Every Critical and High finding is fixed and covered by tests. What remains open is exactly the "Open items" list
above: T2 (visit-outlier definition), M10/M11/M12 (GAM specification, deferred by decision), M18 (fixation/visit
classification disagreement, a research decision), and the `k` choice inside L6's otherwise-fixed R hygiene.
