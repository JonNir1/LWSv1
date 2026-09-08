### Exclusion Criteria for Trials:
([V]) trial has bad actions  
([V]) trial has false-alarms  
([V]) trial has gaze-coverage below X%  
([V]) optional: trial has very few fixations (below Xth percentile of fixation count distribution)  

### Exclusion Criteria for Subjects:
TODO:
  - verify those are indeed implemented and introduced into the funnel
  - check how many/which subjects are excluded

([V]) too many trials excluded by abovementioned criteria, or  
([V]) too many trials with no actions  


### Planned Analyses:
**General:**
- calc variability within & between subjects for LWS visits and repeated visits
- power analysis for LWS counts + repeated counts
- check Carmel's data: how many reps required to successfully decode a SEEN target. multiply this number by the number of LWS visits to get the number of LWS visits required to get valid decoding in LWS instances.
- ([V]) effects on LWS probability: trial type, target category, target rotation, time in trial / fatigue, target
  location / eccentricity, Attentional Blink / SSM. See `spatial_effects.ipynb`, `stimulus_features.ipynb`,
  `time_on_task.ipynb`, `ssm_and_ab.ipynb` (P0-P9, SF1-7, TOT1 per the now-deleted
  `RESULTS_COMPARISON_2026-04_vs_2026-08.md`, commit `92fb11e4`).

**Trial Type / Target Category:**
- ([V]) differences in hit-rate/d': `hit_rate.ipynb` (HR1-HR3 per the now-deleted
  `RESULTS_COMPARISON_2026-04_vs_2026-08.md`, commit `92fb11e4`)
- ([V]) differences in LWS-visit count/proportion: same notebooks as the "effects on LWS probability" item above
- differences in repeated-visit count/proportion: not found in any notebook, unclear if this was run. Flagging
  rather than marking done or removing.

**LWS vs Repeated Visits:** (within subject)
- visit duration / fixation count
- visit spread (dispersion)
- pupil size in LWS/identification/repeated visits (partially touched in `ssm_and_ab.ipynb`)
- saccade into a LWS/TR/identification visit
- *(`gaze_behavior.ipynb` exists as a title-only stub for this section, not yet implemented)*

**Search Strategies:**
- scan path analysis (`visualizer/scanpath.py` can render one, but no analysis notebook uses it yet)
- number of scanned icons per trial
- exploration/exploitation - fixation duration and saccade sizes over trial time (plot using line plot (also by category and search-array type))

Notes for redesigning the *next* experiment (not this codebase) moved to `NEXT_EXPERIMENT_NOTES.md`.
