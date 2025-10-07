### Determining Outliers (exclusion criteria):
**Bad Trial if:**
- trial has 0 actions
- trial has 1 or more BAD actions (subject hit the wrong key)
- gaze data below X% of the trial samples
- trial has very few fixations (below Xth percentile of fixation count distribution)

**For Specific Trials:** (within subject)
- gaze data below X% of the trial time
- late-to-action trials
- bad ET recordings: too many missing samples, too many blinks low fixation count/rate
- LWS-visit count/rate: extremely high/low

**For Subjects:**
- ([V]) bad-action count: too high / too frequent
- ([]) low coverage of gaze-data: too low (no gaze data --> no data to analyze)
- ([]) target detection: hit-rate/d' too low
- ([]) LWS-visit count/rate: too low (no LWS visits --> no data to analyze)
- ([]) repeated-visit count/rate: too low (no repeated visits --> no data to analyze) - maybe not needed?


### Planned Analyses:
**General:**
- calc variability within & between subjects for LWS visits and repeated visits
- power analysis for LWS counts + repeated counts
- check Carmel's data: how many reps required to successfully decode a SEEN target. multiply this number by the number of LWS visits to get the number of LWS visits required to get valid decoding in LWS instances.

**Trial Category / Target Category:**
- differences in hit-rate/d'
- differences in LWS-visit count/rate
- differences in repeated-visit count/rate

**LWS vs Repeated Visits:** (within subject)
- visit duration / fixation count
- visit spread (dispersion)
- pupil size in LWS/identification/repeated visits

**Search Strategies:**
- scan path analysis
- number of scanned icons per trial
- exploration/exploitation - fixation duration and saccade sizes over trial time (plot using line plot (also by category and search-array type))


### FOR NEXT VERSION:
**General:**
- trigger order: `start recording` -> `trial start` -> `targets on` -> `targets off` -> `stimulus on` -> `stimulus off` -> `stop_recording` -> `trial end`
meaning the `recording` does not properly flank the trial.
- change trial categories and durations randomly, not sequentially

**Target Marking:**
- DO NOT require confirmation for target marking
- DO allow rejecting a marked target
- Use keys in keys from both hands for marking and rejecting targets, to avoid subject looking at keyboard.
- Do not stop the clock when marking a target, but continue until the end of the trial.
- After marking a target, instruct subjects to visit the target-exemplar section and back to the identified target, to verify that there are "return fixations".

**Stimulus:**
- Set up the target-exemplars in a square at the center of the screen, with the icons at the 4 quarters of the screen.
- Allow for targets to be in the same quarter.
- No need to have same number of distractors in each quarter.
- No need to have repeated targets, we can have only one or no target per exemplar in trial.
- Use same icons as targets and distractors, but in different trials.
- Verify that icons are not too similar to each other, to avoid subject misidentifying targets:
    - explanation: we don't want subjects to identify a distractor-clock as the target-clock in clock trials. same for faces, etc.
    - **solution:** run a short SVP experiment to make sure targets are 100% identifiable.
