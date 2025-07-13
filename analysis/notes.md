**TODO - Figures:**
- number of on-target fixations till identifications, split by category & trial type (with and without bad-action trials) - histogram
- number of on-target visits till identifications - histogram (with and without bad-action trials; split by category and overall)
- number of all fixations till first identification - line plot (with and without bad-action trials; split by category and search-array type)
- % identified after bottom-strip visits (include non-identified) - cumulative histogram for -4, -3, ... fixations back from identification
- exploration/exploitation - fixation duration and saccade sizes over trial time - line plot (also by category and search-array type)

**TODO - Analysis Thresholds:**
- Determine bad subjects: num bad-action trials, miss rate, FA rate, distance on hits
- Determine on-target distance: show distance on hits for all (valid) subjects


### FOR NEXT VERSION:
- trigger order: `start recording` -> `trial start` -> `targets on` -> `targets off` -> `stimulus on` -> `stimulus off` -> `stop_recording` -> `trial end`
meaning the `recording` does not properly flank the trial.
- do not require target-marking confirmation
- change trial categories and durations randomly, not sequentially
- do not stop the time when the target is identified, but continue until the end of the trial
- alternatively, continue to next trial upon first identification