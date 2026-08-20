# Notes for the next experiment version

Design changes for a future data-collection round. Not about this codebase or its analysis; moved out of
`plans.md` to keep that file focused on the current dataset's remaining analyses.

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
