# LWSv1

Analysis code for the LWS ("Looking Without Seeing") v1 experiment. See [CLAUDE.md](CLAUDE.md) for architecture,
setup, and commands; [CODE_REVIEW.md](CODE_REVIEW.md) for known issues; [plans.md](plans.md) for open analyses.

## Data quirks (raw data, not the codebase)

- Record/trial triggers are interleaved: `start_trial -> start_record -> end_trial -> end_record` (should be
  record-trial-trial-record).
- There is a ~55 ms offset between BioSemi's `start_trial` trigger and Tobii's first gaze record; it falls in the
  first two rows of each trial's behavioral data.
