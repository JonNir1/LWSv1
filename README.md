# LWSv1

## TODOs & Comments
* trigger for `start_record`, `end_record`, `start_trial`, `end_trial` are intermixed: `start_trial` --> `start_record` --> `end_trial` --> `end_record` (should be record-trial-trial-record).
* there is ~55ms difference between BioSemi's `start_trial` trigger and Tobii's first gaze record. Its the first 2 lines of each trial's behavioral data and should be removed.
* check if Trial's targets are identified multiple times.


## Data Models
- `Subject` contains a list of `Trial` objects.
- Each `Trial` contains gaze+trigger data (`pd.DataFrame`) and a `SearchArray` object.
