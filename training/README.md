# Training records

[Method](../docs/03_METHOD_AND_TRAINING.md) · [Experiment flow](../docs/04_EXPERIMENT_FLOW.md)

Start with [all1061_source_recipe_records_sanitized.csv](all1061_source_recipe_records_sanitized.csv).

Columns include model/family/seed, source protocol, training fraction, training/validation/upstream-label exposure, unlabelled images, batch, completed/selected rounds, updates, trainable parameters and configuration JSON. The latter retains applicable learning rates. Parent/fresh-head work is identified separately.

[427 source families](all427_source_recipe_families_sanitized.csv) and the [schedule summary](source_schedule_summary_sanitized.json) provide broader views.

## Completed original schedules

| Stage | Work |
|---|---|
| Primary neural image | 36 calibration + 180 development records |
| Classical | 70 calibration + 180 development records |
| Dense development | 168 fits |
| UCF matched reference controls | 9 fits |
| CNN optimization stability | 18 fits |
| Representation / scale | 379 records |
| Mechanisms | 125 records |
| Full-method / matched architecture | 69 fits: 60 full architecture + 9 matched DINO |
| Formal efficiency | 97 cases, 388 conditions, 1,164 primary passes |
| ELPV development supplement | 24 outer fits; successful unchanged grids imply 480 inner fits |

These counts have different roles and overlap with the catalogue. Do not sum them as independent models, architectures or total training cost. Unselected upstream HPO trials are not catalogue entries. Teacher/ensemble views share training; deterministic reuse is not a new fit; fresh-head updates exclude the parent stage.

Raw pause-contaminated elapsed timers are not active-compute estimates. Original failures, raw histories and optimizer/RNG remain preserved in the owner workspace and historical handoff.
