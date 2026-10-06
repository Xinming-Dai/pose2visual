# pose2visual
## Neural encoding from Lightning Pose keypoints

Compares encoding of neural activity from `base_keypoints` against
`base_keypoints + additional_keypoints` (one camera per model), using the predictions of a
Lightning Pose checkpoint (`<model_dir>/video_preds/_iblrig_<camera>Camera.downsampled.<eid>.csv`).
Requires `extract_neural_data` (step 0) to have produced `<eid>_aligned.npz`.

1. `scripts/delta/lightningpose_encoding/step0_list_keypoints.sh`: once per `video_preds`; writes
   `video_preds/encoding/keypoint_names/<camera>_<eid>.json` (all, usable and mostly-masked keypoints, with the likelihood threshold).
2. `scripts/delta/lightningpose_encoding/step1_extract_dynamic_vars_trials.sh`: one job for all
   `EIDS` and both `CAMERAS` (overridable; the loop runs in python via `--eids` / `--cameras`,
   `--output-name '{camera}_encoding'`); keypoint speeds binned onto the neural intervals,
   `video_preds/<camera>_encoding/<eid>/dynamic_vars.npz` (`OUTPUT_NAME_SUFFIX` changes
   `encoding`). The yaml `output_name` of step 3 must equal that folder, e.g. `left_encoding`.
3. Edit `configs/encoding/litpose_encoding.yaml` (eid, camera, base / additional keypoints,
   optional `variant_name`), then `scripts/delta/lightningpose_encoding/step2_run_encoding.sh`. The base fit is
   cached in `<output_name>/<eid>/base/<hash8>/` (one folder per base setting, e.g. pca on / off) and reused; the variant writes `encoding_result.npz`,
   `encoding_params.json` and `encoding.log` (checkpoint location, base / additional / unused
   keypoints) to `<output_name>/<eid>/<variant_name>/`, where `variant_name` defaults to the
   sorted `additional_keypoints` joined by `_` (set it in the yaml to choose another name). A
   folder that already holds a result with other settings (base / additional keypoints, pca,
   seed) is never overwritten; a hash suffix is added to the folder name instead. To compare
   another additional set against the same base, change `additional_keypoints` and resubmit
   (`RERUN_BASE=1` forces a base refit). The train/val/test split comes from `<eid>_aligned.npz`,
   so it is identical for all variants. Inside this job the keypoint speeds are z-scored and
   rotated onto principal components (`pca`, `pca_n_components_base`,
   `pca_n_components_base_plus_additional` in the yaml; default n_components = number of
   keypoints of each condition), with the basis fit on the pooled
   train/val/test features; changing them refits the base.

Keypoint features are speeds in pixels/second of the downsampled video frames (ONE's are
half-resolution px/s), so absolute values are not comparable with ONE's or across cameras; the
encoders z-score inputs, so R^2 / BPS do not depend on the unit.
