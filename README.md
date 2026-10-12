# pose2visual
## Neural encoding from Lightning Pose keypoints

Compares encoding of neural activity from `base_keypoints` against
`base_keypoints + additional_keypoints` (one camera per model), using the predictions of a
Lightning Pose checkpoint (`<model_dir>/video_preds/_iblrig_<camera>Camera.downsampled.<eid>.csv`).
Requires `extract_neural_data` (step 0) to have produced `<eid>_aligned.npz`.

1. `scripts/delta/lightningpose_encoding/step0_list_keypoints.sh`: once per `video_preds`; writes
   `video_preds/encoding/keypoint_names/<camera>_<eid>.json` (all, usable and mostly-masked keypoints, with the likelihood threshold).
2. `scripts/delta/lightningpose_encoding/step1_extract_dynamic_vars_trials.sh`: one job for all
   sessions of the yaml's `session.eid_group` (a key of `neural_sessions` in `paths.eid_groups_json`; `EIDS` or
   `EID_GROUP` in the environment override it) and both `CAMERAS` (overridable; the loop runs in python via `--eids` / `--cameras`,
   `--output-name` defaulting to the yaml's `output.output_name`); keypoint speeds (after the optional PCA reprojection) binned onto the neural intervals,
   `video_preds/<camera>_<output_name>/<eid>/dynamic_vars.npz` (the camera is always prefixed; set `OUTPUT_NAME` in the
   environment to override the yaml). Steps 2 and 3 read the same camera-free `output.output_name` and use
   `session.camera` for the prefix, so `output_name: encoding` and `camera: left` give `left_encoding`.
3. Edit `configs/encoding/litpose_encoding.yaml`, which is nested into sections: `session` (`eid`, `eid_group`, `camera`),
   `paths`, `output` (`output_name`, optional `variant_name`), `filter`, `pca` and `anatomical_group`
   (`name`, `base_keypoints`, `additional_keypoints`). `filter.group_set.unfiltered_groups` lists body-part groups (from the group set `filter.group_set.name` of `paths.body_parts_json`, default `nonoverlapping_analysis_groups`) that skip confidence filtering: their likelihood is ignored, only NaN x/y frames are interpolated and `max_masked_frac` counts NaN frames only; step 0 stores them as `no_filter_keypoints` and steps 1 and 2 fail if the yaml no longer matches, so rerun step 0 after changing it. `session.eid` is the single session of one step 2 job
   (`step2_run_encoding.sh <eid>` overrides it), `session.eid_group` the sessions of step 1 and of the plot. With `anatomical_group.name: all` the two lists hold keypoint
   names, with a group set of `body_parts.json` such as `nonoverlapping_analysis_groups` they hold group names, e.g.
   `base_keypoints: [eye_related]` and `additional_keypoints: [mouth_related]`, resolved per session to the group's usable keypoints), then `scripts/delta/lightningpose_encoding/step2_run_encoding.sh`. The base fit is
   cached in `<camera>_<output_name>/<eid>/base/<hash8>/` (one folder per base setting, e.g. pca on / off) and reused; the variant writes `encoding_result.npz`,
   `encoding_params.json` and `encoding.log` (checkpoint location, base / additional / unused
   keypoints) to `<camera>_<output_name>/<eid>/<variant_name>/`, where `output.variant_name` defaults to the
   sorted `additional_keypoints` joined by `_` (set it in the yaml to choose another name). A
   folder that already holds a result with other settings (base / additional keypoints, pca,
   seed) is never overwritten; a hash suffix is added to the folder name instead. To compare
   another additional set against the same base, change `additional_keypoints` and resubmit
   (`RERUN_BASE=1` forces a base refit). The train/val/test split comes from `<eid>_aligned.npz`,
   so it is identical for all variants. Submit step 2 once per session; at the end of every job
   `plot_encoding_base_vs_additional.py --only-if-complete` checks the sessions of `session.eid_group`, and the job
   that finds all of them finished writes `<camera>_<output_name>/plots/.../encoding_barplot.png` (the other jobs log which
   sessions are missing; `SKIP_PLOT=1` turns this off). PCA is applied in step 1, not here: when the `step_order` pipeline contains `2d_pose_pca` (there is no
   separate `pca.enabled` key, so the two cannot disagree; step 1 fails early on a missing or unknown `step_order`), each body-part group (`pca.group_set` of `paths.body_parts_json`, default
   `nonoverlapping_analysis_groups`) has the interpolated x/y of its usable keypoints projected onto
   its top `PCA_n_components` principal components (`PCA_n_components` in
   `scripts/delta/data/body_parts.json`, clamped to the group's x/y feature count) and mapped back
   to pixels before the speed is taken, so the variables keep their keypoint names. Step 2 checks
   that the PCA flag implied by `step_order` equals the flag stored in `dynamic_vars.npz`; use another `OUTPUT_NAME` /
   `output.output_name` for a pca run, which also refits the base.

Keypoint features are speeds in pixels/second of the downsampled video frames (ONE's are
half-resolution px/s), so absolute values are not comparable with ONE's or across cameras; the
encoders z-score inputs, so R^2 / BPS do not depend on the unit.
