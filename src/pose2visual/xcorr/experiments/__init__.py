"""Optional follow-up experiments on the neural-behavior cross-correlation.

Both need only step 0, not a finished Experiment A (baseline, ``pose2visual.xcorr.run_xcorr``):
they recompute the original keypoint-speed cross-correlations with the baseline's own code
(``common.compute_original``), so A, B and C can run at the same time. Each writes to its own
folder inside the session folder; when A has run on the same neurons, its saved neuron order is
used for their figures. They ask whether the similar xcorr patterns of ear, eye, nose and mouth
keypoints come from shared motion:

* ``pca`` (Experiment B, ``experiment_B_pca/``): PCA of the face group (eye, pupil, mouth,
  upperlip, lowerlip, nose, pad, ear; both sides pooled into one fit); the top component
  scores become new behaviors ("PCA on positions") and the face positions reprojected onto those
  same components give a denoised per-keypoint speed ("PCA on speed"). ``left_paw`` / ``right_paw``
  (``wrist``, ``d1``-``d4`` by side) and any keypoint matching neither keep their original speed
  unchanged. The tongue is not part of the fit: it is the binary ``tongue_visible`` feature, not
  keypoint positions.
* ``relative_motion`` (Experiment C, ``experiment_C_relative_motion/``): subtract the centroid of
  each anatomical group with two or more keypoints from the keypoint coordinates, recompute the
  speed and the xcorrs, and compare with the original speed. Groups with a single keypoint
  keep their original speed and are listed in the report.
* ``pca_reconstruction`` (part of Experiment B, ``experiment_B_pca/``): not a xcorr but a look
  at the decomposition: for the whole face (``wholeface`` group, no tongue / forelimb / digits) the
  centered positions, their rank-2 PCA part and the residual are drawn as a three-panel heatmap and
  as a synchronized video (frame with the tracked keypoints | mean + rank-2 | mean + residual) over
  one representative window per camera. Needs no neural data
  (``scripts/delta/lightningpose_xcorr/step3_render_pca_reconstruction.sh``).
* ``masked_crosscorr``: the baseline's Pearson cross-correlation for traces with missing samples
  (used by ``pca`` and ``relative_motion``).
* ``common`` and ``diagnostics``: the original-xcorr computation, neuron order and the
  diagnostic figures.

Each experiment computes and (unless ``--no-plot``) plots in one command line
(``python -m pose2visual.xcorr.experiments.pca`` or ``.relative_motion``); its ``replot_*``
counterpart (``replot_pca`` / ``replot_relative_motion``) redraws the figures of a finished run
from the saved arrays/csv/json with no recomputation, the same split
``pose2visual.xcorr.run_xcorr``/``replot`` uses for the baseline. See
``scripts/delta/lightningpose_xcorr/step1_compute_xcorr.sh`` (computes the baseline and both
experiments) and ``step2_plot_xcorr.sh`` (plots all three). Importing the package is cheap.
"""
