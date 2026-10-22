"""Optional follow-up experiments on the neural-behavior cross-correlation.

Both start from a finished baseline run (``pose2visual.xcorr.run_xcorr``), only read its files and
write to their own folder inside the session folder, so the baseline figures and arrays are never
changed. They ask whether the similar xcorr patterns of ear, eye, nose and mouth keypoints come
from shared motion:

* ``rank1`` (Experiment A, ``experiment_A_rank1/``): subtract the rank-1 approximation of the
  ``(neurons * lags, behaviors)`` xcorr matrix of each camera; heatmaps of the original, the
  rank-1 component and the SVD residual, the explained fraction and the loadings.
* ``relative_motion`` (Experiment B, ``experiment_B_relative_motion/``): subtract the centroid of
  each anatomical group with two or more keypoints from the keypoint coordinates, recompute the
  speed and the xcorrs, and compare with the original speed. Groups with a single keypoint
  keep their original speed and are listed in the report.
* ``masked_crosscorr``: the baseline's Pearson cross-correlation for traces with missing samples
  (used by ``relative_motion``).
* ``common`` and ``diagnostics``: baseline loaders and the diagnostic figures.

Each experiment has its own command line (``python -m pose2visual.xcorr.experiments.rank1`` or
``.relative_motion``); see ``scripts/delta/lightningpose_xcorr/step3a_experiment_rank1.sh`` and
``step3b_experiment_relative_motion.sh``. Importing the package is cheap.
"""
