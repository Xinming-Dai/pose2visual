"""Neural-behavior cross-correlation (Figure 20 of ``spike_assignments.pdf``).

Cross-correlates the 60 Hz binned spikes of every unit with the keypoint-derived behavioral
traces of the encoding pipeline over the whole video, ``r(tau) = Corr(neural(t),
behavior(t + tau))``, and writes per-neuron summary statistics, heatmaps and example curves.

Pipeline (see ``scripts/delta/lightningpose_xcorr``):

* ``extract_neural_continuous`` (step 0): spikes of one session binned on the contiguous 1 s
  intervals of the video, the whole-recording counterpart of ``encoding.extract_neural_data``.
* ``run_xcorr`` (step 1): for every camera (``--cameras``, default left and right) behavior
  (``behavior``), smoothing and cross-correlation (``crosscorr``), circular-shift significance
  (``permutation``), neuron order (``ordering``) and one figure with all behaviors
  (``figures``). ``--exclude`` drops keypoints by glob (``'d*'`` = fingers), ``--n-jobs 2`` runs
  the cameras in parallel, and the yaml is frozen at start (``settings.freeze_config``). The
  figures go to a ``plots`` subfolder of the session folder.
* ``replot``: redraws those figures from the saved arrays and csv without recomputing anything
  (``--cameras``, ``--exclude``, ``--vmax``, ``--order``, ``--plots-subdir``); the drawing code
  is shared with ``run_xcorr`` (``plots``).
* ``extract_trial_events`` (step 0b, ONE) and ``peth`` (step 1b): reward and movement-onset times
  of the session and the event-aligned PETH of every neuron (fractional rate change against the
  pre-event baseline), drawn as two extra columns of the overview like the last columns of
  Figure 20.

Settings live in the ``xcorr`` section of ``configs/encoding/litpose_encoding.yaml``
(``settings``); filtering and imputation come from the encoding sections of that yaml. The
cameras are a command-line choice, ``session.camera`` is not read. Step-0 inputs go to
``<video_preds_dir>/<xcorr.output_name>/<eid>/``; each experiment writes its own subfolder there:
``experiment_A_baseline/`` (``run_xcorr`` and ``peth``), ``experiment_B_pca/`` and
``experiment_C_relative_motion/`` (``experiments``).
Importing the package is cheap: heavy dependencies are imported inside the functions using them.
"""
