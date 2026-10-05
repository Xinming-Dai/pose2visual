"""Neural encoding of IBL dynamic variables: predict spike rates from behavior traces.

Three-step pipeline, each step runnable as ``python -m pose2visual.encoding.<module>``:

Modules:
    extract_neural_data: step 0 (cpu, network). Pull spikes and behaviors for one IBL session
        via ONE and write ``<eid>_aligned.npz``. Uses ``ibl_data_utils``.
    extract_dynamic_vars_trials: step 1 (cpu). Repackage the ``DYNAMIC_VARS`` behavior traces
        stored in ``<eid>_aligned.npz`` into ``dynamic_vars_z_trials.npz``.
    run_encoding: step 2 (gpu). Fit RRR and CNN encoders on that file with Ray Tune and save
        ``encoding_results_dynamic_vars.npy``.
    ibl_data_utils: ONE/brainbox helpers (spike and behavior binning, alignment) for step 0.
    constants: ``DYNAMIC_VARS``, the behavior traces shared by all steps.
    encoder: ``train_{rrr,cnn}_encoder[_with_tune]`` used by ``run_encoding``.
    rrr_encoder: ``RRRGD`` reduced-rank regression model and its LBFGS training loop.
    utils: CLI parsing, seeding, logging setup and encoding metrics (co-bps).

Slurm wrappers live in ``scripts/delta/encoding/``.
"""
