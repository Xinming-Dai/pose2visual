"""Named preprocessing recipes for the Lightning Pose encoding pipeline (step A-C).

A "step order" names the full recipe applied before encoding, recorded as an ordered list of
step names in ``litpose_step_pipelines.json`` sitting next to this module. Every recipe has the
same fixed shape::

    ["confidence_filter", "linear_interpolate", ("2d_pose_pca",)? "keypoint_speed", "encoding"]

A recipe does not say where the keypoint trajectories come from. ``video_preds_dir`` of the
encoding yaml points at a folder of prediction csvs, exactly as for the plain single-seed case:
the csvs may be the output of ``pose2visual.lightningpose.fit_eks_latent_model`` (EKS),
``python -m pose2visual.qc.rainbow --save-ensemble-median`` (ensemble median) or a single model.
Which csv columns are the position (``x, y`` or the ``x_ens_median, y_ens_median`` of an EKS csv)
is the yaml's ``position.columns``, see ``litpose_keypoints.load_position_columns``. Step B
(``extract_litpose_dynamic_vars_trials``) masks and interpolates the low-likelihood frames,
optionally reprojects the x/y of each body-part group onto its top principal components
(``2d_pose_pca``, see ``litpose_pca``; the speed is taken in the original pixel space, so the
reprojection comes before ``keypoint_speed``), and takes the speed. ``2d_pose_pca`` (optional)
decides whether the config applies PCA (there is no separate ``pca.enabled`` yaml key) and
``encoding`` maps onto the RRR/CNN fit of ``run_litpose_encoding``.

``resolve_step_pipeline`` is this module's only real entry point: it looks up a name and derives
the one flag that actually drives behavior (``use_pca``, which
``litpose_keypoints.load_encoding_yaml`` turns into the config's ``pca`` flag) rather than
interpreting the list as a generic, arbitrarily-ordered step sequence — there is nothing else
for a step to vary within this fixed shape.

Adding a new named pipeline:

1. Add a ``name -> [steps...]`` entry to ``litpose_step_pipelines.json``. The list must still
   match the fixed shape above: ``'confidence_filter'``, ``'linear_interpolate'``, then either
   ``'keypoint_speed', 'encoding'`` or ``'2d_pose_pca', 'keypoint_speed', 'encoding'``.
2. Point a ``litpose_encoding.yaml``'s ``step_order:`` at the new name; PCA then
   follows from whether its steps include ``2d_pose_pca``.
3. Add a case to ``tests/encoding/test_litpose_step_pipelines.py``.
"""

import json
from pathlib import Path
from typing import Any

_PIPELINES_JSON = Path(__file__).with_name('litpose_step_pipelines.json')
_MIDDLE_STEPS = ('confidence_filter', 'linear_interpolate')
_PCA_STEP = '2d_pose_pca'
_SPEED_STEP = 'keypoint_speed'
_LAST_STEP = 'encoding'


def load_step_pipelines(path: str | Path = _PIPELINES_JSON) -> dict[str, list[str]]:
    """Load the named step-order pipelines json.

    Args:
        path: path to the json file; defaults to ``litpose_step_pipelines.json`` next to this
            module.

    Returns:
        Mapping of pipeline name to its ordered step list, as stored in the file.
    """
    with Path(path).open(encoding='utf-8') as f:
        return json.load(f)


def _validate_step_list(name: str, steps: list[str]) -> None:
    """Raise unless `steps` matches the fixed pipeline shape.

    Args:
        name: pipeline name, used in the error message.
        steps: ordered step list to validate.

    Raises:
        ValueError: if `steps` does not match
            `[confidence_filter, linear_interpolate, (2d_pose_pca,)? keypoint_speed, encoding]`.
    """
    expected_with_pca = (*_MIDDLE_STEPS, _PCA_STEP, _SPEED_STEP, _LAST_STEP)
    expected_without_pca = (*_MIDDLE_STEPS, _SPEED_STEP, _LAST_STEP)
    if tuple(steps) not in (expected_with_pca, expected_without_pca):
        raise ValueError(
            f'step_order {name!r} has steps {steps}, which does not match the fixed shape '
            f'[{", ".join(_MIDDLE_STEPS)}, ({_PCA_STEP},)? {_SPEED_STEP}, {_LAST_STEP}] '
            f'(e.g. {list(expected_with_pca)} or {list(expected_without_pca)})',
        )


def resolve_step_pipeline(
    name: str,
    path: str | Path = _PIPELINES_JSON,
) -> dict[str, Any]:
    """Look up a named step-order pipeline and derive its operative flag.

    Args:
        name: pipeline name, a key of the json loaded from `path`.
        path: path to the step-order pipelines json; defaults to `litpose_step_pipelines.json`
            next to this module.

    Returns:
        Dict with `name`, `steps` (the ordered list as stored) and `use_pca` (whether
        `2d_pose_pca` is in `steps`).

    Raises:
        ValueError: if `name` is not a key of the loaded pipelines, or its step list does not
            match the fixed shape (see module docstring).
    """
    pipelines = load_step_pipelines(path)
    if name not in pipelines:
        raise ValueError(f'unknown step_order {name!r}; available: {sorted(pipelines)}')
    steps = pipelines[name]
    _validate_step_list(name, steps)
    return {
        'name': name,
        'steps': steps,
        'use_pca': _PCA_STEP in steps,
    }
