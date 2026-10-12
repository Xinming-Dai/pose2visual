"""Named preprocessing recipes for the Lightning Pose encoding pipeline (step A-C).

A "step order" names the full recipe applied before encoding, recorded as an ordered list of
step names in ``litpose_step_pipelines.json`` sitting next to this module. Every recipe has the
same fixed shape::

    [{"ensemble_median" | "eks_mean"}, "confidence_filter", "linear_interpolate",
     ("2d_pose_pca",)? "keypoint_speed", "encoding"]

The first step picks how multiple Lightning Pose seed models' predictions for one session are
combined into a single (x, y, likelihood) trajectory per keypoint. Step B
(``extract_litpose_dynamic_vars_trials``) then masks and interpolates the low-likelihood frames,
optionally reprojects the x/y of each body-part group onto its top principal components
(``2d_pose_pca``, see ``litpose_pca``; the speed is taken in the original pixel space, so the
reprojection comes before ``keypoint_speed``), and takes the speed. The combined csv itself is
produced by a separate, already-existing tool
(``skills/pose-video/plot_comparison.py --save-ensemble-median`` for ``ensemble_median``,
``pose2visual.lightningpose.export_eks_csv --standard-naming`` for ``eks_mean``) and consumed by
pointing ``video_preds_dir`` at its output folder, exactly as for the plain single-seed case —
this module does not resolve or build that csv itself. ``2d_pose_pca`` (optional) decides whether the
config applies PCA (there is no separate ``pca.enabled`` yaml key) and ``encoding`` maps onto the
RRR/CNN fit of ``run_litpose_encoding``.

``resolve_step_pipeline`` is this module's only real entry point: it looks up a name and derives
the two flags that actually drive behavior (``combine_method``, for provenance/logging, and
``use_pca``, which ``litpose_keypoints.load_encoding_yaml`` turns into the config's ``pca`` flag)
rather than interpreting the list as a generic, arbitrarily-ordered step sequence — there is nothing else
for a step to vary within this fixed shape.

Adding a new named pipeline:

1. Add a ``name -> [steps...]`` entry to ``litpose_step_pipelines.json``. The list must still
   match the fixed shape above: ``steps[0]`` in ``{'ensemble_median', 'eks_mean'}``, then
   ``'confidence_filter'``, ``'linear_interpolate'``, then either ``'keypoint_speed',
   'encoding'`` or ``'2d_pose_pca', 'keypoint_speed', 'encoding'``.
2. If it reuses an existing combine method, no code change is needed here —
   ``resolve_step_pipeline`` derives ``combine_method``/``use_pca`` generically from the list.
3. If it needs a genuinely new way to combine keypoints, add that combine method's name to
   ``_COMBINE_METHODS`` below, and write the new combination tool under
   ``src/pose2visual/lightningpose/`` (next to ``ensemble_median.py`` / ``fit_eks_latent_model.py``
   / ``export_eks_csv.py``), producing a ``video_preds_dir``-compatible standard-named csv.
4. Point a ``litpose_encoding.yaml``'s ``step_order:`` at the new name; PCA then
   follows from whether its steps include ``2d_pose_pca``.
5. Add a case to ``tests/encoding/test_litpose_step_pipelines.py``.
"""

import json
from pathlib import Path
from typing import Any

_PIPELINES_JSON = Path(__file__).with_name('litpose_step_pipelines.json')
_COMBINE_METHODS = ('ensemble_median', 'eks_mean')
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
            `[{ensemble_median|eks_mean}, confidence_filter, linear_interpolate,
            (2d_pose_pca,)? keypoint_speed, encoding]`.
    """
    expected_with_pca = ('ensemble_median', *_MIDDLE_STEPS, _PCA_STEP, _SPEED_STEP, _LAST_STEP)
    expected_without_pca = ('ensemble_median', *_MIDDLE_STEPS, _SPEED_STEP, _LAST_STEP)
    valid = (
        len(steps) >= 1 and steps[0] in _COMBINE_METHODS
        and tuple(steps[1:]) in (
            (*_MIDDLE_STEPS, _PCA_STEP, _SPEED_STEP, _LAST_STEP),
            (*_MIDDLE_STEPS, _SPEED_STEP, _LAST_STEP),
        )
    )
    if not valid:
        raise ValueError(
            f'step_order {name!r} has steps {steps}, which does not match the fixed shape '
            f'[{{{"|".join(_COMBINE_METHODS)}}}, {", ".join(_MIDDLE_STEPS)}, '
            f'({_PCA_STEP},)? {_SPEED_STEP}, {_LAST_STEP}] (e.g. {list(expected_with_pca)} or '
            f'{list(expected_without_pca)})',
        )


def resolve_step_pipeline(
    name: str,
    path: str | Path = _PIPELINES_JSON,
) -> dict[str, Any]:
    """Look up a named step-order pipeline and derive its two operative flags.

    Args:
        name: pipeline name, a key of the json loaded from `path`.
        path: path to the step-order pipelines json; defaults to `litpose_step_pipelines.json`
            next to this module.

    Returns:
        Dict with `name`, `steps` (the ordered list as stored), `combine_method` (`steps[0]`)
        and `use_pca` (whether `2d_pose_pca` is in `steps`).

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
        'combine_method': steps[0],
        'use_pca': _PCA_STEP in steps,
    }
