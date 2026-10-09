"""Append-only registry of QC/EKS run manifests, keyed by run folder name.

Scripts that write timestamped run folders (e.g. `plot_comparison.py`,
`fit_eks_latent_model.py`) call `append_entry` after writing their per-run manifest so the
run is also recorded in a single `registry.json` sibling to the run folders. This makes
runs discoverable without opening every folder, and gives `prune_run.py` a place to look up
a run's directory before deleting it.
"""

import json
import shutil
from pathlib import Path
from typing import Any


def append_entry(registry_path: Path, run_name: str, entry: dict[str, Any]) -> None:
    """Add or overwrite one run's entry in a registry.json file.

    Args:
        registry_path: path to the registry.json file (created if missing).
        run_name: run folder name, used as the entry's key.
        entry: JSON-serializable manifest data for this run.
    """
    registry: dict[str, Any] = {}
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())

    registry[run_name] = entry

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + '\n')


def prune_entry(registry_path: Path, run_dir: Path, run_name: str) -> None:
    """Delete a run's output directory and remove its entry from the registry.

    Args:
        registry_path: path to the registry.json file.
        run_dir: run folder to delete.
        run_name: run folder name, used as the entry's key in the registry.

    Raises:
        KeyError: if `run_name` is not present in the registry.
    """
    registry: dict[str, Any] = {}
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())

    if run_name not in registry:
        raise KeyError(f'run {run_name!r} not found in registry {registry_path}')

    if run_dir.exists():
        shutil.rmtree(run_dir)

    del registry[run_name]
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + '\n')
