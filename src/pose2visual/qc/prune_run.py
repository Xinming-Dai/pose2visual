#!/usr/bin/env python3
"""Delete a run's output folder and its entry in registry.json.

Works against either `<model_dir>/rainbow_qc/registry.json` (pose2visual.qc.rainbow runs) or
`<model_dir>/eks/registry.json` (`pose2visual.lightningpose.fit_eks_latent_model` runs); the
registry format is the same for both.

Usage:
    python -m pose2visual.qc.prune_run --registry /path/to/rainbow_qc/registry.json \\
        --run-name 20261009_143201__my-qc 20261009_150322__other-qc
"""

import argparse
import json
import sys
from pathlib import Path

from pose2visual.run_registry import prune_entry


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--registry',
        required=True,
        help='Path to the registry.json file',
    )
    parser.add_argument(
        '--run-name',
        required=True,
        nargs='+',
        help='Run folder name(s) (registry key) to delete',
    )
    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    registry_path = Path(args.registry).resolve()

    if not registry_path.exists():
        print(f'Error: registry not found: {registry_path}', file=sys.stderr)
        sys.exit(1)

    registry = json.loads(registry_path.read_text())

    had_error = False
    for run_name in args.run_name:
        if run_name not in registry:
            print(
                f'Error: run {run_name!r} not found in {registry_path}',
                file=sys.stderr,
            )
            had_error = True
            continue

        run_dir = registry_path.parent / run_name
        if not run_dir.exists():
            print(f'Warning: run directory already gone: {run_dir}', file=sys.stderr)

        prune_entry(registry_path, run_dir, run_name)
        print(f'Pruned {run_name!r}: removed {run_dir} and its registry entry')

    if had_error:
        sys.exit(1)


if __name__ == '__main__':
    main()
