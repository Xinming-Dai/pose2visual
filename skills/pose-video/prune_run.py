#!/usr/bin/env python3
"""Delete a run's output folder and its entry in registry.json.

Works against either `<model_dir>/rainbow_qc/registry.json` (plot_comparison.py runs) or
`<model_dir>/eks/registry.json` (fit_eks_latent_model.py runs) — the registry format is the
same for both.

Usage:
    python prune_run.py --registry /path/to/rainbow_qc/registry.json \\
        --run-name 20261009_143201__my-qc
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add src to path for development/skill usage (project-agnostic import)
_pose2visual_src = Path(__file__).parent.parent.parent / 'src'
if _pose2visual_src.exists() and str(_pose2visual_src) not in sys.path:
    sys.path.insert(0, str(_pose2visual_src))

from pose2visual.run_registry import prune_entry


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
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
        help='Run folder name (registry key) to delete',
    )
    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    registry_path = Path(args.registry).resolve()
    run_dir = registry_path.parent / args.run_name

    if not registry_path.exists():
        print(f'Error: registry not found: {registry_path}', file=sys.stderr)
        sys.exit(1)

    registry = json.loads(registry_path.read_text())
    if args.run_name not in registry:
        print(
            f'Error: run {args.run_name!r} not found in {registry_path}',
            file=sys.stderr,
        )
        sys.exit(1)

    if not run_dir.exists():
        print(f'Warning: run directory already gone: {run_dir}', file=sys.stderr)

    prune_entry(registry_path, run_dir, args.run_name)
    print(f'Pruned {args.run_name!r}: removed {run_dir} and its registry entry')


if __name__ == '__main__':
    main()
