"""Workspace cleaning utility.

Safely removes build/cache artifacts while keeping data, models, and logs intact
by default. Use optional flags to prune heavier artifacts.
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PATTERNS = [
    "**/__pycache__",
    "**/*.pyc",
    "**/*.pyo",
    "**/*.pyd",
    "**/.ipynb_checkpoints",
    "data/debug_detection",
]

HEAVY_PATTERNS = [
    # TensorBoard/event logs inside model dirs
    "models/model_*/logs",
    # Evaluation plots (metrics CSVs are kept)
    "data/evaluation_results/*.png",
]


def remove_path(p: Path) -> None:
    if not p.exists():
        return
    if p.is_file() or p.is_symlink():
        p.unlink(missing_ok=True)
    else:
        shutil.rmtree(p, ignore_errors=True)


def clean(dry_run: bool = True, include_heavy: bool = False) -> None:
    patterns = list(DEFAULT_PATTERNS)
    if include_heavy:
        patterns.extend(HEAVY_PATTERNS)

    to_remove = []
    for pattern in patterns:
        for path in ROOT.glob(pattern):
            to_remove.append(path)

    if not to_remove:
        print("Nothing to clean; workspace already tidy.")
        return

    print(f"Found {len(to_remove)} item(s) to remove:")
    for p in to_remove:
        print(f"  - {p.relative_to(ROOT)}")

    if dry_run:
        print("\nDry run only. Re-run with --apply to delete.")
        return

    for p in to_remove:
        remove_path(p)
    print("\nCleanup complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Clean workspace caches and temporary artifacts.")
    parser.add_argument("--apply", action="store_true", help="Actually delete files (default: dry run)")
    parser.add_argument("--include-heavy", action="store_true", help="Also remove logs/plots (keeps data/models)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    clean(dry_run=not args.apply, include_heavy=args.include_heavy)
