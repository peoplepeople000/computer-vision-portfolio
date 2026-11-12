"""Populate resumable deep-search caches before scene-level inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from nocpt.deep_retrieval import retrieve_deep_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", nargs="+")
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--data-root", type=Path, default=Path("participant"))
    parser.add_argument("--cache", type=Path, default=Path("outputs/cache"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    for scene in args.scenes:
        retrieve_deep_scene(args.data_root, args.split, scene, args.cache,
                            workers=args.workers)
        print(f"Prefetched {scene}", flush=True)


if __name__ == "__main__":
    main()
