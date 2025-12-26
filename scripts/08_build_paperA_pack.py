#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adni_analysis.paper_a import build_paper_a_pack


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Paper A analysis pack from the frozen evidence core.")
    parser.add_argument(
        "--definitions",
        type=Path,
        default=Path("outputs/paperA/definitions.yaml"),
        help="Paper A definitions YAML (tracked)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/paperA"),
        help="Output directory for generated (untracked) Paper A artifacts",
    )
    args = parser.parse_args()

    build_paper_a_pack(definitions_path=args.definitions, out_dir=args.out_dir)


if __name__ == "__main__":
    main()

