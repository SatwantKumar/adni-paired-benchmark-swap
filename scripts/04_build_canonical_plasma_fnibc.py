#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adni_analysis.config import load_yaml
from adni_analysis.pipeline import build_canonical_plasma_fnibc, ensure_pipeline_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical plasma tables from FNIHBC trajectories (long + wide).")
    parser.add_argument("--config", type=Path, default=Path("config/pipeline.yaml"))
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    dirs = ensure_pipeline_dirs(cfg)
    build_canonical_plasma_fnibc(cfg, dirs)


if __name__ == "__main__":
    main()

