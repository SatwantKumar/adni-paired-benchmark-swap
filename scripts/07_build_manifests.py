#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adni_analysis.config import load_yaml
from adni_analysis.manifests import build_signature, build_table_manifest
from adni_analysis.pipeline import collect_core_tables, ensure_pipeline_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build table manifest + build signature from core outputs.")
    parser.add_argument("--config", type=Path, default=Path("config/pipeline.yaml"))
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    dirs = ensure_pipeline_dirs(cfg)
    tables = collect_core_tables(dirs.core_dir)
    build_table_manifest(tables, dirs.manifests_dir / "table_manifest.json")
    build_signature(out_path=dirs.manifests_dir / "build_signature.txt", config_paths=[args.config], table_paths=tables)


if __name__ == "__main__":
    main()

