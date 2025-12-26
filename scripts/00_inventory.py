#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adni_analysis.inventory import build_raw_inventory


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory raw ADNI files (hashes, row counts, columns).")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--audit-dir", type=Path, default=Path("audit"))
    parser.add_argument("--core-dict-dir", type=Path, default=Path("core/dictionaries"))
    parser.add_argument("--manifests-dir", type=Path, default=Path("manifests"))
    args = parser.parse_args()

    build_raw_inventory(
        raw_root=args.raw_root,
        audit_dir=args.audit_dir,
        core_dict_dir=args.core_dict_dir,
        manifests_dir=args.manifests_dir,
    )


if __name__ == "__main__":
    main()
