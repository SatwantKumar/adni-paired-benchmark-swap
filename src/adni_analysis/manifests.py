from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd

from adni_analysis.utils import sha256_file


def _table_shape_fast(path: Path) -> tuple[int, int]:
    df = pd.read_csv(path, nrows=5)
    n_cols = int(df.shape[1])
    with path.open("rb") as f:
        n_rows = sum(1 for _ in f) - 1
    return max(0, int(n_rows)), n_cols


def build_table_manifest(table_paths: list[Path], out_path: Path) -> None:
    tables: dict[str, dict] = {}
    for p in sorted(table_paths):
        if p.suffix.lower() != ".csv":
            continue
        n_rows, n_cols = _table_shape_fast(p)
        tables[p.as_posix()] = {
            "sha256": sha256_file(p),
            "n_rows": n_rows,
            "n_cols": n_cols,
        }

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "tables": tables,
    }
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_signature(*, out_path: Path, config_paths: list[Path], table_paths: list[Path]) -> None:
    lines: list[str] = []
    lines.append(f"generated_utc: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"python: {sys.version.split()[0]}")
    lines.append(f"platform: {platform.platform()}")
    lines.append("")
    for name in ["pandas", "numpy", "pyyaml"]:
        try:
            v = metadata.version(name)
            lines.append(f"{name}: {v}")
        except metadata.PackageNotFoundError:
            lines.append(f"{name}: (not installed)")
    lines.append("")

    for cfg in config_paths:
        lines.append(f"config: {cfg.as_posix()} sha256={sha256_file(cfg)}")
    lines.append("")

    for p in sorted(table_paths):
        if not p.exists():
            continue
        lines.append(f"table: {p.as_posix()} sha256={sha256_file(p)}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

