from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from adni_analysis.utils import ensure_dir, sha256_file, write_columns_txt


@dataclass(frozen=True)
class RawInventoryPaths:
    index_csv: Path
    columns_dir: Path
    key_presence_csv: Path


def _read_csv_header(path: Path) -> tuple[list[str], str | None, str | None]:
    last_err: str | None = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with path.open("r", newline="", encoding=enc) as f:
                header = next(csv.reader(f))
            return header, enc, None
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
    return [], None, last_err


def _count_rows_fast(path: Path) -> int | None:
    try:
        with path.open("rb") as f:
            line_count = 0
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                line_count += chunk.count(b"\n")
        return max(0, line_count - 1)
    except Exception:  # noqa: BLE001
        return None


def build_raw_inventory(
    *,
    raw_root: Path,
    audit_dir: Path,
    core_dict_dir: Path,
    manifests_dir: Path,
) -> RawInventoryPaths:
    raw_inventory_dir = ensure_dir(audit_dir / "raw_inventory")
    columns_dir = ensure_dir(raw_inventory_dir / "raw_columns")
    ensure_dir(core_dict_dir)
    ensure_dir(manifests_dir)

    index_csv = raw_inventory_dir / "raw_file_index.csv"
    key_presence_csv = core_dict_dir / "key_fields_presence.csv"

    records: list[dict] = []
    key_presence: list[dict] = []

    for path in sorted(raw_root.rglob("*")):
        if not path.is_file():
            continue
        if path.name in {".DS_Store"}:
            continue

        relpath = path.as_posix()
        stat = path.stat()
        rec: dict = {
            "path": relpath,
            "size_bytes": stat.st_size,
            "mtime_iso": pd.to_datetime(stat.st_mtime, unit="s").isoformat(),
            "sha256": sha256_file(path),
        }

        if path.suffix.lower() == ".csv":
            header, encoding, err = _read_csv_header(path)
            row_count = _count_rows_fast(path)
            rec.update(
                {
                    "is_csv": True,
                    "encoding": encoding,
                    "row_count": row_count,
                    "column_count": len(header),
                    "read_error": err,
                }
            )
            if header:
                write_columns_txt(header, columns_dir / f"{path.name}.columns.txt")
                upper = {c.upper() for c in header}
                key_presence.append(
                    {
                        "path": relpath,
                        "has_RID": "RID" in upper,
                        "has_PTID": "PTID" in upper,
                        "has_VISCODE": "VISCODE" in upper,
                        "has_VISCODE2": "VISCODE2" in upper,
                        "has_EXAMDATE": "EXAMDATE" in upper,
                        "has_SCANDATE": "SCANDATE" in upper,
                        "has_TRACER": "TRACER" in upper,
                    }
                )
        else:
            rec.update({"is_csv": False})

        records.append(rec)

    pd.DataFrame.from_records(records).to_csv(index_csv, index=False)
    pd.DataFrame.from_records(records).to_csv(manifests_dir / "file_manifest.csv", index=False)
    pd.DataFrame.from_records(key_presence).to_csv(key_presence_csv, index=False)

    return RawInventoryPaths(index_csv=index_csv, columns_dir=columns_dir, key_presence_csv=key_presence_csv)

