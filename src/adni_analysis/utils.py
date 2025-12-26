from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
from pandas.util import hash_pandas_object


@dataclass(frozen=True)
class PipelineDirs:
    core_dir: Path
    audit_dir: Path
    manifests_dir: Path
    core_canonical_dir: Path
    core_paired_dir: Path
    core_dict_dir: Path
    audit_raw_inventory_dir: Path
    audit_raw_columns_dir: Path
    audit_validations_dir: Path
    audit_join_reports_dir: Path
    audit_qc_dir: Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_MULTI_UNDERSCORE = re.compile(r"_+")


def normalize_label(value: object) -> str:
    s = str(value).strip().lower()
    s = _NON_ALNUM.sub("_", s)
    s = _MULTI_UNDERSCORE.sub("_", s).strip("_")
    return s


def make_event_id(df: pd.DataFrame, cols: list[str], *, prefix: str) -> pd.Series:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot compute event_id; missing columns: {missing}")

    key: pd.Series | None = None
    for col in cols:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            part = s.dt.strftime("%Y-%m-%d").astype("string")
        else:
            part = s.astype("string")
        part = part.fillna("").str.strip()
        key = part if key is None else key.str.cat(part, sep="|")

    if key is None:
        raise ValueError("No columns provided for event_id")

    key = (prefix + "|") + key
    return key.map(lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()).astype("string")


def parse_date_series(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    raw = series.astype("string")
    parsed = pd.to_datetime(raw, errors="coerce")
    return parsed, raw


def add_provenance(df: pd.DataFrame, source_file: Path) -> pd.DataFrame:
    df = df.copy()
    df["source_file"] = str(source_file)
    df["source_row"] = (pd.RangeIndex(start=1, stop=len(df) + 1, step=1)).astype("int64")
    return df


def add_row_uid(df: pd.DataFrame, *, include_index: bool = False) -> pd.DataFrame:
    df = df.copy()
    as_str = df.astype("string").fillna("")
    hashed = hash_pandas_object(as_str, index=include_index).astype("uint64")
    df["row_uid"] = hashed.map(lambda x: f"{x:016x}").astype("string")
    return df


def write_columns_txt(columns: Iterable[str], out_path: Path) -> None:
    out_path.write_text("\n".join(columns) + "\n", encoding="utf-8")


def today_iso() -> str:
    return date.today().isoformat()
