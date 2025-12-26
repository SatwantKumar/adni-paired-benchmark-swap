from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from adni_analysis.utils import PipelineDirs, sha256_file, today_iso


def write_join_rules(*, out_path: Path, windows_days: dict, plasma_filter: dict | None, plasma_source: str | None) -> None:
    plasma_filter_str = json.dumps(plasma_filter, sort_keys=True) if plasma_filter else "(none)"
    plasma_source_str = plasma_source if plasma_source else "(default)"
    lines: list[str] = []
    lines.append("# Join rules (deterministic)")
    lines.append("")
    lines.append("This document describes the deterministic tie-break rules used by the pipeline when canonicalizing tables and when pairing modalities into dyads/triads.")
    lines.append("")
    lines.append("## Canonicalization")
    lines.append("")
    lines.append("### Amyloid PET (UCBERKELEY_AMY_6MM)")
    lines.append("- Parse `SCANDATE` and `PROCESSDATE` to dates.")
    lines.append("- Deduplicate by `(RID, SCANDATE, TRACER)` and keep exactly one row using:")
    lines.append("  1) highest `qc_flag` rank (`pass` > numeric/unknown > `fail`)")
    lines.append("  2) latest `PROCESSDATE`")
    lines.append("  3) latest `update_stamp`")
    lines.append("  4) latest file order (`source_row`) as final tie-breaker")
    lines.append("")
    lines.append("### CSF (UPENNBIOMK_ROCHE_ELECSYS)")
    lines.append("- Parse `EXAMDATE` and `RUNDATE` to dates.")
    lines.append("- Deduplicate by `(RID, EXAMDATE)` and keep exactly one row using:")
    lines.append("  1) latest `RUNDATE`")
    lines.append("  2) latest `update_stamp`")
    lines.append("  3) latest file order (`source_row`) as final tie-breaker")
    lines.append("- Derive ratios: `ABETA42/ABETA40`, `PTAU/ABETA42`, `TAU/ABETA42` (no positivity cutpoints applied in the core).")
    lines.append("")
    lines.append("### Plasma (FNIHBC blood biomarker trajectories)")
    lines.append("- Keep a long-form table with `TESTVALUE`, `UNITS`, and `COMMENTS` (no censoring/removal).")
    lines.append("- Create `analyte_std` from `PLASMA_BIOMARKER` + `TESTNAME` and pivot to wide format keyed by:")
    lines.append("  `(RID, EXAMDATE, SAMPLEID, ASSAYPLATFORM, ASSAYVERSION, PERFORMINGLAB, MATRIX)`.")
    lines.append("- When multiple measurements collide for the same `(key, analyte_std)`, keep one row using:")
    lines.append("  1) lowest numeric `CV`")
    lines.append("  2) latest `RUNDATE`")
    lines.append("  3) lowest `SEQUENCE_NO`")
    lines.append("  4) latest file order (`source_row`) as final tie-breaker")
    lines.append("")
    lines.append("### Plasma (C2N PrecivityAD2 APS2 table)")
    lines.append("- Parse `EXAMDATE` to a date.")
    lines.append("- Convert sentinel missing values (e.g., `-4`) to null for numeric fields including `APS2_C2N` and component analytes.")
    lines.append("- Standardize assay metadata fields for pairing (`ASSAYPLATFORM=C2N`, `ASSAYVERSION=PrecivityAD2`, `MATRIX=PLASMA`).")
    lines.append("")
    lines.append("## Pairing into dyads/triads")
    lines.append("")
    lines.append("Windows are symmetric ±days and applied to absolute day gaps.")
    lines.append(f"- PET–plasma window: ±{int(windows_days['pet_plasma'])} days")
    lines.append(f"- PET–CSF window: ±{int(windows_days['pet_csf'])} days")
    lines.append(f"- Plasma–CSF window: ±{int(windows_days['plasma_csf'])} days")
    lines.append(f"- Plasma source used for pairing: {plasma_source_str}")
    lines.append(f"- Plasma filter applied before pairing: {plasma_filter_str}")
    lines.append("")
    lines.append("### Dyads")
    lines.append("- For each anchor event, select the candidate within window that minimizes absolute day gap.")
    lines.append("- Tie-breakers (in order): earlier candidate date, then lexicographically smallest `row_uid`.")
    lines.append("")
    lines.append("### PET-anchored triads")
    lines.append("- For each PET event, consider all plasma candidates within window and all CSF candidates within window.")
    lines.append("- Choose the pair that minimizes (in order):")
    lines.append("  1) the maximum of the three pairwise gaps (PET–plasma, PET–CSF, plasma–CSF)")
    lines.append("  2) the sum of the three pairwise gaps")
    lines.append("  3) PET–plasma gap")
    lines.append("  4) PET–CSF gap")
    lines.append("  5) earlier plasma date, then earlier CSF date")
    lines.append("  6) lexicographically smallest plasma and CSF `row_uid` as final tie-breakers")
    lines.append("")
    lines.append("### Multiple events per participant")
    lines.append("- Canonical PET/CSF/plasma tables may contain multiple events per `RID`.")
    lines.append("- PET-anchored triads are constructed independently per PET event; a participant can contribute multiple triads.")
    lines.append("- The same plasma/CSF event may be selected for multiple PET anchors if it is the closest match within each anchor’s window.")
    lines.append("")
    lines.append("## Event identity")
    lines.append("- `pet_event_id`: SHA256(`RID|SCANDATE|TRACER`) with `pet|` prefix")
    lines.append("- `csf_event_id`: SHA256(`RID|EXAMDATE|RUNDATE|BATCH`) with `csf|` prefix")
    lines.append("- `plasma_event_id`: SHA256(`RID|EXAMDATE|SAMPLEID|ASSAYPLATFORM|ASSAYVERSION|PERFORMINGLAB|MATRIX`) with `plasma|` prefix")
    lines.append("- `clin_event_id`: SHA256(`RID|examdate|VISCODE`) with `clin|` prefix")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _schema_for_csv(path: Path, *, dataset_name: str, primary_key: list[str] | None, allowed_values_cols: set[str]) -> dict:
    df = pd.read_csv(path, low_memory=False)
    cols: list[dict] = []
    for c in df.columns:
        s = df[c]
        dtype = str(s.dtype)
        nulls = int(s.isna().sum())
        n = int(len(s))
        entry: dict = {
            "name": c,
            "dtype": dtype,
            "nullable": nulls > 0,
            "null_fraction": float(nulls / n) if n else 0.0,
            "is_primary_key": bool(primary_key and c in primary_key),
        }
        if c in allowed_values_cols:
            uniq = s.dropna().astype("string").unique().tolist()
            uniq = sorted({str(x).strip() for x in uniq if str(x).strip()})
            if 0 < len(uniq) <= 50:
                entry["allowed_values"] = uniq
        cols.append(entry)

    return {
        "dataset": dataset_name,
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "primary_key": primary_key,
        "columns": cols,
    }


def build_schema_registry(*, out_dir: Path, tables: list[Path]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    pk_map: dict[str, list[str]] = {
        "clinical_adnimerge": ["RID", "VISCODE"],
        "pet_amyloid_ucb_6mm": ["pet_event_id"],
        "csf_elecsys_upenn": ["csf_event_id"],
        "plasma_fnibc_long": ["row_uid"],
        "plasma_fnibc_wide": ["plasma_event_id"],
        "plasma_c2n_precivityad2_score": ["plasma_event_id"],
        "plasma_janssen_ptau217": ["plasma_event_id"],
        "dyad_pet_plasma": ["pet_event_id"],
        "dyad_pet_csf": ["pet_event_id"],
        "dyad_csf_plasma": ["csf_event_id"],
        "triad_pet_anchored": ["pet_event_id"],
    }
    allowed_values_cols = {"dx_simplified", "TRACER", "ASSAYVERSION", "ASSAYPLATFORM", "PTETHCAT", "PTRACCAT"}

    for path in tables:
        if path.suffix.lower() != ".csv":
            continue
        name = path.stem
        schema = _schema_for_csv(path, dataset_name=name, primary_key=pk_map.get(name), allowed_values_cols=allowed_values_cols)
        (out_dir / f"{name}.schema.json").write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_evidence_core_catalog(*, out_path: Path, tables: list[Path]) -> None:
    records: list[dict] = []
    for p in tables:
        if p.suffix.lower() != ".csv":
            continue
        name = p.stem
        df = pd.read_csv(p, nrows=5)
        cols = set(df.columns)

        date_field = None
        for candidate in ["SCANDATE", "EXAMDATE", "pet_date", "csf_date", "plasma_date"]:
            if candidate in cols:
                date_field = candidate
                break

        rid_col = "RID" if "RID" in cols else None
        n_rows = max(0, (sum(1 for _ in p.open("rb")) - 1))
        uniq_rid = None
        if rid_col:
            rid_series = pd.read_csv(p, usecols=[rid_col])[rid_col]
            uniq_rid = int(rid_series.nunique(dropna=True))

        purpose = {
            "clinical_adnimerge": "clinical backbone",
            "pet_amyloid_ucb_6mm": "amyloid PET proxy (UCB 6mm pipeline)",
            "csf_elecsys_upenn": "CSF amyloid/tau (Roche Elecsys, UPenn)",
            "plasma_fnibc_long": "plasma biomarkers (long form, multi-platform)",
            "plasma_fnibc_wide": "plasma biomarkers (wide form, per draw/platform)",
            "plasma_c2n_precivityad2_score": "plasma biomarkers + APS2 (C2N PrecivityAD2)",
            "plasma_janssen_ptau217": "plasma pTau217 (Janssen)",
            "dyad_pet_plasma": "paired dyad (PET–plasma)",
            "dyad_pet_csf": "paired dyad (PET–CSF)",
            "dyad_csf_plasma": "paired dyad (CSF–plasma)",
            "triad_pet_anchored": "paired triad (PET-anchored)",
        }.get(name, "")

        key_fields = {
            "clinical_adnimerge": "RID,VISCODE",
            "pet_amyloid_ucb_6mm": "pet_event_id",
            "csf_elecsys_upenn": "csf_event_id",
            "plasma_fnibc_long": "row_uid",
            "plasma_fnibc_wide": "plasma_event_id",
            "plasma_c2n_precivityad2_score": "plasma_event_id",
            "plasma_janssen_ptau217": "plasma_event_id",
            "dyad_pet_plasma": "pet_event_id",
            "dyad_pet_csf": "pet_event_id",
            "dyad_csf_plasma": "csf_event_id",
            "triad_pet_anchored": "pet_event_id",
        }.get(name, "")

        records.append(
            {
                "dataset_name": name,
                "path": p.as_posix(),
                "purpose": purpose,
                "key_fields": key_fields,
                "n_rows": n_rows,
                "unique_rids": uniq_rid,
                "date_field": date_field,
            }
        )

    pd.DataFrame.from_records(records).sort_values(["dataset_name"]).to_csv(out_path, index=False)


def build_time_gap_distributions(
    *,
    out_path: Path,
    windows_days: dict,
    dyad_pet_plasma: Path,
    dyad_pet_csf: Path,
    dyad_csf_plasma: Path,
    triad: Path,
) -> None:
    specs = [
        ("dyad_pet_plasma", dyad_pet_plasma, "delta_days_abs", int(windows_days["pet_plasma"])),
        ("dyad_pet_csf", dyad_pet_csf, "delta_days_abs", int(windows_days["pet_csf"])),
        ("dyad_csf_plasma", dyad_csf_plasma, "delta_days_abs", int(windows_days["plasma_csf"])),
        ("triad_pet_anchored", triad, "delta_pet_plasma_days", int(windows_days["pet_plasma"])),
        ("triad_pet_anchored", triad, "delta_pet_csf_days", int(windows_days["pet_csf"])),
        ("triad_pet_anchored", triad, "delta_plasma_csf_days", int(windows_days["plasma_csf"])),
    ]

    rows: list[dict] = []
    for dataset, path, col, win in specs:
        df = pd.read_csv(path, usecols=[col])
        s = pd.to_numeric(df[col], errors="coerce")
        n_total = int(len(s))
        s = s.dropna()
        n = int(len(s))
        if n == 0:
            rows.append(
                {
                    "dataset": dataset,
                    "delta_field": col,
                    "window_days": win,
                    "n_total": n_total,
                    "n_nonnull": 0,
                    "frac_nonnull": 0.0,
                    "frac_zero": 0.0,
                    "outside_window": 0,
                }
            )
            continue

        q = s.quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
        rows.append(
            {
                "dataset": dataset,
                "delta_field": col,
                "window_days": win,
                "n_total": n_total,
                "n_nonnull": n,
                "frac_nonnull": float(n / n_total) if n_total else 0.0,
                "frac_zero": float((s == 0).mean()),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)) if n > 1 else 0.0,
                "min": float(q.loc[0.0]),
                "p01": float(q.loc[0.01]),
                "p05": float(q.loc[0.05]),
                "p25": float(q.loc[0.25]),
                "p50": float(q.loc[0.5]),
                "p75": float(q.loc[0.75]),
                "p95": float(q.loc[0.95]),
                "p99": float(q.loc[0.99]),
                "max": float(q.loc[1.0]),
                "outside_window": int((s > win).sum()),
            }
        )

    pd.DataFrame.from_records(rows).to_csv(out_path, index=False)


def build_referential_integrity_report(
    *,
    out_path: Path,
    clinical: Path,
    pet: Path,
    csf: Path,
    plasma_wide: Path,
    dyad_pet_plasma: Path,
    dyad_pet_csf: Path,
    dyad_csf_plasma: Path,
    triad: Path,
) -> None:
    clin = pd.read_csv(clinical, usecols=["RID"])
    clin_rids = set(clin["RID"].dropna().astype(int).tolist())

    def rid_set(path: Path) -> set[int]:
        df = pd.read_csv(path, usecols=["RID"])
        return set(df["RID"].dropna().astype(int).tolist())

    pet_rids = rid_set(pet)
    csf_rids = rid_set(csf)
    plasma_rids = rid_set(plasma_wide)

    lines: list[str] = []
    lines.append("# Referential integrity (must-pass)")
    lines.append("")
    lines.append("## RID coverage (canonical → clinical)")
    for label, rids in [("PET", pet_rids), ("CSF", csf_rids), ("Plasma", plasma_rids)]:
        missing = rids - clin_rids
        lines.append(f"- {label}: {len(missing)} RIDs not found in clinical backbone")
    lines.append("")

    pet_ids = set(pd.read_csv(pet, usecols=["pet_event_id"])["pet_event_id"].dropna().astype(str).tolist())
    csf_ids = set(pd.read_csv(csf, usecols=["csf_event_id"])["csf_event_id"].dropna().astype(str).tolist())
    plasma_ids = set(pd.read_csv(plasma_wide, usecols=["plasma_event_id"])["plasma_event_id"].dropna().astype(str).tolist())

    def check_fk(path: Path, col: str, valid: set[str]) -> tuple[int, int, int]:
        df = pd.read_csv(path, usecols=[col])
        total = int(len(df))
        values = df[col].astype("string")
        nulls = int(values.isna().sum())
        missing = int((~values.isna() & ~values.isin(valid)).sum())
        return total, nulls, missing

    lines.append("## Event-id foreign key checks (paired → canonical)")
    for label, path, col, valid in [
        ("dyad_pet_plasma.pet_event_id", dyad_pet_plasma, "pet_event_id", pet_ids),
        ("dyad_pet_plasma.plasma_event_id", dyad_pet_plasma, "plasma_event_id", plasma_ids),
        ("dyad_pet_csf.pet_event_id", dyad_pet_csf, "pet_event_id", pet_ids),
        ("dyad_pet_csf.csf_event_id", dyad_pet_csf, "csf_event_id", csf_ids),
        ("dyad_csf_plasma.csf_event_id", dyad_csf_plasma, "csf_event_id", csf_ids),
        ("dyad_csf_plasma.plasma_event_id", dyad_csf_plasma, "plasma_event_id", plasma_ids),
        ("triad.pet_event_id", triad, "pet_event_id", pet_ids),
        ("triad.csf_event_id", triad, "csf_event_id", csf_ids),
        ("triad.plasma_event_id", triad, "plasma_event_id", plasma_ids),
    ]:
        total, nulls, missing = check_fk(path, col, valid)
        lines.append(f"- {label}: total={total}, nulls={nulls}, missing={missing}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Must-pass gates
    if (pet_rids - clin_rids) or (csf_rids - clin_rids) or (plasma_rids - clin_rids):
        raise RuntimeError("Referential integrity failed: some canonical RIDs are missing from clinical backbone")

    for _, path, col, valid in [
        ("", dyad_pet_plasma, "pet_event_id", pet_ids),
        ("", dyad_pet_plasma, "plasma_event_id", plasma_ids),
        ("", dyad_pet_csf, "pet_event_id", pet_ids),
        ("", dyad_pet_csf, "csf_event_id", csf_ids),
        ("", dyad_csf_plasma, "csf_event_id", csf_ids),
        ("", dyad_csf_plasma, "plasma_event_id", plasma_ids),
        ("", triad, "pet_event_id", pet_ids),
        ("", triad, "csf_event_id", csf_ids),
        ("", triad, "plasma_event_id", plasma_ids),
    ]:
        df = pd.read_csv(path, usecols=[col])
        values = df[col].astype("string")
        missing = int((~values.isna() & ~values.isin(valid)).sum())
        if missing:
            raise RuntimeError(f"Referential integrity failed: {missing} values in {path} column {col} not found in canonical set")


def _pet_pos_from_status(series: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")

    numeric = pd.to_numeric(series, errors="coerce")
    num_mask = numeric.notna()
    if num_mask.any():
        out.loc[num_mask] = numeric.loc[num_mask] == 1

    str_mask = ~num_mask
    if str_mask.any():
        s = series.astype("string").str.strip().str.upper()
        out.loc[str_mask] = (
            s.loc[str_mask].str.startswith("POS")
            | s.loc[str_mask].isin({"P", "POSITIVE", "1", "TRUE", "T", "YES", "Y"})
        )

    return out


def build_regression_snapshot(
    *,
    triad_path: Path,
    plasma_wide_path: Path,
    csf_path: Path,
    out_path: Path,
    check_path: Path,
    release_id: str | None,
    mode: str,
    float_tol: float,
) -> None:
    triad = pd.read_csv(triad_path, low_memory=False)
    triad_n = int(triad.shape[0])
    triad_rids = int(triad["RID"].nunique(dropna=True)) if "RID" in triad.columns else 0

    deltas = {}
    for col in ["delta_pet_plasma_days", "delta_pet_csf_days", "delta_plasma_csf_days"]:
        s = pd.to_numeric(triad[col], errors="coerce")
        deltas[col] = {
            "median": float(s.median()),
            "p25": float(s.quantile(0.25)),
            "p75": float(s.quantile(0.75)),
        }

    pet_pos = _pet_pos_from_status(triad.get("AMYLOID_STATUS", pd.Series([pd.NA] * len(triad))))
    pet_pos_rate = float(pet_pos.mean()) if len(pet_pos) else float("nan")

    csf = pd.read_csv(csf_path, usecols=["csf_event_id", "ABETA42", "ABETA40", "abeta42_40_ratio"], low_memory=False)
    triad_csf_ids = triad["csf_event_id"].astype("string")
    csf_tri = csf[csf["csf_event_id"].astype("string").isin(triad_csf_ids)].copy()
    csf_missing = {
        "ABETA42": float(pd.to_numeric(csf_tri["ABETA42"], errors="coerce").isna().mean()) if len(csf_tri) else float("nan"),
        "ABETA40": float(pd.to_numeric(csf_tri["ABETA40"], errors="coerce").isna().mean()) if len(csf_tri) else float("nan"),
        "abeta42_40_ratio": float(pd.to_numeric(csf_tri["abeta42_40_ratio"], errors="coerce").isna().mean())
        if len(csf_tri)
        else float("nan"),
    }

    plasma = pd.read_csv(plasma_wide_path, low_memory=False)
    triad_plasma_ids = triad["plasma_event_id"].astype("string")
    plasma_tri = plasma[plasma["plasma_event_id"].astype("string").isin(triad_plasma_ids)].copy()
    default_plasma_cols = [
        "c2n_plasma_abeta40_abeta40",
        "c2n_plasma_abeta42_abeta42",
        "c2n_plasma_abeta42_abeta40_abeta_ratio",
        "c2n_plasma_ptau217_ptau217",
        "c2n_plasma_nptau217_nptau217",
        "c2n_plasma_ptau217_ratio_ptau217_ratio",
    ]
    plasma_missing = {}
    for col in default_plasma_cols:
        if col in plasma_tri.columns:
            plasma_missing[col] = float(pd.to_numeric(plasma_tri[col], errors="coerce").isna().mean()) if len(plasma_tri) else float("nan")

    snapshot = {
        "release_id": release_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "triad": {
            "n_rows": triad_n,
            "n_unique_rid": triad_rids,
            "pet_pos_rate_from_AMYLOID_STATUS": round(pet_pos_rate, 6) if not np.isnan(pet_pos_rate) else None,
            "deltas_days": {k: {kk: round(vv, 6) for kk, vv in v.items()} for k, v in deltas.items()},
        },
        "missingness_in_triads": {
            "csf": {k: round(v, 6) if not np.isnan(v) else None for k, v in csf_missing.items()},
            "plasma": {k: round(v, 6) if not np.isnan(v) else None for k, v in plasma_missing.items()},
        },
    }

    if out_path.exists() and mode != "update":
        previous = json.loads(out_path.read_text(encoding="utf-8"))

        prev_release = previous.get("release_id")
        if release_id is not None and str(prev_release) != str(release_id):
            out_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            check_path.write_text(f"Regression check: SNAPSHOT UPDATED (release_id changed: {prev_release} -> {release_id})\n", encoding="utf-8")
            return

        def cmp(a: object, b: object, path: str) -> list[str]:
            diffs: list[str] = []
            if isinstance(a, dict) and isinstance(b, dict):
                keys = set(a.keys()) | set(b.keys())
                for k in sorted(keys):
                    diffs.extend(cmp(a.get(k), b.get(k), f"{path}.{k}"))
                return diffs
            if isinstance(a, (int, str)) or a is None or isinstance(a, bool):
                if a != b:
                    diffs.append(f"{path}: {a} != {b}")
                return diffs
            if isinstance(a, float) or isinstance(b, float):
                if a is None or b is None:
                    if a != b:
                        diffs.append(f"{path}: {a} != {b}")
                    return diffs
                if abs(float(a) - float(b)) > float_tol:
                    diffs.append(f"{path}: {a} != {b} (tol={float_tol})")
                return diffs
            if a != b:
                diffs.append(f"{path}: {a} != {b}")
            return diffs

        diffs = cmp(previous.get("triad"), snapshot.get("triad"), "triad")
        diffs += cmp(previous.get("missingness_in_triads"), snapshot.get("missingness_in_triads"), "missingness_in_triads")
        if diffs:
            check_path.write_text("Regression check: FAIL\n\n" + "\n".join(f"- {d}" for d in diffs) + "\n", encoding="utf-8")
            raise RuntimeError("Regression snapshot mismatch; see audit/qc/regression_check.md")

        check_path.write_text("Regression check: PASS (matches snapshot)\n", encoding="utf-8")
        return

    out_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    check_path.write_text("Regression check: SNAPSHOT CREATED/UPDATED\n", encoding="utf-8")


def build_value_flags(*, csf_path: Path, plasma_long_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csf = pd.read_csv(csf_path, low_memory=False)
    csf_flags: list[dict] = []
    for biomarker in ["ABETA40", "ABETA42", "PTAU", "TAU", "abeta42_40_ratio", "ptau_abeta42_ratio", "ttau_abeta42_ratio"]:
        if biomarker not in csf.columns:
            continue
        s = pd.to_numeric(csf[biomarker], errors="coerce")
        if s.notna().sum() == 0:
            continue
        hi = float(s.quantile(0.999))
        lo = float(s.quantile(0.001))
        nonpos = (s <= 0) & s.notna()
        high = (s > hi) & s.notna()
        low = (s < lo) & s.notna()
        for idx in np.where(nonpos | high | low)[0]:
            flag = "nonpositive" if bool(nonpos.iloc[idx]) else ("high_outlier" if bool(high.iloc[idx]) else "low_outlier")
            thresh = 0.0 if flag == "nonpositive" else (hi if flag == "high_outlier" else lo)
            csf_flags.append(
                {
                    "RID": csf.at[idx, "RID"],
                    "csf_event_id": csf.at[idx, "csf_event_id"],
                    "EXAMDATE": csf.at[idx, "EXAMDATE"],
                    "biomarker": biomarker,
                    "value": float(s.iloc[idx]),
                    "flag": flag,
                    "threshold": thresh,
                }
            )

    pd.DataFrame.from_records(csf_flags).to_csv(out_dir / "csf_value_flags.csv", index=False)

    plasma = pd.read_csv(plasma_long_path, low_memory=False)
    plasma_flags: list[dict] = []
    plasma["TESTVALUE_num"] = pd.to_numeric(plasma["TESTVALUE"], errors="coerce")
    plasma["COMMENTS_str"] = plasma.get("COMMENTS", pd.Series([pd.NA] * len(plasma))).astype("string")
    plasma["has_lod_comment"] = plasma["COMMENTS_str"].str.contains("LOD", case=False, na=False)

    for analyte, g in plasma.groupby("analyte_std", dropna=False):
        s = g["TESTVALUE_num"]
        s = s.dropna()
        if len(s) < 20:
            continue
        hi = float(s.quantile(0.999))
        lo = float(s.quantile(0.001))
        nonpos_mask = (g["TESTVALUE_num"] <= 0) & g["TESTVALUE_num"].notna()
        high_mask = (g["TESTVALUE_num"] > hi) & g["TESTVALUE_num"].notna()
        low_mask = (g["TESTVALUE_num"] < lo) & g["TESTVALUE_num"].notna()
        lod_mask = g["has_lod_comment"] == True  # noqa: E712

        flagged = g[nonpos_mask | high_mask | low_mask | lod_mask]
        for _, r in flagged.iterrows():
            flag = "lod_comment" if bool(r["has_lod_comment"]) else ("nonpositive" if r["TESTVALUE_num"] <= 0 else None)
            if flag is None:
                flag = "high_outlier" if r["TESTVALUE_num"] > hi else "low_outlier"
            thresh = 0.0 if flag == "nonpositive" else (hi if flag == "high_outlier" else (lo if flag == "low_outlier" else np.nan))
            plasma_flags.append(
                {
                    "RID": r.get("RID"),
                    "plasma_event_id": r.get("plasma_event_id"),
                    "EXAMDATE": r.get("EXAMDATE"),
                    "ASSAYVERSION": r.get("ASSAYVERSION"),
                    "analyte_std": analyte,
                    "TESTVALUE": r.get("TESTVALUE_num"),
                    "UNITS": r.get("UNITS"),
                    "COMMENTS": r.get("COMMENTS"),
                    "flag": flag,
                    "threshold": thresh,
                }
            )

    pd.DataFrame.from_records(plasma_flags).to_csv(out_dir / "plasma_value_flags.csv", index=False)


@dataclass(frozen=True)
class GitInfo:
    commit: str | None
    is_repo: bool
    is_dirty: bool | None


def get_git_info(repo_root: Path) -> GitInfo:
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        return GitInfo(commit=commit, is_repo=True, is_dirty=dirty)
    except Exception:  # noqa: BLE001
        return GitInfo(commit=None, is_repo=False, is_dirty=None)


def build_release_stamp(
    *,
    out_path: Path,
    cfg: dict,
    dirs: PipelineDirs,
    build_signature_path: Path,
) -> None:
    windows = cfg["pairing"]["windows_days"]
    plasma_filter = cfg["pairing"].get("plasma_filter")
    plasma_source = cfg.get("pairing", {}).get("plasma_source")
    filter_label = "nofilter"
    if plasma_filter and isinstance(plasma_filter, dict) and plasma_filter.get("ASSAYVERSION"):
        filter_label = str(plasma_filter["ASSAYVERSION"]).strip().replace(" ", "_")

    release_id = cfg.get("release", {}).get("release_id") if isinstance(cfg.get("release"), dict) else None
    if not release_id:
        release_id = f"adni_core_{today_iso()}_{filter_label}_v1"

    git = get_git_info(Path.cwd())
    build_signature = build_signature_path.read_text(encoding="utf-8") if build_signature_path.exists() else ""

    triad_path = dirs.core_paired_dir / "triad_pet_anchored.csv"
    triad = pd.read_csv(triad_path, usecols=["RID"])

    stamp = {
        "release_id": release_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git": {"is_repo": git.is_repo, "commit": git.commit, "is_dirty": git.is_dirty},
        "inputs": {"raw_root": cfg.get("raw_root"), "sources": cfg.get("sources", {})},
        "pairing": {"windows_days": windows, "plasma_filter": plasma_filter, "plasma_source": plasma_source},
        "definitions": {
            "pet_qc_rule": "Dedup by (RID,SCANDATE,TRACER); prefer qc_flag pass, then latest PROCESSDATE, then latest update_stamp, then latest source_row.",
            "pet_positivity_definition": "No additional thresholding applied in evidence core; includes UCB fields AMYLOID_STATUS and CENTILOIDS for downstream definitions.",
            "csf_positivity_definition": "No additional thresholding applied in evidence core; includes Elecsys ABETA42, ABETA40 and derived ratios for downstream cutpoints.",
            "event_id_definitions": {
                "pet_event_id": "sha256('pet|RID|SCANDATE|TRACER')",
                "csf_event_id": "sha256('csf|RID|EXAMDATE|RUNDATE|BATCH')",
                "plasma_event_id": "sha256('plasma|RID|EXAMDATE|SAMPLEID|ASSAYPLATFORM|ASSAYVERSION|PERFORMINGLAB|MATRIX')",
                "clin_event_id": "sha256('clin|RID|examdate|VISCODE')",
            },
        },
        "outputs": {
            "triad": {"n_rows": int(triad.shape[0]), "n_unique_rid": int(triad["RID"].nunique(dropna=True))},
            "paths": {
                "core_dir": dirs.core_dir.as_posix(),
                "audit_dir": dirs.audit_dir.as_posix(),
                "manifests_dir": dirs.manifests_dir.as_posix(),
                "schema_registry_dir": (dirs.manifests_dir / "schema_registry").as_posix(),
            },
        },
        "build_signature_txt": build_signature,
    }

    out_path.write_text(json.dumps(stamp, indent=2, sort_keys=True) + "\n", encoding="utf-8")
