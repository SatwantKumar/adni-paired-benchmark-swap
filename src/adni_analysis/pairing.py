from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairingWindows:
    pet_plasma: int
    pet_csf: int
    plasma_csf: int


def _abs_days(delta: pd.Series) -> pd.Series:
    return delta.abs() / np.timedelta64(1, "D")


def _filter_exact(df: pd.DataFrame, filters: dict[str, str] | None) -> pd.DataFrame:
    if not filters:
        return df
    out = df.copy()
    for col, wanted in filters.items():
        if col not in out.columns:
            raise ValueError(f"plasma_filter references missing column: {col}")
        out = out[out[col].astype("string").str.strip() == str(wanted).strip()]
    return out


def build_dyad_pet_plasma(
    *,
    pet: pd.DataFrame,
    plasma: pd.DataFrame,
    windows: PairingWindows,
    plasma_filter: dict[str, str] | None,
) -> pd.DataFrame:
    plasma = _filter_exact(plasma, plasma_filter)

    pet = pet[["RID", "pet_event_id", "SCANDATE", "TRACER", "row_uid", "source_file"]].rename(
        columns={"row_uid": "pet_row_uid", "source_file": "pet_source_file"}
    )
    plasma = plasma[
        [
            "RID",
            "plasma_event_id",
            "EXAMDATE",
            "ASSAYPLATFORM",
            "ASSAYVERSION",
            "PERFORMINGLAB",
            "MATRIX",
            "row_uid",
            "source_file",
        ]
    ].rename(columns={"row_uid": "plasma_row_uid", "source_file": "plasma_source_file"})

    out_rows: list[dict] = []
    for rid, pet_g in pet.groupby("RID", dropna=False):
        plasma_g = plasma[plasma["RID"] == rid]
        if plasma_g.empty:
            for _, r in pet_g.iterrows():
                out_rows.append(
                    {
                        "RID": rid,
                        "pet_event_id": r["pet_event_id"],
                        "pet_date": r["SCANDATE"],
                        "TRACER": r["TRACER"],
                        "pet_row_uid": r["pet_row_uid"],
                        "pet_source_file": r["pet_source_file"],
                        "plasma_date": pd.NaT,
                        "plasma_event_id": pd.NA,
                        "plasma_row_uid": pd.NA,
                        "plasma_source_file": pd.NA,
                        "delta_days_abs": pd.NA,
                    }
                )
            continue

        plasma_dates = plasma_g["EXAMDATE"]
        for _, r in pet_g.iterrows():
            if pd.isna(r["SCANDATE"]):
                continue
            deltas = (plasma_dates - r["SCANDATE"]).astype("timedelta64[ns]")
            abs_days = _abs_days(deltas)
            ok = abs_days <= windows.pet_plasma
            if not ok.any():
                out_rows.append(
                    {
                        "RID": rid,
                        "pet_event_id": r["pet_event_id"],
                        "pet_date": r["SCANDATE"],
                        "TRACER": r["TRACER"],
                        "pet_row_uid": r["pet_row_uid"],
                        "pet_source_file": r["pet_source_file"],
                        "plasma_date": pd.NaT,
                        "plasma_event_id": pd.NA,
                        "plasma_row_uid": pd.NA,
                        "plasma_source_file": pd.NA,
                        "delta_days_abs": pd.NA,
                    }
                )
                continue

            candidates = plasma_g.loc[ok].copy()
            candidates["abs_days"] = abs_days[ok].values
            candidates = candidates.sort_values(["abs_days", "EXAMDATE", "plasma_row_uid"], kind="mergesort")
            best = candidates.iloc[0]
            out_rows.append(
                {
                    "RID": rid,
                    "pet_event_id": r["pet_event_id"],
                    "pet_date": r["SCANDATE"],
                    "TRACER": r["TRACER"],
                    "pet_row_uid": r["pet_row_uid"],
                    "pet_source_file": r["pet_source_file"],
                    "plasma_date": best["EXAMDATE"],
                    "plasma_event_id": best["plasma_event_id"],
                    "ASSAYPLATFORM": best["ASSAYPLATFORM"],
                    "ASSAYVERSION": best["ASSAYVERSION"],
                    "PERFORMINGLAB": best["PERFORMINGLAB"],
                    "MATRIX": best["MATRIX"],
                    "plasma_row_uid": best["plasma_row_uid"],
                    "plasma_source_file": best["plasma_source_file"],
                    "delta_days_abs": float(best["abs_days"]),
                }
            )

    return pd.DataFrame(out_rows)


def build_dyad_pet_csf(*, pet: pd.DataFrame, csf: pd.DataFrame, windows: PairingWindows) -> pd.DataFrame:
    pet = pet[["RID", "pet_event_id", "SCANDATE", "TRACER", "row_uid", "source_file"]].rename(
        columns={"row_uid": "pet_row_uid", "source_file": "pet_source_file"}
    )
    csf = csf[["RID", "csf_event_id", "EXAMDATE", "row_uid", "source_file"]].rename(
        columns={"row_uid": "csf_row_uid", "source_file": "csf_source_file"}
    )

    out_rows: list[dict] = []
    for rid, pet_g in pet.groupby("RID", dropna=False):
        csf_g = csf[csf["RID"] == rid]
        if csf_g.empty:
            for _, r in pet_g.iterrows():
                out_rows.append(
                    {
                        "RID": rid,
                        "pet_event_id": r["pet_event_id"],
                        "pet_date": r["SCANDATE"],
                        "TRACER": r["TRACER"],
                        "pet_row_uid": r["pet_row_uid"],
                        "pet_source_file": r["pet_source_file"],
                        "csf_date": pd.NaT,
                        "csf_event_id": pd.NA,
                        "csf_row_uid": pd.NA,
                        "csf_source_file": pd.NA,
                        "delta_days_abs": pd.NA,
                    }
                )
            continue

        csf_dates = csf_g["EXAMDATE"]
        for _, r in pet_g.iterrows():
            if pd.isna(r["SCANDATE"]):
                continue
            deltas = (csf_dates - r["SCANDATE"]).astype("timedelta64[ns]")
            abs_days = _abs_days(deltas)
            ok = abs_days <= windows.pet_csf
            if not ok.any():
                out_rows.append(
                    {
                        "RID": rid,
                        "pet_event_id": r["pet_event_id"],
                        "pet_date": r["SCANDATE"],
                        "TRACER": r["TRACER"],
                        "pet_row_uid": r["pet_row_uid"],
                        "pet_source_file": r["pet_source_file"],
                        "csf_date": pd.NaT,
                        "csf_event_id": pd.NA,
                        "csf_row_uid": pd.NA,
                        "csf_source_file": pd.NA,
                        "delta_days_abs": pd.NA,
                    }
                )
                continue

            candidates = csf_g.loc[ok].copy()
            candidates["abs_days"] = abs_days[ok].values
            candidates = candidates.sort_values(["abs_days", "EXAMDATE", "csf_row_uid"], kind="mergesort")
            best = candidates.iloc[0]
            out_rows.append(
                {
                    "RID": rid,
                    "pet_event_id": r["pet_event_id"],
                    "pet_date": r["SCANDATE"],
                    "TRACER": r["TRACER"],
                    "pet_row_uid": r["pet_row_uid"],
                    "pet_source_file": r["pet_source_file"],
                    "csf_date": best["EXAMDATE"],
                    "csf_event_id": best["csf_event_id"],
                    "csf_row_uid": best["csf_row_uid"],
                    "csf_source_file": best["csf_source_file"],
                    "delta_days_abs": float(best["abs_days"]),
                }
            )

    return pd.DataFrame(out_rows)


def build_dyad_csf_plasma(
    *,
    csf: pd.DataFrame,
    plasma: pd.DataFrame,
    windows: PairingWindows,
    plasma_filter: dict[str, str] | None,
) -> pd.DataFrame:
    plasma = _filter_exact(plasma, plasma_filter)

    csf = csf[["RID", "csf_event_id", "EXAMDATE", "row_uid", "source_file"]].rename(
        columns={"row_uid": "csf_row_uid", "source_file": "csf_source_file"}
    )
    plasma = plasma[
        [
            "RID",
            "plasma_event_id",
            "EXAMDATE",
            "ASSAYPLATFORM",
            "ASSAYVERSION",
            "PERFORMINGLAB",
            "MATRIX",
            "row_uid",
            "source_file",
        ]
    ].rename(columns={"row_uid": "plasma_row_uid", "source_file": "plasma_source_file"})

    out_rows: list[dict] = []
    for rid, csf_g in csf.groupby("RID", dropna=False):
        plasma_g = plasma[plasma["RID"] == rid]
        if plasma_g.empty:
            for _, r in csf_g.iterrows():
                out_rows.append(
                    {
                        "RID": rid,
                        "csf_event_id": r["csf_event_id"],
                        "csf_date": r["EXAMDATE"],
                        "csf_row_uid": r["csf_row_uid"],
                        "csf_source_file": r["csf_source_file"],
                        "plasma_date": pd.NaT,
                        "plasma_event_id": pd.NA,
                        "plasma_row_uid": pd.NA,
                        "plasma_source_file": pd.NA,
                        "delta_days_abs": pd.NA,
                    }
                )
            continue

        plasma_dates = plasma_g["EXAMDATE"]
        for _, r in csf_g.iterrows():
            if pd.isna(r["EXAMDATE"]):
                continue
            deltas = (plasma_dates - r["EXAMDATE"]).astype("timedelta64[ns]")
            abs_days = _abs_days(deltas)
            ok = abs_days <= windows.plasma_csf
            if not ok.any():
                out_rows.append(
                    {
                        "RID": rid,
                        "csf_event_id": r["csf_event_id"],
                        "csf_date": r["EXAMDATE"],
                        "csf_row_uid": r["csf_row_uid"],
                        "csf_source_file": r["csf_source_file"],
                        "plasma_date": pd.NaT,
                        "plasma_event_id": pd.NA,
                        "plasma_row_uid": pd.NA,
                        "plasma_source_file": pd.NA,
                        "delta_days_abs": pd.NA,
                    }
                )
                continue

            candidates = plasma_g.loc[ok].copy()
            candidates["abs_days"] = abs_days[ok].values
            candidates = candidates.sort_values(["abs_days", "EXAMDATE", "plasma_row_uid"], kind="mergesort")
            best = candidates.iloc[0]
            out_rows.append(
                {
                    "RID": rid,
                    "csf_event_id": r["csf_event_id"],
                    "csf_date": r["EXAMDATE"],
                    "csf_row_uid": r["csf_row_uid"],
                    "csf_source_file": r["csf_source_file"],
                    "plasma_date": best["EXAMDATE"],
                    "plasma_event_id": best["plasma_event_id"],
                    "ASSAYPLATFORM": best["ASSAYPLATFORM"],
                    "ASSAYVERSION": best["ASSAYVERSION"],
                    "PERFORMINGLAB": best["PERFORMINGLAB"],
                    "MATRIX": best["MATRIX"],
                    "plasma_row_uid": best["plasma_row_uid"],
                    "plasma_source_file": best["plasma_source_file"],
                    "delta_days_abs": float(best["abs_days"]),
                }
            )

    return pd.DataFrame(out_rows)


def build_triad_pet_anchored(
    *,
    pet: pd.DataFrame,
    csf: pd.DataFrame,
    plasma: pd.DataFrame,
    windows: PairingWindows,
    plasma_filter: dict[str, str] | None,
) -> pd.DataFrame:
    plasma = _filter_exact(plasma, plasma_filter)

    pet_ev = pet[
        [
            "RID",
            "pet_event_id",
            "SCANDATE",
            "TRACER",
            "CENTILOIDS",
            "SUMMARY_SUVR",
            "AMYLOID_STATUS",
            "row_uid",
            "source_file",
        ]
    ].rename(columns={"row_uid": "pet_row_uid", "source_file": "pet_source_file"})
    csf_ev = csf[
        [
            "RID",
            "csf_event_id",
            "EXAMDATE",
            "ABETA42",
            "ABETA40",
            "abeta42_40_ratio",
            "PTAU",
            "TAU",
            "ptau_abeta42_ratio",
            "ttau_abeta42_ratio",
            "row_uid",
            "source_file",
        ]
    ].rename(columns={"row_uid": "csf_row_uid", "source_file": "csf_source_file"})
    plasma_ev = plasma.copy()

    out_rows: list[dict] = []

    for rid, pet_g in pet_ev.groupby("RID", dropna=False):
        csf_g = csf_ev[csf_ev["RID"] == rid].reset_index(drop=True)
        plasma_g = plasma_ev[plasma_ev["RID"] == rid].reset_index(drop=True)
        if csf_g.empty or plasma_g.empty:
            continue

        csf_dates = csf_g["EXAMDATE"]
        plasma_dates = plasma_g["EXAMDATE"]

        for _, p in pet_g.iterrows():
            pet_date = p["SCANDATE"]
            if pd.isna(pet_date):
                continue

            csf_d = _abs_days(csf_dates - pet_date).to_numpy()
            plasma_d = _abs_days(plasma_dates - pet_date).to_numpy()

            csf_ok = np.where(csf_d <= windows.pet_csf)[0]
            plasma_ok = np.where(plasma_d <= windows.pet_plasma)[0]
            if csf_ok.size == 0 or plasma_ok.size == 0:
                continue

            best_score: tuple[float, float, float, float, str, str, str, str] | None = None
            best_pair: tuple[int, int] | None = None
            for i in plasma_ok:
                plasma_date = plasma_dates.iloc[i]
                if pd.isna(plasma_date):
                    continue
                for j in csf_ok:
                    csf_date = csf_dates.iloc[j]
                    if pd.isna(csf_date):
                        continue

                    d_pp = float(abs((plasma_date - pet_date) / np.timedelta64(1, "D")))
                    d_pc = float(abs((csf_date - pet_date) / np.timedelta64(1, "D")))
                    d_bc = float(abs((plasma_date - csf_date) / np.timedelta64(1, "D")))
                    if d_bc > windows.plasma_csf:
                        continue

                    plasma_uid = str(plasma_g.iloc[i].get("row_uid", ""))
                    csf_uid = str(csf_g.iloc[j].get("csf_row_uid", ""))
                    score = (
                        max(d_pp, d_pc, d_bc),
                        d_pp + d_pc + d_bc,
                        d_pp,
                        d_pc,
                        str(plasma_date),
                        str(csf_date),
                        plasma_uid,
                        csf_uid,
                    )
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pair = (i, j)

            if best_pair is None:
                continue

            i, j = best_pair
            plasma_row = plasma_g.iloc[i]
            csf_row = csf_g.iloc[j]

            out_rows.append(
                {
                    "RID": rid,
                    "pet_event_id": p.get("pet_event_id"),
                    "pet_date": pet_date,
                    "TRACER": p.get("TRACER"),
                    "pet_row_uid": p.get("pet_row_uid"),
                    "pet_source_file": p.get("pet_source_file"),
                    "CENTILOIDS": p.get("CENTILOIDS"),
                    "SUMMARY_SUVR": p.get("SUMMARY_SUVR"),
                    "AMYLOID_STATUS": p.get("AMYLOID_STATUS"),
                    "csf_event_id": csf_row.get("csf_event_id"),
                    "csf_date": csf_row.get("EXAMDATE"),
                    "csf_row_uid": csf_row.get("csf_row_uid"),
                    "csf_source_file": csf_row.get("csf_source_file"),
                    "ABETA42": csf_row.get("ABETA42"),
                    "ABETA40": csf_row.get("ABETA40"),
                    "abeta42_40_ratio": csf_row.get("abeta42_40_ratio"),
                    "PTAU": csf_row.get("PTAU"),
                    "TAU": csf_row.get("TAU"),
                    "ptau_abeta42_ratio": csf_row.get("ptau_abeta42_ratio"),
                    "ttau_abeta42_ratio": csf_row.get("ttau_abeta42_ratio"),
                    "plasma_event_id": plasma_row.get("plasma_event_id"),
                    "plasma_date": plasma_row.get("EXAMDATE"),
                    "plasma_row_uid": plasma_row.get("row_uid"),
                    "plasma_source_file": plasma_row.get("source_file"),
                    "ASSAYPLATFORM": plasma_row.get("ASSAYPLATFORM"),
                    "ASSAYVERSION": plasma_row.get("ASSAYVERSION"),
                    "PERFORMINGLAB": plasma_row.get("PERFORMINGLAB"),
                    "MATRIX": plasma_row.get("MATRIX"),
                    "delta_pet_plasma_days": float(abs((plasma_row.get("EXAMDATE") - pet_date) / np.timedelta64(1, "D"))),
                    "delta_pet_csf_days": float(abs((csf_row.get("EXAMDATE") - pet_date) / np.timedelta64(1, "D"))),
                    "delta_plasma_csf_days": float(
                        abs((plasma_row.get("EXAMDATE") - csf_row.get("EXAMDATE")) / np.timedelta64(1, "D"))
                    ),
                }
            )

    expected_cols = [
        "RID",
        "pet_event_id",
        "pet_date",
        "TRACER",
        "pet_row_uid",
        "pet_source_file",
        "CENTILOIDS",
        "SUMMARY_SUVR",
        "AMYLOID_STATUS",
        "csf_event_id",
        "csf_date",
        "csf_row_uid",
        "csf_source_file",
        "ABETA42",
        "ABETA40",
        "abeta42_40_ratio",
        "PTAU",
        "TAU",
        "ptau_abeta42_ratio",
        "ttau_abeta42_ratio",
        "plasma_event_id",
        "plasma_date",
        "plasma_row_uid",
        "plasma_source_file",
        "ASSAYPLATFORM",
        "ASSAYVERSION",
        "PERFORMINGLAB",
        "MATRIX",
        "delta_pet_plasma_days",
        "delta_pet_csf_days",
        "delta_plasma_csf_days",
    ]
    if not out_rows:
        return pd.DataFrame(columns=expected_cols)
    df = pd.DataFrame(out_rows)
    for col in expected_cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df.loc[:, expected_cols]


def write_triad_join_report(
    *,
    out_path: Path,
    pet: pd.DataFrame,
    csf: pd.DataFrame,
    plasma: pd.DataFrame,
    dyad_pet_plasma: pd.DataFrame,
    dyad_pet_csf: pd.DataFrame,
    triads: pd.DataFrame,
    windows: PairingWindows,
    plasma_filter: dict[str, str] | None,
) -> None:
    def n(df: pd.DataFrame) -> int:
        return int(df.shape[0])

    lines: list[str] = []
    lines.append("# Triad join report")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- PET events: {n(pet)}")
    lines.append(f"- CSF events: {n(csf)}")
    lines.append(f"- Plasma wide rows: {n(plasma)}")
    if plasma_filter:
        lines.append(f"- Plasma filter: {plasma_filter}")
    lines.append("")
    lines.append("## Date ranges (min/max)")

    def _minmax(df: pd.DataFrame, col: str) -> tuple[str | None, str | None]:
        if col not in df.columns:
            return (None, None)
        s = pd.to_datetime(df[col], errors="coerce")
        if s.notna().sum() == 0:
            return (None, None)
        return (str(s.min().date()), str(s.max().date()))

    pet_min, pet_max = _minmax(pet, "SCANDATE")
    csf_min, csf_max = _minmax(csf, "EXAMDATE")
    plasma_min, plasma_max = _minmax(plasma, "EXAMDATE")
    lines.append(f"- PET `SCANDATE`: {pet_min} to {pet_max}")
    lines.append(f"- CSF `EXAMDATE`: {csf_min} to {csf_max}")
    lines.append(f"- Plasma `EXAMDATE`: {plasma_min} to {plasma_max}")

    if csf_max is not None and plasma_min is not None:
        try:
            csf_max_dt = pd.to_datetime(csf_max)
            plasma_min_dt = pd.to_datetime(plasma_min)
            gap_days = float((plasma_min_dt - csf_max_dt) / np.timedelta64(1, "D"))
            if gap_days > 0:
                lines.append(f"- Note: plasma begins {gap_days:.0f} days after the latest CSF date; triads within tight windows may be impossible.")
        except Exception:  # noqa: BLE001
            pass
    lines.append("")
    lines.append("## Windows (±days)")
    lines.append(f"- PET–plasma: {windows.pet_plasma}")
    lines.append(f"- PET–CSF: {windows.pet_csf}")
    lines.append(f"- Plasma–CSF: {windows.plasma_csf}")
    lines.append("")
    lines.append("## Matches")
    lines.append(f"- PET→plasma dyads within window: {int(dyad_pet_plasma['plasma_date'].notna().sum())}")
    lines.append(f"- PET→CSF dyads within window: {int(dyad_pet_csf['csf_date'].notna().sum())}")
    lines.append(f"- Triads (PET-anchored, all pairwise windows): {n(triads)}")
    if n(triads) > 0:
        lines.append(f"- Unique participants in triads: {int(triads['RID'].nunique())}")
        lines.append("")
        lines.append("## Time gaps (days, absolute)")
        for col in ["delta_pet_plasma_days", "delta_pet_csf_days", "delta_plasma_csf_days"]:
            s = pd.to_numeric(triads[col], errors="coerce")
            lines.append(
                f"- {col}: median={float(s.median()):.1f}, IQR=({float(s.quantile(0.25)):.1f}, {float(s.quantile(0.75)):.1f}), max={float(s.max()):.1f}"
            )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
