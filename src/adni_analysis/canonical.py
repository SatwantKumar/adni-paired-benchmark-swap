from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from adni_analysis.utils import (
    PipelineDirs,
    add_provenance,
    add_row_uid,
    ensure_dir,
    make_event_id,
    normalize_label,
    parse_date_series,
)


def _qc_rank(value: object) -> int:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0
    s = str(value).strip().lower()
    if s in {"pass", "passed", "p"}:
        return 2
    if s in {"fail", "failed", "f"}:
        return -1
    try:
        return int(float(s))
    except ValueError:
        return 0


def build_pet_ucb_amy_6mm(*, in_path: Path, out_path: Path, audit_qc_dir: Path) -> None:
    df = pd.read_csv(in_path, low_memory=False)
    df = add_provenance(df, in_path)
    df = add_row_uid(df)

    df["SCANDATE"], df["SCANDATE_raw"] = parse_date_series(df["SCANDATE"])
    df["PROCESSDATE"], df["PROCESSDATE_raw"] = parse_date_series(df["PROCESSDATE"])

    df["_qc_rank"] = df["qc_flag"].map(_qc_rank).astype("int64")

    key_cols = ["RID", "SCANDATE", "TRACER"]
    dup_mask = df.duplicated(subset=key_cols, keep=False)
    dup_groups = df.loc[dup_mask, key_cols + ["qc_flag", "_qc_rank", "PROCESSDATE", "update_stamp", "row_uid", "source_row"]]
    if not dup_groups.empty:
        dup_summary = (
            dup_groups.groupby(key_cols, dropna=False)
            .size()
            .reset_index(name="n_rows")
            .sort_values(["n_rows"], ascending=False)
        )
        dup_summary.to_csv(audit_qc_dir / "pet_duplicate_events.csv", index=False)

    df = df.sort_values(
        key_cols + ["_qc_rank", "PROCESSDATE", "update_stamp", "source_row"],
        ascending=[True, True, True, False, False, False, False],
        kind="mergesort",
    )
    df = df.groupby(key_cols, dropna=False, as_index=False).head(1).reset_index(drop=True)
    df = df.drop(columns=["_qc_rank"])

    df["pet_event_id"] = make_event_id(df, ["RID", "SCANDATE", "TRACER"], prefix="pet")

    now = datetime.now(timezone.utc).date()
    min_ok = pd.Timestamp("1990-01-01")
    max_ok = pd.Timestamp(now) + pd.Timedelta(days=366)
    bad_dates = df.loc[(df["SCANDATE"].notna()) & ((df["SCANDATE"] < min_ok) | (df["SCANDATE"] > max_ok)), key_cols + ["SCANDATE_raw", "row_uid"]]
    if not bad_dates.empty:
        bad_dates.to_csv(audit_qc_dir / "pet_date_anomalies.csv", index=False)

    cent = pd.to_numeric(df.get("CENTILOIDS"), errors="coerce")
    bad_cent = df.loc[cent.notna() & ((cent < -50) | (cent > 200)), ["RID", "SCANDATE", "TRACER", "CENTILOIDS", "row_uid"]]
    if not bad_cent.empty:
        bad_cent.to_csv(audit_qc_dir / "pet_centiloid_extremes.csv", index=False)

    keep_cols = [
        "RID",
        "PTID",
        "VISCODE",
        "VISCODE2",
        "pet_event_id",
        "SCANDATE",
        "SCANDATE_raw",
        "TRACER",
        "SITEID",
        "qc_flag",
        "TRACER_SUVR_WARNING",
        "IMAGE_RESOLUTION",
        "PROCESSDATE",
        "PROCESSDATE_raw",
        "AMYLOID_STATUS",
        "AMYLOID_STATUS_COMPOSITE_REF",
        "CENTILOIDS",
        "SUMMARY_SUVR",
        "COMPOSITE_REF_SUVR",
        "CSF_SUVR",
        "update_stamp",
        "source_file",
        "source_row",
        "row_uid",
    ]
    existing = [c for c in keep_cols if c in df.columns]
    df.loc[:, existing].to_csv(out_path, index=False)


def build_csf_elecsys_upenn(*, in_path: Path, out_path: Path, audit_qc_dir: Path) -> None:
    df = pd.read_csv(in_path, low_memory=False)
    df = add_provenance(df, in_path)
    df = add_row_uid(df)

    df["EXAMDATE"], df["EXAMDATE_raw"] = parse_date_series(df["EXAMDATE"])
    df["RUNDATE"], df["RUNDATE_raw"] = parse_date_series(df["RUNDATE"])

    for col in ["ABETA40", "ABETA42", "PTAU", "TAU"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["abeta42_40_ratio"] = df["ABETA42"] / df["ABETA40"]
    df["ptau_abeta42_ratio"] = df["PTAU"] / df["ABETA42"]
    df["ttau_abeta42_ratio"] = df["TAU"] / df["ABETA42"]

    key_cols = ["RID", "EXAMDATE"]
    dup_mask = df.duplicated(subset=key_cols, keep=False)
    if dup_mask.any():
        dup_summary = (
            df.loc[dup_mask, key_cols + ["RUNDATE", "row_uid", "source_row"]]
            .groupby(key_cols, dropna=False)
            .size()
            .reset_index(name="n_rows")
            .sort_values(["n_rows"], ascending=False)
        )
        dup_summary.to_csv(audit_qc_dir / "csf_duplicate_events.csv", index=False)

    df = df.sort_values(
        key_cols + ["RUNDATE", "update_stamp", "source_row"],
        ascending=[True, True, False, False, False],
        kind="mergesort",
    )
    df = df.groupby(key_cols, dropna=False, as_index=False).head(1).reset_index(drop=True)

    csf_id_cols = ["RID", "EXAMDATE"]
    if "RUNDATE" in df.columns:
        csf_id_cols.append("RUNDATE")
    if "BATCH" in df.columns:
        csf_id_cols.append("BATCH")
    df["csf_event_id"] = make_event_id(df, csf_id_cols, prefix="csf")

    bad_den = df.loc[df["ABETA40"].notna() & (df["ABETA40"] <= 0), ["RID", "EXAMDATE", "ABETA40", "row_uid"]]
    if not bad_den.empty:
        bad_den.to_csv(audit_qc_dir / "csf_abeta40_nonpositive.csv", index=False)

    keep_cols = [
        "RID",
        "PTID",
        "VISCODE2",
        "PHASE",
        "csf_event_id",
        "EXAMDATE",
        "EXAMDATE_raw",
        "RUNDATE",
        "RUNDATE_raw",
        "BATCH",
        "COMMENT",
        "ABETA40",
        "ABETA42",
        "PTAU",
        "TAU",
        "abeta42_40_ratio",
        "ptau_abeta42_ratio",
        "ttau_abeta42_ratio",
        "update_stamp",
        "source_file",
        "source_row",
        "row_uid",
    ]
    existing = [c for c in keep_cols if c in df.columns]
    df.loc[:, existing].to_csv(out_path, index=False)


@dataclass(frozen=True)
class PlasmaFnibcOutputs:
    long_path: Path
    wide_path: Path
    analytes_csv: Path


def build_plasma_fnibc(
    *,
    in_path: Path,
    out_long_path: Path,
    out_wide_path: Path,
    core_dict_dir: Path,
    audit_validations_dir: Path,
    audit_qc_dir: Path,
) -> PlasmaFnibcOutputs:
    df = pd.read_csv(in_path, low_memory=False)
    df = add_provenance(df, in_path)
    df = add_row_uid(df)

    df["EXAMDATE"], df["EXAMDATE_raw"] = parse_date_series(df["EXAMDATE"])
    df["RUNDATE"], df["RUNDATE_raw"] = parse_date_series(df["RUNDATE"])

    df["TESTVALUE"] = pd.to_numeric(df["TESTVALUE"], errors="coerce")
    df["CV_num"] = pd.to_numeric(df["CV"], errors="coerce")

    df["analyte_source"] = df["PLASMA_BIOMARKER"].astype("string").fillna("") + "__" + df["TESTNAME"].astype("string").fillna("")
    df["analyte_std"] = df["analyte_source"].map(normalize_label).astype("string")
    df["platform_std"] = (
        df["ASSAYPLATFORM"].astype("string").fillna("") + "__" + df["ASSAYVERSION"].astype("string").fillna("")
    ).map(normalize_label)

    plasma_id_cols = ["RID", "EXAMDATE", "SAMPLEID", "ASSAYPLATFORM", "ASSAYVERSION", "PERFORMINGLAB", "MATRIX"]
    df["plasma_event_id"] = make_event_id(df, plasma_id_cols, prefix="plasma")

    unit_incons = (
        df.groupby(["analyte_std"], dropna=False)["UNITS"]
        .nunique(dropna=True)
        .reset_index(name="n_unique_units")
        .query("n_unique_units > 1")
    )
    if not unit_incons.empty:
        unit_incons.to_csv(audit_qc_dir / "plasma_fnibc_unit_inconsistencies.csv", index=False)

    ensure_dir(core_dict_dir)
    analytes_csv = core_dict_dir / "plasma_fnibc_analytes.csv"
    (
        df.groupby(["ASSAYPLATFORM", "ASSAYVERSION", "PLASMA_BIOMARKER", "TESTNAME", "UNITS", "analyte_std"], dropna=False)
        .size()
        .reset_index(name="n_rows")
        .sort_values(["n_rows"], ascending=False)
        .to_csv(analytes_csv, index=False)
    )

    long_keep = [
        "RID",
        "PTID",
        "VISCODE2",
        "plasma_event_id",
        "EXAMDATE",
        "EXAMDATE_raw",
        "SAMPLEID",
        "MATRIX",
        "ASSAYPLATFORM",
        "ASSAYVERSION",
        "PERFORMINGLAB",
        "INSTRUMENTID",
        "RUN_NO",
        "RUNDATE",
        "RUNDATE_raw",
        "SEQUENCE_NO",
        "STUDY_ID",
        "PLASMA_BIOMARKER",
        "TESTNAME",
        "TESTVALUE",
        "UNITS",
        "CV",
        "COMMENTS",
        "analyte_source",
        "analyte_std",
        "platform_std",
        "update_stamp",
        "source_file",
        "source_row",
        "row_uid",
    ]
    existing_long = [c for c in long_keep if c in df.columns]
    df.loc[:, existing_long].to_csv(out_long_path, index=False)

    key_cols = ["RID", "EXAMDATE", "SAMPLEID", "ASSAYPLATFORM", "ASSAYVERSION", "PERFORMINGLAB", "MATRIX"]
    for col in key_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column for plasma wide pivot: {col}")

    pick_sort_cols = ["CV_num", "RUNDATE", "SEQUENCE_NO", "source_row"]
    for c in pick_sort_cols:
        if c not in df.columns:
            df[c] = pd.NA

    df = df.sort_values(
        key_cols + ["analyte_std"] + pick_sort_cols,
        ascending=[True, True, True, True, True, True, True, True, True, False, True, False],
        kind="mergesort",
    )
    dup_mask = df.duplicated(subset=key_cols + ["analyte_std"], keep=False)
    if dup_mask.any():
        collision_summary = (
            df.loc[dup_mask, key_cols + ["analyte_std"]]
            .value_counts()
            .reset_index(name="n_rows")
            .sort_values(["n_rows"], ascending=False)
        )
        collision_summary.to_csv(audit_validations_dir / "plasma_pivot_collisions.csv", index=False)

    df_best = df.drop_duplicates(subset=key_cols + ["analyte_std"], keep="first").copy()

    meta = df_best[key_cols + ["PTID", "VISCODE2"]].drop_duplicates(subset=key_cols, keep="first")
    values = df_best.pivot(index=key_cols, columns="analyte_std", values="TESTVALUE").reset_index()

    row_uids = (
        df_best.groupby(key_cols, dropna=False)["row_uid"]
        .apply(lambda s: ";".join(sorted({str(x) for x in s.dropna().astype(str)})))
        .reset_index(name="source_row_uids")
    )

    wide = values.merge(meta, on=key_cols, how="left").merge(row_uids, on=key_cols, how="left")
    wide["plasma_event_id"] = make_event_id(wide, key_cols, prefix="plasma")
    wide["source_file"] = str(in_path)
    wide = add_row_uid(wide)
    wide.to_csv(out_wide_path, index=False)

    return PlasmaFnibcOutputs(long_path=out_long_path, wide_path=out_wide_path, analytes_csv=analytes_csv)


def build_plasma_janssen_ptau217(*, in_path: Path, out_path: Path, audit_qc_dir: Path) -> None:
    df = pd.read_csv(in_path, low_memory=False)
    df = add_provenance(df, in_path)
    df = add_row_uid(df)

    df["EXAMDATE"], df["EXAMDATE_raw"] = parse_date_series(df["EXAMDATE"])
    if "DILUTION_CORRECTED_CONC" in df.columns:
        df["DILUTION_CORRECTED_CONC"] = pd.to_numeric(df["DILUTION_CORRECTED_CONC"], errors="coerce")
    df["CV_num"] = pd.to_numeric(df.get("CV"), errors="coerce")

    id_cols = ["RID", "EXAMDATE"]
    if "SAMPLE_ID" in df.columns:
        id_cols.append("SAMPLE_ID")
    if "RUN" in df.columns:
        id_cols.append("RUN")
    df["plasma_event_id"] = make_event_id(df, id_cols, prefix="plasma_janssen")

    key_cols = ["RID", "EXAMDATE"]
    dup_mask = df.duplicated(subset=key_cols, keep=False)
    if dup_mask.any():
        (
            df.loc[dup_mask, key_cols]
            .value_counts()
            .reset_index(name="n_rows")
            .sort_values(["n_rows"], ascending=False)
            .to_csv(audit_qc_dir / "plasma_janssen_duplicate_events.csv", index=False)
        )

    keep_cols = [
        "RID",
        "PTID",
        "VISCODE2",
        "plasma_event_id",
        "EXAMDATE",
        "EXAMDATE_raw",
        "SAMPLE_ID",
        "RUN",
        "DILUTION_CORRECTED_CONC",
        "CV",
        "update_stamp",
        "source_file",
        "source_row",
        "row_uid",
    ]
    existing = [c for c in keep_cols if c in df.columns]
    df.loc[:, existing].to_csv(out_path, index=False)


def build_plasma_c2n_precivityad2(
    *,
    in_path: Path,
    out_path: Path,
    audit_qc_dir: Path,
) -> None:
    df = pd.read_csv(in_path, low_memory=False)
    df = add_provenance(df, in_path)
    df = add_row_uid(df)

    df["EXAMDATE"], df["EXAMDATE_raw"] = parse_date_series(df["EXAMDATE"])

    numeric_cols = [
        "pT217_C2N",
        "npT217_C2N",
        "AB42_C2N",
        "AB40_C2N",
        "AB42_AB40_C2N",
        "pT217_npT217_C2N",
        "APS2_C2N",
    ]
    sentinel = -4
    sentinel_rows: list[dict] = []
    for col in numeric_cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        n_sentinel = int((s == sentinel).sum())
        sentinel_rows.append({"field": col, "sentinel": sentinel, "n_rows": n_sentinel})
        s = s.mask(s == sentinel, pd.NA)
        df[col] = s

    if sentinel_rows:
        pd.DataFrame.from_records(sentinel_rows).to_csv(
            audit_qc_dir / "plasma_c2n_precivityad2_sentinel_missing.csv", index=False
        )

    # Standardize key assay metadata to match pairing expectations.
    df["ASSAYPLATFORM"] = "C2N"
    df["ASSAYVERSION"] = "PrecivityAD2"
    df["PERFORMINGLAB"] = "C2N"
    df["MATRIX"] = "PLASMA"

    df["SAMPLEID"] = (
        df.get("Primary", pd.Series([pd.NA] * len(df))).astype("string").fillna("")
        + "_"
        + df.get("Additive", pd.Series([pd.NA] * len(df))).astype("string").fillna("")
    ).str.strip("_")

    df["plasma_event_id"] = make_event_id(
        df,
        ["RID", "EXAMDATE", "SAMPLEID", "ASSAYPLATFORM", "ASSAYVERSION", "PERFORMINGLAB", "MATRIX"],
        prefix="plasma",
    )

    # Add standardized component fields to align with the FNIHBC wide naming convention.
    rename_map = {
        "pT217_C2N": "c2n_plasma_ptau217_ptau217",
        "npT217_C2N": "c2n_plasma_nptau217_nptau217",
        "AB42_C2N": "c2n_plasma_abeta42_abeta42",
        "AB40_C2N": "c2n_plasma_abeta40_abeta40",
        "AB42_AB40_C2N": "c2n_plasma_abeta42_abeta40_abeta_ratio",
        "pT217_npT217_C2N": "c2n_plasma_ptau217_ratio_ptau217_ratio",
    }
    for raw_col, std_col in rename_map.items():
        if raw_col in df.columns and std_col not in df.columns:
            df[std_col] = df[raw_col]

    aps2 = pd.to_numeric(df.get("APS2_C2N"), errors="coerce")
    bad_aps2 = df.loc[aps2.notna() & ((aps2 < 0) | (aps2 > 100)), ["RID", "EXAMDATE", "APS2_C2N", "row_uid"]]
    if not bad_aps2.empty:
        bad_aps2.to_csv(audit_qc_dir / "plasma_c2n_precivityad2_aps2_out_of_range.csv", index=False)

    keep_cols = [
        "RID",
        "PTID",
        "VISCODE",
        "VISCODE2",
        "PHASE",
        "plasma_event_id",
        "EXAMDATE",
        "EXAMDATE_raw",
        "Primary",
        "Additive",
        "SAMPLEID",
        "ASSAYPLATFORM",
        "ASSAYVERSION",
        "PERFORMINGLAB",
        "MATRIX",
        "APS2_C2N",
        "APOE_C2N",
        "Comments",
        # standardized components
        "c2n_plasma_abeta40_abeta40",
        "c2n_plasma_abeta42_abeta42",
        "c2n_plasma_abeta42_abeta40_abeta_ratio",
        "c2n_plasma_ptau217_ptau217",
        "c2n_plasma_nptau217_nptau217",
        "c2n_plasma_ptau217_ratio_ptau217_ratio",
        # provenance
        "update_stamp",
        "source_file",
        "source_row",
        "row_uid",
    ]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = pd.NA
    df.loc[:, keep_cols].to_csv(out_path, index=False)


def build_clinical_adnimerge(*, in_path: Path, out_path: Path, audit_qc_dir: Path, ptdemog_path: Path | None = None) -> None:
    df = pd.read_csv(in_path, low_memory=False)
    df = add_provenance(df, in_path)
    df = add_row_uid(df)

    if ptdemog_path is not None:
        pt = pd.read_csv(ptdemog_path, low_memory=False)
        if "VISDATE" in pt.columns and "RID" in pt.columns:
            pt = add_provenance(pt, ptdemog_path)
            pt = add_row_uid(pt)
            pt["VISDATE"], pt["VISDATE_raw"] = parse_date_series(pt["VISDATE"])

            # One baseline-like row per RID (earliest VISDATE).
            pt = pt.sort_values(["RID", "VISDATE", "source_row"], ascending=[True, True, True], kind="mergesort")
            pt = pt.groupby(["RID"], dropna=False, as_index=False).head(1).reset_index(drop=True)

            # Keep only RIDs missing from ADNIMERGE.
            adni_rids = set(df["RID"].dropna().astype(int).tolist())
            pt_rids = pt["RID"].dropna().astype(int)
            pt = pt[~pt_rids.isin(adni_rids)].copy()

            if not pt.empty:
                # Map PTDEMOG fields into the clinical schema (minimal coverage rows).
                pt = pt.rename(columns={"VISDATE": "EXAMDATE", "SITEID": "SITE"})
                pt = pt.reindex(columns=df.columns)
                df = pd.concat([df, pt], ignore_index=True)

    df["EXAMDATE"], _examdate_raw = parse_date_series(df["EXAMDATE"])
    df["EXAMDATE_bl"], _examdate_bl_raw = parse_date_series(df["EXAMDATE_bl"]) if "EXAMDATE_bl" in df.columns else (pd.NaT, pd.NA)

    df["examdate"] = df["EXAMDATE"].dt.strftime("%Y-%m-%d").astype("string")

    years_bl = pd.to_numeric(df["Years_bl"], errors="coerce") if "Years_bl" in df.columns else pd.Series([pd.NA] * len(df))
    visit_months = years_bl * 12
    if "EXAMDATE_bl" in df.columns:
        months_from_dates = (df["EXAMDATE"] - df["EXAMDATE_bl"]).dt.days / 30.4375
        visit_months = visit_months.fillna(months_from_dates)
    df["visit_months_from_bl"] = visit_months

    df["protocol_group"] = df["COLPROT"].astype("string").str.strip()

    def dx_simplify(value: object) -> object:
        if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
            return pd.NA
        s = str(value).strip()
        if not s:
            return pd.NA
        s_upper = s.upper()
        if s_upper in {"CN", "NORMAL", "NC", "SMC"}:
            return "CN"
        if "MCI" in s_upper:
            return "MCI"
        if s_upper in {"AD", "DEMENTIA"} or "DEMENTIA" in s_upper:
            return "AD"
        return "Other"

    df["dx_simplified"] = df["DX"].map(dx_simplify).astype("string")
    df["clin_event_id"] = make_event_id(df, ["RID", "examdate", "VISCODE"], prefix="clin")

    keep_cols = [
        # Identity + timing
        "RID",
        "PTID",
        "VISCODE",
        "EXAMDATE",
        "COLPROT",
        "ORIGPROT",
        "SITE",
        "clin_event_id",
        # Demographics
        "AGE",
        "PTGENDER",
        "PTEDUCAT",
        "PTETHCAT",
        "PTRACCAT",
        "APOE4",
        # Diagnosis/stage anchors
        "DX",
        "DX_bl",
        "CDRSB",
        "MMSE",
        "MOCA",
        "FAQ",
        # Cognitive composites (optional but useful)
        "ADAS13",
        "mPACCdigit",
        "mPACCtrailsB",
        # Modality presence flags
        "AV45",
        "FBB",
        "PIB",
        "FDG",
        "ABETA",
        "TAU",
        "PTAU",
        # ADNI-provided time-from-baseline columns
        "Years_bl",
        "Month_bl",
        "Month",
        # Derived
        "examdate",
        "visit_months_from_bl",
        "protocol_group",
        "dx_simplified",
        # Provenance
        "update_stamp",
        "source_file",
        "source_row",
        "row_uid",
    ]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = pd.NA
    df.loc[:, keep_cols].to_csv(out_path, index=False)

    def uniqueness_row(keys: list[str], label: str) -> dict:
        key_df = df.loc[:, keys]
        n_rows = int(len(df))
        n_unique = int(key_df.drop_duplicates().shape[0])
        counts = key_df.value_counts(dropna=False)
        n_keys_with_dups = int((counts > 1).sum()) if not counts.empty else 0
        max_mult = int(counts.max()) if not counts.empty else 0
        return {
            "key": label,
            "n_rows": n_rows,
            "n_unique_keys": n_unique,
            "n_duplicate_rows": n_rows - n_unique,
            "n_keys_with_duplicates": n_keys_with_dups,
            "max_multiplicity": max_mult,
        }

    uniq = pd.DataFrame.from_records(
        [
            uniqueness_row(["RID", "EXAMDATE"], "RID+EXAMDATE"),
            uniqueness_row(["RID", "VISCODE"], "RID+VISCODE"),
        ]
    )
    uniq.to_csv(audit_qc_dir / "adnimerge_key_uniqueness.csv", index=False)
