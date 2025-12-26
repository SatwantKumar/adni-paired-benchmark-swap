from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml

from adni_analysis.config import load_yaml
from adni_analysis.pairing import PairingWindows, build_dyad_pet_plasma
from adni_analysis.utils import ensure_dir


TriCategory = Literal["Negative", "Indeterminate", "Positive"]
Direction = Literal["higher_is_more_positive", "lower_is_more_positive"]


@dataclass(frozen=True)
class PaperAArtifacts:
    cohort_triads: Path
    plasma_thresholds: Path
    plasma_labels: Path
    metrics_summary: Path
    benchmark_swap: Path


_Z_95 = 1.959963984540054


def _wilson_ci_95(k: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + (_Z_95**2) / n
    center = (p + (_Z_95**2) / (2 * n)) / denom
    half = (_Z_95 * np.sqrt((p * (1 - p) + (_Z_95**2) / (4 * n)) / n)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (float(lo), float(hi))


def _qc_rank(value: object) -> int:
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
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


def _confusion_counts(y: np.ndarray, x: np.ndarray) -> dict[str, int]:
    tp = int(np.sum((x == 1) & (y == 1)))
    fn = int(np.sum((x == 0) & (y == 1)))
    fp = int(np.sum((x == 1) & (y == 0)))
    tn = int(np.sum((x == 0) & (y == 0)))
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn}


def _ppa_npa(counts: dict[str, int]) -> dict[str, float]:
    tp = counts["TP"]
    fn = counts["FN"]
    fp = counts["FP"]
    tn = counts["TN"]

    ppa_den = tp + fn
    npa_den = tn + fp

    ppa = float(tp / ppa_den) if ppa_den else float("nan")
    npa = float(tn / npa_den) if npa_den else float("nan")

    ppa_lo, ppa_hi = _wilson_ci_95(tp, ppa_den) if ppa_den else (float("nan"), float("nan"))
    npa_lo, npa_hi = _wilson_ci_95(tn, npa_den) if npa_den else (float("nan"), float("nan"))
    return {
        "PPA": ppa,
        "PPA_lo": float(ppa_lo),
        "PPA_hi": float(ppa_hi),
        "NPA": npa,
        "NPA_lo": float(npa_lo),
        "NPA_hi": float(npa_hi),
    }


def _require_columns(df: pd.DataFrame, cols: list[str], *, label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _as_date(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _attach_nearest_clinical(triads: pd.DataFrame, clinical: pd.DataFrame) -> pd.DataFrame:
    triads = _as_date(triads, ["pet_date"])
    clinical = _as_date(clinical, ["EXAMDATE"])

    triads = triads.copy()
    clinical = clinical.copy()
    triads["RID"] = pd.to_numeric(triads.get("RID"), errors="coerce")
    clinical["RID"] = pd.to_numeric(clinical.get("RID"), errors="coerce")
    triads = triads.dropna(subset=["RID"]).copy()
    clinical = clinical.dropna(subset=["RID"]).copy()
    triads["RID"] = triads["RID"].astype("int64")
    clinical["RID"] = clinical["RID"].astype("int64")

    clinical = clinical.copy()
    clinical = clinical.rename(columns={"EXAMDATE": "clin_examdate"})
    clinical = clinical.dropna(subset=["clin_examdate"]).copy()
    keep_cols = [c for c in clinical.columns if c not in {"row_uid", "source_file", "source_row"}]
    clinical = clinical.loc[:, keep_cols]

    rename_map: dict[str, str] = {}
    for c in clinical.columns:
        if c in {"RID", "clin_examdate"}:
            continue
        if c == "examdate":
            rename_map[c] = "clin_examdate_str"
        else:
            rename_map[c] = f"clin_{c}"
    clinical = clinical.rename(columns=rename_map)

    triads_work = triads.copy()
    triads_work["_triad_row"] = np.arange(len(triads_work), dtype="int64")
    triads_work = triads_work.dropna(subset=["pet_date"]).copy()

    # `merge_asof` requires the join keys to be sorted; with `by=RID`, Pandas expects global sorting by the time key.
    triads_sorted = triads_work.sort_values(["pet_date", "RID"], kind="mergesort")
    clinical_sorted = clinical.sort_values(["clin_examdate", "RID"], kind="mergesort")

    merged = pd.merge_asof(
        triads_sorted,
        clinical_sorted,
        left_on="pet_date",
        right_on="clin_examdate",
        by="RID",
        direction="nearest",
    )

    merged["delta_pet_clin_days"] = (
        (merged["clin_examdate"] - merged["pet_date"]).abs() / np.timedelta64(1, "D")
    ).astype("float64")
    merged = merged.sort_values(["_triad_row"], kind="mergesort").drop(columns=["_triad_row"])
    return merged


def _select_one_triad_per_rid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["RID"] = pd.to_numeric(df["RID"], errors="coerce")
    df = df.dropna(subset=["RID"]).copy()
    df["RID"] = df["RID"].astype("int64")

    csf_ratio = df["abeta42_40_ratio"] if "abeta42_40_ratio" in df.columns else df.get("csf_a_pos")
    df["_has_csf_a"] = pd.to_numeric(csf_ratio, errors="coerce").notna().astype("int64")

    pet_qc_fail = df.get("pet_qc_fail")
    if pet_qc_fail is None:
        df["_pet_qc_bad"] = 0
    else:
        df["_pet_qc_bad"] = pd.to_numeric(pet_qc_fail, errors="coerce").fillna(0).astype("int64")

    pet_centiloid_extreme = df.get("pet_centiloid_extreme")
    if pet_centiloid_extreme is None:
        df["_pet_centiloid_extreme"] = 0
    else:
        df["_pet_centiloid_extreme"] = (
            pd.to_numeric(pet_centiloid_extreme, errors="coerce").fillna(0).astype("int64")
        )

    deltas: list[pd.Series] = []
    for col in ["delta_pet_plasma_days", "delta_pet_csf_days", "delta_plasma_csf_days"]:
        if col in df.columns:
            deltas.append(pd.to_numeric(df[col], errors="coerce"))
        else:
            deltas.append(pd.Series(np.nan, index=df.index, dtype="float64"))

    delta_pet_plasma = deltas[0].fillna(np.inf)
    delta_pet_csf = deltas[1].fillna(np.inf)
    delta_plasma_csf = deltas[2].fillna(np.inf)

    df["_max_gap"] = np.maximum.reduce(
        [delta_pet_plasma.to_numpy(), delta_pet_csf.to_numpy(), delta_plasma_csf.to_numpy()]
    )
    df["_sum_gap"] = delta_pet_plasma + delta_pet_csf + delta_plasma_csf
    df["_delta_pet_plasma"] = delta_pet_plasma
    df["_delta_pet_csf"] = delta_pet_csf
    df["_delta_plasma_csf"] = delta_plasma_csf

    if "pet_date" in df.columns:
        df["_pet_date"] = pd.to_datetime(df["pet_date"], errors="coerce")
    else:
        df["_pet_date"] = pd.NaT

    if "pet_event_id" in df.columns:
        df["_pet_event_id"] = df["pet_event_id"].astype("string")
    else:
        df["_pet_event_id"] = pd.NA

    df = df.sort_values(
        [
            "RID",
            "_has_csf_a",
            "_max_gap",
            "_sum_gap",
            "_pet_qc_bad",
            "_pet_centiloid_extreme",
            "_delta_plasma_csf",
            "_delta_pet_csf",
            "_delta_pet_plasma",
            "_pet_date",
            "_pet_event_id",
        ],
        ascending=[True, False, True, True, True, True, True, True, True, True, True],
        kind="mergesort",
    )
    out = df.groupby("RID", sort=False, as_index=False).head(1).copy()
    return out.drop(
        columns=[
            "_has_csf_a",
            "_max_gap",
            "_sum_gap",
            "_pet_qc_bad",
            "_pet_centiloid_extreme",
            "_delta_pet_plasma",
            "_delta_pet_csf",
            "_delta_plasma_csf",
            "_pet_date",
            "_pet_event_id",
        ],
        errors="ignore",
    )


def _pet_pos_from_status(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    out = pd.Series(pd.NA, index=s.index, dtype="Int64")
    out[v == 1] = 1
    out[v == 0] = 0
    return out


def _derive_csf_benchmarks(df: pd.DataFrame, defs: dict) -> pd.DataFrame:
    df = df.copy()
    cut_a = float(defs["benchmarks"]["csf"]["primary_csf_a"]["cutpoint"])
    cut_t = float(defs["benchmarks"]["csf"]["secondary_csf_at"]["csf_t_component"]["cutpoint"])

    ratio = pd.to_numeric(df["abeta42_40_ratio"], errors="coerce")
    ptau = pd.to_numeric(df["PTAU"], errors="coerce")

    csf_a = pd.Series(pd.NA, index=df.index, dtype="Int64")
    csf_a[ratio.notna()] = (ratio[ratio.notna()] < cut_a).astype("int64")

    csf_t = pd.Series(pd.NA, index=df.index, dtype="Int64")
    csf_t[ptau.notna()] = (ptau[ptau.notna()] > cut_t).astype("int64")

    csf_at = pd.Series(pd.NA, index=df.index, dtype="Int64")
    both = csf_a.notna() & csf_t.notna()
    csf_at[both] = ((csf_a[both] == 1) & (csf_t[both] == 1)).astype("int64")

    df["csf_a_pos"] = csf_a
    df["csf_t_pos"] = csf_t
    df["csf_at_pos"] = csf_at
    return df


def _quantile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype="float64")
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q, method="linear"))


def _derive_two_thresholds(
    *,
    x: pd.Series,
    y: pd.Series,
    direction: Direction,
    rule_out_target_ppa: float,
    rule_in_target_npa: float,
) -> tuple[float, float]:
    if not (0.0 < rule_out_target_ppa < 1.0):
        raise ValueError("rule_out_target_ppa must be between 0 and 1 (exclusive)")
    if not (0.0 < rule_in_target_npa < 1.0):
        raise ValueError("rule_in_target_npa must be between 0 and 1 (exclusive)")

    y_num = pd.to_numeric(y, errors="coerce")
    x_num = pd.to_numeric(x, errors="coerce")
    mask = y_num.isin([0, 1]) & x_num.notna()
    y_num = y_num[mask].astype("int64")
    x_num = x_num[mask].astype("float64")

    if direction == "higher_is_more_positive":
        pos = x_num[y_num == 1]
        neg = x_num[y_num == 0]
        t_neg = _quantile(pos, 1.0 - rule_out_target_ppa)
        t_pos = _quantile(neg, rule_in_target_npa)
        if not np.isnan(t_neg) and not np.isnan(t_pos) and t_neg > t_pos:
            t_neg = t_pos
        return (float(t_neg), float(t_pos))

    if direction == "lower_is_more_positive":
        z = -x_num
        pos = z[y_num == 1]
        neg = z[y_num == 0]
        t_neg_z = _quantile(pos, 1.0 - rule_out_target_ppa)
        t_pos_z = _quantile(neg, rule_in_target_npa)
        t_neg = -t_neg_z
        t_pos = -t_pos_z
        if not np.isnan(t_neg) and not np.isnan(t_pos) and t_pos > t_neg:
            t_pos = t_neg
        return (float(t_neg), float(t_pos))

    raise ValueError(f"Unknown direction: {direction}")


def _assign_tri_category(x: pd.Series, *, direction: Direction, t_neg: float, t_pos: float) -> pd.Series:
    x_num = pd.to_numeric(x, errors="coerce")
    out = pd.Series(pd.NA, index=x.index, dtype="string")

    if np.isnan(t_neg) or np.isnan(t_pos):
        return out

    if direction == "higher_is_more_positive":
        out[x_num < t_neg] = "Negative"
        out[(x_num >= t_neg) & (x_num < t_pos)] = "Indeterminate"
        out[x_num >= t_pos] = "Positive"
        return out

    if direction == "lower_is_more_positive":
        out[x_num > t_neg] = "Negative"
        out[(x_num <= t_neg) & (x_num > t_pos)] = "Indeterminate"
        out[x_num <= t_pos] = "Positive"
        return out

    raise ValueError(f"Unknown direction: {direction}")


def _binary_from_tri_category(cat: pd.Series, *, policy: str) -> tuple[np.ndarray, np.ndarray, int]:
    s = cat.astype("string")
    valid = s.isin(["Negative", "Indeterminate", "Positive"])
    s = s.where(valid, pd.NA)

    n_indet = int((s == "Indeterminate").sum())

    if policy == "determinates_only":
        keep = s.isin(["Negative", "Positive"])
        x = s.map({"Negative": 0, "Positive": 1})
        return (x[keep].astype("int64").to_numpy(), keep.to_numpy(), n_indet)

    if policy == "indet_to_negative":
        keep = s.isin(["Negative", "Indeterminate", "Positive"])
        x = s.map({"Negative": 0, "Indeterminate": 0, "Positive": 1})
        return (x[keep].astype("int64").to_numpy(), keep.to_numpy(), n_indet)

    if policy == "indet_to_positive":
        keep = s.isin(["Negative", "Indeterminate", "Positive"])
        x = s.map({"Negative": 0, "Indeterminate": 1, "Positive": 1})
        return (x[keep].astype("int64").to_numpy(), keep.to_numpy(), n_indet)

    raise ValueError(f"Unknown policy: {policy}")


def _bootstrap_benchmark_swap(
    *,
    df: pd.DataFrame,
    rid_col: str,
    cat_col: str,
    ref_a_col: str,
    ref_b_col: str,
    policy: str,
    n_rep: int,
    seed: int,
) -> dict[str, float]:
    base = df[[rid_col, cat_col, ref_a_col, ref_b_col]].copy()
    base = base.dropna(subset=[rid_col, ref_a_col, ref_b_col])
    base[rid_col] = pd.to_numeric(base[rid_col], errors="coerce").astype("Int64")
    base = base.dropna(subset=[rid_col])

    x_bin, keep_mask, _ = _binary_from_tri_category(base[cat_col], policy=policy)
    base = base.loc[keep_mask].copy()
    base[ref_a_col] = pd.to_numeric(base[ref_a_col], errors="coerce").astype("int64")
    base[ref_b_col] = pd.to_numeric(base[ref_b_col], errors="coerce").astype("int64")

    y_a = base[ref_a_col].to_numpy(dtype="int64")
    y_b = base[ref_b_col].to_numpy(dtype="int64")
    rid = base[rid_col].to_numpy(dtype="int64")

    ok = np.isin(y_a, [0, 1]) & np.isin(y_b, [0, 1])
    y_a = y_a[ok]
    y_b = y_b[ok]
    x_bin = x_bin[ok]
    rid = rid[ok]

    unique_rids = np.unique(rid)
    n_rids = int(unique_rids.size)
    if n_rids == 0:
        return {
            "n_rids": 0,
            "n_rows": 0,
            "delta_PPA": float("nan"),
            "delta_PPA_lo": float("nan"),
            "delta_PPA_hi": float("nan"),
            "delta_NPA": float("nan"),
            "delta_NPA_lo": float("nan"),
            "delta_NPA_hi": float("nan"),
        }

    # Per-RID confusion counts (vectorized by grouping index).
    rid_index = {rid_val: i for i, rid_val in enumerate(unique_rids.tolist())}
    idx = np.fromiter((rid_index[int(r)] for r in rid), dtype="int64", count=rid.size)

    def per_rid_counts(y: np.ndarray) -> dict[str, np.ndarray]:
        tp = np.bincount(idx, weights=((x_bin == 1) & (y == 1)).astype("int64"), minlength=n_rids)
        fn = np.bincount(idx, weights=((x_bin == 0) & (y == 1)).astype("int64"), minlength=n_rids)
        fp = np.bincount(idx, weights=((x_bin == 1) & (y == 0)).astype("int64"), minlength=n_rids)
        tn = np.bincount(idx, weights=((x_bin == 0) & (y == 0)).astype("int64"), minlength=n_rids)
        return {"TP": tp, "FN": fn, "FP": fp, "TN": tn}

    a = per_rid_counts(y_a)
    b = per_rid_counts(y_b)

    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n_rids, np.repeat(1.0 / n_rids, n_rids), size=n_rep).astype("int64")

    def metric_vectors(counts: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        tp = weights @ counts["TP"]
        fn = weights @ counts["FN"]
        fp = weights @ counts["FP"]
        tn = weights @ counts["TN"]

        ppa_den = tp + fn
        npa_den = tn + fp
        ppa = np.full(ppa_den.shape, np.nan, dtype="float64")
        npa = np.full(npa_den.shape, np.nan, dtype="float64")
        np.divide(tp, ppa_den, out=ppa, where=ppa_den > 0)
        np.divide(tn, npa_den, out=npa, where=npa_den > 0)
        return ppa, npa

    ppa_a, npa_a = metric_vectors(a)
    ppa_b, npa_b = metric_vectors(b)
    d_ppa = ppa_b - ppa_a
    d_npa = npa_b - npa_a

    def ci(vec: np.ndarray) -> tuple[float, float, float]:
        lo, med, hi = np.nanquantile(vec, [0.025, 0.5, 0.975])
        return (float(med), float(lo), float(hi))

    dppa_med, dppa_lo, dppa_hi = ci(d_ppa)
    dnpa_med, dnpa_lo, dnpa_hi = ci(d_npa)
    return {
        "n_rids": n_rids,
        "n_rows": int(rid.size),
        "delta_PPA": dppa_med,
        "delta_PPA_lo": dppa_lo,
        "delta_PPA_hi": dppa_hi,
        "delta_NPA": dnpa_med,
        "delta_NPA_lo": dnpa_lo,
        "delta_NPA_hi": dnpa_hi,
    }


def _rid_split(
    *,
    df: pd.DataFrame,
    rid_col: str,
    method: str,
    derivation_fraction: float | None,
    seed: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame]:
    df = df.copy()
    df[rid_col] = pd.to_numeric(df[rid_col], errors="coerce").astype("Int64")

    split_method = str(method or "none").strip().lower()
    if split_method in {"none", "", "null"}:
        df["split"] = "evaluation"
        split_meta = {
            "method": "none",
            "derivation_fraction": None,
            "seed": None,
            "n_total_rids": int(df[rid_col].nunique(dropna=True)),
            "n_derivation_rids": int(df[rid_col].nunique(dropna=True)),
            "n_evaluation_rids": int(df[rid_col].nunique(dropna=True)),
        }
        return (df.copy(), df.copy(), split_meta, df)

    if split_method != "rid_random":
        raise ValueError(f"Unsupported split.method: {split_method}")

    frac = float(derivation_fraction if derivation_fraction is not None else 0.60)
    seed_split = int(seed if seed is not None else 20250102)
    if not (0.0 < frac < 1.0):
        raise ValueError("split.derivation_fraction must be between 0 and 1 (exclusive)")

    unique_rids = df[rid_col].dropna().astype("int64").unique()
    unique_rids = np.array(sorted(unique_rids.tolist()), dtype="int64")
    rng = np.random.default_rng(seed_split)
    rng.shuffle(unique_rids)
    n_rids = int(unique_rids.size)
    n_deriv = int(np.floor(frac * n_rids))
    n_deriv = max(1, min(n_rids - 1, n_deriv)) if n_rids >= 2 else 0
    deriv_rids = set(unique_rids[:n_deriv].tolist())

    df["split"] = np.where(df[rid_col].isin(list(deriv_rids)), "derivation", "evaluation")
    df_train = df[df["split"] == "derivation"].copy()
    df_eval = df[df["split"] == "evaluation"].copy()
    split_meta = {
        "method": "rid_random",
        "derivation_fraction": frac,
        "seed": seed_split,
        "n_total_rids": int(n_rids),
        "n_derivation_rids": int(n_deriv),
        "n_evaluation_rids": int(n_rids - n_deriv),
    }
    return (df_train, df_eval, split_meta, df)


def _build_aps2_pet_only_appendix(*, defs: dict, out_dir: Path) -> None:
    appendix = (defs.get("appendices") or {}).get("aps2_pet_only")
    if not isinstance(appendix, dict) or not bool(appendix.get("enabled")):
        return

    appendix_dir = ensure_dir(out_dir / "aps2_pet_only")
    ensure_dir(appendix_dir / "2x2_tables")
    audit_dir = ensure_dir(appendix_dir / "audit")

    tables = appendix.get("tables") or {}
    pet_path = Path(tables["pet"])
    plasma_path = Path(tables["plasma"])

    pet = pd.read_csv(pet_path, low_memory=False, parse_dates=["SCANDATE"])
    plasma = pd.read_csv(plasma_path, low_memory=False, parse_dates=["EXAMDATE"])

    windows_days = appendix.get("windows_days") or {}
    windows = PairingWindows(
        pet_plasma=int(windows_days.get("pet_plasma", 90)),
        pet_csf=0,
        plasma_csf=0,
    )
    plasma_filter = appendix.get("plasma_filter")

    dyads = build_dyad_pet_plasma(pet=pet, plasma=plasma, windows=windows, plasma_filter=plasma_filter)
    dyads = dyads[dyads["plasma_event_id"].notna()].copy()

    pet_keep = ["pet_event_id", "AMYLOID_STATUS", "CENTILOIDS", "SUMMARY_SUVR", "qc_flag"]
    pet_keep = [c for c in pet_keep if c in pet.columns]
    dyads = dyads.merge(pet.loc[:, pet_keep], on="pet_event_id", how="left", validate="m:1")

    endpoint = appendix.get("endpoint") or {}
    endpoint_name = str(endpoint.get("name", "APS2_C2N"))
    endpoint_direction = str(endpoint.get("direction", "higher_is_more_positive"))
    if endpoint_direction not in {"higher_is_more_positive", "lower_is_more_positive"}:
        raise ValueError(f"appendices.aps2_pet_only.endpoint.direction invalid: {endpoint_direction}")

    plasma_keep = ["plasma_event_id", endpoint_name]
    for c in [
        "c2n_plasma_abeta42_abeta40_abeta_ratio",
        "c2n_plasma_ptau217_ratio_ptau217_ratio",
        "c2n_plasma_ptau217_ptau217",
        "c2n_plasma_nptau217_nptau217",
    ]:
        if c in plasma.columns:
            plasma_keep.append(c)
    plasma_keep = sorted(set(plasma_keep))
    dyads = dyads.merge(plasma.loc[:, plasma_keep], on="plasma_event_id", how="left", validate="m:1")

    dyads["pet_pos"] = _pet_pos_from_status(dyads["AMYLOID_STATUS"])
    dyads = dyads.dropna(subset=["RID", "pet_pos", endpoint_name]).copy()

    split_cfg = (appendix.get("tri_category") or {}).get("split") or {}
    dyads_train, dyads_eval, split_meta, dyads = _rid_split(
        df=dyads,
        rid_col="RID",
        method=str(split_cfg.get("method", "none")),
        derivation_fraction=split_cfg.get("derivation_fraction"),
        seed=split_cfg.get("seed"),
    )

    mfg = appendix.get("manufacturer_locked_binary") or {}
    mfg_op = str(mfg.get("operating_point", "aps2_manufacturer_48"))
    mfg_cutoff = float(mfg.get("cutoff", 48))
    mfg_cat = _assign_tri_category(
        dyads[endpoint_name],
        direction=endpoint_direction,  # type: ignore[arg-type]
        t_neg=mfg_cutoff,
        t_pos=mfg_cutoff,
    )
    mfg_cat_col = f"plasma_cat__{endpoint_name}__{mfg_op}"
    dyads[mfg_cat_col] = mfg_cat

    tri_cfg = appendix.get("tri_category") or {}
    method = str(tri_cfg.get("method", "tail_targets")).strip().lower()
    if method != "tail_targets":
        raise ValueError(f"appendices.aps2_pet_only.tri_category.method unsupported: {method}")

    operating_points = tri_cfg.get("operating_points") or []
    if not operating_points:
        raise ValueError("appendices.aps2_pet_only.tri_category.operating_points is required when enabled")

    thresholds_rows: list[dict] = []
    for op in operating_points:
        op_name = str(op["name"])
        rule_out_target_ppa = float(op["rule_out_target_ppa"])
        rule_in_target_npa = float(op["rule_in_target_npa"])

        t_neg, t_pos = _derive_two_thresholds(
            x=dyads_train[endpoint_name],
            y=dyads_train["pet_pos"],
            direction=endpoint_direction,  # type: ignore[arg-type]
            rule_out_target_ppa=rule_out_target_ppa,
            rule_in_target_npa=rule_in_target_npa,
        )

        thresholds_rows.append(
            {
                "operating_point": op_name,
                "benchmark_for_thresholds": "pet",
                "threshold_split_method": split_meta.get("method"),
                "split_derivation_fraction": split_meta.get("derivation_fraction"),
                "split_seed": split_meta.get("seed"),
                "n_train_rids": int(dyads_train["RID"].nunique(dropna=True)),
                "n_train_rows": int(len(dyads_train)),
                "endpoint": endpoint_name,
                "direction": endpoint_direction,
                "rule_out_target_ppa": rule_out_target_ppa,
                "rule_in_target_npa": rule_in_target_npa,
                "t_negative": t_neg,
                "t_positive": t_pos,
            }
        )

        dyads[f"plasma_cat__{endpoint_name}__{op_name}"] = _assign_tri_category(
            dyads[endpoint_name],
            direction=endpoint_direction,  # type: ignore[arg-type]
            t_neg=t_neg,
            t_pos=t_pos,
        )

    thresholds_rows.append(
        {
            "operating_point": mfg_op,
            "benchmark_for_thresholds": "manufacturer_locked",
            "threshold_split_method": "manufacturer_locked",
            "split_derivation_fraction": split_meta.get("derivation_fraction"),
            "split_seed": split_meta.get("seed"),
            "n_train_rids": int(dyads_train["RID"].nunique(dropna=True)),
            "n_train_rows": int(len(dyads_train)),
            "endpoint": endpoint_name,
            "direction": endpoint_direction,
            "rule_out_target_ppa": None,
            "rule_in_target_npa": None,
            "t_negative": mfg_cutoff,
            "t_positive": mfg_cutoff,
            "source": str(mfg.get("source")) if mfg.get("source") else None,
        }
    )

    dyads.to_csv(appendix_dir / "cohort_pet_plasma_dyads.csv", index=False)
    pd.DataFrame.from_records(thresholds_rows).sort_values(["operating_point", "endpoint"], kind="mergesort").to_csv(
        appendix_dir / "plasma_thresholds_used.csv", index=False
    )

    dyads_eval = dyads[dyads["split"] == "evaluation"].copy()
    policies = [str(x) for x in defs["index_test"]["plasma"].get("binary_policies", [])]
    if not policies:
        policies = ["determinates_only", "indet_to_negative", "indet_to_positive"]

    metric_rows: list[dict] = []
    for op in list(operating_points) + [{"name": mfg_op}]:
        op_name = str(op["name"])
        cat_col = f"plasma_cat__{endpoint_name}__{op_name}"
        if cat_col not in dyads_eval.columns:
            continue

        y_num = pd.to_numeric(dyads_eval["pet_pos"], errors="coerce")
        for policy in policies:
            x_bin, keep_mask, n_indet = _binary_from_tri_category(dyads_eval[cat_col], policy=policy)
            y_keep = y_num.to_numpy()[keep_mask]
            ok = np.isin(y_keep, [0, 1]) & np.isin(x_bin, [0, 1])
            y_arr = y_keep[ok].astype("int64")
            x_arr = x_bin[ok].astype("int64")

            counts = _confusion_counts(y_arr, x_arr)
            stats = _ppa_npa(counts)

            n_total = int(dyads_eval[cat_col].notna().sum())
            n_confirm_primary = int(dyads_eval[cat_col].isin(["Positive", "Indeterminate"]).sum())
            n_neg = int(dyads_eval[cat_col].isin(["Negative"]).sum())
            ref_pos = int(((y_num == 1) & dyads_eval[cat_col].notna()).sum())
            ref_neg = int(((y_num == 0) & dyads_eval[cat_col].notna()).sum())
            miss_rate = float("nan")
            if ref_pos > 0:
                miss_rate = float(((dyads_eval[cat_col] == "Negative") & (y_num == 1)).sum() / ref_pos)

            n_used = counts["TP"] + counts["FP"] + counts["TN"] + counts["FN"]
            ppa_den = counts["TP"] + counts["FN"]
            npa_den = counts["TN"] + counts["FP"]

            metric_rows.append(
                {
                    "operating_point": op_name,
                    "endpoint": endpoint_name,
                    "benchmark": "PET",
                    "policy": policy,
                    "threshold_split_method": str(split_meta.get("method")),
                    "split_derivation_fraction": split_meta.get("derivation_fraction"),
                    "split_seed": split_meta.get("seed"),
                    "n_eval_rids": int(dyads_eval["RID"].nunique(dropna=True)),
                    "n_eval_pet_events": int(len(dyads_eval)),
                    "n_total_with_cat": n_total,
                    "n_indeterminate": n_indet,
                    "indeterminate_fraction": float(n_indet / n_total) if n_total else float("nan"),
                    "confirm_rate_primary": float(n_confirm_primary / n_total) if n_total else float("nan"),
                    "negative_rate": float(n_neg / n_total) if n_total else float("nan"),
                    "ref_pos": ref_pos,
                    "ref_neg": ref_neg,
                    "n_used": n_used,
                    "ppa_den": ppa_den,
                    "npa_den": npa_den,
                    **counts,
                    **stats,
                    "miss_rate_under_reflex_primary": miss_rate,
                }
            )

            tpath = appendix_dir / "2x2_tables" / f"2x2__{endpoint_name}__PET__{op_name}__{policy}.csv"
            pd.DataFrame([counts]).to_csv(tpath, index=False)

    pd.DataFrame.from_records(metric_rows).to_csv(appendix_dir / "metrics_summary.csv", index=False)

    s = pd.to_numeric(dyads_eval["delta_days_abs"], errors="coerce").dropna()
    gap_rows: list[dict] = []
    if s.empty:
        gap_rows.append({"split": "evaluation", "delta_field": "delta_days_abs", "n": 0})
    else:
        q = s.quantile([0, 0.25, 0.5, 0.75, 0.95, 1.0])
        gap_rows.append(
            {
                "split": "evaluation",
                "delta_field": "delta_days_abs",
                "n": int(len(s)),
                "frac_zero": float((s == 0).mean()),
                "min": float(q.loc[0.0]),
                "q25": float(q.loc[0.25]),
                "median": float(q.loc[0.5]),
                "q75": float(q.loc[0.75]),
                "q95": float(q.loc[0.95]),
                "max": float(q.loc[1.0]),
            }
        )
    pd.DataFrame.from_records(gap_rows).to_csv(audit_dir / "time_gap_summary.csv", index=False)

    split_summary_lines: list[str] = []
    split_summary_lines.append("# APS2 PET-only appendix split summary")
    split_summary_lines.append("")
    split_summary_lines.append(f"- split_method: {split_meta.get('method')}")
    split_summary_lines.append(f"- derivation_fraction: {split_meta.get('derivation_fraction')}")
    split_summary_lines.append(f"- split_seed: {split_meta.get('seed')}")
    split_summary_lines.append(f"- total_rids: {split_meta.get('n_total_rids')}")
    split_summary_lines.append(f"- derivation_rids: {split_meta.get('n_derivation_rids')}")
    split_summary_lines.append(f"- evaluation_rids: {split_meta.get('n_evaluation_rids')}")
    split_summary_lines.append("")
    split_summary_lines.append(f"- evaluation PET events: {int(len(dyads_eval))}")
    split_summary_lines.append("")
    (audit_dir / "split_summary.md").write_text("\n".join(split_summary_lines) + "\n", encoding="utf-8")


def build_paper_a_pack(*, definitions_path: Path, out_dir: Path) -> PaperAArtifacts:
    defs = load_yaml(definitions_path)

    tables = defs["evidence_core"]["tables"]
    triads_path = Path(tables["triads"])
    pet_path = Path(tables["pet"])
    plasma_wide_path = Path(tables["plasma_wide"])
    clinical_path = Path(tables["clinical"])

    ensure_dir(out_dir)
    ensure_dir(out_dir / "2x2_tables")
    audit_dir = ensure_dir(out_dir / "audit")
    figures_dir = ensure_dir(out_dir / "figures")

    triads = pd.read_csv(triads_path, low_memory=False)
    triads = _as_date(triads, ["pet_date", "csf_date", "plasma_date"])

    # Join plasma endpoints (wide) by plasma_event_id.
    plasma = pd.read_csv(plasma_wide_path, low_memory=False)
    plasma_endpoints = [
        d["name"] if isinstance(d, dict) else str(d)
        for d in defs["index_test"]["plasma"]["co_primary_endpoints"]
    ]
    plasma_secondary = [str(x) for x in defs["index_test"]["plasma"].get("secondary_endpoints", [])]
    plasma_keep = ["plasma_event_id"] + sorted(set(plasma_endpoints + plasma_secondary))
    plasma_keep = [c for c in plasma_keep if c in plasma.columns]
    triads = triads.merge(plasma.loc[:, plasma_keep], on="plasma_event_id", how="left", validate="m:1")

    # Attach nearest clinical row (prefixed).
    clinical = pd.read_csv(clinical_path, low_memory=False)
    triads = _attach_nearest_clinical(triads, clinical)

    # Derived benchmark columns (primary cutpoints only).
    triads["pet_pos"] = _pet_pos_from_status(triads["AMYLOID_STATUS"])
    triads = _derive_csf_benchmarks(triads, defs)

    # Attach PET QC (for sensitivity analyses).
    pet = pd.read_csv(pet_path, low_memory=False)
    pet_keep = ["pet_event_id"]
    if "qc_flag" in pet.columns:
        pet_keep.append("qc_flag")
    pet = pet.loc[:, pet_keep].rename(columns={"qc_flag": "pet_qc_flag"})
    triads = triads.merge(pet, on="pet_event_id", how="left", validate="m:1")
    triads["pet_qc_rank"] = triads.get("pet_qc_flag").map(_qc_rank).astype("int64") if "pet_qc_flag" in triads.columns else 0
    triads["pet_qc_fail"] = triads["pet_qc_rank"] < 0

    cent = pd.to_numeric(triads.get("CENTILOIDS"), errors="coerce")
    triads["pet_centiloid_extreme"] = cent.notna() & ((cent < -50) | (cent > 200))

    # Build tri-category plasma labels per endpoint and operating point.
    cls = defs["index_test"]["plasma"]["classification"]
    method = str(cls.get("method", "tail_targets"))
    benchmark_for_thresholds = str(cls.get("benchmark_for_thresholds", "pet")).strip().lower()
    if benchmark_for_thresholds not in {"pet", "csf_a"}:
        raise ValueError("classification.benchmark_for_thresholds must be one of: pet, csf_a")

    # RID-level derivation/evaluation split for threshold selection vs evaluation.
    triads["RID"] = pd.to_numeric(triads["RID"], errors="coerce").astype("Int64")
    split_cfg = cls.get("split") or {}
    split_method = str(split_cfg.get("method", "none")).strip().lower()
    if split_method in {"none", "", "null"}:
        triads["split"] = "evaluation"
        triads_train_all = triads.copy()
        triads_eval_all = triads.copy()
        split_meta = {
            "method": "none",
            "derivation_fraction": None,
            "seed": None,
            "n_total_rids": int(triads["RID"].nunique(dropna=True)),
            "n_derivation_rids": int(triads_train_all["RID"].nunique(dropna=True)),
            "n_evaluation_rids": int(triads_eval_all["RID"].nunique(dropna=True)),
        }
    elif split_method == "rid_random":
        frac = float(split_cfg.get("derivation_fraction", 0.60))
        seed_split = int(split_cfg.get("seed", 20250102))
        if not (0.0 < frac < 1.0):
            raise ValueError("split.derivation_fraction must be between 0 and 1 (exclusive)")

        unique_rids = triads["RID"].dropna().astype("int64").unique()
        unique_rids = np.array(sorted(unique_rids.tolist()), dtype="int64")
        rng = np.random.default_rng(seed_split)
        rng.shuffle(unique_rids)
        n_rids = int(unique_rids.size)
        n_deriv = int(np.floor(frac * n_rids))
        n_deriv = max(1, min(n_rids - 1, n_deriv)) if n_rids >= 2 else 0
        deriv_rids = set(unique_rids[:n_deriv].tolist())

        triads["split"] = np.where(triads["RID"].isin(list(deriv_rids)), "derivation", "evaluation")
        triads_train_all = triads[triads["split"] == "derivation"].copy()
        triads_eval_all = triads[triads["split"] == "evaluation"].copy()
        split_meta = {
            "method": "rid_random",
            "derivation_fraction": frac,
            "seed": seed_split,
            "n_total_rids": int(n_rids),
            "n_derivation_rids": int(n_deriv),
            "n_evaluation_rids": int(n_rids - n_deriv),
        }
    else:
        raise ValueError(f"Unsupported split.method: {split_method}")

    triads_train_for_thresholds = _select_one_triad_per_rid(triads_train_all)
    y_threshold = (
        triads_train_for_thresholds["pet_pos"]
        if benchmark_for_thresholds == "pet"
        else triads_train_for_thresholds["csf_a_pos"]
    )

    operating_points = cls.get("operating_points")
    if not operating_points:
        raise ValueError("definitions.yaml is missing index_test.plasma.classification.operating_points")

    directions = cls.get("endpoint_directions", {})
    if not isinstance(directions, dict):
        raise ValueError("index_test.plasma.classification.endpoint_directions must be a mapping")

    thresholds_rows: list[dict] = []
    labels_rows: list[dict] = []

    for op in operating_points:
        op_name = str(op["name"])
        rule_out_target_ppa = float(op["rule_out_target_ppa"])
        rule_in_target_npa = float(op["rule_in_target_npa"])

        for endpoint in plasma_endpoints:
            direction = str(directions.get(endpoint))
            if direction not in {"higher_is_more_positive", "lower_is_more_positive"}:
                raise ValueError(f"Missing/invalid endpoint direction for {endpoint}: {direction}")

            if method != "tail_targets":
                raise ValueError(f"Unsupported classification.method: {method}")

            x_train = triads_train_for_thresholds[endpoint]
            y_train = y_threshold
            y_num = pd.to_numeric(y_train, errors="coerce")
            x_num = pd.to_numeric(x_train, errors="coerce")
            fit_mask = y_num.isin([0, 1]) & x_num.notna()
            n_fit = int(fit_mask.sum())
            n_fit_pos = int(((y_num == 1) & fit_mask).sum())
            n_fit_neg = int(((y_num == 0) & fit_mask).sum())

            t_neg, t_pos = _derive_two_thresholds(
                x=x_train,
                y=y_train,
                direction=direction,  # type: ignore[arg-type]
                rule_out_target_ppa=rule_out_target_ppa,
                rule_in_target_npa=rule_in_target_npa,
            )

            thresholds_rows.append(
                {
                    "operating_point": op_name,
                    "benchmark_for_thresholds": benchmark_for_thresholds,
                    "threshold_split_method": split_method,
                    "split_derivation_fraction": split_meta.get("derivation_fraction"),
                    "split_seed": split_meta.get("seed"),
                    "train_selection": "one_triad_per_rid",
                    "n_train_rids": int(triads_train_for_thresholds["RID"].nunique(dropna=True)),
                    "n_train_triads": int(len(triads_train_for_thresholds)),
                    "n_train_triads_in_split": int(len(triads_train_all)),
                    "n_fit_rows": n_fit,
                    "n_fit_pos": n_fit_pos,
                    "n_fit_neg": n_fit_neg,
                    "endpoint": endpoint,
                    "direction": direction,
                    "rule_out_target_ppa": rule_out_target_ppa,
                    "rule_in_target_npa": rule_in_target_npa,
                    "t_negative": t_neg,
                    "t_positive": t_pos,
                }
            )

            cat = _assign_tri_category(triads[endpoint], direction=direction, t_neg=t_neg, t_pos=t_pos)  # type: ignore[arg-type]
            label_col = f"plasma_cat__{endpoint}__{op_name}"
            triads[label_col] = cat

            for rid, triad_id, val, lab in zip(
                triads["RID"].tolist(),
                triads["pet_event_id"].tolist(),
                pd.to_numeric(triads[endpoint], errors="coerce").tolist(),
                cat.tolist(),
                strict=True,
            ):
                labels_rows.append(
                    {
                        "pet_event_id": triad_id,
                        "RID": rid,
                        "operating_point": op_name,
                        "endpoint": endpoint,
                        "value": val,
                        "tri_category": lab,
                    }
                )

    # Optional: manufacturer-locked dichotomy (APS2).
    mfg = defs["index_test"]["plasma"].get("manufacturer_locked_binary")
    mfg_op_name = None
    if isinstance(mfg, dict) and mfg.get("endpoint") and mfg.get("cutoff") is not None:
        mfg_endpoint = str(mfg["endpoint"])
        mfg_direction = str(mfg.get("direction", "higher_is_more_positive"))
        mfg_cutoff = float(mfg["cutoff"])
        mfg_op_name = str(mfg.get("operating_point", "manufacturer_locked")).strip()
        if not mfg_op_name:
            mfg_op_name = "manufacturer_locked"

        if mfg_endpoint not in triads.columns:
            raise ValueError(f"manufacturer_locked_binary endpoint not found in cohort table: {mfg_endpoint}")
        if mfg_direction not in {"higher_is_more_positive", "lower_is_more_positive"}:
            raise ValueError(f"manufacturer_locked_binary has invalid direction: {mfg_direction}")

        mfg_cat = _assign_tri_category(
            triads[mfg_endpoint],
            direction=mfg_direction,  # type: ignore[arg-type]
            t_neg=mfg_cutoff,
            t_pos=mfg_cutoff,
        )
        mfg_col = f"plasma_cat__{mfg_endpoint}__{mfg_op_name}"
        triads[mfg_col] = mfg_cat

        thresholds_rows.append(
            {
                "operating_point": mfg_op_name,
                "benchmark_for_thresholds": "manufacturer_locked",
                "threshold_split_method": "manufacturer_locked",
                "split_derivation_fraction": split_meta.get("derivation_fraction"),
                "split_seed": split_meta.get("seed"),
                "train_selection": "one_triad_per_rid",
                "n_train_rids": int(triads_train_for_thresholds["RID"].nunique(dropna=True)),
                "n_train_triads": int(len(triads_train_for_thresholds)),
                "n_train_triads_in_split": int(len(triads_train_all)),
                "n_fit_rows": 0,
                "n_fit_pos": 0,
                "n_fit_neg": 0,
                "endpoint": mfg_endpoint,
                "direction": mfg_direction,
                "rule_out_target_ppa": None,
                "rule_in_target_npa": None,
                "t_negative": mfg_cutoff,
                "t_positive": mfg_cutoff,
                "source": str(mfg.get("source")) if mfg.get("source") else None,
            }
        )

        for rid, triad_id, val, lab in zip(
            triads["RID"].tolist(),
            triads["pet_event_id"].tolist(),
            pd.to_numeric(triads[mfg_endpoint], errors="coerce").tolist(),
            mfg_cat.tolist(),
            strict=True,
        ):
            labels_rows.append(
                {
                    "pet_event_id": triad_id,
                    "RID": rid,
                    "operating_point": mfg_op_name,
                    "endpoint": mfg_endpoint,
                    "value": val,
                    "tri_category": lab,
                }
            )

    # Write cohort table (wide) for reproducibility.
    cohort_path = out_dir / "cohort_triads.csv"
    triads.to_csv(cohort_path, index=False)

    thresholds_path = out_dir / "plasma_thresholds_used.csv"
    pd.DataFrame.from_records(thresholds_rows).sort_values(
        ["operating_point", "endpoint"], kind="mergesort"
    ).to_csv(thresholds_path, index=False)

    labels_path = out_dir / "plasma_labels.csv"
    pd.DataFrame.from_records(labels_rows).to_csv(labels_path, index=False)

    # Re-slice after adding plasma_cat_* columns.
    triads_eval_all = triads[triads["split"] == "evaluation"].copy()
    triads_eval = _select_one_triad_per_rid(triads_eval_all)

    triads_train_all_post = triads[triads["split"] == "derivation"].copy()
    if split_meta.get("method") == "none":
        triads_train_all_post = triads.copy()
    triads_train = _select_one_triad_per_rid(triads_train_all_post) if len(triads_train_all_post) else triads_train_all_post.copy()

    policies = [str(x) for x in defs["index_test"]["plasma"].get("binary_policies", [])]
    if not policies:
        policies = ["determinates_only", "indet_to_negative", "indet_to_positive"]

    bootstrap = defs.get("statistics", {}).get("bootstrap", {})
    n_rep = int(bootstrap.get("replicates", 10000))
    seed = int(bootstrap.get("seed", 20250101))

    def _compute_metrics_and_swap(
        eval_df: pd.DataFrame, *, analysis_set: str, out_2x2_dir: Path | None
    ) -> tuple[list[dict], list[dict]]:
        metric_rows: list[dict] = []
        swap_rows: list[dict] = []

        if out_2x2_dir is not None:
            ensure_dir(out_2x2_dir)

        for op in operating_points:
            op_name = str(op["name"])
            for endpoint in plasma_endpoints:
                cat_col = f"plasma_cat__{endpoint}__{op_name}"
                if cat_col not in eval_df.columns:
                    continue

                for benchmark_name, ref_col in [
                    ("PET", "pet_pos"),
                    ("CSF-A", "csf_a_pos"),
                    ("CSF-A+T", "csf_at_pos"),
                ]:
                    y_ref = eval_df[ref_col]
                    y_num = pd.to_numeric(y_ref, errors="coerce")

                    for policy in policies:
                        x_bin, keep_mask, n_indet = _binary_from_tri_category(eval_df[cat_col], policy=policy)
                        y_keep = y_num.to_numpy()[keep_mask]
                        ok = np.isin(y_keep, [0, 1]) & np.isin(x_bin, [0, 1])
                        y_arr = y_keep[ok].astype("int64")
                        x_arr = x_bin[ok].astype("int64")

                        counts = _confusion_counts(y_arr, x_arr)
                        stats = _ppa_npa(counts)

                        n_total = int(eval_df[cat_col].notna().sum())
                        n_confirm_primary = int(eval_df[cat_col].isin(["Positive", "Indeterminate"]).sum())
                        n_neg = int(eval_df[cat_col].isin(["Negative"]).sum())
                        ref_pos = int(((y_num == 1) & eval_df[cat_col].notna()).sum())
                        ref_neg = int(((y_num == 0) & eval_df[cat_col].notna()).sum())
                        miss_rate = float("nan")
                        if ref_pos > 0:
                            miss_rate = float(((eval_df[cat_col] == "Negative") & (y_num == 1)).sum() / ref_pos)

                        n_used = counts["TP"] + counts["FP"] + counts["TN"] + counts["FN"]
                        ppa_den = counts["TP"] + counts["FN"]
                        npa_den = counts["TN"] + counts["FP"]

                        metric_rows.append(
                            {
                                "analysis_set": analysis_set,
                                "operating_point": op_name,
                                "endpoint": endpoint,
                                "benchmark": benchmark_name,
                                "policy": policy,
                                "threshold_split_method": split_method,
                                "split_derivation_fraction": split_meta.get("derivation_fraction"),
                                "split_seed": split_meta.get("seed"),
                                "n_eval_rids": int(eval_df["RID"].nunique(dropna=True)),
                                "n_eval_triads": int(len(eval_df)),
                                "n_total_with_cat": n_total,
                                "n_indeterminate": n_indet,
                                "indeterminate_fraction": float(n_indet / n_total) if n_total else float("nan"),
                                "confirm_rate_primary": float(n_confirm_primary / n_total) if n_total else float("nan"),
                                "negative_rate": float(n_neg / n_total) if n_total else float("nan"),
                                "ref_pos": ref_pos,
                                "ref_neg": ref_neg,
                                "n_used": n_used,
                                "ppa_den": ppa_den,
                                "npa_den": npa_den,
                                **counts,
                                **stats,
                                "miss_rate_under_reflex_primary": miss_rate,
                            }
                        )

                        if out_2x2_dir is not None and benchmark_name in {"PET", "CSF-A"}:
                            tpath = out_2x2_dir / f"2x2__{endpoint}__{benchmark_name}__{op_name}__{policy}.csv"
                            pd.DataFrame([counts]).to_csv(tpath, index=False)

                for policy in policies:
                    swap = _bootstrap_benchmark_swap(
                        df=eval_df,
                        rid_col="RID",
                        cat_col=cat_col,
                        ref_a_col="pet_pos",
                        ref_b_col="csf_a_pos",
                        policy=policy,
                        n_rep=n_rep,
                        seed=seed,
                    )
                    swap_rows.append(
                        {
                            "analysis_set": analysis_set,
                            "operating_point": op_name,
                            "endpoint": endpoint,
                            "policy": policy,
                            "swap": "CSF-A minus PET",
                            "threshold_split_method": split_method,
                            "split_derivation_fraction": split_meta.get("derivation_fraction"),
                            "split_seed": split_meta.get("seed"),
                            **swap,
                        }
                    )

        return metric_rows, swap_rows

    metric_rows, swap_rows = _compute_metrics_and_swap(
        triads_eval, analysis_set="one_triad_per_rid", out_2x2_dir=out_dir / "2x2_tables"
    )
    metric_rows_all, swap_rows_all = _compute_metrics_and_swap(
        triads_eval_all, analysis_set="all_triads", out_2x2_dir=audit_dir / "2x2_tables_all_triads"
    )

    # Manufacturer-locked APS2 dichotomy metrics (evaluated on held-out split).
    if mfg_op_name is not None:
        mfg_endpoint = str(mfg["endpoint"])
        mfg_col = f"plasma_cat__{mfg_endpoint}__{mfg_op_name}"
        for benchmark_name, ref_col in [("PET", "pet_pos"), ("CSF-A", "csf_a_pos"), ("CSF-A+T", "csf_at_pos")]:
            y_num = pd.to_numeric(triads_eval[ref_col], errors="coerce")
            for policy in policies:
                x_bin, keep_mask, n_indet = _binary_from_tri_category(triads_eval[mfg_col], policy=policy)
                y_keep = y_num.to_numpy()[keep_mask]
                ok = np.isin(y_keep, [0, 1]) & np.isin(x_bin, [0, 1])
                y_arr = y_keep[ok].astype("int64")
                x_arr = x_bin[ok].astype("int64")

                counts = _confusion_counts(y_arr, x_arr)
                stats = _ppa_npa(counts)

                n_total = int(triads_eval[mfg_col].notna().sum())
                n_used = counts["TP"] + counts["FP"] + counts["TN"] + counts["FN"]
                ppa_den = counts["TP"] + counts["FN"]
                npa_den = counts["TN"] + counts["FP"]

                metric_rows.append(
                    {
                        "analysis_set": "one_triad_per_rid",
                        "operating_point": mfg_op_name,
                        "endpoint": mfg_endpoint,
                        "benchmark": benchmark_name,
                        "policy": policy,
                        "threshold_split_method": "manufacturer_locked",
                        "split_derivation_fraction": split_meta.get("derivation_fraction"),
                        "split_seed": split_meta.get("seed"),
                        "n_eval_rids": int(triads_eval["RID"].nunique(dropna=True)),
                        "n_eval_triads": int(len(triads_eval)),
                        "n_total_with_cat": n_total,
                        "n_indeterminate": n_indet,
                        "indeterminate_fraction": float(n_indet / n_total) if n_total else float("nan"),
                        "confirm_rate_primary": float(triads_eval[mfg_col].isin(["Positive", "Indeterminate"]).mean()),
                        "negative_rate": float(triads_eval[mfg_col].isin(["Negative"]).mean()),
                        "ref_pos": int(((y_num == 1) & triads_eval[mfg_col].notna()).sum()),
                        "ref_neg": int(((y_num == 0) & triads_eval[mfg_col].notna()).sum()),
                        "n_used": n_used,
                        "ppa_den": ppa_den,
                        "npa_den": npa_den,
                        **counts,
                        **stats,
                        "miss_rate_under_reflex_primary": float("nan"),
                    }
                )

                if benchmark_name in {"PET", "CSF-A"}:
                    tpath = out_dir / "2x2_tables" / f"2x2__{mfg_endpoint}__{benchmark_name}__{mfg_op_name}__{policy}.csv"
                    pd.DataFrame([counts]).to_csv(tpath, index=False)

        for policy in policies:
            swap = _bootstrap_benchmark_swap(
                df=triads_eval,
                rid_col="RID",
                cat_col=mfg_col,
                ref_a_col="pet_pos",
                ref_b_col="csf_a_pos",
                policy=policy,
                n_rep=n_rep,
                seed=seed,
            )
            swap_rows.append(
                {
                    "analysis_set": "one_triad_per_rid",
                    "operating_point": mfg_op_name,
                    "endpoint": mfg_endpoint,
                    "policy": policy,
                    "swap": "CSF-A minus PET",
                    "threshold_split_method": "manufacturer_locked",
                    "split_derivation_fraction": split_meta.get("derivation_fraction"),
                    "split_seed": split_meta.get("seed"),
                    **swap,
                }
            )

    metrics_path = out_dir / "metrics_summary.csv"
    pd.DataFrame.from_records(metric_rows).to_csv(metrics_path, index=False)

    swap_path = out_dir / "benchmark_swap.csv"
    pd.DataFrame.from_records(swap_rows).to_csv(swap_path, index=False)

    pd.DataFrame.from_records(metric_rows_all).to_csv(audit_dir / "all_triads_metrics_summary.csv", index=False)
    pd.DataFrame.from_records(swap_rows_all).to_csv(audit_dir / "all_triads_benchmark_swap.csv", index=False)

    # Persist the exact (possibly updated) definitions used for this build.
    defs_copy_path = out_dir / "definitions_used.yaml"
    defs_copy_path.write_text(yaml.safe_dump(defs, sort_keys=False), encoding="utf-8")

    # Paper A audit/QC artifacts (untracked by default).
    split_assign = triads[["RID", "split"]].drop_duplicates().sort_values(["RID"], kind="mergesort")
    rid_counts = triads.groupby(["RID"], dropna=False).size().reset_index(name="n_triads")
    split_assign = split_assign.merge(rid_counts, on="RID", how="left")
    split_assign.to_csv(audit_dir / "split_assignment.csv", index=False)

    triads_one_per_rid = _select_one_triad_per_rid(triads)
    triads_one_per_rid_scores = triads_one_per_rid.copy()
    triads_one_per_rid_scores["has_csf_a"] = triads_one_per_rid_scores["csf_a_pos"].notna().astype("int64")
    for col in ["delta_pet_plasma_days", "delta_pet_csf_days", "delta_plasma_csf_days"]:
        if col in triads_one_per_rid_scores.columns:
            triads_one_per_rid_scores[col] = pd.to_numeric(triads_one_per_rid_scores[col], errors="coerce")
    triads_one_per_rid_scores["selection_max_gap_days"] = (
        triads_one_per_rid_scores[["delta_pet_plasma_days", "delta_pet_csf_days", "delta_plasma_csf_days"]]
        .max(axis=1, skipna=True)
        .astype("float64")
    )
    triads_one_per_rid_scores["selection_sum_gap_days"] = (
        triads_one_per_rid_scores[["delta_pet_plasma_days", "delta_pet_csf_days", "delta_plasma_csf_days"]]
        .sum(axis=1, skipna=True)
        .astype("float64")
    )
    triads_one_per_rid_scores.to_csv(audit_dir / "one_triad_per_rid.csv", index=False)

    (audit_dir / "one_triad_per_rid_summary.md").write_text(
        "\n".join(
            [
                "# One-triad-per-RID primary analysis set",
                "",
                "Selection is deterministic: prefer triads with CSF Aβ42/40 available (if present for that RID), then minimize time-gap scores (max gap, then sum gap).",
                "",
                f"- triads (all): {int(len(triads))} (RIDs={int(triads['RID'].nunique(dropna=True))})",
                f"- triads (one per RID): {int(len(triads_one_per_rid))} (RIDs={int(triads_one_per_rid['RID'].nunique(dropna=True))})",
                f"- CSF-A available (one per RID): {int(triads_one_per_rid['csf_a_pos'].notna().sum())} (RIDs={int(triads_one_per_rid.loc[triads_one_per_rid['csf_a_pos'].notna(), 'RID'].nunique(dropna=True))})",
                "",
                "Primary Paper A metrics use the held-out evaluation split restricted to one triad per RID; sensitivity analyses use all evaluation triads with RID-clustered inference.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def _rate(s: pd.Series) -> float:
        v = pd.to_numeric(s, errors="coerce").dropna()
        return float(v.mean()) if len(v) else float("nan")

    split_summary_lines: list[str] = []
    split_summary_lines.append("# Paper A split summary")
    split_summary_lines.append("")
    split_summary_lines.append(f"- split_method: {split_meta.get('method')}")
    split_summary_lines.append(f"- derivation_fraction: {split_meta.get('derivation_fraction')}")
    split_summary_lines.append(f"- split_seed: {split_meta.get('seed')}")
    split_summary_lines.append(f"- total_rids: {split_meta.get('n_total_rids')}")
    split_summary_lines.append(f"- derivation_rids: {split_meta.get('n_derivation_rids')}")
    split_summary_lines.append(f"- evaluation_rids: {split_meta.get('n_evaluation_rids')}")
    split_summary_lines.append("")

    def _with_csf_a(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["csf_a_pos"].notna()].copy() if "csf_a_pos" in df.columns else df.iloc[0:0].copy()

    def _write_split_block(*, label: str, all_triads: pd.DataFrame, one_per_rid: pd.DataFrame) -> None:
        split_summary_lines.append(f"## {label}")
        split_summary_lines.append(f"- triads (all): {int(len(all_triads))} (RIDs={int(all_triads['RID'].nunique(dropna=True))})")
        split_summary_lines.append(
            f"- triads (one per RID): {int(len(one_per_rid))} (RIDs={int(one_per_rid['RID'].nunique(dropna=True))})"
        )

        split_summary_lines.append(f"- PET+ rate (all triads): {_rate(all_triads['pet_pos']):.3f}")
        split_summary_lines.append(f"- PET+ rate (one per RID): {_rate(one_per_rid['pet_pos']):.3f}")
        split_summary_lines.append(f"- CSF-A+ rate (all triads): {_rate(all_triads['csf_a_pos']):.3f}")
        split_summary_lines.append(f"- CSF-A+ rate (one per RID): {_rate(one_per_rid['csf_a_pos']):.3f}")
        split_summary_lines.append(f"- CSF A+T+ rate (all triads): {_rate(all_triads['csf_at_pos']):.3f}")
        split_summary_lines.append(f"- CSF A+T+ rate (one per RID): {_rate(one_per_rid['csf_at_pos']):.3f}")

        csf_all = _with_csf_a(all_triads)
        csf_one = _with_csf_a(one_per_rid)
        split_summary_lines.append(
            f"- CSF-A available (all triads): {int(len(csf_all))} (RIDs={int(csf_all['RID'].nunique(dropna=True))})"
        )
        split_summary_lines.append(
            f"- CSF-A available (one per RID): {int(len(csf_one))} (RIDs={int(csf_one['RID'].nunique(dropna=True))})"
        )
        split_summary_lines.append("")

    if split_meta.get("method") == "rid_random":
        _write_split_block(label="derivation", all_triads=triads_train_all_post, one_per_rid=triads_train)
        _write_split_block(label="evaluation", all_triads=triads_eval_all, one_per_rid=triads_eval)
    else:
        _write_split_block(label="evaluation", all_triads=triads_eval_all, one_per_rid=triads_eval)
    (audit_dir / "split_summary.md").write_text("\n".join(split_summary_lines), encoding="utf-8")

    def _gap_rows(df: pd.DataFrame, *, split_label: str) -> list[dict]:
        rows: list[dict] = []
        for col in ["delta_pet_plasma_days", "delta_pet_csf_days", "delta_plasma_csf_days", "delta_pet_clin_days"]:
            if col not in df.columns:
                continue
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.empty:
                rows.append({"split": split_label, "delta_field": col, "n": 0})
                continue
            q = s.quantile([0, 0.25, 0.5, 0.75, 0.95, 1.0])
            rows.append(
                {
                    "split": split_label,
                    "delta_field": col,
                    "n": int(len(s)),
                    "frac_zero": float((s == 0).mean()),
                    "min": float(q.loc[0.0]),
                    "q25": float(q.loc[0.25]),
                    "median": float(q.loc[0.5]),
                    "q75": float(q.loc[0.75]),
                    "q95": float(q.loc[0.95]),
                    "max": float(q.loc[1.0]),
                }
            )
        return rows

    gap_rows: list[dict] = []
    gap_rows.extend(_gap_rows(triads, split_label="all"))
    gap_rows.extend(_gap_rows(triads_train_all_post, split_label="derivation_all_triads"))
    gap_rows.extend(_gap_rows(triads_eval_all, split_label="evaluation_all_triads"))
    gap_rows.extend(_gap_rows(triads_train, split_label="derivation_one_triad_per_rid"))
    gap_rows.extend(_gap_rows(triads_eval, split_label="evaluation_one_triad_per_rid"))
    pd.DataFrame.from_records(gap_rows).to_csv(audit_dir / "time_gap_summary.csv", index=False)

    # STARD-forward cohort accounting (counts table + a simple flow diagram).
    flow_rows: list[dict] = []

    def _flow_row(*, step: str, df: pd.DataFrame, note: str | None = None) -> dict:
        return {
            "step": step,
            "n_rows": int(len(df)),
            "n_rids": int(df["RID"].nunique(dropna=True)) if "RID" in df.columns else None,
            "note": note,
        }

    flow_rows.append(_flow_row(step="triads_all", df=triads, note="PET-anchored triads (all)"))
    flow_rows.append(_flow_row(step="triads_all_csf_a_available", df=triads[triads["csf_a_pos"].notna()].copy(), note="ABETA42/40 ratio available"))
    if split_meta.get("method") == "rid_random":
        flow_rows.append(_flow_row(step="derivation_all_triads", df=triads_train_all_post, note="Derivation split (all triads)"))
        flow_rows.append(_flow_row(step="evaluation_all_triads", df=triads_eval_all, note="Evaluation split (all triads)"))
    flow_rows.append(_flow_row(step="one_triad_per_rid_all", df=triads_one_per_rid, note="Deterministic one-triad-per-RID selection (all splits)"))
    flow_rows.append(
        _flow_row(
            step="one_triad_per_rid_all_csf_a_available",
            df=triads_one_per_rid[triads_one_per_rid["csf_a_pos"].notna()].copy(),
            note="One per RID, restricted to RIDs with CSF Aβ42/40 available",
        )
    )
    flow_rows.append(_flow_row(step="evaluation_one_triad_per_rid", df=triads_eval, note="Primary analysis set (evaluation split)"))
    flow_rows.append(
        _flow_row(
            step="evaluation_one_triad_per_rid_csf_a_available",
            df=triads_eval[triads_eval["csf_a_pos"].notna()].copy(),
            note="Primary swap-eligible subset (CSF-A available)",
        )
    )
    pd.DataFrame.from_records(flow_rows).to_csv(audit_dir / "stard_flow_counts.csv", index=False)

    (audit_dir / "stard_flow_summary.md").write_text(
        "\n".join(
            [
                "# STARD-style cohort accounting (Paper A)",
                "",
                "Primary analyses use the held-out evaluation split with one triad per RID (deterministic selection).",
                "Benchmark-swap analyses additionally require CSF Aβ42/40 ratio availability (ABETA42 and ABETA40).",
                "",
                f"- Evaluation (one per RID): {int(len(triads_eval))} triads (RIDs={int(triads_eval['RID'].nunique(dropna=True))})",
                f"- Evaluation (one per RID, CSF-A available): {int(triads_eval['csf_a_pos'].notna().sum())} triads (RIDs={int(triads_eval.loc[triads_eval['csf_a_pos'].notna(), 'RID'].nunique(dropna=True))})",
                "",
                "See `outputs/paperA/audit/stard_flow_counts.csv` for the full accounting table.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        import matplotlib.pyplot as plt  # type: ignore

        def _fmt_counts(*, triads_n: int, rids_n: int) -> str:
            return f"Triads: {triads_n}; Participants: {rids_n}"

        eval_rids = int(triads_eval["RID"].nunique(dropna=True))
        eval_rows = int(len(triads_eval))
        eval_swap_rids = int(triads_eval.loc[triads_eval["csf_a_pos"].notna(), "RID"].nunique(dropna=True))
        eval_swap_rows = int(triads_eval["csf_a_pos"].notna().sum())

        eval_all_rids = int(triads_eval_all["RID"].nunique(dropna=True))
        eval_all_rows = int(len(triads_eval_all))

        deriv_all_rids = int(triads_train_all_post["RID"].nunique(dropna=True))
        deriv_all_rows = int(len(triads_train_all_post))
        deriv_one_rids = int(triads_train["RID"].nunique(dropna=True))
        deriv_one_rows = int(len(triads_train))

        fig, ax = plt.subplots(figsize=(10.5, 7.0))
        ax.axis("off")

        def _box(x: float, y: float, text: str, *, fontsize: int = 10) -> None:
            ax.text(
                x,
                y,
                text,
                ha="center",
                va="center",
                fontsize=fontsize,
                bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F4F6F8", "edgecolor": "#4C566A"},
            )

        def _arrow(x0: float, y0: float, x1: float, y1: float) -> None:
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#2E3440"})

        # Layout: top → split → (derivation, evaluation) → swap-eligible (evaluation branch).
        top_x, top_y = 0.5, 0.90
        split_x, split_y = 0.5, 0.74
        deriv_x, deriv_y = 0.28, 0.50
        eval_x, eval_y = 0.72, 0.50
        swap_x, swap_y = 0.72, 0.24

        _box(
            top_x,
            top_y,
            "PET-anchored triads\n"
            f"{_fmt_counts(triads_n=int(len(triads)), rids_n=int(triads['RID'].nunique(dropna=True)))}",
            fontsize=11,
        )

        if split_meta.get("method") == "rid_random":
            _box(
                split_x,
                split_y,
                "Participant-level split (RID)\n"
                f"Derivation: {split_meta.get('n_derivation_rids')} participants\n"
                f"Evaluation: {split_meta.get('n_evaluation_rids')} participants",
            )
            _arrow(top_x, top_y - 0.07, split_x, split_y + 0.07)

            _box(
                deriv_x,
                deriv_y,
                "Derivation set (threshold selection)\n"
                f"All triads: {deriv_all_rows}; Participants: {deriv_all_rids}\n"
                f"One triad per participant: {deriv_one_rows}",
                fontsize=9,
            )
            _box(
                eval_x,
                eval_y,
                "Evaluation set (primary reporting)\n"
                f"All triads: {eval_all_rows}; Participants: {eval_all_rids}\n"
                f"One triad per participant: {eval_rows}",
                fontsize=9,
            )
            _arrow(split_x + 0.06, split_y - 0.07, eval_x - 0.06, eval_y + 0.09)
            _arrow(split_x - 0.06, split_y - 0.07, deriv_x + 0.06, deriv_y + 0.09)
        else:
            _arrow(top_x, top_y - 0.07, eval_x, eval_y + 0.09)
            _box(
                eval_x,
                eval_y,
                "Evaluation set (primary reporting)\n"
                f"All triads: {eval_all_rows}; Participants: {eval_all_rids}\n"
                f"One triad per participant: {eval_rows}",
                fontsize=9,
            )

        _box(
            swap_x,
            swap_y,
            "Swap-eligible subset (primary)\nCSF Aβ42/40 available\n"
            f"Triads: {eval_swap_rows}; Participants: {eval_swap_rids}",
        )
        _arrow(eval_x, eval_y - 0.09, swap_x, swap_y + 0.09)

        ax.text(
            0.01,
            0.02,
            "Note: participants are unique RIDs; APS2 analyses are PET-only appendix and not part of PET↔CSF benchmark swap.",
            ha="left",
            va="bottom",
            fontsize=9,
            color="#444444",
        )

        fig.tight_layout()
        fig.savefig(figures_dir / "figure1_cohort_flow.svg")
        fig.savefig(figures_dir / "figure1_cohort_flow.png", dpi=200)
        plt.close(fig)
    except Exception:
        pass

    # Pooled paired benchmark swap (precision-enhancing): all swap-eligible RIDs using locked thresholds (no refitting).
    pooled_swap_one = triads_one_per_rid[triads_one_per_rid["csf_a_pos"].notna()].copy()
    pooled_swap_all_triads = triads[triads["csf_a_pos"].notna()].copy()

    pooled_swap_rows: list[dict] = []
    for op in operating_points:
        op_name = str(op["name"])
        for endpoint in plasma_endpoints:
            cat_col = f"plasma_cat__{endpoint}__{op_name}"
            if cat_col not in pooled_swap_one.columns:
                continue
            for policy in policies:
                swap_one = _bootstrap_benchmark_swap(
                    df=pooled_swap_one,
                    rid_col="RID",
                    cat_col=cat_col,
                    ref_a_col="pet_pos",
                    ref_b_col="csf_a_pos",
                    policy=policy,
                    n_rep=n_rep,
                    seed=seed,
                )
                pooled_swap_rows.append(
                    {
                        "analysis_set": "pooled_swap_eligible_one_triad_per_rid",
                        "includes_splits": "derivation+evaluation",
                        "operating_point": op_name,
                        "endpoint": endpoint,
                        "policy": policy,
                        "swap": "CSF-A minus PET",
                        "threshold_split_method": split_method,
                        "split_derivation_fraction": split_meta.get("derivation_fraction"),
                        "split_seed": split_meta.get("seed"),
                        **swap_one,
                    }
                )

                swap_all = _bootstrap_benchmark_swap(
                    df=pooled_swap_all_triads,
                    rid_col="RID",
                    cat_col=cat_col,
                    ref_a_col="pet_pos",
                    ref_b_col="csf_a_pos",
                    policy=policy,
                    n_rep=n_rep,
                    seed=seed,
                )
                pooled_swap_rows.append(
                    {
                        "analysis_set": "pooled_swap_eligible_all_triads",
                        "includes_splits": "derivation+evaluation",
                        "operating_point": op_name,
                        "endpoint": endpoint,
                        "policy": policy,
                        "swap": "CSF-A minus PET",
                        "threshold_split_method": split_method,
                        "split_derivation_fraction": split_meta.get("derivation_fraction"),
                        "split_seed": split_meta.get("seed"),
                        **swap_all,
                    }
                )

    pd.DataFrame.from_records(pooled_swap_rows).to_csv(audit_dir / "pooled_swap_eligible_benchmark_swap.csv", index=False)
    (audit_dir / "pooled_swap_eligible_summary.md").write_text(
        "\n".join(
            [
                "# Pooled paired benchmark swap (precision-enhancing)",
                "",
                "This analysis increases precision for the paired benchmark-swap estimate by using all RIDs with CSF Aβ42/40 available.",
                "Plasma thresholds are locked from the derivation split (no refitting).",
                "",
                f"- swap-eligible triads (all): {int(len(pooled_swap_all_triads))} (RIDs={int(pooled_swap_all_triads['RID'].nunique(dropna=True))})",
                f"- swap-eligible (one per RID): {int(len(pooled_swap_one))} (RIDs={int(pooled_swap_one['RID'].nunique(dropna=True))})",
                "",
                "Primary (leakage-controlled) benchmark-swap results remain the evaluation-only one-triad-per-RID analysis in `outputs/paperA/benchmark_swap.csv`.",
                "Use this pooled analysis as a pre-labeled precision-enhancing supplement/appendix table.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Benchmark-swap forest plot (high-impact summary figure).
    try:
        import matplotlib.pyplot as plt  # type: ignore

        def _swap_label(policy: str) -> str:
            return {"indet_to_negative": "Indeterminate→Negative", "indet_to_positive": "Indeterminate→Positive"}.get(
                policy, policy
            )

        endpoint_labels = {
            "c2n_plasma_abeta42_abeta40_abeta_ratio": "Plasma Aβ42/40 ratio",
            "c2n_plasma_ptau217_ratio_ptau217_ratio": "Plasma %p-tau217",
        }
        endpoint_order = [
            "c2n_plasma_abeta42_abeta40_abeta_ratio",
            "c2n_plasma_ptau217_ratio_ptau217_ratio",
        ]
        op_order = ["op_ppa0.95_npa0.95", "op_ppa0.90_npa0.90"]
        op_style = {
            "op_ppa0.95_npa0.95": {"color": "#1f77b4", "marker": "o", "label": "op 0.95/0.95"},
            "op_ppa0.90_npa0.90": {"color": "#ff7f0e", "marker": "s", "label": "op 0.90/0.90"},
        }
        op_offset = {"op_ppa0.95_npa0.95": -0.12, "op_ppa0.90_npa0.90": 0.12}
        policies_for_plot = ["indet_to_negative", "indet_to_positive"]

        def _forest(
            df_plot: pd.DataFrame,
            *,
            out_stem: str,
            title: str,
            note: str,
        ) -> None:
            df_plot = df_plot.copy()
            df_plot = df_plot[df_plot["policy"].isin(policies_for_plot)].copy()
            df_plot = df_plot[df_plot["operating_point"].isin(op_order)].copy()
            df_plot = df_plot[df_plot["endpoint"].isin(endpoint_order)].copy()

            if df_plot.empty:
                return

            y_positions = {ep: i for i, ep in enumerate(endpoint_order)}
            fig, axes = plt.subplots(
                nrows=len(policies_for_plot),
                ncols=2,
                figsize=(11, 6),
                sharey=True,
            )
            fig.suptitle(title, fontsize=12)

            for row_i, policy in enumerate(policies_for_plot):
                df_pol = df_plot[df_plot["policy"] == policy].copy()
                for col_i, metric in enumerate(["PPA", "NPA"]):
                    ax = axes[row_i][col_i]
                    ax.axvline(0.0, color="#666666", lw=1.0, zorder=0)
                    ax.grid(axis="x", color="#dddddd", lw=0.8, zorder=0)

                    for ep in endpoint_order:
                        for op in op_order:
                            row = df_pol[(df_pol["endpoint"] == ep) & (df_pol["operating_point"] == op)]
                            if row.empty:
                                continue
                            r0 = row.iloc[0]
                            delta = float(r0[f"delta_{metric}"])
                            lo = float(r0[f"delta_{metric}_lo"])
                            hi = float(r0[f"delta_{metric}_hi"])
                            y = float(y_positions[ep]) + float(op_offset.get(op, 0.0))
                            style = op_style.get(op, {"color": "black", "marker": "o", "label": op})
                            ax.errorbar(
                                delta,
                                y,
                                xerr=[[delta - lo], [hi - delta]],
                                fmt=style["marker"],
                                color=style["color"],
                                ecolor=style["color"],
                                elinewidth=1.2,
                                capsize=3,
                                markersize=6,
                                label=style["label"],
                                zorder=3,
                            )

                    ax.set_xlabel(f"Δ{metric} (CSF-A − PET)")
                    if col_i == 0:
                        ax.set_title(_swap_label(policy), loc="left", fontsize=11)

                    ax.set_yticks([y_positions[ep] for ep in endpoint_order])
                    ax.set_yticklabels([endpoint_labels.get(ep, ep) for ep in endpoint_order])
                    ax.invert_yaxis()

                    # Tight x-limits for readability.
                    lows = []
                    highs = []
                    for metric2 in [metric]:
                        lows.extend(df_pol[f"delta_{metric2}_lo"].dropna().astype("float64").tolist())
                        highs.extend(df_pol[f"delta_{metric2}_hi"].dropna().astype("float64").tolist())
                    if lows and highs:
                        x_min = float(min(lows))
                        x_max = float(max(highs))
                        pad = max(0.02, 0.1 * (x_max - x_min))
                        ax.set_xlim(x_min - pad, x_max + pad)

            # De-duplicate legend entries.
            handles, labels = axes[0][0].get_legend_handles_labels()
            uniq = {}
            for h, lab in zip(handles, labels, strict=True):
                uniq.setdefault(lab, h)
            fig.legend(
                uniq.values(),
                uniq.keys(),
                loc="lower center",
                ncol=2,
                frameon=False,
                bbox_to_anchor=(0.5, 0.01),
            )

            fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=9)
            fig.tight_layout(rect=[0.0, 0.05, 1.0, 0.95])
            fig.savefig(figures_dir / f"{out_stem}.svg")
            fig.savefig(figures_dir / f"{out_stem}.png", dpi=200)
            plt.close(fig)

        pooled_df = pd.DataFrame.from_records(pooled_swap_rows)
        pooled_one = pooled_df[pooled_df["analysis_set"] == "pooled_swap_eligible_one_triad_per_rid"].copy()
        pooled_one_for_figure = pooled_one[
            pooled_one["policy"].isin(policies_for_plot)
            & pooled_one["operating_point"].isin(op_order)
            & pooled_one["endpoint"].isin(endpoint_order)
        ].copy()
        pooled_one_for_figure.to_csv(audit_dir / "benchmark_swap_forest_data_pooled_one_triad_per_rid.csv", index=False)
        _forest(
            pooled_one,
            out_stem="figure2_benchmark_swap_forest_pooled_one_triad_per_rid",
            title="Paired benchmark swap (CSF-A − PET) with locked thresholds (pooled swap-eligible)",
            note="Pooled swap-eligible participants (one triad per RID). Plasma thresholds locked from derivation split (no refitting). 95% CIs: RID bootstrap.",
        )

        eval_df = pd.DataFrame.from_records(swap_rows)
        eval_df = eval_df[eval_df["analysis_set"] == "one_triad_per_rid"].copy()
        eval_df_for_figure = eval_df[
            eval_df["policy"].isin(policies_for_plot)
            & eval_df["operating_point"].isin(op_order)
            & eval_df["endpoint"].isin(endpoint_order)
        ].copy()
        eval_df_for_figure.to_csv(audit_dir / "benchmark_swap_forest_data_evaluation_one_triad_per_rid.csv", index=False)
        _forest(
            eval_df,
            out_stem="figure2_benchmark_swap_forest_evaluation_one_triad_per_rid",
            title="Paired benchmark swap (CSF-A − PET) with locked thresholds (evaluation split)",
            note="Held-out evaluation split (one triad per RID). Plasma thresholds locked from derivation split. 95% CIs: RID bootstrap.",
        )

    except Exception:
        pass

    # Stage stratification (Neurology reviewer expectation).
    if "clin_dx_simplified" in triads_eval.columns:
        stage_counts = (
            triads_eval["clin_dx_simplified"]
            .astype("string")
            .fillna("Missing")
            .value_counts(dropna=False)
            .rename_axis("stage")
            .reset_index(name="n_rids")
        )
        stage_counts.to_csv(audit_dir / "stage_counts.csv", index=False)

        stage_metric_rows: list[dict] = []
        stage_swap_rows: list[dict] = []
        for stage in sorted(stage_counts["stage"].tolist()):
            df_stage = triads_eval[triads_eval["clin_dx_simplified"].astype("string").fillna("Missing") == stage].copy()
            mr, sr = _compute_metrics_and_swap(df_stage, analysis_set="one_triad_per_rid", out_2x2_dir=None)
            for row in mr:
                row["stage"] = stage
            for row in sr:
                row["stage"] = stage
            stage_metric_rows.extend(mr)
            stage_swap_rows.extend(sr)

        pd.DataFrame.from_records(stage_metric_rows).to_csv(audit_dir / "stage_stratified_metrics.csv", index=False)
        pd.DataFrame.from_records(stage_swap_rows).to_csv(audit_dir / "stage_stratified_benchmark_swap.csv", index=False)

    # Borderline strata (prespecified): PET Centiloid 20–40 and CSF Aβ42/40 within ±10% of cutpoint.
    try:
        cut_a = float(defs["benchmarks"]["csf"]["primary_csf_a"]["cutpoint"])
    except Exception:
        cut_a = 0.0525
    csf_band = 0.10
    cl_lo, cl_hi = 20.0, 40.0

    cent_eval = pd.to_numeric(triads_eval.get("CENTILOIDS"), errors="coerce")
    ratio_eval = pd.to_numeric(triads_eval.get("abeta42_40_ratio"), errors="coerce")
    pet_borderline = cent_eval.notna() & (cent_eval >= cl_lo) & (cent_eval <= cl_hi)
    csf_borderline = ratio_eval.notna() & (ratio_eval >= cut_a * (1.0 - csf_band)) & (ratio_eval <= cut_a * (1.0 + csf_band))

    strata_defs = [
        ("pet_centiloid_20_40", pet_borderline),
        ("csf_abeta42_40_within_10pct_cutpoint", csf_borderline),
        ("both_pet_and_csf_borderline", pet_borderline & csf_borderline),
    ]

    borderline_summary_lines = ["# Borderline strata (prespecified)", ""]
    borderline_summary_lines.append(f"- PET Centiloid band: [{cl_lo:g}, {cl_hi:g}]")
    borderline_summary_lines.append(f"- CSF Aβ42/40 band: cutpoint={cut_a:g} with ±{csf_band:.0%}")
    borderline_summary_lines.append("")
    borderline_counts: list[dict] = []
    borderline_metric_rows: list[dict] = []
    borderline_swap_rows: list[dict] = []
    for name, mask in strata_defs:
        df_sub = triads_eval.loc[mask].copy()
        borderline_counts.append({"stratum": name, "n_rids": int(df_sub["RID"].nunique(dropna=True)), "n_triads": int(len(df_sub))})
        borderline_summary_lines.append(f"- {name}: {int(len(df_sub))} triads (RIDs={int(df_sub['RID'].nunique(dropna=True))})")
        mr, sr = _compute_metrics_and_swap(df_sub, analysis_set="one_triad_per_rid", out_2x2_dir=None)
        for row in mr:
            row["stratum"] = name
        for row in sr:
            row["stratum"] = name
        borderline_metric_rows.extend(mr)
        borderline_swap_rows.extend(sr)

    pd.DataFrame.from_records(borderline_counts).to_csv(audit_dir / "borderline_strata_counts.csv", index=False)
    pd.DataFrame.from_records(borderline_metric_rows).to_csv(audit_dir / "borderline_strata_metrics.csv", index=False)
    pd.DataFrame.from_records(borderline_swap_rows).to_csv(audit_dir / "borderline_strata_benchmark_swap.csv", index=False)
    (audit_dir / "borderline_strata_summary.md").write_text("\n".join(borderline_summary_lines) + "\n", encoding="utf-8")

    # PET vs CSF-A discordance mechanics (paired subset) + McNemar test.
    try:
        from scipy import stats as _sp_stats  # type: ignore

        disc = triads_eval[triads_eval["pet_pos"].notna() & triads_eval["csf_a_pos"].notna()].copy()
        pet_bin = pd.to_numeric(disc["pet_pos"], errors="coerce").astype("int64")
        csf_bin = pd.to_numeric(disc["csf_a_pos"], errors="coerce").astype("int64")
        a = int(((pet_bin == 1) & (csf_bin == 1)).sum())
        b = int(((pet_bin == 1) & (csf_bin == 0)).sum())
        c = int(((pet_bin == 0) & (csf_bin == 1)).sum())
        d = int(((pet_bin == 0) & (csf_bin == 0)).sum())
        n = a + b + c + d

        try:
            p_exact = float(_sp_stats.binomtest(min(b, c), b + c, 0.5).pvalue) if (b + c) > 0 else float("nan")
        except Exception:
            p_exact = float(_sp_stats.binom_test(min(b, c), b + c, 0.5)) if (b + c) > 0 else float("nan")

        chi2_cc = float(((abs(b - c) - 1) ** 2) / (b + c)) if (b + c) > 0 else float("nan")
        p_chi2_cc = float(_sp_stats.chi2.sf(chi2_cc, df=1)) if (b + c) > 0 else float("nan")

        pd.DataFrame(
            [
                {
                    "n": n,
                    "PET1_CSF1": a,
                    "PET1_CSF0": b,
                    "PET0_CSF1": c,
                    "PET0_CSF0": d,
                    "mcnemar_exact_p": p_exact,
                    "mcnemar_chi2_cc": chi2_cc,
                    "mcnemar_chi2_cc_p": p_chi2_cc,
                }
            ]
        ).to_csv(audit_dir / "pet_csf_discordance.csv", index=False)

        (audit_dir / "pet_csf_mcnemar.md").write_text(
            "\n".join(
                [
                    "# PET vs CSF-A discordance (McNemar)",
                    "",
                    f"- n (paired, one per RID): {n}",
                    f"- discordant counts: PET+/CSF- = {b}, PET-/CSF+ = {c}",
                    f"- McNemar exact p-value: {p_exact:.6g}" if p_exact == p_exact else "- McNemar exact p-value: NA",
                    f"- McNemar chi-square (cc) p-value: {p_chi2_cc:.6g}" if p_chi2_cc == p_chi2_cc else "- McNemar chi-square (cc) p-value: NA",
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        if n > 0:
            quad = disc.copy()
            quad["quadrant"] = (
                "PET" + pet_bin.astype("string") + "_CSF" + csf_bin.astype("string")
            )
            summary_rows: list[dict] = []
            for q in ["PET1_CSF1", "PET1_CSF0", "PET0_CSF1", "PET0_CSF0"]:
                dsub = quad[quad["quadrant"] == q]
                summary_rows.append(
                    {
                        "quadrant": q,
                        "n": int(len(dsub)),
                        "median_CENTILOIDS": float(pd.to_numeric(dsub.get("CENTILOIDS"), errors="coerce").median()),
                        "median_abeta42_40_ratio": float(pd.to_numeric(dsub.get("abeta42_40_ratio"), errors="coerce").median()),
                        "median_PTAU": float(pd.to_numeric(dsub.get("PTAU"), errors="coerce").median()),
                        "median_plasma_abeta42_40": float(pd.to_numeric(dsub.get("c2n_plasma_abeta42_abeta40_abeta_ratio"), errors="coerce").median()),
                        "median_plasma_pct_ptau217": float(pd.to_numeric(dsub.get("c2n_plasma_ptau217_ratio_ptau217_ratio"), errors="coerce").median()),
                        "median_delta_pet_csf_days": float(pd.to_numeric(dsub.get("delta_pet_csf_days"), errors="coerce").median()),
                    }
                )
            pd.DataFrame.from_records(summary_rows).to_csv(audit_dir / "pet_csf_quadrant_summary.csv", index=False)
    except Exception:
        pass

    # Time-gap robustness beyond the window (≤7 days vs >7 days).
    timegap_fields = ["delta_pet_plasma_days", "delta_pet_csf_days"]
    timegap_metric_rows: list[dict] = []
    timegap_swap_rows: list[dict] = []
    for delta_field in timegap_fields:
        if delta_field not in triads_eval.columns:
            continue
        s = pd.to_numeric(triads_eval[delta_field], errors="coerce")
        for stratum, mask in [("le_7d", s <= 7), ("gt_7d", s > 7)]:
            df_sub = triads_eval.loc[mask.fillna(False)].copy()
            mr, sr = _compute_metrics_and_swap(df_sub, analysis_set="one_triad_per_rid", out_2x2_dir=None)
            for row in mr:
                row["delta_field"] = delta_field
                row["delta_stratum"] = stratum
            for row in sr:
                row["delta_field"] = delta_field
                row["delta_stratum"] = stratum
            timegap_metric_rows.extend(mr)
            timegap_swap_rows.extend(sr)
    pd.DataFrame.from_records(timegap_metric_rows).to_csv(audit_dir / "timegap_stratified_metrics.csv", index=False)
    pd.DataFrame.from_records(timegap_swap_rows).to_csv(audit_dir / "timegap_stratified_benchmark_swap.csv", index=False)

    missing_rows: list[dict] = []
    for endpoint in sorted(set(plasma_endpoints + plasma_secondary)):
        if endpoint not in triads_eval.columns:
            continue
        miss = triads_eval[endpoint].isna()
        total = int(len(triads_eval))
        missing_rows.append(
            {
                "endpoint": endpoint,
                "n_eval": total,
                "n_missing": int(miss.sum()),
                "missing_fraction": float(miss.mean()) if total else float("nan"),
                "missing_fraction_pet_pos": float(miss[triads_eval["pet_pos"] == 1].mean()),
                "missing_fraction_pet_neg": float(miss[triads_eval["pet_pos"] == 0].mean()),
                "missing_fraction_csf_a_pos": float(miss[triads_eval["csf_a_pos"] == 1].mean()),
                "missing_fraction_csf_a_neg": float(miss[triads_eval["csf_a_pos"] == 0].mean()),
            }
        )
    pd.DataFrame.from_records(missing_rows).to_csv(audit_dir / "missingness_summary.csv", index=False)

    # APS2 vs component concordance (evaluation split; exploratory).
    if "APS2_C2N" in triads_eval.columns:
        aps2 = pd.to_numeric(triads_eval["APS2_C2N"], errors="coerce")
        corr_rows: list[dict] = []
        for endpoint in sorted(set(plasma_endpoints + plasma_secondary)):
            if endpoint == "APS2_C2N" or endpoint not in triads_eval.columns:
                continue
            x = pd.to_numeric(triads_eval[endpoint], errors="coerce")
            mask = aps2.notna() & x.notna()
            if int(mask.sum()) < 3:
                corr_rows.append({"endpoint": endpoint, "n": int(mask.sum()), "pearson_r": None, "spearman_r": None})
                continue
            pearson = float(np.corrcoef(aps2[mask].to_numpy(), x[mask].to_numpy())[0, 1])
            spearman = float(np.corrcoef(aps2[mask].rank().to_numpy(), x[mask].rank().to_numpy())[0, 1])
            corr_rows.append({"endpoint": endpoint, "n": int(mask.sum()), "pearson_r": pearson, "spearman_r": spearman})
        pd.DataFrame.from_records(corr_rows).to_csv(audit_dir / "aps2_component_correlations.csv", index=False)

    if mfg_op_name is not None:
        mfg_endpoint = str(mfg["endpoint"])
        mfg_col = f"plasma_cat__{mfg_endpoint}__{mfg_op_name}"
        if mfg_col in triads_eval.columns:
            conc_rows: list[dict] = []
            aps2_cat = triads_eval[mfg_col].astype("string")
            for op in operating_points:
                op_name = str(op["name"])
                for endpoint in plasma_endpoints:
                    if endpoint == mfg_endpoint:
                        continue
                    comp_col = f"plasma_cat__{endpoint}__{op_name}"
                    if comp_col not in triads_eval.columns:
                        continue
                    comp_cat = triads_eval[comp_col].astype("string")
                    tab = (
                        pd.crosstab(comp_cat, aps2_cat, dropna=False)
                        .rename_axis(index="component_cat", columns="aps2_cat")
                        .stack()
                        .reset_index(name="n")
                    )
                    tab["operating_point"] = op_name
                    tab["endpoint"] = endpoint
                    conc_rows.extend(tab.to_dict(orient="records"))
            pd.DataFrame.from_records(conc_rows).to_csv(audit_dir / "aps2_component_concordance.csv", index=False)

    (audit_dir / "duplicate_handling.md").write_text(
        "\n".join(
            [
                "# Duplicate handling note",
                "",
                "Canonicalization and deterministic tie-break rules for duplicate events are documented in:",
                f"- {defs['evidence_core']['join_rules']}",
                "",
                "Paper A uses the frozen evidence core outputs; no additional ad hoc deduplication is performed in the Paper A pack.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    (audit_dir / "paired_analysis_note.md").write_text(
        "\n".join(
            [
                "# Paired benchmark swap (Paper A)",
                "",
                "Benchmark swap (CSF-A minus PET) is computed within the same held-out evaluation split.",
                "Uncertainty uses a participant-level (RID-clustered) paired bootstrap as defined in `outputs/paperA/definitions.yaml`.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    def _compute_metrics_for_eval(eval_df: pd.DataFrame, *, qc_filter: str) -> list[dict]:
        rows: list[dict] = []
        for op in operating_points:
            op_name = str(op["name"])
            for endpoint in plasma_endpoints:
                cat_col = f"plasma_cat__{endpoint}__{op_name}"
                if cat_col not in eval_df.columns:
                    continue
                for benchmark_name, ref_col in [("PET", "pet_pos"), ("CSF-A", "csf_a_pos"), ("CSF-A+T", "csf_at_pos")]:
                    y_ref = eval_df[ref_col]
                    y_num = pd.to_numeric(y_ref, errors="coerce")
                    for policy in policies:
                        x_bin, keep_mask, n_indet = _binary_from_tri_category(eval_df[cat_col], policy=policy)
                        y_keep = y_num.to_numpy()[keep_mask]
                        ok = np.isin(y_keep, [0, 1]) & np.isin(x_bin, [0, 1])
                        y_arr = y_keep[ok].astype("int64")
                        x_arr = x_bin[ok].astype("int64")

                        counts = _confusion_counts(y_arr, x_arr)
                        stats = _ppa_npa(counts)
                        n_used = counts["TP"] + counts["FP"] + counts["TN"] + counts["FN"]
                        ppa_den = counts["TP"] + counts["FN"]
                        npa_den = counts["TN"] + counts["FP"]

                        rows.append(
                            {
                                "qc_filter": qc_filter,
                                "operating_point": op_name,
                                "endpoint": endpoint,
                                "benchmark": benchmark_name,
                                "policy": policy,
                                "n_eval_rids": int(eval_df["RID"].nunique(dropna=True)),
                                "n_eval_triads": int(len(eval_df)),
                                "n_total_with_cat": int(eval_df[cat_col].notna().sum()),
                                "n_indeterminate": n_indet,
                                "indeterminate_fraction": float(n_indet / int(eval_df[cat_col].notna().sum()))
                                if int(eval_df[cat_col].notna().sum())
                                else float("nan"),
                                "n_used": n_used,
                                "ppa_den": ppa_den,
                                "npa_den": npa_den,
                                **counts,
                                **stats,
                            }
                        )
        return rows

    qc_rows: list[dict] = []
    qc_rows.extend(_compute_metrics_for_eval(triads_eval, qc_filter="none"))
    qc_mask = (~triads_eval["pet_qc_fail"].fillna(False)) & (~triads_eval["pet_centiloid_extreme"].fillna(False))
    qc_rows.extend(_compute_metrics_for_eval(triads_eval.loc[qc_mask].copy(), qc_filter="pet_qc_nonfail_and_centiloid_nonextreme"))
    pd.DataFrame.from_records(qc_rows).to_csv(audit_dir / "qc_sensitivity_metrics.csv", index=False)

    # Tight-window subset sensitivity (subset restriction; no rematching).
    tight_cfg = (
        defs.get("evidence_core", {})
        .get("pairing", {})
        .get("sensitivity_windows_days", {})
        .get("tight")
    )
    if isinstance(tight_cfg, dict):
        wp = float(tight_cfg.get("pet_plasma", 60))
        wc = float(tight_cfg.get("pet_csf", 120))
        wbc = float(tight_cfg.get("plasma_csf", 120))

        tight_mask = (
            (pd.to_numeric(triads_eval["delta_pet_plasma_days"], errors="coerce") <= wp)
            & (pd.to_numeric(triads_eval["delta_pet_csf_days"], errors="coerce") <= wc)
            & (pd.to_numeric(triads_eval["delta_plasma_csf_days"], errors="coerce") <= wbc)
        )
        triads_eval_tight = triads_eval.loc[tight_mask].copy()

        (audit_dir / "tight_window_subset_summary.md").write_text(
            "\n".join(
                [
                    "# Tight-window subset sensitivity (no rematching)",
                    "",
                    f"- Window thresholds (days, absolute): PET–plasma ≤ {wp:g}, PET–CSF ≤ {wc:g}, plasma–CSF ≤ {wbc:g}",
                    f"- Evaluation triads (full): {int(len(triads_eval))} (RIDs={int(triads_eval['RID'].nunique(dropna=True))})",
                    f"- Evaluation triads (tight subset): {int(len(triads_eval_tight))} (RIDs={int(triads_eval_tight['RID'].nunique(dropna=True))})",
                    "",
                    "This sensitivity restricts to evaluation triads that already satisfy the tighter windows; it does not re-match events under the tighter windows.",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        pd.DataFrame.from_records(_compute_metrics_for_eval(triads_eval_tight, qc_filter="tight_window_subset")).to_csv(
            audit_dir / "tight_window_subset_metrics.csv", index=False
        )

        tight_swap_rows: list[dict] = []
        for op in operating_points:
            op_name = str(op["name"])
            for endpoint in plasma_endpoints:
                cat_col = f"plasma_cat__{endpoint}__{op_name}"
                if cat_col not in triads_eval_tight.columns:
                    continue
                for policy in policies:
                    swap = _bootstrap_benchmark_swap(
                        df=triads_eval_tight,
                        rid_col="RID",
                        cat_col=cat_col,
                        ref_a_col="pet_pos",
                        ref_b_col="csf_a_pos",
                        policy=policy,
                        n_rep=n_rep,
                        seed=seed,
                    )
                    tight_swap_rows.append(
                        {
                            "operating_point": op_name,
                            "endpoint": endpoint,
                            "policy": policy,
                            "swap": "CSF-A minus PET",
                            "subset": "tight_window_subset",
                            **swap,
                        }
                    )
        pd.DataFrame.from_records(tight_swap_rows).to_csv(audit_dir / "tight_window_subset_benchmark_swap.csv", index=False)

    _build_aps2_pet_only_appendix(defs=defs, out_dir=out_dir)

    return PaperAArtifacts(
        cohort_triads=cohort_path,
        plasma_thresholds=thresholds_path,
        plasma_labels=labels_path,
        metrics_summary=metrics_path,
        benchmark_swap=swap_path,
    )
