from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _format_mean_sd(values: pd.Series, digits: int = 1) -> str:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric.dropna()
    if numeric.empty:
        return ""
    mean = float(numeric.mean())
    sd = float(numeric.std(ddof=1))
    return f"{mean:.{digits}f} ({sd:.{digits}f})"


def _format_median_iqr_abs(values: pd.Series, digits: int = 0) -> str:
    numeric = pd.to_numeric(values, errors="coerce").abs()
    numeric = numeric.dropna()
    if numeric.empty:
        return ""
    q25 = float(numeric.quantile(0.25))
    med = float(numeric.quantile(0.50))
    q75 = float(numeric.quantile(0.75))
    return f"{med:.{digits}f} [{q25:.{digits}f}, {q75:.{digits}f}]"


def _format_n_pct(n: int, denom: int, digits: int = 1) -> str:
    if denom <= 0:
        return f"{n} (NA)"
    pct = 100.0 * n / denom
    return f"{n} ({pct:.{digits}f}%)"


def _format_rate(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return ""
    return f"{100.0 * float(value):.{digits}f}%"


def _format_metric_with_ci(value: float, lo: float, hi: float, digits: int = 1) -> str:
    if pd.isna(value) or pd.isna(lo) or pd.isna(hi):
        return ""
    return f"{100.0 * float(value):.{digits}f}% ({100.0 * float(lo):.{digits}f}–{100.0 * float(hi):.{digits}f})"


def _format_delta_pp(value: float, lo: float, hi: float, digits: int = 1) -> str:
    if pd.isna(value) or pd.isna(lo) or pd.isna(hi):
        return ""
    return f"{100.0 * float(value):+.{digits}f} pp ({100.0 * float(lo):+.{digits}f}–{100.0 * float(hi):+.{digits}f})"


def _write_table(df: pd.DataFrame, out_csv: Path, out_md: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    out_md.write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")


def build_table1(one_triad_per_rid: pd.DataFrame) -> pd.DataFrame:
    eval_df = one_triad_per_rid.loc[one_triad_per_rid["split"] == "evaluation"].copy()
    swap_eval_df = eval_df.loc[eval_df["has_csf_a"] == True].copy()  # noqa: E712

    def col(values_eval: str, values_swap: str) -> dict[str, str]:
        return {"Evaluation (n=128)": values_eval, "Swap-eligible† (n=50)": values_swap}

    n_eval = int(eval_df["RID"].nunique())
    n_swap = int(swap_eval_df["RID"].nunique())

    rows: list[dict[str, str]] = []
    rows.append({"Characteristic": "Participants, n", **col(str(n_eval), str(n_swap))})
    rows.append(
        {
            "Characteristic": "Age, years",
            **col(_format_mean_sd(eval_df["clin_AGE"]), _format_mean_sd(swap_eval_df["clin_AGE"])),
        }
    )
    rows.append(
        {
            "Characteristic": "Female sex, n (%)",
            **col(
                _format_n_pct(int((eval_df["clin_PTGENDER"] == "Female").sum()), n_eval),
                _format_n_pct(int((swap_eval_df["clin_PTGENDER"] == "Female").sum()), n_swap),
            ),
        }
    )
    rows.append(
        {
            "Characteristic": "Education, years",
            **col(
                _format_mean_sd(eval_df["clin_PTEDUCAT"]),
                _format_mean_sd(swap_eval_df["clin_PTEDUCAT"]),
            ),
        }
    )
    rows.append(
        {
            "Characteristic": "APOE ε4 carrier (≥1), n (%)",
            **col(
                _format_n_pct(int((eval_df["clin_APOE4"] >= 1).sum()), n_eval),
                _format_n_pct(int((swap_eval_df["clin_APOE4"] >= 1).sum()), n_swap),
            ),
        }
    )
    for dx_label, display in [("CN", "Diagnosis: CU, n (%)"), ("MCI", "Diagnosis: MCI, n (%)"), ("AD", "Diagnosis: Dementia, n (%)")]:
        rows.append(
            {
                "Characteristic": display,
                **col(
                    _format_n_pct(int((eval_df["clin_dx_simplified"] == dx_label).sum()), n_eval),
                    _format_n_pct(int((swap_eval_df["clin_dx_simplified"] == dx_label).sum()), n_swap),
                ),
            }
        )
    rows.append(
        {
            "Characteristic": "CDR-SB",
            **col(
                _format_mean_sd(eval_df["clin_CDRSB"]),
                _format_mean_sd(swap_eval_df["clin_CDRSB"]),
            ),
        }
    )
    rows.append(
        {
            "Characteristic": "MMSE",
            **col(_format_mean_sd(eval_df["clin_MMSE"]), _format_mean_sd(swap_eval_df["clin_MMSE"])),
        }
    )
    rows.append(
        {
            "Characteristic": "PET positive, n (%)",
            **col(
                _format_n_pct(int((eval_df["pet_pos"] == 1).sum()), n_eval),
                _format_n_pct(int((swap_eval_df["pet_pos"] == 1).sum()), n_swap),
            ),
        }
    )
    rows.append(
        {
            "Characteristic": "CSF-A positive (Aβ42/40), n (%)",
            **col(
                "NA (not required)",
                _format_n_pct(int((swap_eval_df["csf_a_pos"] == 1).sum()), n_swap),
            ),
        }
    )
    rows.append(
        {
            "Characteristic": "|Δ| PET–plasma, days (median [IQR])",
            **col(
                _format_median_iqr_abs(eval_df["delta_pet_plasma_days"]),
                _format_median_iqr_abs(swap_eval_df["delta_pet_plasma_days"]),
            ),
        }
    )
    rows.append(
        {
            "Characteristic": "|Δ| PET–CSF, days (median [IQR])",
            **col(
                _format_median_iqr_abs(eval_df["delta_pet_csf_days"]),
                _format_median_iqr_abs(swap_eval_df["delta_pet_csf_days"]),
            ),
        }
    )
    rows.append(
        {
            "Characteristic": "|Δ| plasma–CSF, days (median [IQR])",
            **col(
                _format_median_iqr_abs(eval_df["delta_plasma_csf_days"]),
                _format_median_iqr_abs(swap_eval_df["delta_plasma_csf_days"]),
            ),
        }
    )

    out = pd.DataFrame(rows)
    out.attrs["footnote"] = "†Swap-eligible subset requires Elecsys CSF Aβ42/40 ratio (ABETA40 available)."
    return out


def build_table2_primary_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    df = metrics.copy()
    df = df.loc[
        (df["analysis_set"] == "one_triad_per_rid")
        & (df["operating_point"] == "op_ppa0.95_npa0.95")
        & (df["policy"] == "indet_to_negative")
        & (df["benchmark"].isin(["PET", "CSF-A"]))
        & (df["endpoint"].isin(["c2n_plasma_abeta42_abeta40_abeta_ratio", "c2n_plasma_ptau217_ratio_ptau217_ratio"]))
    ].copy()

    pretty_endpoint = {
        "c2n_plasma_abeta42_abeta40_abeta_ratio": "Plasma Aβ42/40 (PrecivityAD2 component)",
        "c2n_plasma_ptau217_ratio_ptau217_ratio": "Plasma %pTau217 (PrecivityAD2 component)",
    }
    df["endpoint_pretty"] = df["endpoint"].map(pretty_endpoint).fillna(df["endpoint"])

    rows: list[dict[str, str]] = []
    for _, r in df.sort_values(["endpoint_pretty", "benchmark"]).iterrows():
        rows.append(
            {
                "Endpoint": str(r["endpoint_pretty"]),
                "Benchmark": str(r["benchmark"]),
                "n used": str(int(r["n_used"])),
                "Indeterminate fraction": _format_rate(r["indeterminate_fraction"]),
                "Confirmatory test rate*": _format_rate(r["confirm_rate_primary"]),
                "Miss rate under reflex*": _format_rate(r["miss_rate_under_reflex_primary"]),
                "PPA (95% CI)": _format_metric_with_ci(r["PPA"], r["PPA_lo"], r["PPA_hi"]),
                "NPA (95% CI)": _format_metric_with_ci(r["NPA"], r["NPA_lo"], r["NPA_hi"]),
            }
        )

    out = pd.DataFrame(rows)
    out.attrs["footnote"] = "*Primary reflex strategy: confirmatory testing if plasma is Positive or Indeterminate."
    return out


def build_table3_swap(df_swap: pd.DataFrame) -> pd.DataFrame:
    df = df_swap.copy()
    df = df.loc[
        (df["analysis_set"] == "one_triad_per_rid")
        & (df["operating_point"] == "op_ppa0.95_npa0.95")
        & (df["policy"] == "indet_to_negative")
        & (df["swap"] == "CSF-A minus PET")
        & (df["endpoint"].isin(["c2n_plasma_abeta42_abeta40_abeta_ratio", "c2n_plasma_ptau217_ratio_ptau217_ratio"]))
    ].copy()

    pretty_endpoint = {
        "c2n_plasma_abeta42_abeta40_abeta_ratio": "Plasma Aβ42/40",
        "c2n_plasma_ptau217_ratio_ptau217_ratio": "Plasma %pTau217",
    }

    rows: list[dict[str, str]] = []
    for _, r in df.sort_values(["endpoint"]).iterrows():
        rows.append(
            {
                "Endpoint": pretty_endpoint.get(str(r["endpoint"]), str(r["endpoint"])),
                "n RIDs": str(int(r["n_rids"])),
                "ΔPPA (CSF-A − PET)": _format_delta_pp(r["delta_PPA"], r["delta_PPA_lo"], r["delta_PPA_hi"]),
                "ΔNPA (CSF-A − PET)": _format_delta_pp(r["delta_NPA"], r["delta_NPA_lo"], r["delta_NPA_hi"]),
            }
        )
    return pd.DataFrame(rows)


def build_tableS2_pooled_swap(pooled_swap: pd.DataFrame) -> pd.DataFrame:
    df = pooled_swap.copy()
    df = df.loc[
        (df["analysis_set"] == "pooled_swap_eligible_one_triad_per_rid")
        & (df["includes_splits"] == "derivation+evaluation")
        & (df["operating_point"] == "op_ppa0.95_npa0.95")
        & (df["policy"] == "indet_to_negative")
        & (df["swap"] == "CSF-A minus PET")
        & (df["endpoint"].isin(["c2n_plasma_abeta42_abeta40_abeta_ratio", "c2n_plasma_ptau217_ratio_ptau217_ratio"]))
    ].copy()

    pretty_endpoint = {
        "c2n_plasma_abeta42_abeta40_abeta_ratio": "Plasma Aβ42/40",
        "c2n_plasma_ptau217_ratio_ptau217_ratio": "Plasma %pTau217",
    }

    rows: list[dict[str, str]] = []
    for _, r in df.sort_values(["endpoint"]).iterrows():
        rows.append(
            {
                "Endpoint": pretty_endpoint.get(str(r["endpoint"]), str(r["endpoint"])),
                "n RIDs": str(int(r["n_rids"])),
                "ΔPPA (CSF-A − PET)": _format_delta_pp(r["delta_PPA"], r["delta_PPA_lo"], r["delta_PPA_hi"]),
                "ΔNPA (CSF-A − PET)": _format_delta_pp(r["delta_NPA"], r["delta_NPA_lo"], r["delta_NPA_hi"]),
            }
        )
    return pd.DataFrame(rows)


def build_tableS_denominators(two_by_two_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []
    for f in sorted(two_by_two_dir.glob("2x2__*.csv")):
        name = f.name
        # Expected: 2x2__{endpoint}__{benchmark}__{op}__{policy}.csv
        parts = name.removesuffix(".csv").split("__")
        if len(parts) != 5:
            continue
        _, endpoint, benchmark, operating_point, policy = parts
        if endpoint == "APS2_C2N":
            continue  # keep APS2 denominators in PET-only appendix
        df = pd.read_csv(f)
        if df.shape[0] != 1:
            continue
        tp = int(df.loc[0, "TP"])
        fn = int(df.loc[0, "FN"])
        fp = int(df.loc[0, "FP"])
        tn = int(df.loc[0, "TN"])
        rows.append(
            {
                "endpoint": endpoint,
                "benchmark": benchmark,
                "operating_point": operating_point,
                "policy": policy,
                "TP": tp,
                "FN": fn,
                "FP": fp,
                "TN": tn,
            }
        )
    return pd.DataFrame(rows).sort_values(["endpoint", "benchmark", "operating_point", "policy"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Paper A manuscript tables from existing outputs.")
    parser.add_argument("--paperA-dir", type=Path, default=Path("outputs/paperA"))
    parser.add_argument("--out-dir", type=Path, default=Path("manuscript/paperA_neurology"))
    args = parser.parse_args()

    paperA_dir: Path = args.paperA_dir
    out_dir: Path = args.out_dir
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    one_triad_path = paperA_dir / "audit" / "one_triad_per_rid.csv"
    metrics_path = paperA_dir / "metrics_summary.csv"
    swap_path = paperA_dir / "benchmark_swap.csv"
    pooled_swap_path = paperA_dir / "audit" / "pooled_swap_eligible_benchmark_swap.csv"
    two_by_two_dir = paperA_dir / "2x2_tables"

    one_triad = pd.read_csv(one_triad_path)
    metrics = pd.read_csv(metrics_path)
    swap = pd.read_csv(swap_path)
    pooled_swap = pd.read_csv(pooled_swap_path)

    t1 = build_table1(one_triad)
    _write_table(t1, tables_dir / "table1_cohort_characteristics.csv", tables_dir / "table1_cohort_characteristics.md")
    if "footnote" in t1.attrs:
        (tables_dir / "table1_footnote.txt").write_text(str(t1.attrs["footnote"]) + "\n", encoding="utf-8")

    t2 = build_table2_primary_metrics(metrics)
    _write_table(t2, tables_dir / "table2_primary_metrics.csv", tables_dir / "table2_primary_metrics.md")
    if "footnote" in t2.attrs:
        (tables_dir / "table2_footnote.txt").write_text(str(t2.attrs["footnote"]) + "\n", encoding="utf-8")

    t3 = build_table3_swap(swap)
    _write_table(t3, tables_dir / "table3_benchmark_swap_evaluation_primary.csv", tables_dir / "table3_benchmark_swap_evaluation_primary.md")

    tS2 = build_tableS2_pooled_swap(pooled_swap)
    _write_table(tS2, tables_dir / "tableS2_benchmark_swap_pooled_primary.csv", tables_dir / "tableS2_benchmark_swap_pooled_primary.md")

    denom = build_tableS_denominators(two_by_two_dir)
    _write_table(denom, tables_dir / "tableS_denominators_2x2_components.csv", tables_dir / "tableS_denominators_2x2_components.md")

    # Copy-through of key source tables used to build manuscript tables (for traceability).
    (tables_dir / "source_metrics_summary.csv").write_text(metrics.to_csv(index=False), encoding="utf-8")
    (tables_dir / "source_benchmark_swap_evaluation.csv").write_text(swap.to_csv(index=False), encoding="utf-8")
    (tables_dir / "source_pooled_swap_eligible_benchmark_swap.csv").write_text(pooled_swap.to_csv(index=False), encoding="utf-8")

    readme = (
        "# Paper A manuscript tables (Neurology)\n\n"
        "Generated from existing Paper A outputs (no recomputation of triads/thresholds).\n\n"
        "## Inputs\n"
        f"- `{one_triad_path}`\n"
        f"- `{metrics_path}`\n"
        f"- `{swap_path}`\n"
        f"- `{pooled_swap_path}`\n"
        f"- `{two_by_two_dir}/`\n\n"
        "## Outputs\n"
        "- `table1_cohort_characteristics.*`\n"
        "- `table2_primary_metrics.*`\n"
        "- `table3_benchmark_swap_evaluation_primary.*`\n"
        "- `tableS2_benchmark_swap_pooled_primary.*`\n"
        "- `tableS_denominators_2x2_components.*`\n"
    )
    (tables_dir / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
