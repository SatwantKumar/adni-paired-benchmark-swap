from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from adni_analysis.canonical import (
    build_clinical_adnimerge,
    build_csf_elecsys_upenn,
    build_pet_ucb_amy_6mm,
    build_plasma_c2n_precivityad2,
    build_plasma_fnibc,
    build_plasma_janssen_ptau217,
)
from adni_analysis.pairing import (
    PairingWindows,
    build_dyad_csf_plasma,
    build_dyad_pet_csf,
    build_dyad_pet_plasma,
    build_triad_pet_anchored,
    write_triad_join_report,
)
from adni_analysis.utils import PipelineDirs, ensure_dir


def ensure_pipeline_dirs(cfg: dict) -> PipelineDirs:
    core_dir = ensure_dir(Path(cfg["outputs"]["core_dir"]))
    audit_dir = ensure_dir(Path(cfg["outputs"]["audit_dir"]))
    manifests_dir = ensure_dir(Path(cfg["outputs"]["manifests_dir"]))

    core_canonical_dir = ensure_dir(core_dir / "canonical")
    core_paired_dir = ensure_dir(core_dir / "paired")
    core_dict_dir = ensure_dir(core_dir / "dictionaries")

    audit_raw_inventory_dir = ensure_dir(audit_dir / "raw_inventory")
    audit_raw_columns_dir = ensure_dir(audit_raw_inventory_dir / "raw_columns")
    audit_validations_dir = ensure_dir(audit_dir / "validations")
    audit_join_reports_dir = ensure_dir(audit_dir / "join_reports")
    audit_qc_dir = ensure_dir(audit_dir / "qc")

    return PipelineDirs(
        core_dir=core_dir,
        audit_dir=audit_dir,
        manifests_dir=manifests_dir,
        core_canonical_dir=core_canonical_dir,
        core_paired_dir=core_paired_dir,
        core_dict_dir=core_dict_dir,
        audit_raw_inventory_dir=audit_raw_inventory_dir,
        audit_raw_columns_dir=audit_raw_columns_dir,
        audit_validations_dir=audit_validations_dir,
        audit_join_reports_dir=audit_join_reports_dir,
        audit_qc_dir=audit_qc_dir,
    )


def build_canonical_pet_ucb(cfg: dict, dirs: PipelineDirs) -> None:
    build_pet_ucb_amy_6mm(
        in_path=Path(cfg["sources"]["pet_ucb_amy_6mm"]),
        out_path=dirs.core_canonical_dir / "pet_amyloid_ucb_6mm.csv",
        audit_qc_dir=dirs.audit_qc_dir,
    )


def build_canonical_csf_elecsys(cfg: dict, dirs: PipelineDirs) -> None:
    build_csf_elecsys_upenn(
        in_path=Path(cfg["sources"]["csf_elecsys_upenn"]),
        out_path=dirs.core_canonical_dir / "csf_elecsys_upenn.csv",
        audit_qc_dir=dirs.audit_qc_dir,
    )


def build_canonical_plasma_fnibc(cfg: dict, dirs: PipelineDirs) -> None:
    build_plasma_fnibc(
        in_path=Path(cfg["sources"]["plasma_fnibc"]),
        out_long_path=dirs.core_canonical_dir / "plasma_fnibc_long.csv",
        out_wide_path=dirs.core_canonical_dir / "plasma_fnibc_wide.csv",
        core_dict_dir=dirs.core_dict_dir,
        audit_validations_dir=dirs.audit_validations_dir,
        audit_qc_dir=dirs.audit_qc_dir,
    )


def build_canonical_plasma_janssen(cfg: dict, dirs: PipelineDirs) -> None:
    build_plasma_janssen_ptau217(
        in_path=Path(cfg["sources"]["plasma_janssen_ptau217"]),
        out_path=dirs.core_canonical_dir / "plasma_janssen_ptau217.csv",
        audit_qc_dir=dirs.audit_qc_dir,
    )


def build_canonical_plasma_c2n_precivityad2(cfg: dict, dirs: PipelineDirs) -> None:
    build_plasma_c2n_precivityad2(
        in_path=Path(cfg["sources"]["plasma_c2n_precivityad2"]),
        out_path=dirs.core_canonical_dir / "plasma_c2n_precivityad2_score.csv",
        audit_qc_dir=dirs.audit_qc_dir,
    )


def build_canonical_clinical(cfg: dict, dirs: PipelineDirs) -> None:
    build_clinical_adnimerge(
        in_path=Path(cfg["sources"]["adnimerge"]),
        out_path=dirs.core_canonical_dir / "clinical_adnimerge.csv",
        audit_qc_dir=dirs.audit_qc_dir,
        ptdemog_path=Path(cfg["sources"]["ptdemog"]) if cfg.get("sources", {}).get("ptdemog") else None,
    )


def build_pairing_outputs(cfg: dict, dirs: PipelineDirs) -> None:
    pet = pd.read_csv(dirs.core_canonical_dir / "pet_amyloid_ucb_6mm.csv", parse_dates=["SCANDATE"])
    csf = pd.read_csv(dirs.core_canonical_dir / "csf_elecsys_upenn.csv", parse_dates=["EXAMDATE"])
    plasma_source = str(cfg.get("pairing", {}).get("plasma_source", "fnibc_wide")).strip().lower()
    if plasma_source == "fnibc_wide":
        plasma_path = dirs.core_canonical_dir / "plasma_fnibc_wide.csv"
    elif plasma_source in {"c2n_precivityad2", "precivityad2", "c2n_aps2"}:
        plasma_path = dirs.core_canonical_dir / "plasma_c2n_precivityad2_score.csv"
    else:
        raise ValueError(f"Unsupported pairing.plasma_source: {plasma_source}")

    plasma = pd.read_csv(plasma_path, parse_dates=["EXAMDATE"])

    win = cfg["pairing"]["windows_days"]
    windows = PairingWindows(pet_plasma=int(win["pet_plasma"]), pet_csf=int(win["pet_csf"]), plasma_csf=int(win["plasma_csf"]))
    plasma_filter = cfg["pairing"].get("plasma_filter")

    dyad_pet_plasma = build_dyad_pet_plasma(pet=pet, plasma=plasma, windows=windows, plasma_filter=plasma_filter)
    dyad_pet_csf = build_dyad_pet_csf(pet=pet, csf=csf, windows=windows)
    dyad_csf_plasma = build_dyad_csf_plasma(csf=csf, plasma=plasma, windows=windows, plasma_filter=plasma_filter)
    triads = build_triad_pet_anchored(pet=pet, csf=csf, plasma=plasma, windows=windows, plasma_filter=plasma_filter)

    dyad_pet_plasma.to_csv(dirs.core_paired_dir / "dyad_pet_plasma.csv", index=False)
    dyad_pet_csf.to_csv(dirs.core_paired_dir / "dyad_pet_csf.csv", index=False)
    dyad_csf_plasma.to_csv(dirs.core_paired_dir / "dyad_csf_plasma.csv", index=False)
    triads.to_csv(dirs.core_paired_dir / "triad_pet_anchored.csv", index=False)

    write_triad_join_report(
        out_path=dirs.audit_join_reports_dir / "triad_join_report.md",
        pet=pet,
        csf=csf,
        plasma=plasma,
        dyad_pet_plasma=dyad_pet_plasma,
        dyad_pet_csf=dyad_pet_csf,
        triads=triads,
        windows=windows,
        plasma_filter=plasma_filter,
    )


def collect_core_tables(core_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for p in sorted(core_dir.rglob("*.csv")):
        if p.is_file():
            paths.append(p)
    return paths


def write_qc_summary(dirs: PipelineDirs) -> None:
    qc_files = [
        ("PET duplicate events", dirs.audit_qc_dir / "pet_duplicate_events.csv"),
        ("PET date anomalies", dirs.audit_qc_dir / "pet_date_anomalies.csv"),
        ("PET centiloid extremes", dirs.audit_qc_dir / "pet_centiloid_extremes.csv"),
        ("CSF duplicate events", dirs.audit_qc_dir / "csf_duplicate_events.csv"),
        ("CSF Aβ40 nonpositive", dirs.audit_qc_dir / "csf_abeta40_nonpositive.csv"),
        ("Plasma unit inconsistencies", dirs.audit_qc_dir / "plasma_fnibc_unit_inconsistencies.csv"),
        ("Plasma C2N sentinel missing", dirs.audit_qc_dir / "plasma_c2n_precivityad2_sentinel_missing.csv"),
        ("Plasma C2N APS2 out of range", dirs.audit_qc_dir / "plasma_c2n_precivityad2_aps2_out_of_range.csv"),
        ("Plasma pivot collisions", dirs.audit_validations_dir / "plasma_pivot_collisions.csv"),
        ("Clinical key uniqueness", dirs.audit_qc_dir / "adnimerge_key_uniqueness.csv"),
        ("Referential integrity", dirs.audit_qc_dir / "referential_integrity.md"),
        ("Time gap distributions", dirs.audit_qc_dir / "time_gap_distributions.csv"),
        ("CSF value flags", dirs.audit_qc_dir / "csf_value_flags.csv"),
        ("Plasma value flags", dirs.audit_qc_dir / "plasma_value_flags.csv"),
        ("Regression snapshot", dirs.audit_qc_dir / "regression_snapshot.json"),
        ("Regression check", dirs.audit_qc_dir / "regression_check.md"),
    ]

    lines: list[str] = []
    lines.append("# QC summary")
    lines.append("")
    for title, path in qc_files:
        if path.exists():
            try:
                if path.suffix.lower() == ".csv":
                    n = sum(1 for _ in path.open("rb")) - 1
                    n = max(0, n)
                    lines.append(f"- {title}: {n} rows (`{path.as_posix()}`)")
                else:
                    size = path.stat().st_size
                    lines.append(f"- {title}: present ({size} bytes) (`{path.as_posix()}`)")
            except Exception:  # noqa: BLE001
                lines.append(f"- {title}: (unreadable) (`{path.as_posix()}`)")
        else:
            lines.append(f"- {title}: 0 rows (file not created)")

    (dirs.audit_qc_dir / "qc_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
