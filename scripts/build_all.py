#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adni_analysis.config import load_yaml
from adni_analysis.inventory import build_raw_inventory
from adni_analysis.hardening import (
    build_evidence_core_catalog,
    build_referential_integrity_report,
    build_release_stamp,
    build_regression_snapshot,
    build_schema_registry,
    build_time_gap_distributions,
    build_value_flags,
    write_join_rules,
)
from adni_analysis.manifests import build_signature, build_table_manifest
from adni_analysis.pipeline import (
    build_canonical_clinical,
    build_canonical_csf_elecsys,
    build_canonical_pet_ucb,
    build_canonical_plasma_c2n_precivityad2,
    build_canonical_plasma_fnibc,
    build_canonical_plasma_janssen,
    build_pairing_outputs,
    collect_core_tables,
    ensure_pipeline_dirs,
    write_qc_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ADNI evidence core from raw tables.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/pipeline.yaml"),
        help="Pipeline YAML config path",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    dirs = ensure_pipeline_dirs(cfg)

    build_raw_inventory(
        raw_root=Path(cfg["raw_root"]),
        audit_dir=dirs.audit_dir,
        core_dict_dir=dirs.core_dict_dir,
        manifests_dir=dirs.manifests_dir,
    )

    build_canonical_pet_ucb(cfg, dirs)
    build_canonical_csf_elecsys(cfg, dirs)
    build_canonical_plasma_fnibc(cfg, dirs)
    build_canonical_plasma_c2n_precivityad2(cfg, dirs)
    build_canonical_plasma_janssen(cfg, dirs)
    build_canonical_clinical(cfg, dirs)

    build_pairing_outputs(cfg, dirs)

    core_tables = collect_core_tables(dirs.core_dir)
    build_table_manifest(core_tables, dirs.manifests_dir / "table_manifest.json")
    build_signature(
        out_path=dirs.manifests_dir / "build_signature.txt",
        config_paths=[args.config],
        table_paths=core_tables,
    )

    # Hardening artifacts (reviewer-proof pack)
    write_join_rules(
        out_path=dirs.audit_join_reports_dir / "join_rules.md",
        windows_days=cfg["pairing"]["windows_days"],
        plasma_filter=cfg["pairing"].get("plasma_filter"),
        plasma_source=str(cfg.get("pairing", {}).get("plasma_source", "fnibc_wide")),
    )
    build_schema_registry(out_dir=dirs.manifests_dir / "schema_registry", tables=core_tables)
    build_evidence_core_catalog(out_path=dirs.audit_dir / "evidence_core_catalog.csv", tables=core_tables)
    build_time_gap_distributions(
        out_path=dirs.audit_qc_dir / "time_gap_distributions.csv",
        windows_days=cfg["pairing"]["windows_days"],
        dyad_pet_plasma=dirs.core_paired_dir / "dyad_pet_plasma.csv",
        dyad_pet_csf=dirs.core_paired_dir / "dyad_pet_csf.csv",
        dyad_csf_plasma=dirs.core_paired_dir / "dyad_csf_plasma.csv",
        triad=dirs.core_paired_dir / "triad_pet_anchored.csv",
    )
    build_value_flags(
        csf_path=dirs.core_canonical_dir / "csf_elecsys_upenn.csv",
        plasma_long_path=dirs.core_canonical_dir / "plasma_fnibc_long.csv",
        out_dir=dirs.audit_qc_dir,
    )

    plasma_source = str(cfg.get("pairing", {}).get("plasma_source", "fnibc_wide")).strip().lower()
    if plasma_source == "fnibc_wide":
        plasma_wide_path = dirs.core_canonical_dir / "plasma_fnibc_wide.csv"
    elif plasma_source in {"c2n_precivityad2", "precivityad2", "c2n_aps2"}:
        plasma_wide_path = dirs.core_canonical_dir / "plasma_c2n_precivityad2_score.csv"
    else:
        raise ValueError(f"Unsupported pairing.plasma_source: {plasma_source}")

    build_referential_integrity_report(
        out_path=dirs.audit_qc_dir / "referential_integrity.md",
        clinical=dirs.core_canonical_dir / "clinical_adnimerge.csv",
        pet=dirs.core_canonical_dir / "pet_amyloid_ucb_6mm.csv",
        csf=dirs.core_canonical_dir / "csf_elecsys_upenn.csv",
        plasma_wide=plasma_wide_path,
        dyad_pet_plasma=dirs.core_paired_dir / "dyad_pet_plasma.csv",
        dyad_pet_csf=dirs.core_paired_dir / "dyad_pet_csf.csv",
        dyad_csf_plasma=dirs.core_paired_dir / "dyad_csf_plasma.csv",
        triad=dirs.core_paired_dir / "triad_pet_anchored.csv",
    )
    build_regression_snapshot(
        triad_path=dirs.core_paired_dir / "triad_pet_anchored.csv",
        plasma_wide_path=plasma_wide_path,
        csf_path=dirs.core_canonical_dir / "csf_elecsys_upenn.csv",
        out_path=dirs.audit_qc_dir / "regression_snapshot.json",
        check_path=dirs.audit_qc_dir / "regression_check.md",
        release_id=str(cfg.get("release", {}).get("release_id"))
        if isinstance(cfg.get("release"), dict) and cfg.get("release", {}).get("release_id")
        else None,
        mode=str(cfg.get("regression", {}).get("mode", "check")),
        float_tol=float(cfg.get("regression", {}).get("float_tol", 1e-6)),
    )
    build_release_stamp(
        out_path=dirs.manifests_dir / "evidence_core_release.json",
        cfg=cfg,
        dirs=dirs,
        build_signature_path=dirs.manifests_dir / "build_signature.txt",
    )

    # Update QC summary after generating hardening artifacts
    write_qc_summary(dirs)


if __name__ == "__main__":
    main()
