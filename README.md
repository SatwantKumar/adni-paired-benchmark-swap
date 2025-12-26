# ADNI paired benchmark swap + gray-zone workflow (code-only)

This repository contains **code only** to build an auditable ADNI “evidence core” and reproduce a paired **PET vs CSF comparator-swap** analysis with explicit **indeterminate (“gray-zone”) policies**.

## Data policy (important)

- This repository **does not** include any ADNI data (raw or derived).
- You must obtain ADNI data through the official ADNI process and comply with ADNI data use agreements.
- Do **not** commit any data or derived outputs to git. The `.gitignore` is intentionally strict.

## Quickstart

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2) Place ADNI files locally

Edit `config/pipeline.yaml` if your raw ADNI filenames/paths differ, then place your ADNI downloads under `data/raw/` to match that config.

### 3) Build the evidence core

```bash
python scripts/build_all.py --config config/pipeline.yaml
```

### 4) Run Paper A pack

```bash
python scripts/08_build_paperA_pack.py \
  --definitions config/paperA_definitions.yaml \
  --out-dir outputs/paperA
```

## What gets produced (locally)

Running the pipeline will generate local, reproducible artifacts under:
- `core/` (canonical tables + paired dyads/triads)
- `audit/` and `manifests/` (QC, join reports, inventories, release stamps)
- `outputs/paperA/` (Paper A analysis pack: figures + auditable 2×2 tables + swap estimates)

None of these are committed to git.

## Contact

Satwant Kumar — Satwant.Dagar@gmail.com

