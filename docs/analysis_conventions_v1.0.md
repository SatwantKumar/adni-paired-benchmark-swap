# Paper A — Analysis conventions v1.0

This document is intended to be “pre-registered style”: it locks primary cohort construction, benchmark definitions, and statistical reporting for Paper A (paired benchmark dependence + gray-zone workflow) using the frozen ADNI evidence core.

## Neurology manuscript locks (packaging only)

Working title (locked):

- Comparator Choice and Indeterminate-Result Policy Alter Reported Performance of Plasma Amyloid Biomarkers: A Paired PET–CSF Study

Main-text primaries (locked for manuscript presentation; analyses unchanged):

- Primary operating point: `op_ppa0.95_npa0.95`
- Primary gray-zone policy: `indet_to_negative` (headline swap); `indet_to_positive` as prespecified sensitivity
- Main Figure 2 (evaluation-only; leakage-controlled): `outputs/paperA/figures/figure2_benchmark_swap_forest_evaluation_one_triad_per_rid.svg`
- Figure S2 (pooled swap-eligible; precision-enhancing; thresholds locked; no refitting): `outputs/paperA/figures/figure2_benchmark_swap_forest_pooled_one_triad_per_rid.svg`
- `determinates_only` is reported as a labeled sensitivity (supplement) due to denominator loss
- APS2 is restricted to a PET-only appendix and is not used for PET↔CSF benchmark-swap claims in this evidence core

## Evidence core release

Analyses use the frozen evidence core release:

- `adni_core_2025-12-23_precivityAD2_fnibc_triads_plus_aps2_v1` (`manifests/evidence_core_release.json`)

Core tables (continuous measures; no Paper-A-specific cutpoints applied):

- PET: `core/canonical/pet_amyloid_ucb_6mm.csv`
- CSF: `core/canonical/csf_elecsys_upenn.csv`
- Plasma (paired triads; multi-platform components): `core/canonical/plasma_fnibc_wide.csv` (paired triads are filtered to `ASSAYVERSION=PrecivityAD2`)
- Plasma (ADNI4 APS2 score + components; PET-only appendix): `core/canonical/plasma_c2n_precivityad2_score.csv`
- Clinical backbone: `core/canonical/clinical_adnimerge.csv`

Paired evidence core:

- PET-anchored triads: `core/paired/triad_pet_anchored.csv`

Deterministic tie-break rules and event-id definitions:

- `audit/join_reports/join_rules.md`

## Aims

1. Quantify within-person benchmark dependence of plasma–amyloid agreement when the comparator is PET vs CSF within paired triads.
2. Quantify gray-zone policy effects (tri-category vs forced binary) on agreement metrics and confirmatory-testing burden.

## Unit of analysis

Primary analysis set uses one triad per participant (`RID`) selected deterministically from the PET-anchored triad evidence core (preference for triads with CSF Aβ42/40 available when present for that participant, then minimizing time gaps). This prevents frequent visitors from dominating results.

Sensitivity analyses use all eligible triads with participant-level (`RID`) clustered inference.

## Timing windows and deterministic matching

Primary pairing windows are symmetric ±days:

- PET–plasma: ±90 days
- PET–CSF: ±180 days
- Plasma–CSF: ±180 days (required for triad construction)

Matching is deterministic and documented in `audit/join_reports/join_rules.md`. Briefly:

- Dyads: select the within-window candidate minimizing absolute day gap; ties break by earlier candidate date, then stable sort by `row_uid`.
- PET-anchored triads: choose the (plasma, CSF) pair minimizing (in order) the maximum of the three pairwise gaps, then the sum of gaps, then PET–plasma gap, then PET–CSF gap; ties break by earlier plasma/CSF date, then stable sort by `row_uid`.

Event identity columns are stable hashes and are used as join keys:

- `pet_event_id`, `csf_event_id`, `plasma_event_id`, `clin_event_id`

## PET benchmark definition (primary)

PET positivity uses UCB’s binary field `AMYLOID_STATUS` from `core/canonical/pet_amyloid_ucb_6mm.csv` (carried into `core/paired/triad_pet_anchored.csv`):

- `pet_pos = 1` if `AMYLOID_STATUS == 1`
- `pet_pos = 0` if `AMYLOID_STATUS == 0`

Continuous PET measures (e.g., `CENTILOIDS`, `SUMMARY_SUVR`) are retained for descriptive and sensitivity analyses but are not thresholded in the evidence core.

## CSF benchmark definitions (Paper A)

No CSF positivity cutpoint is applied in the evidence core. CSF positivity is defined in the Paper A pack (and recorded in `outputs/paperA/definitions.yaml`) using Elecsys continuous values (e.g., `abeta42_40_ratio` or `ABETA42`).

### Primary CSF comparator: CSF-A (amyloid-only)

Define Elecsys CSF amyloid positivity using the Aβ42/40 ratio:

- `abeta42_40_ratio = ABETA42 / ABETA40`
- `csf_a_pos = 1` if `abeta42_40_ratio < c_A`, else `0`

Cutpoint strategy (prespecified):

- Primary `c_A` is the published ADNI Elecsys CSF-only (“algorithm”) cutpoint `c_A = 0.0525` (recorded in `outputs/paperA/definitions.yaml`).
- Sensitivity band: evaluate `c_A × {0.90, 0.95, 1.00, 1.05, 1.10}`.
- Secondary (PET-based) cutpoint from the same source (`~0.0528`) is included as a named sensitivity value (not primary).
- Optional (explicitly labeled “PET-calibrated”): include an internally calibrated cutpoint that matches CSF-A positivity prevalence to PET positivity prevalence within the triad cohort (descriptive sensitivity, not primary).

### Secondary CSF comparator: CSF A+T (composite benchmark definition)

Define a composite “A+T” CSF benchmark (different clinical meaning than amyloid-only):

- `csf_a_pos` as defined above (Aβ42/40)
- `csf_t_pos = 1` if `PTAU > c_T`, else `0` (Elecsys CSF p-tau181)
- `csf_at_pos = 1` if `(csf_a_pos == 1) AND (csf_t_pos == 1)`, else `0`

Cutpoint strategy (prespecified):

- Primary `c_T` is the published ADNI Elecsys CSF-only (“algorithm”) cutpoint `c_T = 22.0 pg/mL` for p-tau181 (recorded in `outputs/paperA/definitions.yaml`).
- Sensitivity band: evaluate `c_T × {0.90, 0.95, 1.00, 1.05, 1.10}`.
- Secondary (PET-based) cutpoint from the same source (`24.3 pg/mL`) is included as a named sensitivity value (not primary).
- For CSF A+T sensitivity, vary one cutpoint at a time (A band holding T fixed; then T band holding A fixed) to avoid an unnecessarily large cutpoint cross-product.

### Additional CSF sensitivity (legacy / signature ratios)

- Legacy comparator: `ABETA42 < c` (reported as sensitivity only).
- Optional signature ratio sensitivity: `ptau_abeta42_ratio = PTAU / ABETA42` (direction and cutpoint recorded in `outputs/paperA/definitions.yaml` if used).

## Plasma index test definition (Paper A)

The paired triad build uses the FNIH BC plasma table (multi-platform components) filtered to `ASSAYVERSION=PrecivityAD2`.

Primary plasma endpoints are direct component assay outputs (no manufacturer composite score is present in the triad cohort):

- Co-primary (amyloid-axis): `c2n_plasma_abeta42_abeta40_abeta_ratio` (plasma Aβ42/40 ratio)
- Co-primary (tau-axis): `c2n_plasma_ptau217_ratio_ptau217_ratio` (plasma %p-tau217)

Supportive (not headline): `c2n_plasma_ptau217_ptau217` and `c2n_plasma_nptau217_nptau217`.

Plasma “positivity” (and any indeterminate/gray-zone definition) is not applied in the evidence core and must be defined and locked in `outputs/paperA/definitions.yaml` prior to generating Paper A results.

Practical note (data coverage): in the current raw drop, the C2N APS2 table is ADNI4-era (2023–) while the Elecsys CSF table in this evidence core is ADNI1–3-era (≤2022). Under the prespecified windows, APS2 cannot be evaluated against CSF in paired triads. Paper A therefore treats APS2 as a PET-only appendix analysis unless an ADNI4 CSF results table is added (or explicitly lagged/window-widened analyses are run and labeled as such).

### Plasma tri-category (gray-zone) operationalization (Paper A)

Paper A reports two complementary plasma views:

- Primary (triad cohort): operational tri-category workflow on the co-primary component endpoints, using a two-threshold Negative/Indeterminate/Positive definition derived once (primary: against PET) and then applied unchanged when evaluating against PET vs CSF benchmarks.
- Appendix (APS2 vs PET only): locked manufacturer dichotomy `APS2_C2N >= 48` (0–47 negative; 48–100 positive; see `outputs/paperA/definitions.yaml` for provenance), plus the same operational tri-category construction for workflow curves.

- Negative (rule-out): values in the endpoint’s low-risk tail
- Positive (rule-in): values in the endpoint’s high-risk tail
- Indeterminate: values between the two thresholds

Thresholds are derived using prespecified tail targets recorded in `outputs/paperA/definitions.yaml`:

- Rule-out targets PPA (sensitivity) for “not Negative”
- Rule-in targets NPA (specificity) for “not Positive”

To avoid optimistic bias from threshold selection on the evaluation cohort, plasma thresholds are derived on a RID-level derivation split and all primary Paper A performance and benchmark-swap results are evaluated on the held-out split (split parameters recorded in `outputs/paperA/definitions.yaml` and the generated Paper A audit outputs).

## Metrics

Binary agreement metrics are reported as:

- PPA (positive percent agreement) = TP / (TP + FN)
- NPA (negative percent agreement) = TN / (TN + FP)

Benchmark swap (“reference swap”) is summarized within the same paired triad cohort by comparing plasma agreement vs PET and plasma agreement vs CSF.

## Uncertainty

- PPA/NPA: Wilson 95% confidence intervals (single proportions).
- Benchmark-swap deltas: paired bootstrap clustered by `RID` (default 10,000 replicates; fixed seed recorded in `outputs/paperA/definitions.yaml`).

## Precision-enhancing pooled swap (supplement)

Because CSF Aβ42/40 is only available for a subset of participants, the evaluation split may contain a limited number of swap-eligible participants. As a pre-labeled precision-enhancing supplement (not replacing the leakage-controlled evaluation-only analysis), we additionally report pooled paired benchmark-swap estimates among all participants with CSF Aβ42/40 available using the same locked plasma thresholds (no refitting).

This pooled swap output is written to `outputs/paperA/audit/pooled_swap_eligible_benchmark_swap.csv` and should be interpreted as a precision aid alongside the primary evaluation-only swap in `outputs/paperA/benchmark_swap.csv`.

## Gray-zone policy (if applicable)

If the plasma definition yields an indeterminate/gray-zone category:

- Primary reporting is tri-category (Positive / Indeterminate / Negative).
- Secondary binary mappings:
  - determinates-only (exclude indeterminates)
  - indeterminate → negative
  - indeterminate → positive

Confirmatory-testing burden is summarized under a reflex strategy (e.g., “confirm PET/LP if plasma is Positive or Indeterminate”), with the policy choices recorded in `outputs/paperA/definitions.yaml`.

## Inclusion/exclusion (Paper A)

- Include PET-anchored triads with non-missing PET benchmark and plasma fields required for the chosen plasma definition.
- For CSF-based analyses, include only triads with the CSF fields required for the chosen CSF definition.
- PET QC handling: primary benchmark uses UCB `AMYLOID_STATUS` after canonical deduplication; sensitivity analyses may additionally exclude rows with failing QC flags if desired (must be recorded in `outputs/paperA/definitions.yaml`).
