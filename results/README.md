# Manuscript result data

This directory contains the shareable, derived result tables used to support
the manuscript *Multitask Bayesian deep learning predicts cancer progression
and identifies resistance-associated molecular features while quantifying
patient-level uncertainty* and its Supplementary Information.

It is deliberately a **results-only** release. It does not contain original
TCGA or METABRIC records, patient/sample identifiers, outcomes or predictions
at patient level, molecular input matrices, pathway or gene-state inputs,
checkpoints, or fitted models.

The files are grouped by the result they support:

- `data/cohort/` — aggregate cohort characteristics and modality overlap.
- `data/external_validation/` — seed-level and aggregate METABRIC performance
  summaries for the three manuscript input configurations.
- `data/early_risk_and_km/` — aggregate early-event enrichment and 18-month
  Kaplan–Meier curve data.
- `data/calibration/` — per-seed and across-seed calibration-bin summaries.
- `data/wasserstein/` — aggregate WFP/Wasserstein feature scores and the
  figure/table source rows for shared and cancer-specific features.
- `data/treatment/` — fixed-support treatment-conditioned WFP rankings,
  fixed-support feature-stratified treatment-pair WFP summaries, and separate
  aggregate mutation-by-treatment risk contrasts.

`results_manifest.csv` maps each manuscript table/figure to the available
release file(s). `NOT_FOUND_OR_NOT_RELEASED.md` is intentional: it documents
source-data components that were either not present in the supplied workspace
or should not be published as-is. It prevents this partial results package
from being mistaken for a release of raw data or patient-level results.

`DATA_DICTIONARY.md` defines the result-table fields and
`METHOD_AND_VERSION_NOTES.md` records the support rules used for each
Wasserstein/treatment analysis. `CHECKSUMS.sha256` permits file-integrity
checks after download.

## Scope and interpretation

All model-derived quantities are observational/model-based summaries. In
particular, treatment-profile contrasts are paired changes in model-predicted
PFI risk under alternate encoded treatment profiles; they are not causal
treatment-effect estimates or treatment recommendations. METABRIC uses
recurrence-free survival (RFS), not TCGA PFI.

The original TCGA and METABRIC data are not redistributed here. Please obtain
them from their original providers and comply with their access and reuse
terms. The repository has no reuse license yet; none is implied by this data
directory.

## Rebuilding this package

The checked-in CSVs are generated from the local aggregate outputs with:

```bash
python3 results/scripts/build_results_release.py --source-root /path/to/TCGA-NPJ-workspace
```

The script names only audited aggregate inputs; it does not search for or copy
patient-level result artifacts. It is included for provenance, not as a
substitute for the original training data or full model rerun environment.
