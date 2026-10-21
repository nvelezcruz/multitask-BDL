# Data dictionary

All CSV files in this release contain aggregate results only. Counts such as
`n_patients`, `n_at_risk`, and `n_feature_evaluated_seeds` are group counts;
they are not identifiers and cannot be linked back to individual records.

## Cohort and external validation

- `data/cohort/cohort_summary.csv` is a long-format version of aggregate
  cohort summaries. `statistic` identifies whether `value` is an `n`, percent,
  mean, standard deviation, median, first quartile, or third quartile.
- `data/external_validation/metabric_external_metrics_by_seed_all_configurations.csv`
  contains one aggregate validation row per configuration, seed, and horizon.
  The configuration names follow the manuscript: `PathClin`,
  `PathNoUDP_MutTMB`, and `PathUDP_MutTMB`.
- `*_summary_*` files summarize the ten seeds (42--51). METABRIC outcomes are
  recurrence-free survival (RFS); internal TCGA results use progression-free
  interval (PFI).

## Early-risk, Kaplan--Meier, and calibration results

- Early-event enrichment tables report seed-level and across-seed summaries of
  the high-risk group’s early-event enrichment.
- `km_H18_*` provides the plotted Kaplan--Meier step coordinates and the
  group-level number at risk at 18 months.
- Calibration tables give binned observed and predicted event probabilities.
  `ipcw_weight_sum` and `weighted_n_early_events` are aggregate inverse-
  probability-of-censoring-weighted quantities.

## Wasserstein feature results

The `data/wasserstein/` files are the fixed-contrast WFP analyses used for the
paper’s shared and cancer-specific molecular-feature findings. `feature_type`
distinguishes pathway, clinical, mutation, and TMB features. The principal
ranking field is `mean_W_feature`, the average Wasserstein feature-perturbation
score across evaluable seeds. `mean_Delta_feature` and related `Delta` fields
are signed model-output changes under the defined feature perturbation; they
are not treatment effects.

`n_seeds_evaluable`, `n_seeds_selected`, and `selection_frequency` communicate
how consistently a feature could be evaluated or was selected across seeds.
`fixed_contrast_treatment_support.csv` records the treatment-profile support
criterion for that analysis.

## Treatment result tables

- `fixed_support_*treatment_profile_wasserstein_ranking.csv` ranks molecular
  features within each encoded treatment profile under the manuscript's fixed
  support rule. Its `mean_W_evaluated` is a feature score, not a treatment
  contrast.
- `fixed_support_*feature_stratified_treatment_pair_*.csv*` summarizes the
  fixed-support WFP treatment-pair analysis. Its `psi` compares the two
  treatment profiles across the feature-defined strata; it is a different
  estimand from the mutation-by-treatment risk contrast below.
- `overall_treatment_profile_contrast_consensus.csv` contains aggregate
  predicted PFI-risk differences for treatment profile A versus B.
- `mutation_treatment_interactions_consensus.csv.gz` is the full aggregate
  mutation-by-treatment comparison table. For a feature, `mean_D_WT` and
  `mean_D_MUT` are the mean model-predicted risk differences for treatment A
  minus B among wild-type and mutated groups, respectively. `mean_Gamma` is
  `mean_D_MUT - mean_D_WT`; `mean_Gamma_pp` expresses that difference in
  percentage points. The accompanying Figure-5 and Supplementary-Table-27
  CSVs are compact selected subsets.

These are model-derived, observational summaries. They are not randomized or
causal treatment-effect estimates and should not be used as treatment advice.
