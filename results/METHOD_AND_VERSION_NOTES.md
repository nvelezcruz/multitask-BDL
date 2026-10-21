# Analysis and version notes

This release freezes the paper-source tables to one support-rule family rather
than blending numerically different analysis versions.

## Fixed-contrast WFP feature scores

`data/wasserstein/` contains the fixed-contrast WFP outputs used to support
the manuscript’s general, cross-cancer, and cancer-specific feature findings
(including the source rows for Supplementary Tables 23--26). In this workflow,
an active treatment profile was supported only when the outer training split
had at least 12 patients and at least 3 PFI events:

`outer_train_n >= 12 and PFI_events >= 3`.

## Later no-minimum-count analysis

The workspace also contains a later no-minimum-count treatment-profile/pair
analysis, which admits profiles/pairs with at least one available record. It
is intentionally **not included** here because it cannot be numerically mixed
with the fixed-support analysis used for the manuscript’s treatment WFP source
tables. If it is adopted for a revision, regenerate the corresponding figures,
tables, caption language, and this source-data release together.

## Seeds and scale

All across-seed summaries use the ten splits/seeds 42--51. Risk contrasts are
on the calibrated model-predicted PFI event-probability scale; `*_pp` columns
are percentage points. They remain observational/model-based contrasts, not
causal effects.
