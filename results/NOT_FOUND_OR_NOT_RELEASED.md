# Items intentionally absent from this release

This is an audit record, not an error log. The following files are needed for a
fully exhaustive source-data release but were not found as shareable aggregate
outputs in the supplied workspace, or should not be published without another
permission decision.

| Manuscript result | What is missing | Why it is not substituted here |
| --- | --- | --- |
| Main Tables 2–7 and Supplementary Tables 1–21 | A complete internal, per-split DGP-and-baseline metric table with status/evaluable-split fields that reconciles to the submitted ten-split tables. | An August 2026 recovery backup contains only 6–7 DGP splits per configuration, while a separate baseline archive has ten splits but does not yet reconcile numerically to the submitted tables. The combined collector directories are empty. Do not merge those run streams or reconstruct results from display-rounded PDF values. |
| Supplementary Figures 13–17 at 24 months | Aggregate 24-month KM curve/statistics tables for the PathNoUDP model. | Only the rendered figures and 18-month aggregate KM source data were found. Do not publish the source patient-assignment files. |
| Figure 3 | The exact posterior-draw data for the displayed example patients, with a share-safe panel-ID mapping. | The available artifacts are patient-level. An anonymized, panel-only export can be added later if approved. |
| Complete calibration provenance | Per-seed calibration-intercept table and bin edges. | Calibration-bin summaries are included. The exact intercept export was not located as a compact non-patient-level table. |
| Supplementary Table 28 | A reconciled final C-index source table for the confidence-selected METABRIC analysis. | The available manuscript/LaTeX and aggregate-analysis versions disagree on C-index values. No version is released until that discrepancy is resolved. |
| Reproduction inputs | TCGA/METABRIC clinical or molecular matrices; pathway/gene-state inputs; raw predictions/outcomes; checkpoints. | These are raw, patient-level, collaborator-provided, or terms-restricted inputs and are deliberately excluded. |

The manuscript’s pathway and gene-state provenance should be documented and
permission checked before publishing those input matrices or related trained
artifacts. The aggregate figures/tables included in this release remain useful
source data, but this manifest makes the remaining gaps visible.

## Recovery material reviewed but not released

After this package was assembled, we reviewed an archived August 2026 rerun.
It includes aggregate DGP metric summaries for a partial set of seeds
(PathClin: 42--47; PathNoUDP-MutTMB: 42--48; PathUDP-MutTMB: 42--47) and a
separate ten-seed baseline archive. These files are aggregate-only after local
path removal, but they are not released here because they are not the complete,
reconciled source for the submitted ten-split internal-performance tables.
They should be released only after their run provenance and numerical agreement
with the manuscript have been confirmed.
