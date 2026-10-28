# wasserstein_FC

This folder runs the saved models for Wasserstein feature and treatment
comparisons. It does not train a DGP or use donor patients.

The original September 16, 2026 runner was not retained. This version was
rebuilt from saved outputs and code. Compare any newly generated table with
the released aggregate tables in `results/data/wasserstein/` and
`results/data/treatment/` before using it.

## Settings

- Cancers: BRCA, COAD, and LUAD; seeds 42--51; horizons 18, 24, and 36 months.
- Mutation contrast: wild type `0` versus mutated `1`.
- Pathway and TMB contrast: cancer-specific outer-training Q25 versus Q75.
- Each counterfactual scenario for a patient is jointly sampled through both GP
  layers (256 posterior draws by default).
- Calibration: cancer-specific, fixed-slope intercepts estimated from matched
  outer-training OOF predictions.
- A treatment profile is supported only when its outer-training cancer cell has
  at least 12 patients **and** 3 PFI events.
- Supported untreated profiles are included in per-profile feature summaries,
  but treatment comparisons are active-treatment-profile versus
  active-treatment-profile only. No treatment-versus-none comparison is made.

## Required inputs (not included)

The original TCGA matrices are **not** direct inputs to this post-fit runner.
For each configuration and seed, it needs a matched saved model artifact and
OOF calibration table instead. Both are restricted because they contain
patient-level arrays, metadata, outcomes, and/or predictions. Do not add them
to this repository.

| Required item | One per | Why it is needed | Share publicly? |
| --- | --- | --- | --- |
| Matching DGP class module | configuration | Loads the serialized model class | Already in `code/` |
| `august_fitted_artifacts_seed{seed}.pt` | configuration x seed | Fitted model, feature matrices/scaling, outcomes, groups, and metadata | No |
| `posthoc_crossfit_oof_native_risks.csv` | configuration x seed | OOF native risks used for calibration | No |
| Writable output directory | run | Partition summaries and QC | Review before sharing |

The artifact must contain these fields: `model`, `Xtr_np_sub`, `Xte_np_sub`,
`Xte_t`, `standardize_mu`, `standardize_sd`, `feat_names`, `t_tr`, `e_tr`,
`g_tr`, `g_te`, `groups`, `meta_tr`, `meta_te`, `alpha_dt`, and `dt_edges`.
Only load artifacts from a trusted source: PyTorch artifact loading uses pickle.
The model-definition module still imports the historical WFC helper so an
existing serialized model can deserialize, but `wasserstein_FC` never calls its
donor-selection or donor-replacement functions.

The OOF CSV must contain:

```text
train_index,time,event,group_idx,group,
p_native_oof_H18,p_native_oof_H24,p_native_oof_H36
```

Its rows must cover every outer-training row exactly once and align exactly to
the artifact's training time, event, and group arrays.

## Configure and preflight

Copy the appropriate example JSON and replace the restricted paths. The JSON
file is the complete input manifest for a run. Paths are resolved relative to
the JSON file, so if you move it outside this folder, update `model_script` or
make that path absolute. The supplied patterns expect:

```text
<artifact_root>/seed{seed}/cfg_TOP0_S2_P2_MT2_OUT1/
  august_fitted_artifacts_seed{seed}.pt
<oof_root>/seed{seed}/cfg_TOP0_S2_P2_MT2_OUT1/
  split_seed{seed}_analysis/posthoc_crossfit_oof_native_risks.csv
```

Check every required file before starting analysis:

```bash
python3 code/wasserstein_FC/run_wasserstein_FC.py \
  --config-file /secure/PathNoUDP.json \
  --seeds 42-51 \
  --output-root /secure/wasserstein_FC_run \
  --check-inputs
```

Then run one configuration at a time:

```bash
python3 code/wasserstein_FC/run_wasserstein_FC.py \
  --config-file /secure/PathNoUDP.json \
  --seeds 42-51 \
  --output-root /secure/wasserstein_FC_run \
  --write-patient-level
```

`--write-patient-level` is required by the current finalizer because it expects
the patient-level partition files. Those files must remain restricted. The
aggregate per-seed summaries, treatment support, feature mapping, and quantile
audit can be considered for sharing only after small-cell and disclosure review.
Remove absolute paths and hashes from `calibration_qc.csv` and `qc.json` before
release.

## Files in this folder

- `run_wasserstein_FC.py` -- post-fit molecular, branch, and
  treatment-comparison analysis.
- `wasserstein_FC_io.py` -- portable JSON configuration and saved-artifact
  loading. It contains no donor workflow.
- `wasserstein_FC_sampling.py` -- joint posterior sampling helpers.
- `wasserstein_FC_calibration.py` -- OOF calibration fit and self-test.
- `wasserstein_FC_finalize.py` -- combines seed partitions into aggregate
  tables.
- `input_manifest.*.example.json` -- one complete input manifest per model
  configuration.
