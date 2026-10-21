# TCGA DGP and baseline code

This folder contains the code for the PathClin, PathNoUDP, and PathUDP input configurations and the corresponding baseline models. It contains no TCGA data, fitted models, predictions, or manuscript results.

## Manuscript result data

The reviewed, non-patient-level tables that support the manuscript are in
[`results/`](results/). That directory has its own manifest, data dictionary,
provenance notes, and an explicit list of source-data artifacts that are not
available or are intentionally not released.

| Configuration | Model inputs |
| --- | --- |
| PathClin | Pathway activity, clinical variables, treatment |
| PathNoUDP | PathClin inputs, somatic mutations, tumor mutational burden |
| PathUDP | PathNoUDP inputs, gene-state probabilities |

The files in `code/` are grouped by their `TCGA_preprocess_`, `TCGA_dgp_`, and `TCGA_baselines_` prefixes. The two `TCGA_collect_` scripts combine results after seeds 42–51 finish. The DGP scripts import `TCGA_treatment_conditioned_WFC_NPJ_August.py`. That helper is not the later fixed-contrast analysis. Use Python 3.10 or newer.

## Inputs and environment

Set `TCGA_ROOT` to the directory containing the original `TCGA_BRCA2`, `LUAD`, and `COAD` inputs. The code does not download them. Set `RUN_ROOT` to a **new** directory under `TCGA_ROOT` or `NPJ_SCRATCH_ROOT`. Its path must contain `August` and the configuration name; the scripts check for both. Keep outputs outside this source folder.

`requirements.txt` lists package names, but the Sol package versions were not saved here. Record those versions before claiming an exact rerun. Set `BASELINE_DEVICE=cpu` or `cuda` as needed.

## Run one split

From this folder, for example:

```bash
export TCGA_ROOT=/path/to/TCGA
export NPJ_SCRATCH_ROOT=/path/to/scratch
export RUN_ROOT=/path/to/scratch/NPJ_August_PathClin
bash run_one_seed.sh PathClin 42
```

Valid configuration names are `PathClin`, `PathNoUDP_MutTMB`, and `PathUDP_MutTMB`. Repeat for seeds 42–51, with a separate `RUN_ROOT` for each configuration. The script preprocesses the data, fits the DGP and baselines, and runs the four Wasserstein jobs required by the DGP collector. It checks the outputs and writes completion records. This runner has not been tested with the full TCGA inputs. Do not use test outcomes to tune the models.

After all ten seeds for one configuration finish, collect that configuration's results:

```bash
python3 code/TCGA_collect_NPJ_August.py \
  --config PathClin --base "$RUN_ROOT" \
  --shared-outdir "$RUN_ROOT/master_collected" --expected-seeds 42-51
python3 code/TCGA_collect_baselines_NPJ_August.py \
  --config PathClin --base "$RUN_ROOT/baselines" --expected-seeds 42-51
```

Change `PathClin` to the relevant configuration. With custom output paths, collect one configuration at a time. The collectors' `--config all` mode expects the original Sol directory names.

To run the copied WFC contract test from this folder, set `PYTHONPATH=code` and run `python3 tests/test_corrected_wfc_contracts.py`.

The scripts save patient-level predictions and outcomes. Check those files before sharing anything. The `.gitignore` excludes common data and result files; still check `git status` before an upload.

## Source files

The DGP scripts, preprocessors, WFC helper, and DGP collector came from `NPJ_August_WFC_corrected`. The baseline scripts and collector came from `NPJ_Sol_run_package_2026-09-08_v2`. In this copy, local paths were replaced with `TCGA_ROOT` and optional `NPJ_SCRATCH_ROOT` variables. The DGP collector was also fixed to count all four C-index groups. Model code and training settings were not changed.

This folder does not include the later fixed-contrast Wasserstein analysis, paired treatment analysis, calibration figures, or METABRIC validation.

No license has been selected for public reuse.
