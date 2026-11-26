#!/usr/bin/env python3
"""Build a results-only manuscript data release from audited aggregate files.

This script deliberately names every accepted source. It does not search for or
copy raw data, patient-level predictions, sample identifiers, or checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path


SEEDS = tuple(range(42, 52))


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required audited source file was not found: {path}")
    return path


def copy_file(source_root: Path, source_relative: str, output_root: Path, output_relative: str, *, gzip_output: bool = False) -> None:
    source = require_file(source_root / source_relative)
    output = output_root / output_relative
    output.parent.mkdir(parents=True, exist_ok=True)
    if gzip_output:
        with source.open("rb") as reader, gzip.open(output, "wb", compresslevel=9) as writer:
            shutil.copyfileobj(reader, writer)
    else:
        shutil.copyfile(source, output)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def build_cohort_summary(source_root: Path, output_root: Path) -> None:
    source = require_file(source_root / "work/npj_cohort_analysis/cohort_summaries.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for summary in payload["summaries"]:
        configuration = str(summary["label"])
        rows.extend(
            [
                {"configuration": configuration, "section": "overall", "variable": "Patients", "level": "all", "statistic": "n", "value": summary["n"], "unit": "patients"},
                {"configuration": configuration, "section": "overall", "variable": "Unique patients", "level": "all", "statistic": "n", "value": summary["unique_patients"], "unit": "patients"},
                {"configuration": configuration, "section": "overall", "variable": "Age", "level": "all", "statistic": "mean", "value": summary["age_mean"], "unit": "years"},
                {"configuration": configuration, "section": "overall", "variable": "Age", "level": "all", "statistic": "sd", "value": summary["age_sd"], "unit": "years"},
                {"configuration": configuration, "section": "overall", "variable": "PFI time", "level": "all", "statistic": "median", "value": summary["pfi_time_median"], "unit": "months"},
                {"configuration": configuration, "section": "overall", "variable": "PFI time", "level": "all", "statistic": "q1", "value": summary["pfi_time_q1"], "unit": "months"},
                {"configuration": configuration, "section": "overall", "variable": "PFI time", "level": "all", "statistic": "q3", "value": summary["pfi_time_q3"], "unit": "months"},
            ]
        )
        if summary.get("tmb_median") is not None:
            rows.extend(
                [
                    {"configuration": configuration, "section": "overall", "variable": "TMB", "level": "all", "statistic": "median", "value": summary["tmb_median"], "unit": "mutations per Mb"},
                    {"configuration": configuration, "section": "overall", "variable": "TMB", "level": "all", "statistic": "q1", "value": summary["tmb_q1"], "unit": "mutations per Mb"},
                    {"configuration": configuration, "section": "overall", "variable": "TMB", "level": "all", "statistic": "q3", "value": summary["tmb_q3"], "unit": "mutations per Mb"},
                ]
            )
        for variable, levels in summary["categories"].items():
            for level, values in levels.items():
                rows.extend(
                    [
                        {"configuration": configuration, "section": "category", "variable": variable, "level": level, "statistic": "n", "value": values["n"], "unit": "patients"},
                        {"configuration": configuration, "section": "category", "variable": variable, "level": level, "statistic": "percent", "value": values["pct"], "unit": "percent"},
                    ]
                )
    write_rows(
        output_root / "data/cohort/cohort_summary.csv",
        ["configuration", "section", "variable", "level", "statistic", "value", "unit"],
        rows,
    )
    copy_file(source_root, "work/npj_cohort_analysis/modality_overlap.csv", output_root, "data/cohort/modality_overlap.csv")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_external_metrics(source_root: Path, output_root: Path) -> None:
    source_templates = {
        "PathClin": "outputs/metabric_external_validation_pathclin_local_2026-09-14/PathClin/seed{seed}/primary_size_derived_AJCC_T/metabric_external_seed_metrics.csv",
        "PathNoUDP_MutTMB": "outputs/metabric_external_validation_corrected_2026-09-14/PathNoUDP_MutTMB/seed{seed}/primary_size_derived_AJCC_T/metabric_external_seed_metrics.csv",
        "PathUDP_MutTMB": "outputs/metabric_external_validation_pathclin_local_2026-09-14/PathUDP_MutTMB/seed{seed}/primary_size_derived_AJCC_T/metabric_external_seed_metrics.csv",
    }
    rows: list[dict[str, str]] = []
    columns: set[str] = set()
    for configuration, template in source_templates.items():
        for seed in SEEDS:
            source_rows = read_csv(require_file(source_root / template.format(seed=seed)))
            for row in source_rows:
                row["source_configuration"] = configuration
                row["seed"] = str(seed)
                rows.append(row)
                columns.update(row)
    fieldnames = sorted(columns)
    output = output_root / "data/external_validation/metabric_external_metrics_by_seed_all_configurations.csv"
    write_rows(output, fieldnames, rows)

    metric_columns = [
        "harrell_c_index", "somers_dxy", "ipcw_auroc", "ipcw_auprc", "ipcw_brier",
        "ipcw_event_prevalence", "ipcw_auprc_lift_over_prevalence",
        "integrated_Brier_score_3_to_60_months",
    ]
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["source_configuration"], row.get("endpoint", ""), row.get("horizon_months", ""))].append(row)
    summary_rows: list[dict[str, object]] = []
    for (configuration, endpoint, horizon), group in sorted(grouped.items()):
        for metric in metric_columns:
            values = [value for row in group if (value := as_float(row.get(metric))) is not None]
            if not values:
                continue
            summary_rows.append(
                {
                    "configuration": configuration,
                    "endpoint": endpoint,
                    "horizon_months": horizon,
                    "metric": metric,
                    "mean_across_seeds": statistics.mean(values),
                    "sd_across_seeds": statistics.stdev(values) if len(values) > 1 else "",
                    "n_seeds": len(values),
                }
            )
    write_rows(
        output_root / "data/external_validation/metabric_external_metrics_summary_all_configurations.csv",
        ["configuration", "endpoint", "horizon_months", "metric", "mean_across_seeds", "sd_across_seeds", "n_seeds"],
        summary_rows,
    )
    copy_file(source_root, "outputs/metabric_external_validation_corrected_2026-09-14/collected/METABRIC_external_final_metrics_with_bootstrap_CI.csv", output_root, "data/external_validation/metabric_pathnoudp_final_metrics_with_bootstrap_ci.csv")
    copy_file(source_root, "outputs/metabric_external_validation_corrected_2026-09-14/collected/METABRIC_external_seed_specific_Cindex.csv", output_root, "data/external_validation/metabric_pathnoudp_harrell_cindex_by_seed.csv")
    copy_file(source_root, "outputs/metabric_external_validation_corrected_2026-09-14/collected/METABRIC_external_overall_Cindex.csv", output_root, "data/external_validation/metabric_pathnoudp_harrell_cindex_summary.csv")
    for filename in [
        "METABRIC_Cindex_AUROC_AUPRC_mean_sd.csv",
        "METABRIC_horizon_specific_Harrell_Cindex.csv",
        "METABRIC_external_seed_specific_metrics.csv",
    ]:
        copy_file(
            source_root,
            f"outputs/metabric_external_validation_corrected_2026-09-14/collected/{filename}",
            output_root,
            f"data/external_validation/{filename}",
        )


def copy_curated_outputs(source_root: Path, output_root: Path) -> None:
    file_map = {
        "outputs/pathnoudp_km_early_failure_2026-09-15/early_failure_lift_by_seed.csv": "data/early_risk_and_km/early_failure_lift_by_seed.csv",
        "outputs/pathnoudp_km_early_failure_2026-09-15/early_failure_lift_mean_sd_across_seeds.csv": "data/early_risk_and_km/early_failure_lift_mean_sd_across_seeds.csv",
        "outputs/pathnoudp_km_early_failure_2026-09-15/km_H18_curves_by_seed.csv": "data/early_risk_and_km/km_H18_curves_by_seed.csv",
        "outputs/pathnoudp_km_early_failure_2026-09-15/km_H18_curves_mean_sd_across_seeds.csv": "data/early_risk_and_km/km_H18_curves_mean_sd_across_seeds.csv",
        "outputs/pathnoudp_early_failure_table_2026-09-15/pathnoudp_early_failure_event_counts_by_seed.csv": "data/early_risk_and_km/early_failure_event_counts_by_seed.csv",
        "outputs/pathnoudp_early_failure_table_2026-09-15/pathnoudp_early_failure_event_counts_summary.csv": "data/early_risk_and_km/early_failure_event_counts_summary.csv",
        "outputs/pathnoudp_early_failure_table_2026-09-15/pathnoudp_early_failure_support_and_lift_summary.csv": "data/early_risk_and_km/early_failure_support_and_lift_summary.csv",
        "outputs/pathnoudp_calibration_across_splits_2026-09-15/pathnoudp_calibration_bins_by_seed.csv": "data/calibration/pathnoudp_calibration_bins_by_seed.csv",
        "outputs/pathnoudp_calibration_across_splits_2026-09-15/pathnoudp_calibration_bins_across_seeds.csv": "data/calibration/pathnoudp_calibration_bins_across_seeds.csv",
        "outputs/pathnoudp_calibration_across_splits_2026-09-15/pathnoudp_calibration_error_summary.csv": "data/calibration/pathnoudp_calibration_error_summary.csv",
        "outputs/Wasserstein_FIXED_CONTRAST_2026-09-16/final_results/PathNoUDP_fixed_contrast_across_seed_feature_summary.csv": "data/wasserstein/pathnoudp_feature_scores_across_seeds.csv",
        "outputs/Wasserstein_FIXED_CONTRAST_2026-09-16/final_results/PathNoUDP_cross_cancer_fixed_contrast_summary.csv": "data/wasserstein/pathnoudp_cross_cancer_feature_summary.csv",
        "outputs/Wasserstein_FIXED_CONTRAST_2026-09-16/final_results/PathNoUDP_fixed_contrast_seed_feature_summary.csv": "data/wasserstein/pathnoudp_feature_scores_by_seed.csv",
        "outputs/Wasserstein_FIXED_CONTRAST_2026-09-16/final_results/PathClin_fixed_contrast_across_seed_feature_summary.csv": "data/wasserstein/pathclin_feature_scores_across_seeds.csv",
        "outputs/Wasserstein_FIXED_CONTRAST_2026-09-16/final_results/PathClin_cross_cancer_fixed_contrast_summary.csv": "data/wasserstein/pathclin_cross_cancer_feature_summary.csv",
        "outputs/Wasserstein_FIXED_CONTRAST_2026-09-16/final_results/PathClin_fixed_contrast_seed_feature_summary.csv": "data/wasserstein/pathclin_feature_scores_by_seed.csv",
        "outputs/Wasserstein_FIXED_CONTRAST_2026-09-16/final_results/fixed_contrast_treatment_support.csv": "data/wasserstein/fixed_contrast_treatment_support.csv",
        "outputs/Wasserstein_FIXED_CONTRAST_2026-09-16/final_results/fixed_contrast_feature_identity_mapping.csv": "data/wasserstein/fixed_contrast_feature_identity_mapping.csv",
        "outputs/Fixed_contrast_corrected_postprocessing_2026-09-16/PathNoUDP_cross_cancer_shared_features_top10_by_horizon.csv": "data/wasserstein/pathnoudp_cross_cancer_shared_top10_by_horizon.csv",
        "outputs/Fixed_contrast_corrected_postprocessing_2026-09-16/PathNoUDP_cross_cancer_shared_features_top10_across_horizons.csv": "data/wasserstein/pathnoudp_cross_cancer_shared_top10_across_horizons.csv",
        "outputs/Fixed_contrast_corrected_postprocessing_2026-09-16/PathNoUDP_cancer_preferential_features_top10_by_horizon.csv": "data/wasserstein/pathnoudp_cancer_specific_top10_by_horizon.csv",
        "outputs/Fixed_contrast_corrected_postprocessing_2026-09-16/PathNoUDP_cancer_preferential_features_top10_across_horizons.csv": "data/wasserstein/pathnoudp_cancer_specific_top10_across_horizons.csv",
        "outputs/Fixed_contrast_corrected_postprocessing_2026-09-16/PathClin_cross_cancer_shared_features_top10_by_horizon.csv": "data/wasserstein/pathclin_cross_cancer_shared_top10_by_horizon.csv",
        "outputs/Fixed_contrast_corrected_postprocessing_2026-09-16/PathClin_cross_cancer_shared_features_top10_across_horizons.csv": "data/wasserstein/pathclin_cross_cancer_shared_top10_across_horizons.csv",
        "outputs/Fixed_contrast_corrected_postprocessing_2026-09-16/PathClin_cancer_preferential_features_top10_by_horizon.csv": "data/wasserstein/pathclin_cancer_specific_top10_by_horizon.csv",
        "outputs/Fixed_contrast_corrected_postprocessing_2026-09-16/PathClin_cancer_preferential_features_top10_across_horizons.csv": "data/wasserstein/pathclin_cancer_specific_top10_across_horizons.csv",
        "outputs/final_consensus_figure_workflow_2026-09-17/rankings/PathNoUDP_treatment_profile_wasserstein_consensus_ranking.csv": "data/treatment/fixed_support_pathnoudp_treatment_profile_wasserstein_ranking.csv",
        "outputs/final_consensus_figure_workflow_2026-09-17/rankings/PathClin_treatment_profile_wasserstein_consensus_ranking.csv": "data/treatment/fixed_support_pathclin_treatment_profile_wasserstein_ranking.csv",
        "outputs/Fixed_contrast_corrected_postprocessing_2026-09-16/PathNoUDP_molecular_feature_treatment_contrast_consensus_top10_corrected.csv": "data/treatment/fixed_support_pathnoudp_feature_stratified_treatment_pair_top10.csv",
        "outputs/Fixed_contrast_corrected_postprocessing_2026-09-16/PathClin_molecular_feature_treatment_contrast_consensus_top10_corrected.csv": "data/treatment/fixed_support_pathclin_feature_stratified_treatment_pair_top10.csv",
        "outputs/PathNoUDP_OBSERVED_PATIENT_TREATMENT_RISK_2026-09-19/mutation_best_treatment_pair_per_feature_ranking_all_horizons.csv": "data/treatment/mutation_best_treatment_pair_per_feature_all_horizons.csv",
        "outputs/PathNoUDP_OBSERVED_PATIENT_TREATMENT_RISK_2026-09-19/top_unique_mutation_features_24m.csv": "data/treatment/mutation_treatment_figure5_top_features_24m.csv",
        "outputs/PathNoUDP_OBSERVED_PATIENT_TREATMENT_RISK_2026-09-19/treatment_contrast_consensus.csv": "data/treatment/overall_treatment_profile_contrast_consensus.csv",
    }
    for source_relative, output_relative in file_map.items():
        copy_file(source_root, source_relative, output_root, output_relative)
    copy_file(
        source_root,
        "outputs/PathNoUDP_OBSERVED_PATIENT_TREATMENT_RISK_2026-09-19/mutation_treatment_interactions_consensus.csv",
        output_root,
        "data/treatment/mutation_treatment_interactions_consensus.csv.gz",
        gzip_output=True,
    )
    for configuration in ["PathNoUDP", "PathClin"]:
        copy_file(
            source_root,
            f"outputs/Fixed_contrast_corrected_postprocessing_2026-09-16/{configuration}_molecular_feature_treatment_contrast_consensus_ranking_corrected.csv",
            output_root,
            f"data/treatment/fixed_support_{configuration.lower()}_feature_stratified_treatment_pair_consensus.csv.gz",
            gzip_output=True,
        )


def write_manifest(output_root: Path) -> None:
    rows = [
        {"manuscript_component": "Main Table 1", "status": "available", "release_files": "data/cohort/cohort_summary.csv; data/cohort/modality_overlap.csv", "note": "Aggregate, non-patient-level cohort characteristics."},
        {"manuscript_component": "Main Figure 1", "status": "not-applicable", "release_files": "", "note": "Workflow schematic; no numeric source-data table identified."},
        {"manuscript_component": "Main Tables 2-7; Supplementary Tables 1-21", "status": "missing", "release_files": "", "note": "Complete internal per-split DGP-and-baseline collector outputs were not found; see NOT_FOUND_OR_NOT_RELEASED.md."},
        {"manuscript_component": "Main Table 8", "status": "available", "release_files": "data/external_validation/", "note": "Aggregate METABRIC performance data, including all three input configurations."},
        {"manuscript_component": "Main Figure 2a-c; Supplementary Table 22", "status": "available", "release_files": "data/early_risk_and_km/early_failure_lift_by_seed.csv; data/early_risk_and_km/early_failure_lift_mean_sd_across_seeds.csv", "note": "Seed-level and across-seed early-event enrichment data."},
        {"manuscript_component": "Main Figure 2d-f; Supplementary Figures 13-17", "status": "partial", "release_files": "data/early_risk_and_km/km_H18_curves_by_seed.csv; data/early_risk_and_km/km_H18_curves_mean_sd_across_seeds.csv", "note": "18-month aggregate KM data are included; 24-month aggregate source tables were not found."},
        {"manuscript_component": "Supplementary Figures 1-12", "status": "partial", "release_files": "data/calibration/", "note": "Per-seed and across-seed calibration-bin summaries; calibration-intercept and bin-edge exports were not located."},
        {"manuscript_component": "Main Figure 3", "status": "not-released", "release_files": "", "note": "Exact supporting artifact is patient-level posterior data; no anonymized panel-only export was found."},
        {"manuscript_component": "Main Figure 4; Supplementary Figures 18-20; Supplementary Tables 23-26", "status": "available", "release_files": "data/wasserstein/", "note": "All aggregate WFP feature scores plus selected figure/table rows."},
        {"manuscript_component": "Supplementary Figures 21-23", "status": "available-fixed-support-version", "release_files": "data/treatment/fixed_support_*", "note": "Fixed-support treatment-profile and feature-stratified treatment-pair WFP source tables. See METHOD_AND_VERSION_NOTES.md."},
        {"manuscript_component": "Main Figure 5; Supplementary Table 27", "status": "available", "release_files": "data/treatment/", "note": "Aggregate wild-type/mutated risk contrasts; exhaustive consensus is gzip-compressed."},
        {"manuscript_component": "Supplementary Table 28", "status": "unresolved", "release_files": "", "note": "Available source versions disagree on C-index values; the final source table needs reconciliation before release."},
    ]
    write_rows(
        output_root / "results_manifest.csv",
        ["manuscript_component", "status", "release_files", "note"],
        rows,
    )


def write_checksums(output_root: Path) -> None:
    """Write portable SHA-256 hashes for every released file except this list."""
    checksum_path = output_root / "CHECKSUMS.sha256"
    entries: list[str] = []
    for path in sorted(item for item in output_root.rglob("*") if item.is_file() and item != checksum_path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(f"{digest}  {path.relative_to(output_root).as_posix()}")
    checksum_path.write_text("\n".join(entries) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path, help="Root of the audited local TCGA NPJ workspace")
    args = parser.parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = Path(__file__).resolve().parents[1]
    build_cohort_summary(source_root, output_root)
    build_external_metrics(source_root, output_root)
    copy_curated_outputs(source_root, output_root)
    write_manifest(output_root)
    write_checksums(output_root)
    print(f"Wrote curated result release under {output_root}")


if __name__ == "__main__":
    main()
