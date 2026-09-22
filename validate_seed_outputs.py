#!/usr/bin/env python3
"""Check outputs required by the original August seed collectors."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CFG_TAG = "cfg_TOP0_S2_P2_MT2_OUT1"
HORIZONS = {3.0, 6.0, 12.0, 18.0, 24.0, 36.0, 60.0}
BASELINE_MODELS = {
    "LR", "RF", "SVM", "STL-BNN", "MTL-BNN", "CoxPH", "DeepCox",
    "MT-DeepCox", "MT-DiscHazard",
}
WFC_FILES = (
    "corrected_wfc_patient_level.csv.gz",
    "corrected_wfc_seed_cancer_feature_summary.csv",
    "corrected_wfc_seed_treatment_profile_summary.csv",
    "corrected_wfc_seed_coverage.csv",
    "corrected_wfc_patient_donor_support.csv",
    "corrected_wfc_donor_audit.csv.gz",
    "corrected_wfc_feature_block_manifest.csv",
    "corrected_wfc_integrity_checks.csv",
    "corrected_wfc_partition_status.csv",
)


def check_dgp(seed: int, seed_dir: Path) -> None:
    path = seed_dir / CFG_TAG / f"{CFG_TAG}_results.csv"
    result = pd.read_csv(path)
    if result.empty or "status" not in result or not result["status"].astype(str).eq("ok").all():
        raise RuntimeError(f"Seed {seed} has no fully successful DGP result: {path}")


def check_baseline(seed: int, seed_dir: Path) -> None:
    checks = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({
            "seed": seed, "required_output": name,
            "status": "ok" if ok else "failed", "detail": detail,
        })

    metrics_path = seed_dir / "tcga_baselines_npj_allmodels_pfi.csv"
    predictions_path = seed_dir / "tcga_baselines_patient_predictions.csv.gz"
    metrics = pd.read_csv(metrics_path) if metrics_path.is_file() else pd.DataFrame()
    predictions = pd.read_csv(predictions_path) if predictions_path.is_file() else pd.DataFrame()
    record("baseline metrics CSV", not metrics.empty, f"rows={len(metrics)}")
    record("baseline patient predictions CSV", not predictions.empty, f"rows={len(predictions)}")
    if not metrics.empty:
        observed_models = set(metrics["model"].dropna().astype(str))
        missing_models = sorted(BASELINE_MODELS - observed_models)
        record("required baseline models", not missing_models, f"missing={missing_models}")
        for model in sorted(BASELINE_MODELS & observed_models):
            got = set(pd.to_numeric(
                metrics.loc[metrics["model"].astype(str).eq(model), "eval_horizon_months"],
                errors="coerce",
            ).dropna().astype(float))
            missing = sorted(HORIZONS - got)
            record(f"{model} horizon coverage", not missing, f"missing={missing}")
        errors = metrics.get("error", pd.Series("", index=metrics.index)).fillna("").astype(str).str.strip()
        record("baseline model execution errors", not errors.ne("").any(),
               f"error_rows={int(errors.ne('').sum())}")
    contract = pd.DataFrame(checks)
    contract.to_csv(seed_dir / "baseline_required_output_contract.csv", index=False)
    failed = contract[contract["status"].ne("ok")]
    if len(failed):
        raise RuntimeError("Baseline output contract failed: " + " | ".join(
            f"{r.required_output}: {r.detail}" for r in failed.itertuples()
        ))


def check_wfc(horizon: int, part_dir: Path) -> None:
    checks = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({
            "horizon_months": horizon, "required_output": name,
            "status": "ok" if ok else "failed", "detail": detail,
        })

    for name in WFC_FILES:
        path = part_dir / name
        ok = path.is_file() and path.stat().st_size > 1
        record(name, ok, f"bytes={path.stat().st_size}" if ok else "missing_or_empty")

    summary_path = part_dir / "corrected_wfc_seed_cancer_feature_summary.csv"
    status_path = part_dir / "corrected_wfc_partition_status.csv"
    integrity_path = part_dir / "corrected_wfc_integrity_checks.csv"
    if all(path.is_file() for path in (summary_path, status_path, integrity_path)):
        summary = pd.read_csv(summary_path)
        cancers = set(summary["cancer"].astype(str))
        record("BRCA_COAD_LUAD_results", cancers == {"BRCA", "COAD", "LUAD"},
               f"observed={sorted(cancers)}")
        status = pd.read_csv(status_path)
        integrity = pd.read_csv(integrity_path)
        versions = set(status.get("implementation_version", pd.Series(dtype=str)).astype(str))
        replay_ok = bool(integrity.loc[
            integrity["check"].astype(str).eq("common_random_numbers_null_replay"),
            "status",
        ].astype(str).eq("ok").any())
        version_ok = versions == {"full_batch_crn_v2"}
        record("full_batch_common_random_numbers", version_ok and replay_ok,
               f"versions={sorted(versions)}; null_replay_ok={replay_ok}")

    contract = pd.DataFrame(checks)
    contract.to_csv(part_dir / "corrected_wfc_required_output_contract.csv", index=False)
    if contract["status"].ne("ok").any():
        raise RuntimeError(f"Corrected WFC output contract failed for H={horizon}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("dgp", "baseline", "wfc"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int)
    args = parser.parse_args()
    if args.kind == "dgp":
        check_dgp(args.seed, args.output_dir)
    elif args.kind == "baseline":
        check_baseline(args.seed, args.output_dir)
    else:
        if args.horizon is None:
            parser.error("--horizon is required for wfc")
        check_wfc(args.horizon, args.output_dir)


if __name__ == "__main__":
    main()
