#!/usr/bin/env python3
"""Collect parallel August TCGA baseline jobs without hiding failed seeds/models."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = [
    "c_test",
    "uno_c_ipcw",
    "somers_dxy_test",
    "brier_at_H",
    "ibs_up_to_H",
    "auroc",
    "auprc",
]
GROUPS = ["Breast", "Colon", "Lung"]
TCGA_ROOT = Path(os.environ["TCGA_ROOT"]).expanduser().resolve()
SCRATCH_ROOT = Path(os.environ.get(
    "NPJ_SCRATCH_ROOT", str(TCGA_ROOT)
))
BASE_ROOTS = {
    "PathClin": TCGA_ROOT / "NPJ_August_2026_FULLRERUN_Baselines_PathClin_MT2",
    "PathNoUDP_MutTMB": TCGA_ROOT / "NPJ_August_2026_FULLRERUN_Baselines_PathNoUDP_MutTMB_MT2",
    "PathUDP_MutTMB": TCGA_ROOT / "NPJ_August_2026_FULLRERUN_Baselines_PathUDP_MutTMB_MT2",
}
ALL_CONFIG_OUTDIR = SCRATCH_ROOT / "NPJ_August_2026_FULLRERUN_Baselines_ALL_CONFIGS_MT2" / "collected"


def parse_seeds(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def metric_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    d = df.copy()
    for metric in METRICS:
        if metric not in d.columns:
            d[metric] = np.nan
        d[metric] = pd.to_numeric(d[metric], errors="coerce")
    if "error" not in d.columns:
        d["error"] = ""
    d["failed"] = d["error"].fillna("").astype(str).str.len() > 0
    named = {
        "n_seeds": ("seed", "nunique"),
        "n_rows": ("seed", "size"),
        "n_failed_rows": ("failed", "sum"),
    }
    for metric in METRICS:
        named[f"{metric}_mean"] = (metric, "mean")
        named[f"{metric}_sd"] = (metric, "std")
        named[f"{metric}_median"] = (metric, "median")
        named[f"{metric}_n"] = (
            metric,
            lambda x: int(pd.Series(x).notna().sum()),
        )
    out = d.groupby(group_cols, dropna=False).agg(**named).reset_index()
    return out.sort_values(group_cols).reset_index(drop=True)


def macro_cindices(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute equal-cancer and patient-weighted C-indices within each split."""
    if metrics.empty:
        return pd.DataFrame()
    d = metrics[metrics["eval_group"].astype(str).isin(GROUPS)].copy()
    keys = [
        "input_config",
        "seed",
        "model",
        "train_mode",
        "train_group",
        "eval_horizon_months",
    ]
    rows = []
    for key, part in d.groupby(keys, dropna=False):
        by_group = (
            part.sort_values("eval_group")
            .drop_duplicates("eval_group", keep="last")
            .set_index("eval_group")
        )
        base = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        for metric in ("c_test", "uno_c_ipcw"):
            vals = pd.to_numeric(by_group.get(metric), errors="coerce")
            complete = all(g in by_group.index for g in GROUPS) and all(
                np.isfinite(vals.get(g, np.nan)) for g in GROUPS
            )
            if complete:
                x = np.asarray([float(vals.loc[g]) for g in GROUPS], dtype=float)
                if "n_eval" in by_group.columns:
                    w = np.asarray(
                        [float(by_group.loc[g, "n_eval"]) for g in GROUPS],
                        dtype=float,
                    )
                else:
                    w = np.ones(len(GROUPS), dtype=float)
                equal = float(np.mean(x))
                weighted = float(np.average(x, weights=w)) if np.sum(w) > 0 else np.nan
                reason = ""
            else:
                equal = weighted = np.nan
                reason = "one or more cancer-specific values were non-estimable"
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "equal_cancer_macro": equal,
                    "patient_weighted_cancer_stratified": weighted,
                    "complete_three_cancers": bool(complete),
                    "nonestimable_reason": reason,
                }
            )
    return pd.DataFrame(rows)


def macro_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    keys = [
        "input_config",
        "model",
        "train_mode",
        "train_group",
        "eval_horizon_months",
        "metric",
    ]
    out = (
        df.groupby(keys, dropna=False)
        .agg(
            equal_cancer_macro_mean=("equal_cancer_macro", "mean"),
            equal_cancer_macro_sd=("equal_cancer_macro", "std"),
            patient_weighted_cancer_stratified_mean=(
                "patient_weighted_cancer_stratified",
                "mean",
            ),
            patient_weighted_cancer_stratified_sd=(
                "patient_weighted_cancer_stratified",
                "std",
            ),
            n_complete_splits=("complete_three_cancers", "sum"),
            n_splits=("seed", "nunique"),
        )
        .reset_index()
    )
    return out.sort_values(keys).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base", type=Path)
    ap.add_argument("--expected-seeds", default="42,43,44,45,46,47,48,49,50,51")
    args = ap.parse_args()

    expected = parse_seeds(args.expected_seeds)
    if args.config == "all":
        bases = {name: path.resolve() for name, path in BASE_ROOTS.items()}
        missing_roots = [str(path) for path in bases.values() if not path.exists()]
        if missing_roots:
            raise FileNotFoundError(
                "All-configuration baseline collection requires all three roots. Missing: "
                + "; ".join(missing_roots)
            )
        collected = ALL_CONFIG_OUTDIR
    else:
        if args.config not in BASE_ROOTS:
            raise ValueError(f"Unknown baseline configuration: {args.config}")
        base = (args.base or BASE_ROOTS[args.config]).expanduser().resolve()
        bases = {args.config: base}
        collected = base / "collected"
    collected.mkdir(parents=True, exist_ok=True)

    metric_parts = []
    prediction_parts = []
    contract_parts = []
    status_rows = []
    for config, base in bases.items():
      for seed in expected:
        seed_dir = base / f"seed{seed}"
        metric_path = seed_dir / "tcga_baselines_npj_allmodels_pfi.csv"
        prediction_path = seed_dir / "tcga_baselines_patient_predictions.csv.gz"
        status_path = seed_dir / "seed_status.csv"
        contract_path = seed_dir / "baseline_required_output_contract.csv"

        if status_path.exists():
            try:
                status = pd.read_csv(status_path).iloc[-1].to_dict()
            except Exception as exc:
                status = {
                    "configuration": config,
                    "seed": seed,
                    "status": "invalid_status_file",
                    "exit_code": np.nan,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        else:
            status = {
                "configuration": config,
                "seed": seed,
                "status": "missing",
                "exit_code": np.nan,
                "error": "seed_status.csv was not found",
            }

        if metric_path.exists():
            try:
                part = pd.read_csv(metric_path)
                part["source_file"] = str(metric_path)
                metric_parts.append(part)
            except Exception as exc:
                status["status"] = "invalid_metrics"
                status["error"] = f"{type(exc).__name__}: {exc}"
        else:
            status["status"] = "missing_metrics"
            status["error"] = "tcga_baselines_npj_allmodels_pfi.csv was not found"

        if prediction_path.exists():
            try:
                part = pd.read_csv(prediction_path)
                part["source_file"] = str(prediction_path)
                prediction_parts.append(part)
            except Exception as exc:
                status["prediction_error"] = f"{type(exc).__name__}: {exc}"
        else:
            status["prediction_error"] = "patient prediction CSV was not found"
        if contract_path.exists():
            try:
                part = pd.read_csv(contract_path)
                part["configuration"] = config
                part["source_file"] = str(contract_path)
                contract_parts.append(part)
            except Exception as exc:
                status["contract_error"] = f"{type(exc).__name__}: {exc}"
        else:
            status["contract_error"] = "baseline_required_output_contract.csv was not found"
        status_rows.append(status)

    metrics = pd.concat(metric_parts, ignore_index=True) if metric_parts else pd.DataFrame()
    predictions = (
        pd.concat(prediction_parts, ignore_index=True)
        if prediction_parts
        else pd.DataFrame()
    )
    statuses = pd.DataFrame(status_rows)
    contracts = (
        pd.concat(contract_parts, ignore_index=True)
        if contract_parts
        else pd.DataFrame()
    )

    metrics.to_csv(collected / "master_baseline_metrics_all_splits.csv", index=False)
    predictions.to_csv(
        collected / "master_baseline_patient_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    statuses.to_csv(collected / "baseline_seed_status.csv", index=False)
    contracts.to_csv(
        collected / "master_baseline_required_output_contract.csv", index=False
    )

    pooled = metric_summary(
        metrics[metrics.get("eval_group", pd.Series(dtype=str)).astype(str) == "ALL"],
        [
            "input_config",
            "model",
            "train_mode",
            "train_group",
            "eval_horizon_months",
        ],
    )
    by_cancer = metric_summary(
        metrics[metrics.get("eval_group", pd.Series(dtype=str)).astype(str).isin(GROUPS)],
        [
            "input_config",
            "model",
            "train_mode",
            "train_group",
            "eval_group",
            "eval_horizon_months",
        ],
    )
    pooled.to_csv(collected / "master_baseline_summary_pooled.csv", index=False)
    by_cancer.to_csv(collected / "master_baseline_summary_by_cancer.csv", index=False)
    pooled_c = pooled.copy()
    if not pooled_c.empty:
        pooled_c["scope"] = "pooled"
        pooled_c["eval_group"] = "ALL"
    cancer_c = by_cancer.copy()
    if not cancer_c.empty:
        cancer_c["scope"] = "within_cancer"
    pd.concat([pooled_c, cancer_c], ignore_index=True, sort=False).to_csv(
        collected / "master_baseline_cindex_summary.csv", index=False
    )

    cindex_cols = [
        c
        for c in [
            "input_config",
            "seed",
            "model",
            "train_mode",
            "train_group",
            "eval_group",
            "eval_horizon_months",
            "c_test",
            "uno_c_ipcw",
            "n_eval",
            "n_events_total",
            "n_early_events_by_H",
            "n_censored_before_H",
            "error",
        ]
        if c in metrics.columns
    ]
    metrics[cindex_cols].to_csv(
        collected / "master_baseline_cindices_by_seed.csv", index=False
    )
    macro = macro_cindices(metrics)
    macro.to_csv(collected / "master_baseline_macro_cindices_by_seed.csv", index=False)
    macro_summary(macro).to_csv(
        collected / "master_baseline_macro_cindex_summary.csv", index=False
    )

    if "error" in metrics.columns:
        failures = metrics[
            metrics["error"].fillna("").astype(str).str.len() > 0
        ].copy()
    else:
        failures = pd.DataFrame()
    failures.to_csv(collected / "baseline_model_failures.csv", index=False)
    if not failures.empty:
        (
            failures.groupby(
                ["input_config", "seed", "model", "train_mode", "train_group"],
                dropna=False,
            )
            .agg(
                failed_rows=("error", "size"),
                error=("error", lambda x: " | ".join(sorted(set(map(str, x))))),
            )
            .reset_index()
            .to_csv(collected / "baseline_model_failure_summary.csv", index=False)
        )
    else:
        pd.DataFrame().to_csv(
            collected / "baseline_model_failure_summary.csv", index=False
        )

    expected_pairs = {(config, int(seed)) for config in bases for seed in expected}
    if len(metrics) and all(c in metrics for c in ["input_config", "seed"]):
        metric_pairs = set(zip(
            metrics["input_config"].astype(str),
            pd.to_numeric(metrics["seed"], errors="coerce").fillna(-1).astype(int),
        ))
    else:
        metric_pairs = set()
    complete_seeds = sorted({seed for _, seed in metric_pairs if seed >= 0})
    contract_ok = (
        len(contracts) > 0
        and "status" in contracts
        and contracts["status"].astype(str).eq("ok").all()
        and all(c in contracts for c in ["configuration", "seed"])
        and set(zip(
            contracts["configuration"].astype(str),
            pd.to_numeric(contracts["seed"], errors="coerce").fillna(-1).astype(int),
        )) == expected_pairs
    )
    statuses_ok = (
        len(statuses) == len(expected_pairs)
        and statuses["status"].astype(str).eq("complete").all()
    )
    manifest = {
        "configuration": args.config,
        "configurations": sorted(bases),
        "expected_seeds": expected,
        "seeds_with_metrics": complete_seeds,
        "missing_configuration_seed_pairs": sorted(
            f"{config}:seed{seed}" for config, seed in (expected_pairs - metric_pairs)
        ),
        "metric_rows": int(len(metrics)),
        "prediction_rows": int(len(predictions)),
        "model_failure_rows": int(len(failures)),
        "required_output_contract_complete": bool(contract_ok),
        "seed_status_complete": bool(statuses_ok),
        "complete": bool(
            expected_pairs == metric_pairs
            and contract_ok
            and statuses_ok
            and len(failures) == 0
        ),
    }
    with open(collected / "baseline_collection_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not manifest["complete"]:
        raise RuntimeError(
            "Baseline collection is incomplete. Inspect baseline_seed_status.csv, "
            "baseline_model_failures.csv, and master_baseline_required_output_contract.csv."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
