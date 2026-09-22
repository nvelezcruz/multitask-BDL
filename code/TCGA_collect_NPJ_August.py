#!/usr/bin/env python3
"""Collect NPJ August TCGA repeated-split outputs without refitting models."""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from TCGA_treatment_conditioned_WFC_NPJ_August import (
    APPROVED_HORIZONS,
    collect_corrected_wfc,
)

TCGA_ROOT = Path(os.environ["TCGA_ROOT"]).expanduser().resolve()
NPJ_SCRATCH_ROOT = Path(
    os.getenv("NPJ_SCRATCH_ROOT", str(TCGA_ROOT))
)
APPROVED_OUTPUT_ROOTS = [TCGA_ROOT, NPJ_SCRATCH_ROOT]
SHARED_DEFAULT = TCGA_ROOT / "NPJ_August_2026_FULLRERUN_master_collected_outputs_MT2"
CONFIG_ROOTS = {
    "PathClin": TCGA_ROOT / "NPJ_August_2026_FULLRERUN_PathClin_pathway_clinical_treatment_MT2",
    "PathNoUDP_MutTMB": TCGA_ROOT / "NPJ_August_2026_FULLRERUN_PathNoUDP_MutTMB_pathway_clinical_treatment_mutation_TMB_MT2",
    "PathUDP_MutTMB": TCGA_ROOT / "NPJ_August_2026_FULLRERUN_PathUDP_MutTMB_pathway_UDP_clinical_treatment_mutation_TMB_MT2",
}
EXPECTED_MASTER_FILES = [
    "master_run_status.csv", "master_cohort_attrition.csv", "master_treatment_timing_audit.csv",
    "master_c_indices_by_seed.csv", "master_c_indices_summary.csv",
    "master_c_indices_explicit_by_seed_horizon_group.csv",
    "master_c_indices_explicit_summary.csv",
    "master_configuration_comparisons.csv", "master_performance_metrics.csv",
    "master_calibration_metrics.csv", "master_feature_selection_stability.csv",
    "master_shared_vs_specific_features.csv", "master_treatment_profile_support.csv",
    "master_treatment_contrasts.csv", "master_feature_treatment_interactions.csv",
    "master_treatment_contrast_stability.csv",
    "master_corrected_wfc_seed_cancer_feature.csv",
    "master_corrected_wfc_final_cancer_unique_patient.csv",
    "master_corrected_wfc_final_treatment_profile_unique_patient.csv",
    "master_corrected_wfc_final_shared_across_cancers.csv",
    "master_corrected_wfc_seed_shared_across_cancers.csv",
    "master_corrected_wfc_cross_seed_rank_consistency.csv",
    "master_corrected_wfc_cross_cancer_rank_consistency.csv",
    "master_corrected_wfc_coverage.csv", "master_corrected_wfc_unique_patient_coverage.csv",
    "master_corrected_wfc_patient_donor_support.csv", "master_corrected_wfc_donor_audit.csv.gz",
    "master_km_tertile_summary.csv",
    "master_km_pairwise_logrank_results.csv",
    "master_cholesky_retry_audit.csv", "master_figure_manifest.csv",
    "master_plot_data_manifest.csv", "master_plot_generation_failures.csv",
    "master_required_output_contract.csv",
    "master_corrected_wfc_required_output_contract.csv",
]


def _inside(path: Path, root: Path) -> bool:
    p, r = path.expanduser().resolve(), root.resolve()
    return p == r or r in p.parents


def _inside_approved_output_root(path: Path) -> bool:
    return any(_inside(path, root) for root in APPROVED_OUTPUT_ROOTS)


def validate_config_root(path: Path, config: str) -> Path:
    p = path.expanduser().resolve()
    if not _inside_approved_output_root(p) or "august" not in str(p).lower() or config.lower() not in str(p).lower():
        raise ValueError(f"Unsafe configuration OUTDIR for {config}: {p}")
    return p


def validate_shared_root(path: Path) -> Path:
    p = path.expanduser().resolve()
    if not _inside_approved_output_root(p) or "august" not in str(p).lower() or "master_collected" not in str(p).lower():
        raise ValueError(f"Unsafe shared collector OUTDIR: {p}")
    return p


def safe_read(path: Path) -> pd.DataFrame:
    try:
        d = pd.read_csv(path)
        d["source_file"] = str(path)
        return d
    except Exception as exc:
        return pd.DataFrame([{"source_file": str(path), "read_error": f"{type(exc).__name__}: {exc}"}])


def collect_named(roots, filename) -> pd.DataFrame:
    files = []
    for config, root in roots.items():
        for p in root.glob(f"seed*/**/{filename}"):
            d = safe_read(p)
            if "configuration" not in d:
                d.insert(0, "configuration", config)
            m = re.search(r"seed(\d+)", str(p))
            if "seed" not in d and m:
                d.insert(1, "seed", int(m.group(1)))
            files.append(d)
    return pd.concat(files, ignore_index=True, sort=False) if files else pd.DataFrame()


def collect_patterns(roots, patterns, exclude=()) -> pd.DataFrame:
    chunks, seen = [], set()
    for config, root in roots.items():
        for pattern in patterns:
            for p in root.glob(pattern):
                if p in seen or any(x.lower() in p.name.lower() for x in exclude):
                    continue
                seen.add(p)
                d = safe_read(p)
                if "configuration" not in d:
                    d.insert(0, "configuration", config)
                m = re.search(r"seed(\d+)", str(p))
                if "seed" not in d and m:
                    d.insert(1, "seed", int(m.group(1)))
                chunks.append(d)
    return pd.concat(chunks, ignore_index=True, sort=False) if chunks else pd.DataFrame()


def write_table(df, path, columns=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None or len(df) == 0:
        df = pd.DataFrame(columns=list(columns) or ["configuration", "status", "nonestimable_reason"])
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    df.to_csv(tmp, index=False, compression="gzip" if path.suffix == ".gz" else None)
    tmp.replace(path)


def atomic_write_text(path: Path, text: str):
    """Replace a collector text artifact atomically to avoid partial shared files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text)
    tmp.replace(path)


def summarize_numeric(df, value_col, group_cols, expected_seeds=10):
    if df.empty or value_col not in df:
        return pd.DataFrame()
    work = df.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    rows = []
    for keys, sub in work.groupby(group_cols, dropna=False):
        vals = sub[value_col].dropna().to_numpy(float)
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update({"metric_value_column": value_col, "mean_across_splits": np.mean(vals) if len(vals) else np.nan,
                    "sd_across_splits": np.std(vals, ddof=1) if len(vals) > 1 else (0.0 if len(vals) == 1 else np.nan),
                    "median_across_splits": np.median(vals) if len(vals) else np.nan,
                    "q05_across_splits": np.quantile(vals, .05) if len(vals) else np.nan,
                    "q95_across_splits": np.quantile(vals, .95) if len(vals) else np.nan,
                    "minimum": np.min(vals) if len(vals) else np.nan, "maximum": np.max(vals) if len(vals) else np.nan,
                    "successful_split_count": len(vals), "failed_split_count": max(0, expected_seeds - len(vals))})
        rows.append(row)
    return pd.DataFrame(rows)


def truthy_series(values):
    """Parse CSV booleans without treating the string 'False' as true."""
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def complete_seed_horizon_coverage(df, roots, expected_seeds, horizon_col, horizons, extra_keys=()):
    """Verify every requested configuration/split/context contains every horizon."""
    required_cols = ["configuration", "seed", horizon_col, *extra_keys]
    if df.empty or any(c not in df for c in required_cols):
        return False
    expected_h = {float(x) for x in horizons}
    observed_keys = 0
    for _, sub in df.groupby(["configuration", "seed", *extra_keys], dropna=False):
        got = set(pd.to_numeric(sub[horizon_col], errors="coerce").dropna().astype(float))
        if not all(any(np.isclose(h, x) for x in got) for h in expected_h):
            return False
        observed_keys += 1
    multiplier = 1
    if extra_keys == ("context",):
        multiplier = 4
    elif extra_keys == ("scope",):
        expected_scopes = {"pooled", "Breast", "Colon", "Lung"}
        for _, seed_rows in df.groupby(["configuration", "seed"], dropna=False):
            if set(seed_rows["scope"].astype(str)) != expected_scopes:
                return False
        multiplier = len(expected_scopes)
    return observed_keys == len(roots) * len(expected_seeds) * multiplier


def collect_status(roots, expected_seeds):
    rows = []
    for config, root in roots.items():
        for seed in expected_seeds:
            sdir = root / f"seed{seed}"
            status_files = list(sdir.glob("manifests/seed_status.*")) + list(sdir.glob("seed_status.*"))
            status, error = "missing", "seed directory or completion record missing"
            if status_files:
                try:
                    if status_files[0].suffix == ".json":
                        rec = json.loads(status_files[0].read_text())
                    else:
                        rec = pd.read_csv(status_files[0]).iloc[0].to_dict()
                    status, error = str(rec.get("status", "unknown")), str(rec.get("error", ""))
                except Exception as exc:
                    status, error = "invalid_status", str(exc)
            elif sdir.exists() and list(sdir.glob("**/*_results.csv")):
                d = safe_read(list(sdir.glob("**/*_results.csv"))[0])
                status = "completed" if ("status" in d and d["status"].astype(str).eq("ok").any()) else "failed"
                error = "derived from DGP results"
            rows.append({"configuration": config, "seed": seed, "status": status, "error": error,
                         "phase": "fit_and_primary_outputs", "seed_dir": str(sdir),
                         "collected_utc": datetime.now(timezone.utc).isoformat()})
            _wfc_seen = set()
            for wf in sorted((sdir / "manifests").glob("corrected_wfc_status_H*_all_cancers.csv")) if (sdir / "manifests").exists() else []:
                try:
                    wr = pd.read_csv(wf).iloc[0].to_dict()
                    _wfc_seen.add(float(wr.get("horizon")))
                    rows.append({"configuration": config, "seed": seed, "status": wr.get("status", "unknown"),
                                 "error": "", "phase": "corrected_treatment_conditioned_WFC", "horizon": wr.get("horizon"),
                                 "context": "all_cancers_separate", "seed_dir": str(sdir),
                                 "collected_utc": datetime.now(timezone.utc).isoformat()})
                except Exception as exc:
                    rows.append({"configuration": config, "seed": seed, "status": "invalid_status",
                                 "error": str(exc), "phase": "corrected_treatment_conditioned_WFC", "seed_dir": str(sdir),
                                 "collected_utc": datetime.now(timezone.utc).isoformat()})
            for _h in APPROVED_HORIZONS:
                if not any(np.isclose(_h, observed) for observed in _wfc_seen):
                    rows.append({"configuration": config, "seed": seed, "status": "missing",
                                 "error": "missing corrected WFC partition completion record",
                                 "phase": "corrected_treatment_conditioned_WFC",
                                 "horizon": _h, "context": "all_cancers_separate", "seed_dir": str(sdir),
                                 "collected_utc": datetime.now(timezone.utc).isoformat()})
    return pd.DataFrame(rows)


def collect_performance(roots, expected_count):
    raw = collect_patterns(roots, ["seed*/**/cfg_*_results.csv"], exclude=("sorted", "progress"))
    long = []
    idcols = [c for c in ["configuration", "seed", "status", "source_file"] if c in raw]
    for _, row in raw.iterrows():
        for col in raw.columns:
            lc = col.lower()
            if col in idcols or not any(k in lc for k in ("c_test", "auc", "auprc", "brier", "ibs", "somers", "dxy")):
                continue
            val = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            m = re.search(r"H(\d+)", col)
            scope = re.sub(r"_?(c_test|auroc|auc|auprc|brier|ibs|somers|dxy).*", "", col, flags=re.I).strip("_") or "pooled"
            long.append({**{c: row.get(c) for c in idcols}, "metric": col, "metric_family":
                         ("C_index" if "c_test" in lc else "AUROC" if "auc" in lc and "auprc" not in lc else
                          "AUPRC" if "auprc" in lc else "Brier" if "brier" in lc else "IBS" if "ibs" in lc else "Somers_Dxy"),
                         "scope": scope, "horizon_months": float(m.group(1)) if m else np.nan,
                         "value": val, "estimable": bool(np.isfinite(val)),
                         "nonestimable_reason": "" if np.isfinite(val) else "missing/nonfinite model metric"})
    perf = pd.DataFrame(long)
    august = collect_named(roots, "august_performance_metrics.csv")
    if len(august):
        august["metric_family"] = np.select(
            [august["metric"].astype(str).str.contains("C_index", case=False),
             august["metric"].astype(str).eq("AUROC"), august["metric"].astype(str).eq("AUPRC"),
             august["metric"].astype(str).str.contains("Brier", case=False)],
            ["C_index", "AUROC", "AUPRC", "Brier"], default="other")
        perf = pd.concat([perf, august], ignore_index=True, sort=False)
    cidx = perf[perf.get("metric_family", pd.Series(dtype=str)).eq("C_index")].copy() if not perf.empty else pd.DataFrame()
    csum = summarize_numeric(cidx, "value", ["configuration", "metric", "scope", "horizon_months"], expected_count)
    return cidx, csum, perf


def wfc_summaries(wfc, expected_count):
    if wfc.empty:
        return (pd.DataFrame(),) * 5
    keys = ["configuration", "horizon_months", "context", "feature", "feature_family"]
    if "feature_scope" in wfc:
        keys.append("feature_scope")
    summary = summarize_numeric(wfc, "W1_event_probability", keys, expected_count)
    signed = summarize_numeric(wfc, "signed_delta_mean_event_probability", keys, expected_count)
    if len(summary) and len(signed):
        summary = summary.merge(signed[keys + ["mean_across_splits", "sd_across_splits", "q05_across_splits", "q95_across_splits"]],
                                on=keys, how="left", suffixes=("", "_signed_delta"))
        _sign = wfc.copy()
        _sign["_delta"] = pd.to_numeric(_sign.get("signed_delta_mean_event_probability"), errors="coerce")
        _sign = _sign.dropna(subset=["_delta"]).groupby(keys, dropna=False)["_delta"].agg(
            positive_split_fraction=lambda x: float(np.mean(x > 0)),
            negative_split_fraction=lambda x: float(np.mean(x < 0))).reset_index()
        _sign["sign_consistency_across_splits"] = _sign[["positive_split_fraction", "negative_split_fraction"]].max(axis=1)
        summary = summary.merge(_sign, on=keys, how="left")
    rankable = wfc[pd.to_numeric(wfc.get("W1_event_probability"), errors="coerce").notna()].copy()
    if "eligible_for_context" in rankable:
        rankable = rankable[truthy_series(rankable["eligible_for_context"])].copy()
    if len(rankable):
        rankable["rank_within_seed_horizon_context"] = rankable.groupby(
            ["configuration", "seed", "horizon_months", "context"])["W1_event_probability"].rank(method="average", ascending=False)
        rankable["top10"] = rankable["rank_within_seed_horizon_context"] <= 10
        rankstab = rankable.groupby(keys, dropna=False).agg(
            mean_rank=("rank_within_seed_horizon_context", "mean"), sd_rank=("rank_within_seed_horizon_context", "std"),
            top10_selection_frequency=("top10", "mean"), successful_splits=("seed", "nunique")).reset_index()
        top = rankable.sort_values(["configuration", "context", "horizon_months", "rank_within_seed_horizon_context"]).groupby(
            ["configuration", "context", "horizon_months"], dropna=False).head(20)
        cross = rankable.groupby(["configuration", "context", "feature", "feature_family"], dropna=False).agg(
            mean_within_horizon_rank=("rank_within_seed_horizon_context", "mean"),
            top10_frequency_across_seed_horizons=("top10", "mean"), horizons_supported=("horizon_months", "nunique"),
            splits_supported=("seed", "nunique")).reset_index()
    else:
        rankstab = top = cross = pd.DataFrame()
    return summary, rankstab, top, rankable, cross


def treatment_fc_summaries(txfc, expected_count):
    """Aggregate treatment-conditioned FC scores and ranks across outer splits."""
    if txfc.empty or "FC_score" not in txfc:
        return pd.DataFrame(), pd.DataFrame()
    d = txfc.copy()
    d["FC_score"] = pd.to_numeric(d["FC_score"], errors="coerce")
    if "FC_conditioning_feature_not_ranked" in d:
        d = d[~truthy_series(d["FC_conditioning_feature_not_ranked"])].copy()
    elif "FC_skipped_feature" in d:
        d = d[~truthy_series(d["FC_skipped_feature"])].copy()
    d = d[np.isfinite(d["FC_score"].to_numpy(float))].copy()
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()
    for col, default in [
        ("H", np.nan), ("scope", ""), ("group", ""), ("treatment_profile", ""),
        ("feature", ""), ("feature_family", ""), ("feature_scope", ""),
    ]:
        if col not in d:
            d[col] = default
    keys = [
        "configuration", "H", "scope", "group", "treatment_profile",
        "feature", "feature_family", "feature_scope",
    ]
    summary = summarize_numeric(d, "FC_score", keys, expected_count)
    rank_keys = ["configuration", "seed", "H", "scope", "group", "treatment_profile"]
    d["rank_within_seed_context"] = d.groupby(rank_keys, dropna=False)["FC_score"].rank(
        method="min", ascending=False
    )
    d["top10"] = d["rank_within_seed_context"] <= 10
    rank_stability = (
        d.groupby(keys, dropna=False)
        .agg(
            mean_rank_across_splits=("rank_within_seed_context", "mean"),
            sd_rank_across_splits=("rank_within_seed_context", "std"),
            top10_selection_frequency=("top10", "mean"),
            successful_split_count=("seed", "nunique"),
        )
        .reset_index()
        .sort_values(
            ["top10_selection_frequency", "mean_rank_across_splits"],
            ascending=[False, True],
        )
    )
    return summary, rank_stability


def treatment_contrast_stability(txcontrast):
    """Summarize paired treatment-profile effects across completed outer splits."""
    if txcontrast.empty or "effect_mean" not in txcontrast:
        return pd.DataFrame()
    d = txcontrast.copy()
    d["effect_mean"] = pd.to_numeric(d["effect_mean"], errors="coerce")
    d = d[np.isfinite(d["effect_mean"].to_numpy(float))].copy()
    if d.empty:
        return pd.DataFrame()
    keys = [
        "configuration", "H", "scope", "group", "row_kind", "contrast_type",
        "reference_profile", "treatment_profile", "contrast_label",
        "feature", "feature_family", "feature_scope", "stratum_id", "stratum_label",
    ]
    for col in keys:
        if col not in d:
            d[col] = ""
    rows = []
    for values, sub in d.groupby(keys, dropna=False):
        effects = sub["effect_mean"].to_numpy(float)
        directions = np.sign(effects[np.abs(effects) > 1e-8])
        n_splits = int(sub["seed"].nunique()) if "seed" in sub else int(len(sub))
        pos, neg = int(np.sum(directions > 0)), int(np.sum(directions < 0))
        dominant = 1 if pos >= neg else -1
        dominant_n = max(pos, neg)
        row = dict(zip(keys, values if isinstance(values, tuple) else (values,)))
        row.update({
            "n_splits_with_estimate": n_splits,
            "effect_mean_across_splits": float(np.mean(effects)),
            "effect_sd_across_splits": float(np.std(effects, ddof=1)) if len(effects) > 1 else np.nan,
            "effect_median_across_splits": float(np.median(effects)),
            "effect_q05_across_splits": float(np.quantile(effects, 0.05)),
            "effect_q95_across_splits": float(np.quantile(effects, 0.95)),
            "effect_mean_pp_across_splits": float(100.0 * np.mean(effects)),
            "direction_positive_splits": pos,
            "direction_negative_splits": neg,
            "dominant_direction": dominant,
            "direction_stability": float(dominant_n / n_splits) if n_splits >= 2 else np.nan,
            "repeated_split_stability_estimable": bool(n_splits >= 2),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["direction_stability", "n_splits_with_estimate"],
        ascending=[False, False],
    ).reset_index(drop=True)


def shared_specific(wfc):
    """Summarize explicit shared features and directly comparable private branches."""
    if wfc.empty:
        return pd.DataFrame()
    context_map = {"BRCA": "Breast", "COAD": "Colon", "LUAD": "Lung"}
    cancer_contexts = {"Breast", "Colon", "Lung", *context_map}
    d = wfc[wfc["context"].astype(str).isin(cancer_contexts)].copy()
    d["context"] = d["context"].astype(str).replace(context_map)
    if "eligible_for_context" in d:
        d = d[truthy_series(d["eligible_for_context"])].copy()
    if "feature_scope" not in d:
        d["feature_scope"] = np.where(
            d["feature"].astype(str).str.startswith(("act_shared::", "udp_shared::", "mut_shared::")),
            "shared_across_cancers",
            np.where(
                d["feature"].astype(str).str.startswith(("act_specific_", "udp_specific_", "mut_specific_")),
                "cancer_specific",
                "other",
            ),
        )
    d["signed"] = pd.to_numeric(d.get("signed_delta_mean_event_probability"), errors="coerce")
    d["w1"] = pd.to_numeric(d.get("W1_event_probability"), errors="coerce")
    d["mechanism_key"] = d["feature"].astype(str).str.replace(
        r"^(?:act|udp|mut)_(?:shared|specific_(?:BRCA|COAD|LUAD))::", "", regex=True
    )

    # Common-trait candidates must be explicit shared inputs and must have the
    # same signed perturbation direction in all three cancers within a split.
    shared_d = d[d["feature_scope"].astype(str).eq("shared_across_cancers")].copy()
    shared_rows = []
    for keys, sub in shared_d.groupby(["configuration", "seed", "horizon_months", "feature", "mechanism_key"], dropna=False):
        s = sub.dropna(subset=["signed"]).drop_duplicates("context")
        signs = np.sign(s["signed"].to_numpy(float))
        all3 = set(s["context"]) == {"Breast", "Colon", "Lung"} and (
            np.all(signs > 0) or np.all(signs < 0)
        )
        shared_rows.append(dict(
            zip(["configuration", "seed", "horizon_months", "feature", "mechanism_key"], keys),
            cancers_supported=int(s["context"].nunique()),
            all_three_cancers_same_direction=bool(all3),
            common_resistance_trait_candidate=bool(all3),
            mean_W1_within_split=float(sub["w1"].mean()) if sub["w1"].notna().any() else np.nan,
            direct_cancer_signed_contrast_range=(
                float(s["signed"].max() - s["signed"].min()) if len(s) else np.nan
            ),
        ))
    shared_detail = pd.DataFrame(shared_rows)
    if len(shared_detail):
        shared_out = shared_detail.groupby(
            ["configuration", "horizon_months", "feature", "mechanism_key"], dropna=False
        ).agg(
            splits_supported=("seed", "nunique"),
            mean_W1_across_splits=("mean_W1_within_split", "mean"),
            sd_W1_across_splits=("mean_W1_within_split", "std"),
            all_three_same_direction_split_frequency=("all_three_cancers_same_direction", "mean"),
            common_candidate_split_count=("common_resistance_trait_candidate", "sum"),
            mean_direct_cancer_signed_contrast_range=("direct_cancer_signed_contrast_range", "mean"),
            sd_direct_cancer_signed_contrast_range=("direct_cancer_signed_contrast_range", "std"),
            minimum_cancers_supported=("cancers_supported", "min"),
        ).reset_index()
        shared_out["trait_class"] = "shared_across_cancers"
        shared_out["origin_cancer"] = "all_three"
        shared_out["direct_specificity_estimable_split_count"] = 0
        shared_out["cancer_specific_candidate"] = False
        shared_out["common_resistance_trait_candidate"] = (
            (shared_out["splits_supported"] >= 2)
            & (shared_out["all_three_same_direction_split_frequency"] >= 0.80)
        )
        shared_out["note"] = (
            "Common candidates require explicit shared encoding plus the same signed direction in Breast, Colon, and Lung; W1 alone is unsigned."
        )
    else:
        shared_out = pd.DataFrame()

    # Cancer-private features are evaluated only in their matching cancer.  A
    # direct specificity contrast is estimable only when the same normalized
    # mechanism was selected in both of the other cancer-private branches for
    # the same split and horizon.
    private = d[d["feature_scope"].astype(str).eq("cancer_specific")].copy()
    private["origin_code"] = private["feature"].astype(str).str.extract(
        r"^(?:act|udp|mut)_specific_(BRCA|COAD|LUAD)::", expand=False
    )
    private["origin_cancer"] = private["origin_code"].map(context_map)
    private = private[private["context"].eq(private["origin_cancer"])].copy()
    private["direct_specificity_contrast_W1"] = np.nan
    private["direct_specificity_estimable"] = False
    peer_keys = ["configuration", "seed", "horizon_months", "mechanism_key"]
    for _, idx in private.groupby(peer_keys, dropna=False).groups.items():
        block = private.loc[idx]
        for ridx, row in block.iterrows():
            peers = block[block["context"].ne(row["context"])].dropna(subset=["w1"])
            if set(peers["context"]) == ({"Breast", "Colon", "Lung"} - {row["context"]}) and np.isfinite(row["w1"]):
                private.at[ridx, "direct_specificity_contrast_W1"] = float(row["w1"] - peers["w1"].mean())
                private.at[ridx, "direct_specificity_estimable"] = True
    if len(private):
        private["signed_positive"] = private["signed"] > 0
        private["signed_negative"] = private["signed"] < 0
        private["direct_positive"] = np.where(
            private["direct_specificity_estimable"],
            private["direct_specificity_contrast_W1"] > 0,
            np.nan,
        )
        private_out = private.groupby(
            ["configuration", "horizon_months", "feature", "mechanism_key", "origin_cancer"],
            dropna=False,
        ).agg(
            splits_supported=("seed", "nunique"),
            mean_W1_across_splits=("w1", "mean"),
            sd_W1_across_splits=("w1", "std"),
            mean_signed_change_across_splits=("signed", "mean"),
            sd_signed_change_across_splits=("signed", "std"),
            signed_positive_split_fraction=("signed_positive", "mean"),
            signed_negative_split_fraction=("signed_negative", "mean"),
            direct_specificity_estimable_split_count=("direct_specificity_estimable", "sum"),
            mean_direct_specificity_contrast_W1=("direct_specificity_contrast_W1", "mean"),
            sd_direct_specificity_contrast_W1=("direct_specificity_contrast_W1", "std"),
            direct_specificity_positive_split_frequency=("direct_positive", "mean"),
        ).reset_index()
        private_out["trait_class"] = "cancer_specific_branch"
        private_out["signed_direction_consistency"] = private_out[
            ["signed_positive_split_fraction", "signed_negative_split_fraction"]
        ].max(axis=1)
        private_out["common_candidate_split_count"] = 0
        private_out["common_resistance_trait_candidate"] = False
        private_out["cancer_specific_candidate"] = (
            (private_out["direct_specificity_estimable_split_count"] >= 2)
            & (private_out["direct_specificity_positive_split_frequency"] >= 0.80)
        )
        private_out["note"] = np.where(
            private_out["direct_specificity_estimable_split_count"] >= 2,
            "Direct specificity compares W1 with the homologous private mechanism in both other cancers on the same splits; signed direction is reported separately.",
            "Private-branch origin is descriptive only because homologous branches in both other cancers were not available on at least two splits.",
        )
    else:
        private_out = pd.DataFrame()
    return pd.concat([shared_out, private_out], ignore_index=True, sort=False)


def manifests(roots):
    figs, data = [], []
    fig_ext = {".png", ".pdf", ".svg"}
    for config, root in roots.items():
        for p in root.glob("seed*/**/*"):
            if not p.is_file():
                continue
            m = re.search(r"seed(\d+)", str(p)); seed = int(m.group(1)) if m else np.nan
            h = re.search(r"H(\d+)", p.name); horizon = float(h.group(1)) if h else np.nan
            base = {"configuration": config, "seed": seed, "horizon": horizon,
                    "analysis": p.stem, "timestamp": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()}
            if p.suffix.lower() in fig_ext:
                candidates = [q for q in p.parent.glob("*.csv*") if q.stem.lower() in p.stem.lower() or p.stem.lower() in q.stem.lower()]
                figs.append({**base, "source_csv_path": str(candidates[0]) if candidates else "",
                             "figure_path": str(p), "row_count": np.nan,
                             "status": "success" if candidates else "figure_present_source_match_not_automatic", "error": ""})
            elif ".csv" in p.name:
                try: n = sum(1 for _ in p.open("rt", errors="ignore")) - 1 if not p.name.endswith(".gz") else np.nan
                except Exception: n = np.nan
                data.append({**base, "source_csv_path": str(p), "figure_path": "", "row_count": n,
                             "status": "success", "error": ""})
    return pd.DataFrame(figs), pd.DataFrame(data)


def configuration_comparisons(perf):
    if perf.empty:
        return pd.DataFrame()
    keys = ["seed", "metric", "metric_family", "scope", "horizon_months"]
    work = perf.copy()
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    # Enforce one estimand per configuration/split/metric/scope/horizon before
    # pairing.  This prevents duplicated plot-source or compatibility rows from
    # creating Cartesian products in the configuration comparison table.
    work = work.groupby(["configuration", *keys], dropna=False).agg(
        value=("value", "mean"), source_row_count=("value", "size")
    ).reset_index()
    rows = []
    configs = sorted(work["configuration"].dropna().unique())
    for i, a in enumerate(configs):
        for b in configs[i + 1:]:
            da = work[work.configuration.eq(a)][keys + ["value", "source_row_count"]].rename(
                columns={"value": "value_A", "source_row_count": "source_row_count_A"}
            )
            db = work[work.configuration.eq(b)][keys + ["value", "source_row_count"]].rename(
                columns={"value": "value_B", "source_row_count": "source_row_count_B"}
            )
            z = da.merge(db, on=keys, how="outer")
            z["configuration_A"], z["configuration_B"] = a, b
            z["paired_difference_A_minus_B"] = z["value_A"] - z["value_B"]
            z["estimable"] = z[["value_A", "value_B"]].notna().all(axis=1)
            rows.append(z)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def collect_all(roots, outdirs, expected_seeds):
    expected_count = len(expected_seeds)
    status = collect_status(roots, expected_seeds)
    cidx, csum, perf = collect_performance(roots, expected_count)
    cidx_explicit = collect_named(roots, "c_index_by_horizon_and_group.csv")
    cidx_explicit_summary = summarize_numeric(
        cidx_explicit,
        "value",
        ["configuration", "metric", "scope", "horizon_months"],
        expected_count,
    ) if len(cidx_explicit) else pd.DataFrame()
    posterior = collect_named(roots, "patient_posterior_risk_by_seed_horizon.csv")
    if len(posterior) and "event_probability_mean" in posterior:
        _pkeys = [c for c in ["configuration", "patient_id", "sample_id", "cancer", "horizon_months"] if c in posterior]
        posterior_across = posterior.groupby(_pkeys, dropna=False).agg(
            event_probability_mean_across_splits=("event_probability_mean", "mean"),
            event_probability_sd_across_splits=("event_probability_mean", "std"),
            mean_within_split_posterior_sd=("event_probability_sd", "mean"),
            rms_within_split_posterior_sd=(
                "event_probability_sd",
                lambda x: float(np.sqrt(np.nanmean(np.square(pd.to_numeric(x, errors="coerce"))))),
            ),
            mean_posterior_median_across_splits=("event_probability_median", "mean"),
            mean_posterior_q05_across_splits=("event_probability_q05", "mean"),
            mean_posterior_q95_across_splits=("event_probability_q95", "mean"),
            median_across_splits=("event_probability_mean", "median"),
            q05_across_splits=("event_probability_mean", lambda x: x.quantile(.05)),
            q95_across_splits=("event_probability_mean", lambda x: x.quantile(.95)),
            heldout_split_count=("seed", "nunique")).reset_index()
        # Law of total variance: posterior uncertainty within each fit plus
        # between-split uncertainty in the posterior mean.
        posterior_across["total_event_probability_sd_across_posterior_and_splits"] = np.sqrt(
            np.square(posterior_across["rms_within_split_posterior_sd"])
            + np.square(posterior_across["event_probability_sd_across_splits"].fillna(0.0))
        )
    else:
        posterior_across = pd.DataFrame()
    corrected_wfc_tables, corrected_wfc_ok = collect_corrected_wfc(
        roots, expected_seeds, bootstrap_samples=1000
    )
    km_assign = collect_named(roots, "km_patient_assignments_train_defined_tertiles.csv")
    km_cut = collect_named(roots, "km_training_cutpoints.csv")
    km_support = collect_named(roots, "km_heldout_group_support.csv")
    km_log = collect_named(roots, "km_logrank_results.csv")
    km_pairwise = collect_named(roots, "km_pairwise_logrank_results.csv")
    km = km_support.merge(km_log, on=[c for c in ["configuration", "seed", "horizon_months", "scope", "cancer"] if c in km_support and c in km_log], how="left") if len(km_support) else km_log
    chol = collect_patterns(roots, ["seed*/**/cholesky_retry_audit_seed*.csv"])
    attrition = collect_patterns(roots, ["preprocessing/**/cohort_attrition_table.csv"])
    txaudit = collect_patterns(roots, ["preprocessing/**/treatment_timing_audit.csv"])
    calibration = collect_patterns(roots, ["seed*/**/*calibration*.csv"], exclude=("manifest", "failure"))
    selected = collect_patterns(roots, ["preprocessing/**/*selected*.csv"])
    if len(selected):
        _fcol = "feat" if "feat" in selected else ("feature" if "feature" in selected else None)
        if _fcol:
            selected["feature"] = selected[_fcol].astype(str)
            selected["selection_family_file"] = selected["source_file"].map(lambda x: Path(str(x)).name)
            selected = selected.groupby(["configuration", "selection_family_file", "feature"], dropna=False).agg(
                selected_split_count=("seed", "nunique"),
                selection_frequency=("seed", lambda x: x.nunique() / max(expected_count, 1))).reset_index()
    txsupport = collect_patterns(roots, ["seed*/**/*treatment*support*.csv", "seed*/**/*treatment*count*.csv"])
    txcontrast = collect_patterns(
        roots,
        ["seed*/**/cfg_*_pairwise_feature_tx_contrasts_seed*.csv"],
        exclude=("WITH_DGP_BRANCH_CALLS", "skipped"),
    )
    txcontrast_stability = treatment_contrast_stability(txcontrast)
    required_contract = collect_patterns(
        roots,
        ["seed*/**/required_npj_output_contract_seed*.csv"],
    )
    if len(txcontrast) and "row_kind" in txcontrast:
        txinteractions = txcontrast[
            txcontrast["row_kind"].astype(str).eq("delta_delta_feature_treatment_interaction")
        ].copy()
    else:
        txinteractions = pd.DataFrame()
    figman, dataman = manifests(roots)
    figfail = collect_patterns(roots, ["seed*/**/*plot*failure*.csv", "seed*/**/*exception*.csv"])
    _txt_fail = []
    for _config, _root in roots.items():
        for _p in list(_root.glob("seed*/**/*exception*.txt")) + list(_root.glob("seed*/**/*failure*.txt")):
            _m = re.search(r"seed(\d+)", str(_p))
            _txt_fail.append({"configuration": _config, "seed": int(_m.group(1)) if _m else np.nan,
                              "source_file": str(_p), "status": "failed",
                              "error": _p.read_text(errors="ignore")[:4000]})
    if _txt_fail:
        figfail = pd.concat([figfail, pd.DataFrame(_txt_fail)], ignore_index=True, sort=False)
    if len(km_cut) and "estimable" in km_cut:
        _km_ne = km_cut[~km_cut["estimable"].astype(bool)].copy()
        if len(_km_ne):
            _km_ne["analysis"] = "KM_training_defined_tertiles"
            _km_ne["status"] = "nonestimable_no_figure"
            _km_ne["error"] = _km_ne.get("nonestimable_reason", "")
            figfail = pd.concat([figfail, _km_ne], ignore_index=True, sort=False)
    shared = corrected_wfc_tables["master_corrected_wfc_final_cancer_unique_patient.csv"]
    comparison_input = perf.copy()
    if len(calibration):
        _cal_rows = []
        for _, _r in calibration.iterrows():
            for _col in calibration.columns:
                if not any(k in _col.lower() for k in ("slope", "intercept")):
                    continue
                _v = pd.to_numeric(pd.Series([_r.get(_col)]), errors="coerce").iloc[0]
                _cal_rows.append({"configuration": _r.get("configuration"), "seed": _r.get("seed"),
                                  "metric": _col, "metric_family": "calibration",
                                  "scope": _r.get("scope", _r.get("group", "pooled")),
                                  "horizon_months": _r.get("horizon_months", _r.get("H", np.nan)),
                                  "value": _v})
        if _cal_rows:
            comparison_input = pd.concat([comparison_input, pd.DataFrame(_cal_rows)], ignore_index=True, sort=False)
    comparisons = configuration_comparisons(comparison_input)
    tables = {
        "master_run_status.csv": status, "master_cohort_attrition.csv": attrition,
        "master_treatment_timing_audit.csv": txaudit, "master_c_indices_by_seed.csv": cidx,
        "master_c_indices_summary.csv": csum, "master_configuration_comparisons.csv": comparisons,
        "master_c_indices_explicit_by_seed_horizon_group.csv": cidx_explicit,
        "master_c_indices_explicit_summary.csv": cidx_explicit_summary,
        "master_performance_metrics.csv": perf, "master_calibration_metrics.csv": calibration,
        "master_feature_selection_stability.csv": selected, "master_shared_vs_specific_features.csv": shared,
        "master_treatment_profile_support.csv": txsupport, "master_treatment_contrasts.csv": txcontrast,
        "master_feature_treatment_interactions.csv": txinteractions,
        "master_treatment_contrast_stability.csv": txcontrast_stability,
        "master_km_tertile_summary.csv": km, "master_cholesky_retry_audit.csv": chol,
        "master_km_pairwise_logrank_results.csv": km_pairwise,
        "master_figure_manifest.csv": figman, "master_plot_data_manifest.csv": dataman,
        "master_plot_generation_failures.csv": figfail,
        "patient_posterior_risk_across_splits.csv": posterior_across,
        "patient_posterior_risk_by_seed_horizon_master.csv": posterior,
        "master_km_patient_assignments.csv": km_assign, "master_km_training_cutpoints.csv": km_cut,
        "master_required_output_contract.csv": required_contract,
        **corrected_wfc_tables,
    }
    expected_contract_seeds = int(len(roots) * expected_count)
    contract_ok = (
        len(required_contract) > 0
        and "status" in required_contract
        and required_contract["status"].astype(str).eq("ok").all()
        and required_contract[["configuration", "seed"]].drop_duplicates().shape[0]
        == expected_contract_seeds
    )
    wfc_contract_ok = bool(corrected_wfc_ok)
    run_status_ok = (
        len(status) > 0
        and status["status"].astype(str).str.contains(
            "complete|success", case=False
        ).all()
    )
    master_analysis_ok = all([
        complete_seed_horizon_coverage(
            cidx_explicit, roots, expected_seeds, "horizon_months",
            [3, 6, 12, 18, 24, 36, 60], extra_keys=("scope",),
        ),
        complete_seed_horizon_coverage(
            posterior, roots, expected_seeds, "horizon_months", [3, 6, 12, 18, 24, 36, 60]
        ),
        complete_seed_horizon_coverage(
            txcontrast, roots, expected_seeds, "H", [3, 6, 12, 18, 24, 36]
        ),
        len(csum) > 0,
        len(calibration) > 0,
        len(km) > 0,
        len(txcontrast_stability) > 0,
        len(corrected_wfc_tables["master_corrected_wfc_final_cancer_unique_patient.csv"]) > 0,
        len(corrected_wfc_tables["master_corrected_wfc_final_shared_across_cancers.csv"]) > 0,
        corrected_wfc_ok,
    ])
    for outdir in outdirs:
        outdir.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            write_table(table, outdir / name)
        summary = {"collected_utc": datetime.now(timezone.utc).isoformat(),
                   "configurations": list(roots), "expected_seeds": expected_seeds,
                   "completed": int(status.status.astype(str).str.contains("complete|success", case=False).sum()),
                   "failed_or_missing": int((~status.status.astype(str).str.contains("complete|success", case=False)).sum()),
                   "required_output_contract_complete": bool(contract_ok),
                   "wfc_output_contract_complete": bool(wfc_contract_ok),
                   "run_and_wfc_status_complete": bool(run_status_ok),
                   "master_analysis_coverage_complete": bool(master_analysis_ok),
                   "collection_complete": bool(contract_ok and wfc_contract_ok and run_status_ok and master_analysis_ok),
                   "master_files": sorted(tables)}
        atomic_write_text(outdir / "collector_summary.json", json.dumps(summary, indent=2))
        failed = status[~status.status.astype(str).str.contains("complete|success", case=False)]
        commands = [f"bash run_{r.configuration}_NPJ_August.sbatch resubmit-failed" for r in failed.itertuples()]
        atomic_write_text(
            outdir / "resubmission_instructions.txt",
            "\n".join(sorted(set(commands))) + ("\n" if commands else "No failed seeds detected.\n"),
        )
    if not (contract_ok and wfc_contract_ok and run_status_ok and master_analysis_ok):
        raise RuntimeError(
            "Collection is incomplete: one or more fit/WFC jobs failed or a required NPJ/WFC "
            "output contract/master-analysis coverage check is missing or failed. Inspect master_run_status.csv, "
            "master_required_output_contract.csv, and master_corrected_wfc_required_output_contract.csv."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["all", *CONFIG_ROOTS], default="all")
    ap.add_argument("--base", type=Path)
    ap.add_argument("--shared-outdir", type=Path, default=SHARED_DEFAULT)
    ap.add_argument("--expected-seeds", default="42,43,44,45,46,47,48,49,50,51")
    ap.add_argument(
        "--write-shared-single-config",
        action="store_true",
        help="Also write a single-configuration collection into the shared root. Off by default to prevent cross-configuration overwrite.",
    )
    args = ap.parse_args()
    shared = validate_shared_root(args.shared_outdir)
    if args.config == "all":
        missing = [str(v) for v in CONFIG_ROOTS.values() if not v.exists()]
        if missing:
            raise FileNotFoundError(
                "Final --config all collection requires all three approved August roots. Missing: "
                + "; ".join(missing)
            )
        roots = {k: validate_config_root(v, k) for k, v in CONFIG_ROOTS.items()}
    else:
        roots = {args.config: validate_config_root(args.base or CONFIG_ROOTS[args.config], args.config)}
    if not roots:
        raise FileNotFoundError("No approved August configuration output roots exist")
    if re.fullmatch(r"\d+\s*-\s*\d+", args.expected_seeds.strip()):
        _a, _b = [int(x) for x in re.split(r"\s*-\s*", args.expected_seeds.strip())]
        seeds = list(range(_a, _b + 1))
    else:
        seeds = [int(x) for x in args.expected_seeds.split(",") if x.strip()]
    if args.config == "all":
        outdirs = [shared]
    else:
        outdirs = [next(iter(roots.values())) / "collected"]
        if args.write_shared_single_config:
            outdirs.insert(0, shared)
    collect_all(roots, outdirs, seeds)
    print(f"Collected {list(roots)} into {', '.join(map(str, outdirs))}")


if __name__ == "__main__":
    main()
