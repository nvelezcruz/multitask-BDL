#!/usr/bin/env python3
"""Finalize the fixed-contrast PathNoUDP/PathClin Wasserstein analyses.

The inference runner writes one self-contained partition for each outer split::

    <partition-root>/<configuration>/seed<seed>/

This module deliberately summarizes *seed-level* estimates.  It never pools
patients from different test splits to estimate a feature effect.  Patient-level
files are concatenated only to make the requested audit exports.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


CONFIGS = ("PathNoUDP", "PathClin")
DEFAULT_SEEDS = tuple(range(42, 52))
HORIZONS = (18, 24, 36)
CANCERS = ("BRCA", "COAD", "LUAD")
BRANCH_TIE_TOLERANCE = 1e-12
INTERACTION_TIE_TOLERANCE = 1e-12
PREDICTION_SCALE = (
    "cancer_specific_fixed_slope_intercept_recalibrated_PFI_event_probability"
)
CALIBRATION_KIND = "cancer_specific_intercept_fixed_slope_1"

PARTITION_FILES = (
    "patient_feature.csv.gz",
    "patient_branch.csv.gz",
    "patient_interactions.csv.gz",
    "seed_feature_summary.csv",
    "seed_branch_summary.csv",
    "seed_profile_summary.csv",
    "seed_interaction_summary.csv",
    "feature_identity_mapping.csv",
    "quantile_audit.csv",
    "calibration_qc.csv",
    "treatment_support.csv",
    "qc.json",
)

FINAL_NAMES = {
    "patient_feature": "{config}_fixed_contrast_patient_level_feature_importance.csv",
    "feature_summary": "{config}_fixed_contrast_across_seed_feature_summary.csv",
    "branch_summary": "{config}_fixed_contrast_branch_attribution.csv",
    "cross_cancer": "{config}_cross_cancer_fixed_contrast_summary.csv",
    "profile_summary": "{config}_treatment_profile_fixed_contrast_feature_summary.csv",
    "patient_interactions": "{config}_feature_treatment_interactions.csv",
    "interaction_summary": "{config}_feature_treatment_interactions_across_seed.csv",
}


FEATURE_ALIASES = {
    "analysis_config": ("config", "configuration", "model_configuration"),
    "seed": ("split_seed", "outer_seed"),
    "cancer": ("group", "cancer_group", "cohort"),
    "horizon": ("H", "horizon_months", "month"),
    "feature": ("canonical_feature_name", "feature_name", "canonical_feature"),
    "feature_type": ("molecular_feature_type", "type"),
    "mean_W_feature": (
        "mean_w_feature", "mean_wasserstein", "mean_overall_wasserstein",
        "W_feature_mean",
    ),
    "sd_W_feature": (
        "sd_w_feature", "sd_wasserstein", "sd_overall_wasserstein",
        "W_feature_sd",
    ),
    "median_W_feature": (
        "median_w_feature", "median_wasserstein", "median_overall_wasserstein",
        "W_feature_median",
    ),
    "mean_Delta_feature": (
        "mean_delta_feature", "mean_signed_change", "mean_delta",
        "Delta_feature_mean",
    ),
    "sd_Delta_feature": (
        "sd_delta_feature", "sd_signed_change", "sd_delta",
        "Delta_feature_sd",
    ),
}

BRANCH_ALIASES = {
    **FEATURE_ALIASES,
    "mean_W_shared": ("mean_w_shared", "W_shared_mean", "mean_shared_wasserstein"),
    "sd_W_shared": ("sd_w_shared", "W_shared_sd", "sd_shared_wasserstein"),
    "mean_W_private": ("mean_w_private", "W_private_mean", "mean_private_wasserstein"),
    "sd_W_private": ("sd_w_private", "W_private_sd", "sd_private_wasserstein"),
    "mean_branch_delta": ("branch_delta", "mean_shared_minus_private"),
    "sd_branch_delta": ("branch_delta_sd", "sd_shared_minus_private"),
    "mean_shared_signed_change": (
        "shared_signed_change", "mean_Delta_shared", "mean_delta_shared",
    ),
    "mean_private_signed_change": (
        "private_signed_change", "mean_Delta_private", "mean_delta_private",
    ),
}

PROFILE_ALIASES = {
    **FEATURE_ALIASES,
    "treatment_profile": ("profile", "treatment_profile_label"),
    "treatment_profile_code": ("profile_code",),
    "treatment_profile_vector": ("profile_vector",),
}

INTERACTION_ALIASES = {
    **FEATURE_ALIASES,
    "treatment_A": ("treatment_a", "profile_A", "profile_a"),
    "treatment_B": ("treatment_b", "profile_B", "profile_b"),
    "treatment_A_code": ("treatment_a_code", "profile_A_code", "profile_a_code"),
    "treatment_B_code": ("treatment_b_code", "profile_B_code", "profile_b_code"),
    "treatment_A_vector": ("treatment_a_vector", "profile_A_vector"),
    "treatment_B_vector": ("treatment_b_vector", "profile_B_vector"),
    "mean_psi": ("psi_mean", "posterior_mean_psi"),
    "median_psi": ("psi_median", "posterior_median_psi"),
    "sd_patient_psi": ("sd_psi", "psi_sd", "posterior_sd_psi"),
    "mean_abs_psi": ("abs_mean_psi", "mean_absolute_psi", "abs_psi_mean"),
    "mean_Pr_psi_gt_0": ("Pr_psi_gt_0", "mean_pr_psi_gt_0", "prob_psi_positive"),
    "mean_Pr_psi_lt_0": ("Pr_psi_lt_0", "mean_pr_psi_lt_0", "prob_psi_negative"),
}


def _token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _canonicalize_columns(df: pd.DataFrame, aliases: dict[str, Sequence[str]]) -> pd.DataFrame:
    """Rename known schema variants without silently clobbering a real column."""
    out = df.copy()
    token_to_columns: dict[str, list[str]] = {}
    for col in out.columns:
        token_to_columns.setdefault(_token(col), []).append(col)
    for target, variants in aliases.items():
        if target in out.columns:
            continue
        candidates: list[str] = []
        for variant in (target, *variants):
            candidates.extend(token_to_columns.get(_token(variant), []))
        candidates = list(dict.fromkeys(candidates))
        if candidates:
            out = out.rename(columns={candidates[0]: target})
    return out


def _canonical_feature(value: object) -> str:
    text = str(value).strip()
    if "::" in text:
        text = text.split("::", 1)[1]
    return text


def _canonical_cancer(value: object) -> str:
    text = str(value).strip().upper()
    return {
        "BREAST": "BRCA", "BRCA": "BRCA",
        "COLON": "COAD", "COAD": "COAD",
        "LUNG": "LUAD", "LUAD": "LUAD",
        "ALL": "ALL", "POOLED": "ALL",
    }.get(text, text)


def _canonical_feature_type(value: object, feature: object = "") -> str:
    text = str(value).strip().lower()
    if text in {"mut", "mutation", "mutations"}:
        return "mutation"
    if text in {"path", "pathway", "pathways", "activity", "consistency"}:
        return "pathway"
    if text in {"tmb", "tumor_mutational_burden", "tumour_mutational_burden"}:
        return "tmb"
    if _token(feature) in {"tmb", "tumormutationalburden", "tumourmutationalburden"}:
        return "tmb"
    return text


def _bool_series(series: pd.Series, default: bool = False) -> pd.Series:
    if series is None:
        return pd.Series(dtype=bool)
    lowered = series.astype(str).str.strip().str.lower()
    result = lowered.isin({"true", "1", "yes", "y", "pass", "passed", "supported"})
    missing = series.isna() | lowered.isin({"", "nan", "none"})
    if default:
        result.loc[missing] = True
    return result


def _numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _normalize_common(df: pd.DataFrame, config: str, seed: int | None = None) -> pd.DataFrame:
    out = df.copy()
    if "analysis_config" not in out:
        out["analysis_config"] = config
    else:
        out["analysis_config"] = out["analysis_config"].fillna(config).astype(str)
    if "seed" not in out and seed is not None:
        out["seed"] = seed
    if "seed" in out:
        out["seed"] = pd.to_numeric(out["seed"], errors="coerce").astype("Int64")
    if "cancer" in out:
        out["cancer"] = out["cancer"].map(_canonical_cancer)
    if "feature" in out:
        out["feature"] = out["feature"].map(_canonical_feature)
    if "feature_type" in out:
        out["feature_type"] = [
            _canonical_feature_type(t, f)
            for t, f in zip(out["feature_type"], out.get("feature", pd.Series("", index=out.index)))
        ]
    if "horizon" in out:
        out["horizon"] = pd.to_numeric(out["horizon"], errors="coerce").astype("Int64")
    if "prediction_scale" not in out:
        out["prediction_scale"] = PREDICTION_SCALE
    if "calibration_kind" not in out:
        out["calibration_kind"] = CALIBRATION_KIND
    return out


def _read_partition_csv(path: Path, aliases: dict[str, Sequence[str]], config: str, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = _canonicalize_columns(df, aliases)
    return _normalize_common(df, config, seed)


def _join_unique(values: Iterable[object]) -> str:
    items: set[str] = set()
    for value in values:
        if pd.isna(value):
            continue
        for item in str(value).split(";"):
            item = item.strip()
            if item and item.lower() not in {"nan", "none"}:
                items.add(item)
    return ";".join(sorted(items))


def _first_nonmissing(values: pd.Series) -> object:
    valid = values.dropna()
    return valid.iloc[0] if not valid.empty else np.nan


def _scope_names(config: str, feature_type: str) -> tuple[str, ...]:
    if config == "PathNoUDP":
        scopes: list[str] = []
        if feature_type in {"pathway", "mutation", "tmb"}:
            scopes.append("molecular_all")
        if feature_type == "mutation":
            scopes.append("mutation_only")
        elif feature_type == "pathway":
            scopes.append("pathway_only")
        return tuple(scopes)
    return ("pathway_only",) if feature_type == "pathway" else tuple()


def _strip_scope_duplicates(df: pd.DataFrame, id_columns: Sequence[str]) -> pd.DataFrame:
    """Remove identical rows repeated only because the runner pre-expanded scopes."""
    out = df.drop(columns=["ranking_scope"], errors="ignore").copy()
    duplicated = out.duplicated(list(id_columns), keep=False)
    if not duplicated.any():
        return out
    rows: list[pd.Series] = []
    for _, group in out.groupby(list(id_columns), dropna=False, sort=False):
        if len(group) == 1:
            rows.append(group.iloc[0])
            continue
        comparison = group.drop(columns=list(id_columns), errors="ignore")
        conflict = False
        for col in comparison:
            vals = comparison[col].dropna()
            if vals.empty:
                continue
            if pd.api.types.is_numeric_dtype(vals):
                if not np.allclose(vals.astype(float), float(vals.iloc[0]), rtol=1e-9, atol=1e-12):
                    conflict = True
                    break
            elif vals.astype(str).nunique() > 1:
                conflict = True
                break
        if conflict:
            key = {c: group.iloc[0][c] for c in id_columns}
            raise ValueError(f"Conflicting duplicate canonical feature rows: {key}")
        rows.append(group.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def _expand_scopes(df: pd.DataFrame, id_columns: Sequence[str]) -> pd.DataFrame:
    base = _strip_scope_duplicates(df, id_columns)
    pieces: list[pd.DataFrame] = []
    for (config, feature_type), group in base.groupby(
        ["analysis_config", "feature_type"], dropna=False, sort=False
    ):
        for scope in _scope_names(str(config), str(feature_type)):
            part = group.copy()
            part["ranking_scope"] = scope
            pieces.append(part)
    if not pieces:
        return base.iloc[0:0].assign(ranking_scope=pd.Series(dtype=str))
    return pd.concat(pieces, ignore_index=True)


def _importance_percentile(rank: pd.Series, n: pd.Series) -> pd.Series:
    rank_f = pd.to_numeric(rank, errors="coerce").astype(float)
    n_f = pd.to_numeric(n, errors="coerce").astype(float)
    return pd.Series(
        np.where(n_f <= 1, 1.0, 1.0 - (rank_f - 1.0) / (n_f - 1.0)),
        index=rank.index,
    )


def _rank(
    df: pd.DataFrame,
    group_columns: Sequence[str],
    value_column: str,
    rank_column: str,
    percentile_column: str,
) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby(list(group_columns), dropna=False, sort=False)[value_column]
    out[rank_column] = grouped.rank(method="min", ascending=False)
    n = grouped.transform("count")
    out[percentile_column] = _importance_percentile(out[rank_column], n)
    return out


def _require_columns(df: pd.DataFrame, required: Sequence[str], context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def _load_seed_tables(
    partition_root: Path,
    config: str,
    seeds: Sequence[int],
    filename: str,
    aliases: dict[str, Sequence[str]],
) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        path = partition_root / config / f"seed{seed}" / filename
        frames.append(_read_partition_csv(path, aliases, config, seed))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _mapping_summary(mapping: pd.DataFrame, n_expected_seeds: int) -> pd.DataFrame:
    required = ["analysis_config", "seed", "cancer", "feature", "feature_type"]
    _require_columns(mapping, required, "feature_identity_mapping.csv")
    data = mapping.copy()
    if "selected" not in data:
        data["selected"] = True
    else:
        data["selected"] = _bool_series(data["selected"], default=True)
    if "evaluable" not in data:
        data["evaluable"] = data["selected"]
    else:
        data["evaluable"] = _bool_series(data["evaluable"], default=False)
    if "inclusion_type" not in data:
        data["inclusion_type"] = np.where(
            data["feature_type"].eq("tmb"), "design_included", "screened"
        )
    data["inclusion_type"] = data["inclusion_type"].astype(str).str.strip().str.lower()
    data.loc[data["feature_type"].eq("tmb"), "inclusion_type"] = "design_included"
    for col in (
        "source_feature_names", "source_columns", "selection_origins",
        "specific_cancers", "model_input_scale", "non_evaluable_reason",
    ):
        if col not in data:
            data[col] = ""

    keys = ["analysis_config", "cancer", "feature", "feature_type"]
    rows: list[dict[str, object]] = []
    for values, group in data.groupby(keys, dropna=False, sort=False):
        inclusion_values = set(group["inclusion_type"].dropna().astype(str))
        inclusion_type = (
            "design_included" if "design_included" in inclusion_values
            else (sorted(inclusion_values)[0] if len(inclusion_values) == 1 else "screened")
        )
        selected_seeds = group.loc[group["selected"], "seed"].dropna().astype(int).unique()
        eval_seeds = group.loc[group["evaluable"], "seed"].dropna().astype(int).unique()
        row = dict(zip(keys, values))
        row.update({
            "inclusion_type": inclusion_type,
            "n_seeds_selected": int(len(selected_seeds)),
            "n_seeds_evaluable_from_mapping": int(len(eval_seeds)),
            "selection_frequency": (
                np.nan if inclusion_type == "design_included"
                else len(selected_seeds) / n_expected_seeds
            ),
            "design_inclusion_frequency": (
                len(selected_seeds) / n_expected_seeds
                if inclusion_type == "design_included" else np.nan
            ),
            "source_feature_names": _join_unique(group["source_feature_names"]),
            "source_columns": _join_unique(group["source_columns"]),
            "selection_origins": _join_unique(group["selection_origins"]),
            "specific_cancers": _join_unique(group["specific_cancers"]),
            "model_input_scales": _join_unique(group["model_input_scale"]),
            "non_evaluable_reasons": _join_unique(group["non_evaluable_reason"]),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _attach_mapping(summary: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    keys = ["analysis_config", "cancer", "feature", "feature_type"]
    keep = [c for c in mapping.columns if c not in summary.columns or c in keys]
    return summary.merge(mapping[keep], on=keys, how="left", validate="many_to_one")


def _prepare_seed_features(df: pd.DataFrame) -> pd.DataFrame:
    data = _canonicalize_columns(df, FEATURE_ALIASES)
    data = data.drop(
        columns=["within_seed_rank", "within_seed_importance_percentile"],
        errors="ignore",
    )
    data = _numeric(data, (
        "mean_W_feature", "sd_W_feature", "median_W_feature",
        "mean_Delta_feature", "sd_Delta_feature", "n_test_patients",
    ))
    required = [
        "analysis_config", "seed", "cancer", "horizon", "feature", "feature_type",
        "mean_W_feature", "mean_Delta_feature",
    ]
    _require_columns(data, required, "seed_feature_summary.csv")
    ids = ["analysis_config", "seed", "cancer", "horizon", "feature", "feature_type"]
    data = _expand_scopes(data, ids)
    group = ["analysis_config", "seed", "cancer", "horizon", "ranking_scope"]
    data = _rank(
        data, group, "mean_W_feature", "within_seed_rank",
        "within_seed_importance_percentile",
    )
    return data


def _summarize_features(
    seed_features: pd.DataFrame,
    mapping: pd.DataFrame,
    n_expected_seeds: int = len(DEFAULT_SEEDS),
) -> pd.DataFrame:
    keys = [
        "analysis_config", "cancer", "horizon", "feature", "feature_type",
        "ranking_scope",
    ]
    aggregations: dict[str, tuple[str, object]] = {
        "n_seeds_evaluable": ("seed", "nunique"),
        "mean_W_feature": ("mean_W_feature", "mean"),
        "sd_W_feature": ("mean_W_feature", "std"),
        "median_W_feature": ("mean_W_feature", "median"),
        "mean_Delta_feature": ("mean_Delta_feature", "mean"),
        "sd_Delta_feature": ("mean_Delta_feature", "std"),
        "mean_rank": ("within_seed_rank", "mean"),
        "sd_rank": ("within_seed_rank", "std"),
        "mean_importance_percentile": ("within_seed_importance_percentile", "mean"),
        "sd_importance_percentile": ("within_seed_importance_percentile", "std"),
    }
    for source, target in (
        ("sd_W_feature", "mean_within_seed_sd_W_feature"),
        ("median_W_feature", "mean_within_seed_median_W_feature"),
        ("sd_Delta_feature", "mean_within_seed_sd_Delta_feature"),
        ("n_test_patients", "mean_n_test_patients"),
    ):
        if source in seed_features:
            aggregations[target] = (source, "mean")
    out = seed_features.groupby(keys, as_index=False, dropna=False).agg(**aggregations)
    out = _attach_mapping(out, mapping)
    out["evaluable_frequency"] = out["n_seeds_evaluable"] / n_expected_seeds
    out["prediction_scale"] = PREDICTION_SCALE
    out["calibration_kind"] = CALIBRATION_KIND
    out = out.sort_values(
        ["analysis_config", "cancer", "horizon", "ranking_scope", "mean_W_feature", "feature"],
        ascending=[True, True, True, True, False, True],
    ).reset_index(drop=True)
    out["across_seed_rank_by_mean_W_feature"] = out.groupby(
        ["analysis_config", "cancer", "horizon", "ranking_scope"],
        dropna=False, sort=False,
    )["mean_W_feature"].rank(method="min", ascending=False).astype("Int64")
    return out


def _prepare_seed_branches(df: pd.DataFrame) -> pd.DataFrame:
    data = _canonicalize_columns(df, BRANCH_ALIASES)
    data = _numeric(data, (
        "mean_W_feature", "mean_W_shared", "sd_W_shared", "mean_W_private",
        "sd_W_private", "mean_branch_delta", "sd_branch_delta",
        "mean_shared_signed_change", "mean_private_signed_change",
    ))
    required = [
        "analysis_config", "seed", "cancer", "horizon", "feature", "feature_type",
        "mean_W_shared", "mean_W_private", "mean_branch_delta",
    ]
    _require_columns(data, required, "seed_branch_summary.csv")
    ids = ["analysis_config", "seed", "cancer", "horizon", "feature", "feature_type"]
    data = _expand_scopes(data, ids)
    group = ["analysis_config", "seed", "cancer", "horizon", "ranking_scope"]
    data = _rank(data, group, "mean_W_shared", "within_seed_shared_rank", "within_seed_shared_percentile")
    data = _rank(data, group, "mean_W_private", "within_seed_private_rank", "within_seed_private_percentile")
    delta = data["mean_branch_delta"].astype(float)
    data["branch_dominance"] = np.select(
        [delta > BRANCH_TIE_TOLERANCE, delta < -BRANCH_TIE_TOLERANCE],
        ["shared", "private"], default="tie",
    )
    return data


def _summarize_branches(seed_branches: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    data = seed_branches.copy()
    data["shared_dominant"] = data["branch_dominance"].eq("shared")
    data["private_dominant"] = data["branch_dominance"].eq("private")
    data["branch_tie"] = data["branch_dominance"].eq("tie")
    keys = [
        "analysis_config", "cancer", "horizon", "feature", "feature_type",
        "ranking_scope",
    ]
    agg: dict[str, tuple[str, object]] = {
        "n_seeds_evaluable": ("seed", "nunique"),
        "mean_W_shared": ("mean_W_shared", "mean"),
        "sd_W_shared": ("mean_W_shared", "std"),
        "mean_W_private": ("mean_W_private", "mean"),
        "sd_W_private": ("mean_W_private", "std"),
        "mean_branch_delta": ("mean_branch_delta", "mean"),
        "sd_branch_delta": ("mean_branch_delta", "std"),
        "number_seeds_shared_dominant": ("shared_dominant", "sum"),
        "number_seeds_private_dominant": ("private_dominant", "sum"),
        "number_seeds_tied": ("branch_tie", "sum"),
        "mean_shared_rank": ("within_seed_shared_rank", "mean"),
        "sd_shared_rank": ("within_seed_shared_rank", "std"),
        "mean_private_rank": ("within_seed_private_rank", "mean"),
        "sd_private_rank": ("within_seed_private_rank", "std"),
        "mean_shared_importance_percentile": ("within_seed_shared_percentile", "mean"),
        "mean_private_importance_percentile": ("within_seed_private_percentile", "mean"),
    }
    for source in ("mean_shared_signed_change", "mean_private_signed_change"):
        if source in data:
            agg[source] = (source, "mean")
            agg[source.replace("mean_", "sd_")] = (source, "std")
    out = data.groupby(keys, as_index=False, dropna=False).agg(**agg)
    out["proportion_shared_dominant"] = (
        out["number_seeds_shared_dominant"] / out["n_seeds_evaluable"]
    )
    out = _attach_mapping(out, mapping)
    out["prediction_scale"] = PREDICTION_SCALE
    out["calibration_kind"] = CALIBRATION_KIND
    return out.sort_values(
        ["analysis_config", "cancer", "horizon", "ranking_scope", "mean_branch_delta", "feature"],
        ascending=[True, True, True, True, False, True],
    ).reset_index(drop=True)


def _prepare_seed_profiles(df: pd.DataFrame) -> pd.DataFrame:
    data = _canonicalize_columns(df, PROFILE_ALIASES)
    data = data.drop(
        columns=["within_profile_rank", "within_profile_percentile"],
        errors="ignore",
    )
    data = _numeric(data, (
        "mean_W_feature", "sd_W_feature", "median_W_feature", "mean_Delta_feature",
        "n_train_profile", "n_train_profile_events", "n_test_profile",
    ))
    if "profile_supported" in data:
        data = data[_bool_series(data["profile_supported"], default=False)].copy()
    required = [
        "analysis_config", "seed", "cancer", "horizon", "feature", "feature_type",
        "treatment_profile", "mean_W_feature", "mean_Delta_feature",
    ]
    _require_columns(data, required, "seed_profile_summary.csv")
    ids = [
        "analysis_config", "seed", "cancer", "horizon", "treatment_profile",
        "feature", "feature_type",
    ]
    data = _expand_scopes(data, ids)
    group = [
        "analysis_config", "seed", "cancer", "horizon", "treatment_profile",
        "ranking_scope",
    ]
    return _rank(
        data, group, "mean_W_feature", "within_profile_rank",
        "within_profile_percentile",
    )


def _summarize_profiles(
    seed_profiles: pd.DataFrame,
    mapping: pd.DataFrame,
    n_expected_seeds: int = len(DEFAULT_SEEDS),
) -> pd.DataFrame:
    keys = [
        "analysis_config", "cancer", "horizon", "treatment_profile",
        "feature", "feature_type", "ranking_scope",
    ]
    optional_identity = [c for c in ("treatment_profile_code", "treatment_profile_vector") if c in seed_profiles]
    keys_with_identity = keys[:4] + optional_identity + keys[4:]
    agg: dict[str, tuple[str, object]] = {
        "n_seeds_evaluable": ("seed", "nunique"),
        "mean_W_feature": ("mean_W_feature", "mean"),
        "sd_W_feature": ("mean_W_feature", "std"),
        "median_W_feature": ("mean_W_feature", "median"),
        "mean_Delta_feature": ("mean_Delta_feature", "mean"),
        "sd_Delta_feature": ("mean_Delta_feature", "std"),
        "mean_rank": ("within_profile_rank", "mean"),
        "sd_rank": ("within_profile_rank", "std"),
        "mean_importance_percentile": ("within_profile_percentile", "mean"),
    }
    for source in ("n_train_profile", "n_train_profile_events", "n_test_profile"):
        if source in seed_profiles:
            agg[f"mean_{source}"] = (source, "mean")
            agg[f"minimum_{source}"] = (source, "min")
    out = seed_profiles.groupby(keys_with_identity, as_index=False, dropna=False).agg(**agg)
    out = _attach_mapping(out, mapping)
    out["evaluable_frequency"] = out["n_seeds_evaluable"] / n_expected_seeds
    out["prediction_scale"] = PREDICTION_SCALE
    out["calibration_kind"] = CALIBRATION_KIND
    out = out.sort_values(
        ["cancer", "horizon", "treatment_profile", "ranking_scope", "mean_W_feature", "feature"],
        ascending=[True, True, True, True, False, True],
    ).reset_index(drop=True)
    out["across_seed_profile_rank"] = out.groupby(
        ["analysis_config", "cancer", "horizon", "treatment_profile", "ranking_scope"],
        dropna=False, sort=False,
    )["mean_W_feature"].rank(method="min", ascending=False).astype("Int64")
    return out


def _swap_columns(out: pd.DataFrame, reverse: pd.Series, a: str, b: str) -> None:
    if a not in out or b not in out or not reverse.any():
        return
    old_a = out.loc[reverse, a].copy()
    out.loc[reverse, a] = out.loc[reverse, b].to_numpy()
    out.loc[reverse, b] = old_a.to_numpy()


def _negate_quantile_pair(out: pd.DataFrame, reverse: pd.Series, low: str, high: str) -> None:
    if low not in out or high not in out or not reverse.any():
        return
    old_low = pd.to_numeric(out.loc[reverse, low], errors="coerce").copy()
    old_high = pd.to_numeric(out.loc[reverse, high], errors="coerce").copy()
    out.loc[reverse, low] = -old_high.to_numpy()
    out.loc[reverse, high] = -old_low.to_numpy()


def _normalize_treatment_orientation(df: pd.DataFrame) -> pd.DataFrame:
    out = _canonicalize_columns(df, INTERACTION_ALIASES)
    if "treatment_A" not in out or "treatment_B" not in out:
        return out
    key_a = (
        out["treatment_A_code"].astype(str)
        if "treatment_A_code" in out else out["treatment_A"].astype(str)
    )
    key_b = (
        out["treatment_B_code"].astype(str)
        if "treatment_B_code" in out else out["treatment_B"].astype(str)
    )
    reverse = key_a.str.strip() > key_b.str.strip()
    for a, b in (
        ("treatment_A", "treatment_B"),
        ("treatment_A_code", "treatment_B_code"),
        ("treatment_A_vector", "treatment_B_vector"),
    ):
        _swap_columns(out, reverse, a, b)

    # Actual risk summaries are treatment-labelled and must move with treatment.
    known_pairs = []
    for state in ("L", "H", "low", "high"):
        for suffix in ("mean", "median", "sd", "p05", "p95"):
            known_pairs.extend([
                (f"R_A_{state}_{suffix}", f"R_B_{state}_{suffix}"),
                (f"risk_A_{state}_{suffix}", f"risk_B_{state}_{suffix}"),
                (f"{suffix}_R_A_{state}", f"{suffix}_R_B_{state}"),
            ])
    for a, b in known_pairs:
        _swap_columns(out, reverse, a, b)

    # Swapping A/B negates psi and both B-A treatment contrasts.
    for col in list(out.columns):
        low = col.lower()
        if not (pd.api.types.is_numeric_dtype(out[col]) or low in {
            "mean_psi", "median_psi", "psi_p05", "psi_p95",
            "delta_low", "delta_high",
        }):
            continue
        if (
            ("psi" in low and not any(x in low for x in ("abs", "sd", "pr_", "prob")))
            or low.startswith("delta_low") or low.startswith("delta_high")
        ):
            if not any(x in low for x in ("p05", "p95", "lower", "upper")):
                values = pd.to_numeric(out.loc[reverse, col], errors="coerce")
                out.loc[reverse, col] = -values.to_numpy()
    for stem in ("psi", "delta_low", "delta_high"):
        for low_suffix, high_suffix in (("p05", "p95"), ("lower", "upper")):
            candidates_low = [f"{stem}_{low_suffix}", f"{low_suffix}_{stem}"]
            candidates_high = [f"{stem}_{high_suffix}", f"{high_suffix}_{stem}"]
            for lo, hi in zip(candidates_low, candidates_high):
                _negate_quantile_pair(out, reverse, lo, hi)
    for positive, negative in (
        ("mean_Pr_psi_gt_0", "mean_Pr_psi_lt_0"),
        ("Pr_psi_gt_0", "Pr_psi_lt_0"),
        ("prob_psi_positive", "prob_psi_negative"),
    ):
        _swap_columns(out, reverse, positive, negative)
    if "mean_psi" in out:
        psi = pd.to_numeric(out["mean_psi"], errors="coerce")
        out["direction"] = np.select(
            [psi > INTERACTION_TIE_TOLERANCE, psi < -INTERACTION_TIE_TOLERANCE],
            ["positive", "negative"], default="tie",
        )
    out["pair_orientation_rule"] = (
        "A/B sorted by treatment-profile code; "
        "psi=(B_high-A_high)-(B_low-A_low)"
    )
    return out


def _prepare_seed_interactions(df: pd.DataFrame) -> pd.DataFrame:
    data = _normalize_treatment_orientation(df)
    data = data.drop(
        columns=["within_pair_magnitude_rank", "within_pair_magnitude_percentile"],
        errors="ignore",
    )
    data = _numeric(data, (
        "mean_psi", "median_psi", "sd_patient_psi", "mean_abs_psi",
        "mean_Pr_psi_gt_0", "mean_Pr_psi_lt_0", "n_sign_reversals",
        "proportion_sign_reversal",
    ))
    required = [
        "analysis_config", "seed", "cancer", "horizon", "feature", "feature_type",
        "treatment_A", "treatment_B", "mean_psi", "mean_abs_psi",
    ]
    _require_columns(data, required, "seed_interaction_summary.csv")
    ids = [
        "analysis_config", "seed", "cancer", "horizon", "treatment_A", "treatment_B",
        "feature", "feature_type",
    ]
    data = _expand_scopes(data, ids)
    group = [
        "analysis_config", "seed", "cancer", "horizon", "treatment_A", "treatment_B",
        "ranking_scope",
    ]
    data = _rank(
        data, group, "mean_abs_psi", "within_pair_magnitude_rank",
        "within_pair_magnitude_percentile",
    )
    psi = data["mean_psi"].astype(float)
    data["direction"] = np.select(
        [psi > INTERACTION_TIE_TOLERANCE, psi < -INTERACTION_TIE_TOLERANCE],
        ["positive", "negative"], default="tie",
    )
    return data


def _summarize_interactions(
    seed_interactions: pd.DataFrame,
    mapping: pd.DataFrame,
    n_expected_seeds: int = len(DEFAULT_SEEDS),
) -> pd.DataFrame:
    data = seed_interactions.copy()
    data["positive_seed"] = data["direction"].eq("positive")
    data["negative_seed"] = data["direction"].eq("negative")
    data["tie_seed"] = data["direction"].eq("tie")
    pair_identity = [c for c in (
        "treatment_A_code", "treatment_A_vector", "treatment_B_code", "treatment_B_vector"
    ) if c in data]
    keys = [
        "analysis_config", "cancer", "horizon", "treatment_A", "treatment_B",
        *pair_identity, "feature", "feature_type", "ranking_scope",
    ]
    agg: dict[str, tuple[str, object]] = {
        "n_seeds_evaluable": ("seed", "nunique"),
        "mean_psi": ("mean_psi", "mean"),
        "sd_psi": ("mean_psi", "std"),
        "median_psi": ("mean_psi", "median"),
        "mean_abs_psi": ("mean_abs_psi", "mean"),
        "mean_Pr_psi_gt_0": ("mean_Pr_psi_gt_0", "mean"),
        "mean_Pr_psi_lt_0": ("mean_Pr_psi_lt_0", "mean"),
        "number_positive_seeds": ("positive_seed", "sum"),
        "number_negative_seeds": ("negative_seed", "sum"),
        "number_tied_seeds": ("tie_seed", "sum"),
        "mean_magnitude_rank": ("within_pair_magnitude_rank", "mean"),
        "sd_magnitude_rank": ("within_pair_magnitude_rank", "std"),
        "mean_magnitude_percentile": ("within_pair_magnitude_percentile", "mean"),
    }
    for source, target, operation in (
        ("median_psi", "mean_seed_median_psi", "mean"),
        ("sd_patient_psi", "mean_within_seed_patient_sd_psi", "mean"),
        ("n_sign_reversals", "total_patient_sign_reversals", "sum"),
        ("proportion_sign_reversal", "mean_proportion_sign_reversal", "mean"),
        ("n_test_cancer", "mean_n_test_cancer", "mean"),
    ):
        if source in data:
            agg[target] = (source, operation)
    out = data.groupby(keys, as_index=False, dropna=False).agg(**agg)
    counts = out[["number_positive_seeds", "number_negative_seeds", "number_tied_seeds"]]
    out["direction_consistency_across_seeds"] = counts.max(axis=1) / out["n_seeds_evaluable"]
    out["dominant_direction_across_seeds"] = np.select(
        [
            (out["number_positive_seeds"] > out["number_negative_seeds"])
            & (out["number_positive_seeds"] > out["number_tied_seeds"]),
            (out["number_negative_seeds"] > out["number_positive_seeds"])
            & (out["number_negative_seeds"] > out["number_tied_seeds"]),
            (out["number_tied_seeds"] > out["number_positive_seeds"])
            & (out["number_tied_seeds"] > out["number_negative_seeds"]),
        ],
        ["positive", "negative", "tie"], default="mixed_or_tied",
    )
    out = _attach_mapping(out, mapping)
    out["evaluable_frequency"] = out["n_seeds_evaluable"] / n_expected_seeds
    out["prediction_scale"] = PREDICTION_SCALE
    out["calibration_kind"] = CALIBRATION_KIND
    out["pair_orientation_rule"] = (
        "A/B sorted by treatment-profile code; "
        "psi=(B_high-A_high)-(B_low-A_low)"
    )
    return out.sort_values(
        ["cancer", "horizon", "treatment_A", "treatment_B", "ranking_scope", "mean_abs_psi", "feature"],
        ascending=[True, True, True, True, True, False, True],
    ).reset_index(drop=True)


def _cross_cancer_summary(
    feature_summary: pd.DataFrame,
    branch_summary: pd.DataFrame,
) -> pd.DataFrame:
    base = feature_summary[feature_summary["mean_W_feature"].notna()].copy()
    idx = ["analysis_config", "horizon", "ranking_scope", "feature", "feature_type"]
    metrics = ["mean_W_feature", "mean_rank", "mean_importance_percentile"]
    merged: pd.DataFrame | None = None
    for metric in metrics:
        pivot = base.pivot_table(index=idx, columns="cancer", values=metric, aggfunc="first").reset_index()
        pivot = pivot.rename(columns={c: f"{c}_{metric}" for c in CANCERS if c in pivot})
        merged = pivot if merged is None else merged.merge(pivot, on=idx, how="outer", validate="one_to_one")
    branch_base = branch_summary.copy()
    for metric in ("mean_W_shared", "mean_W_private"):
        pivot = branch_base.pivot_table(index=idx, columns="cancer", values=metric, aggfunc="first").reset_index()
        pivot = pivot.rename(columns={c: f"{c}_{metric}" for c in CANCERS if c in pivot})
        merged = pivot if merged is None else merged.merge(pivot, on=idx, how="outer", validate="one_to_one")
    assert merged is not None
    for cancer in CANCERS:
        for metric in (*metrics, "mean_W_shared", "mean_W_private"):
            col = f"{cancer}_{metric}"
            if col not in merged:
                merged[col] = np.nan
    wcols = [f"{c}_mean_W_feature" for c in CANCERS]
    pcols = [f"{c}_mean_importance_percentile" for c in CANCERS]
    merged["n_cancers_evaluable"] = merged[wcols].notna().sum(axis=1)
    merged = merged[merged["n_cancers_evaluable"] >= 2].copy()
    merged["macro_mean_W_feature"] = merged[wcols].mean(axis=1, skipna=True)
    merged["mean_importance_percentile"] = merged[pcols].mean(axis=1, skipna=True)
    merged["minimum_cancer_importance_percentile"] = merged[pcols].min(axis=1, skipna=True)
    merged["prediction_scale"] = PREDICTION_SCALE
    merged["calibration_kind"] = CALIBRATION_KIND
    merged = merged.sort_values(
        ["analysis_config", "horizon", "ranking_scope", "macro_mean_W_feature", "feature"],
        ascending=[True, True, True, False, True],
    ).reset_index(drop=True)
    merged["cross_cancer_rank"] = merged.groupby(
        ["analysis_config", "horizon", "ranking_scope"], dropna=False, sort=False
    )["macro_mean_W_feature"].rank(method="min", ascending=False).astype("Int64")
    return merged


def _chunk_transform_common(chunk: pd.DataFrame, config: str, seed: int) -> pd.DataFrame:
    aliases = dict(FEATURE_ALIASES)
    aliases.update(BRANCH_ALIASES)
    aliases.update(INTERACTION_ALIASES)
    chunk = _canonicalize_columns(chunk, aliases)
    return _normalize_common(chunk, config, seed)


def _concatenate_patient_files(
    sources: Sequence[tuple[Path, str, int]],
    destination: Path | None = None,
    interaction: bool = False,
    *,
    archive: zipfile.ZipFile | None = None,
    archive_member: str | None = None,
) -> dict[str, object]:
    """Concatenate patient rows to a CSV file or directly to a ZIP member.

    Exactly one destination is used.  ZIP mode never creates an intermediate
    uncompressed CSV, which keeps peak disk use close to the compressed size.
    """
    if (destination is None) == (archive is None):
        raise ValueError("Specify exactly one of destination or archive")
    if archive is not None and not archive_member:
        raise ValueError("archive_member is required when archive is provided")

    union: list[str] = []
    for path, _, _ in sources:
        header = pd.read_csv(path, nrows=0)
        header = _canonicalize_columns(header, {**FEATURE_ALIASES, **BRANCH_ALIASES, **INTERACTION_ALIASES})
        for col in header.columns:
            if col not in union:
                union.append(col)
    for extra in ("analysis_config", "seed", "prediction_scale", "calibration_kind"):
        if extra not in union:
            union.append(extra)
    if interaction:
        for extra in ("direction", "pair_orientation_rule"):
            if extra not in union:
                union.append(extra)

    raw_handle = None
    text_handle = None
    try:
        if archive is not None:
            raw_handle = archive.open(str(archive_member), mode="w", force_zip64=True)
            text_handle = io.TextIOWrapper(
                raw_handle, encoding="utf-8", newline="", write_through=False
            )
        else:
            assert destination is not None
            destination.parent.mkdir(parents=True, exist_ok=True)
            text_handle = destination.open("w", encoding="utf-8", newline="")

        wrote = False
        rows = 0
        probability_min = math.inf
        probability_max = -math.inf
        for path, config, seed in sources:
            for chunk in pd.read_csv(path, chunksize=50_000, low_memory=False):
                chunk = _chunk_transform_common(chunk, config, seed)
                if interaction:
                    chunk = _normalize_treatment_orientation(chunk)
                unexpected = [col for col in chunk.columns if col not in union]
                if unexpected:
                    raise ValueError(
                        f"Unexpected transformed columns in {path.name}: {unexpected}"
                    )
                probability_columns = [
                    c for c in chunk.columns
                    if ("risk" in c.lower() or re.match(r"^R_[AB]_[LH]", c))
                    and any(
                        x in c.lower() for x in ("mean", "median", "p05", "p95")
                    )
                ]
                for col in probability_columns:
                    values = pd.to_numeric(chunk[col], errors="coerce")
                    if values.notna().any():
                        probability_min = min(probability_min, float(values.min()))
                        probability_max = max(probability_max, float(values.max()))
                # Headers are expected to be identical. Extra transformed columns
                # indicate a runner-schema mismatch and are rejected above.
                chunk = chunk.reindex(columns=union)
                chunk.to_csv(text_handle, header=not wrote, index=False)
                wrote = True
                rows += len(chunk)
        if not wrote:
            pd.DataFrame(columns=union).to_csv(text_handle, index=False)
        text_handle.flush()
    finally:
        if text_handle is not None:
            text_handle.close()
        elif raw_handle is not None:
            raw_handle.close()
    return {
        "rows": rows,
        "probability_min": np.nan if math.isinf(probability_min) else probability_min,
        "probability_max": np.nan if math.isinf(probability_max) else probability_max,
        "archive_member": archive_member,
    }


def _active_profile(value: object) -> bool | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    bits = re.findall(r"(?<!\d)[01](?!\d)", text)
    if bits:
        return any(bit == "1" for bit in bits)
    compact = re.sub(r"[^01]", "", text)
    if compact and len(compact) <= 8:
        return "1" in compact
    if any(term in text for term in ("untreated", "no treatment", "none", "0000")):
        return False
    if any(term in text for term in ("chemo", "hormone", "radiation", "pharmaceutical", "drug")):
        return True
    return None


def _write_qc_report(
    output_path: Path,
    seeds: Sequence[int],
    feature_summaries: dict[str, pd.DataFrame],
    branch_summaries: dict[str, pd.DataFrame],
    cross_summaries: dict[str, pd.DataFrame],
    interaction_summaries: dict[str, pd.DataFrame],
    quantiles: pd.DataFrame,
    calibration: pd.DataFrame,
    treatment_support: pd.DataFrame,
    qc_payloads: list[dict[str, object]],
    patient_qc: dict[str, dict[str, object]],
    missing_files: list[str],
) -> list[str]:
    failures: list[str] = []
    lines = [
        "FIXED-CONTRAST WASSERSTEIN QC REPORT",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Configurations: {', '.join(CONFIGS)}",
        f"Outer split seeds: {', '.join(map(str, seeds))}",
        f"Horizons: {', '.join(map(str, HORIZONS))} months",
        f"Prediction scale: {PREDICTION_SCALE}",
        f"Calibration: {CALIBRATION_KIND}",
        "DGP retraining performed: NO",
        "Donor sampling/random feature replacement in primary analyses: NO",
        "Across-seed method: seed-level estimates summarized without concatenating test patients",
        "Branch additivity assumed: NO",
        "",
        "QC CHECKS",
    ]

    def check(label: str, passed: bool, detail: str = "") -> None:
        state = "PASS" if passed else "FAIL"
        lines.append(f"[{state}] {label}" + (f" -- {detail}" if detail else ""))
        if not passed:
            failures.append(label + (f": {detail}" if detail else ""))

    check("All required seed partition files are present", not missing_files,
          "; ".join(missing_files[:10]))
    for config, summary in feature_summaries.items():
        observed = set(pd.to_numeric(summary.get("horizon", pd.Series(dtype=float)), errors="coerce").dropna().astype(int))
        check(f"{config}: horizons are exactly 18, 24, and 36 months", observed == set(HORIZONS), str(sorted(observed)))
        mapping_consistent = not summary["mean_W_feature"].isna().any()
        if "n_seeds_evaluable_from_mapping" in summary:
            mapping_consistent = mapping_consistent and bool(
                (
                    summary["n_seeds_evaluable"]
                    <= summary["n_seeds_evaluable_from_mapping"]
                ).all()
            )
        check(
            f"{config}: absent/non-evaluable features were not assigned artificial values",
            mapping_consistent,
        )

    for config, summary in interaction_summaries.items():
        a_col = next((c for c in ("treatment_A_vector", "treatment_A") if c in summary), None)
        b_col = next((c for c in ("treatment_B_vector", "treatment_B") if c in summary), None)
        if a_col and b_col and not summary.empty:
            known_activity = summary.apply(
                lambda r: (
                    _active_profile(r[a_col]) is not False
                    and _active_profile(r[b_col]) is not False
                ),
                axis=1,
            )
            check(
                f"{config}: treatment-modification table uses active-active pairs",
                bool(known_activity.all()),
            )
        code_a = "treatment_A_code" if "treatment_A_code" in summary else "treatment_A"
        code_b = "treatment_B_code" if "treatment_B_code" in summary else "treatment_B"
        if code_a in summary and code_b in summary:
            oriented = summary[code_a].astype(str).str.strip() <= summary[code_b].astype(str).str.strip()
            check(
                f"{config}: treatment A/B orientation is consistent",
                bool(oriented.all()),
            )

    q = _canonicalize_columns(quantiles.copy(), FEATURE_ALIASES) if not quantiles.empty else quantiles
    if not q.empty:
        q = _numeric(q, ("low_value", "high_value", "q25", "q75", "contrast_low", "contrast_high"))
        low_col = next((c for c in ("low_value", "contrast_low", "q25") if c in q), None)
        high_col = next((c for c in ("high_value", "contrast_high", "q75") if c in q), None)
        if low_col is None and "feature_low_raw" in q:
            low_col = "feature_low_raw"
        if high_col is None and "feature_high_raw" in q:
            high_col = "feature_high_raw"
        if low_col and high_col and "feature_type" in q:
            mutation = q[q["feature_type"].astype(str).str.lower().eq("mutation")]

            def raw_values(values: pd.Series) -> np.ndarray:
                parsed: list[float] = []
                for value in values.dropna():
                    parsed.extend(
                        float(item) for item in str(value).split(";") if item.strip()
                    )
                return np.asarray(parsed, dtype=float)

            mutation_low = raw_values(mutation[low_col])
            mutation_high = raw_values(mutation[high_col])
            mutation_ok = mutation.empty or (
                mutation_low.size > 0 and mutation_high.size > 0
                and np.allclose(mutation_low, 0.0)
                and np.allclose(mutation_high, 1.0)
            )
            check("Mutation fixed contrasts are exactly 0 versus 1", mutation_ok)
        if "contrast_definition" in q and "feature_type" in q:
            continuous = q[q["feature_type"].astype(str).str.lower().isin({"pathway", "tmb"})]
            # Non-evaluable screened features may legitimately have no contrast
            # definition (for example, missing/nonfinite outer-training data).
            # The provenance requirement applies to contrasts that were used.
            if "evaluable" in continuous:
                continuous = continuous[_bool_series(continuous["evaluable"], default=False)]
            source_text = continuous["contrast_definition"].astype(str).str.lower()
            check("Pathway/TMB quantiles are identified as outer-training values",
                  continuous.empty or source_text.str.contains("train").all())
    else:
        check("Quantile audit is available", False)

    if not calibration.empty:
        kind_cols = [c for c in calibration if "kind" in c.lower() or "calibration" in c.lower()]
        values = _join_unique(calibration[kind_cols].astype(str).stack()) if kind_cols else ""
        check("Cancer-specific fixed-slope calibration audit is available",
              ("cancer" in values.lower() and ("fixed" in values.lower() or "slope" in values.lower())))
    else:
        check("Calibration audit is available", False)

    for config, detail in patient_qc.items():
        lo = detail.get("probability_min", np.nan)
        hi = detail.get("probability_max", np.nan)
        valid = (pd.isna(lo) and pd.isna(hi)) or (float(lo) >= -1e-8 and float(hi) <= 1.0 + 1e-8)
        check(f"{config}: saved posterior probability summaries remain within [0,1]", valid, f"range={lo} to {hi}")

    if not treatment_support.empty:
        active_cols = [c for c in ("treatment_A_vector", "treatment_A", "treatment_B_vector", "treatment_B") if c in treatment_support]
        if len(active_cols) >= 2:
            a_col = next((c for c in ("treatment_A_vector", "treatment_A") if c in treatment_support), None)
            b_col = next((c for c in ("treatment_B_vector", "treatment_B") if c in treatment_support), None)
            active = treatment_support.apply(
                lambda r: _active_profile(r[a_col]) is not False and _active_profile(r[b_col]) is not False,
                axis=1,
            )
            check("Primary treatment comparisons exclude known untreated profiles", bool(active.all()))

    # Respect the runner's explicit QC schema.  Negative-state fields are not
    # generic pass/fail flags: donor_sampling_used=False is a required success.
    explicit_failures: list[str] = []
    for payload in qc_payloads:
        prefix = f"{payload.get('analysis_config', '?')} seed {payload.get('seed', '?')}"
        if payload.get("no_model_retraining") is not True:
            explicit_failures.append(f"{prefix}: no_model_retraining was not true")
        if payload.get("donor_sampling_used") is not False:
            explicit_failures.append(f"{prefix}: donor_sampling_used was not false")
        payload_horizons = set(int(x) for x in payload.get("horizons", []) if str(x).strip())
        if payload_horizons != set(HORIZONS):
            explicit_failures.append(f"{prefix}: horizons={sorted(payload_horizons)}")
        if payload.get("prediction_scale") != PREDICTION_SCALE:
            explicit_failures.append(
                f"{prefix}: prediction_scale={payload.get('prediction_scale')!r}"
            )
        if payload.get("calibration_kind") != CALIBRATION_KIND:
            explicit_failures.append(
                f"{prefix}: calibration_kind={payload.get('calibration_kind')!r}"
            )
        for item in payload.get("checks", []):
            if isinstance(item, dict) and str(item.get("status", "")).strip().lower() in {
                "fail", "failed", "failure", "error"
            }:
                explicit_failures.append(
                    f"{prefix}: {item.get('check', 'unnamed check')} -- {item.get('detail', '')}"
                )
        check_status = {
            str(item.get("check", "")): str(item.get("status", "")).lower()
            for item in payload.get("checks", []) if isinstance(item, dict)
        }
        if check_status.get("joint_posterior_sampling") != "pass":
            explicit_failures.append(f"{prefix}: joint_posterior_sampling was not confirmed")
    check("All runner-level QC flags passed", not explicit_failures, "; ".join(explicit_failures[:10]))

    lines.extend(["", "FEATURE AVAILABILITY AND EVALUABILITY"])
    for config, summary in feature_summaries.items():
        base = summary[summary["ranking_scope"].eq("molecular_all" if config == "PathNoUDP" else "pathway_only")]
        for cancer in CANCERS:
            c = base[base["cancer"].eq(cancer)]
            n_features = c.groupby("horizon")["feature"].nunique().to_dict()
            lines.append(f"{config} {cancer}: molecular features by horizon {n_features}")
        counts = Counter(base.drop_duplicates(["cancer", "horizon", "feature"])["n_seeds_evaluable"].astype(int))
        lines.append(f"{config}: feature/cancer/horizon evaluability counts {dict(sorted(counts.items(), reverse=True))}")

    lines.extend(["", "STRONGEST FIXED-CONTRAST MOLECULAR FEATURES"])
    for config, summary in feature_summaries.items():
        primary_scope = "molecular_all" if config == "PathNoUDP" else "pathway_only"
        sub = summary[summary["ranking_scope"].eq(primary_scope)]
        for (cancer, horizon), group in sub.groupby(["cancer", "horizon"], sort=False):
            top = group.nlargest(5, "mean_W_feature")
            text = ", ".join(f"{r.feature} ({r.mean_W_feature:.6g})" for r in top.itertuples())
            lines.append(f"{config} {cancer} H={int(horizon)}: {text}")

    lines.extend(["", "STRONGEST CROSS-CANCER FEATURES (EQUAL CANCER WEIGHT)"])
    for config, summary in cross_summaries.items():
        primary_scope = "molecular_all" if config == "PathNoUDP" else "pathway_only"
        for horizon, group in summary[summary["ranking_scope"].eq(primary_scope)].groupby("horizon"):
            top = group.nlargest(5, "macro_mean_W_feature")
            text = ", ".join(f"{r.feature} ({r.macro_mean_W_feature:.6g})" for r in top.itertuples())
            lines.append(f"{config} H={int(horizon)}: {text}")

    lines.extend(["", "STRONGEST BRANCH EFFECTS"])
    for config, summary in branch_summaries.items():
        primary_scope = "molecular_all" if config == "PathNoUDP" else "pathway_only"
        sub = summary[summary["ranking_scope"].eq(primary_scope)]
        top_shared = sub.nlargest(5, "mean_W_shared")
        top_private = sub.nlargest(5, "mean_W_private")
        lines.append(f"{config} shared: " + ", ".join(
            f"{r.cancer}/{r.feature}/H{int(r.horizon)} ({r.mean_W_shared:.6g})" for r in top_shared.itertuples()
        ))
        lines.append(f"{config} private: " + ", ".join(
            f"{r.cancer}/{r.feature}/H{int(r.horizon)} ({r.mean_W_private:.6g})" for r in top_private.itertuples()
        ))

    lines.extend(["", "SUPPORTED ACTIVE TREATMENT COMPARISONS"])
    for config, summary in interaction_summaries.items():
        pairs = summary[["cancer", "treatment_A", "treatment_B"]].drop_duplicates()
        lines.append(f"{config}: " + "; ".join(
            f"{r.cancer}: {r.treatment_A} vs {r.treatment_B}" for r in pairs.itertuples()
        ))
        primary_scope = "molecular_all" if config == "PathNoUDP" else "pathway_only"
        top = summary[summary["ranking_scope"].eq(primary_scope)].nlargest(10, "mean_abs_psi")
        lines.append(f"{config} strongest feature-by-treatment interactions: " + ", ".join(
            f"{r.cancer}/{r.feature}/H{int(r.horizon)} ({r.mean_psi:.6g})" for r in top.itertuples()
        ))
        if "total_patient_sign_reversals" in summary:
            reversed_rows = summary[summary["total_patient_sign_reversals"].fillna(0) > 0]
            lines.append(f"{config} rows with at least one patient-level treatment-contrast reversal: {len(reversed_rows)}")

    lines.extend(["", "FIXED CONTRASTS (OUTER-TRAINING VALUES)"])
    if not quantiles.empty:
        display_cols = [c for c in (
            "analysis_config", "seed", "cancer", "feature", "feature_type",
            "low_value", "high_value", "contrast_low", "contrast_high", "q25", "q75",
            "feature_low_raw", "feature_high_raw", "contrast_definition",
        ) if c in quantiles]
        lines.extend(quantiles[display_cols].to_csv(index=False).strip().splitlines())

    lines.extend(["", "QC FAILURES / UNSUPPORTED RESULTS"])
    lines.extend(f"- {failure}" for failure in failures)
    if not failures:
        lines.append("None.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return failures


def _validate_partitions(partition_root: Path, seeds: Sequence[int]) -> list[str]:
    missing = []
    for config in CONFIGS:
        for seed in seeds:
            part = partition_root / config / f"seed{seed}"
            for filename in PARTITION_FILES:
                path = part / filename
                if not path.is_file():
                    missing.append(str(path))
    return missing


def _read_json_payloads(partition_root: Path, seeds: Sequence[int]) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for config in CONFIGS:
        for seed in seeds:
            path = partition_root / config / f"seed{seed}" / "qc.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                payload = {"payload": payload}
            payload.setdefault("analysis_config", config)
            payload.setdefault("seed", seed)
            payloads.append(payload)
    return payloads


def finalize(
    partition_root: Path,
    output_dir: Path,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    strict: bool = True,
    patient_archive: Path | None = None,
) -> dict[str, Path]:
    """Aggregate all seed partitions and return the generated file paths.

    When ``patient_archive`` is provided, the six large patient-level CSVs are
    streamed as root-level members of that ZIP.  Summary and QC files remain
    ordinary files in ``output_dir``.  Source partitions are read only.
    """
    partition_root = Path(partition_root).resolve()
    output_dir = Path(output_dir).resolve()
    seeds = tuple(int(x) for x in seeds)
    if not seeds:
        raise ValueError("At least one seed is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = _validate_partitions(partition_root, seeds)
    if missing:
        raise FileNotFoundError(
            "Cannot finalize an incomplete run. Missing partition files:\n" + "\n".join(missing)
        )

    mapping_frames = []
    for config in CONFIGS:
        mapping_frames.append(_load_seed_tables(
            partition_root, config, seeds, "feature_identity_mapping.csv", FEATURE_ALIASES
        ))
    mapping_by_seed = pd.concat(mapping_frames, ignore_index=True)
    mapping_summary = _mapping_summary(mapping_by_seed, len(seeds))
    mapping_path = output_dir / "fixed_contrast_feature_identity_mapping.csv"
    mapping_summary.to_csv(mapping_path, index=False)
    # Preserve the seed-specific identity decisions as an additional audit table.
    mapping_by_seed.to_csv(output_dir / "fixed_contrast_feature_identity_mapping_by_seed.csv", index=False)

    outputs: dict[str, Path] = {"feature_identity_mapping": mapping_path}
    known_row_counts: dict[str, int] = {}
    archive_member_names: list[str] = []
    feature_summaries: dict[str, pd.DataFrame] = {}
    branch_summaries: dict[str, pd.DataFrame] = {}
    cross_summaries: dict[str, pd.DataFrame] = {}
    interaction_summaries: dict[str, pd.DataFrame] = {}
    patient_qc: dict[str, dict[str, object]] = {}

    patient_archive_path: Path | None = None
    patient_archive_work: Path | None = None
    if patient_archive is not None:
        patient_archive_path = Path(patient_archive)
        if not patient_archive_path.is_absolute():
            patient_archive_path = output_dir / patient_archive_path
        patient_archive_path = patient_archive_path.resolve()
        if patient_archive_path.suffix.lower() != ".zip":
            raise ValueError("patient_archive must have a .zip suffix")
        patient_archive_path.parent.mkdir(parents=True, exist_ok=True)
        patient_archive_work = patient_archive_path.with_name(
            patient_archive_path.name + ".inprogress"
        )
        if patient_archive_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing patient archive: {patient_archive_path}"
            )
        if patient_archive_work.exists():
            raise FileExistsError(
                f"Incomplete patient archive already exists: {patient_archive_work}"
            )
        patient_names = {
            FINAL_NAMES["patient_feature"].format(config=config)
            for config in CONFIGS
        } | {
            FINAL_NAMES["patient_interactions"].format(config=config)
            for config in CONFIGS
        } | {
            f"{config}_fixed_contrast_patient_level_branch_attribution.csv"
            for config in CONFIGS
        }
        existing_patient_files = [
            str(output_dir / name)
            for name in sorted(patient_names)
            if (output_dir / name).exists()
        ]
        if existing_patient_files:
            raise FileExistsError(
                "Archive mode will not shadow existing uncompressed patient files:\n"
                + "\n".join(existing_patient_files)
            )
        with zipfile.ZipFile(
            patient_archive_work,
            mode="x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ):
            pass

    def emit_patient_csv(
        sources: Sequence[tuple[Path, str, int]],
        member_name: str,
        *,
        interaction: bool = False,
    ) -> tuple[dict[str, object], Path]:
        if patient_archive_work is None:
            destination = output_dir / member_name
            result = _concatenate_patient_files(
                sources, destination=destination, interaction=interaction
            )
            return result, destination
        with zipfile.ZipFile(
            patient_archive_work,
            mode="a",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            result = _concatenate_patient_files(
                sources,
                interaction=interaction,
                archive=archive,
                archive_member=member_name,
            )
        archive_member_names.append(member_name)
        assert patient_archive_path is not None
        return result, patient_archive_path

    for config in CONFIGS:
        seed_feature_raw = _load_seed_tables(
            partition_root, config, seeds, "seed_feature_summary.csv", FEATURE_ALIASES
        )
        seed_features = _prepare_seed_features(seed_feature_raw)
        seed_features.to_csv(output_dir / f"{config}_fixed_contrast_seed_feature_summary.csv", index=False)
        feature_summary = _summarize_features(seed_features, mapping_summary, len(seeds))
        feature_path = output_dir / FINAL_NAMES["feature_summary"].format(config=config)
        feature_summary.to_csv(feature_path, index=False)
        feature_summaries[config] = feature_summary
        outputs[f"{config}_feature_summary"] = feature_path

        seed_branch_raw = _load_seed_tables(
            partition_root, config, seeds, "seed_branch_summary.csv", BRANCH_ALIASES
        )
        seed_branches = _prepare_seed_branches(seed_branch_raw)
        seed_branches.to_csv(output_dir / f"{config}_fixed_contrast_seed_branch_summary.csv", index=False)
        branch_summary = _summarize_branches(seed_branches, mapping_summary)
        branch_path = output_dir / FINAL_NAMES["branch_summary"].format(config=config)
        branch_summary.to_csv(branch_path, index=False)
        branch_summaries[config] = branch_summary
        outputs[f"{config}_branch_summary"] = branch_path

        cross = _cross_cancer_summary(feature_summary, branch_summary)
        cross_path = output_dir / FINAL_NAMES["cross_cancer"].format(config=config)
        cross.to_csv(cross_path, index=False)
        cross_summaries[config] = cross
        outputs[f"{config}_cross_cancer"] = cross_path

        seed_profile_raw = _load_seed_tables(
            partition_root, config, seeds, "seed_profile_summary.csv", PROFILE_ALIASES
        )
        seed_profiles = _prepare_seed_profiles(seed_profile_raw)
        seed_profiles.to_csv(output_dir / f"{config}_fixed_contrast_seed_profile_summary.csv", index=False)
        profile_summary = _summarize_profiles(seed_profiles, mapping_summary, len(seeds))
        profile_path = output_dir / FINAL_NAMES["profile_summary"].format(config=config)
        profile_summary.to_csv(profile_path, index=False)
        outputs[f"{config}_profile_summary"] = profile_path

        seed_interaction_raw = _load_seed_tables(
            partition_root, config, seeds, "seed_interaction_summary.csv", INTERACTION_ALIASES
        )
        seed_interactions = _prepare_seed_interactions(seed_interaction_raw)
        seed_interactions.to_csv(output_dir / f"{config}_fixed_contrast_seed_interaction_summary.csv", index=False)
        interaction_summary = _summarize_interactions(
            seed_interactions, mapping_summary, len(seeds)
        )
        interaction_path = output_dir / FINAL_NAMES["interaction_summary"].format(config=config)
        interaction_summary.to_csv(interaction_path, index=False)
        interaction_summaries[config] = interaction_summary
        outputs[f"{config}_interaction_summary"] = interaction_path

        patient_feature_sources = [
            (partition_root / config / f"seed{seed}" / "patient_feature.csv.gz", config, seed)
            for seed in seeds
        ]
        patient_feature_name = FINAL_NAMES["patient_feature"].format(config=config)
        feature_patient_qc, patient_feature_location = emit_patient_csv(
            patient_feature_sources, patient_feature_name
        )
        known_row_counts[patient_feature_name] = int(feature_patient_qc["rows"])
        outputs[f"{config}_patient_feature"] = patient_feature_location

        # Branch patients are retained as an explicit supplementary audit output;
        # the required branch-attribution file above is the across-seed summary.
        patient_branch_sources = [
            (partition_root / config / f"seed{seed}" / "patient_branch.csv.gz", config, seed)
            for seed in seeds
        ]
        patient_branch_name = (
            f"{config}_fixed_contrast_patient_level_branch_attribution.csv"
        )
        branch_patient_qc, patient_branch_location = emit_patient_csv(
            patient_branch_sources, patient_branch_name
        )
        known_row_counts[patient_branch_name] = int(branch_patient_qc["rows"])
        outputs[f"{config}_patient_branch"] = patient_branch_location

        patient_interaction_sources = [
            (partition_root / config / f"seed{seed}" / "patient_interactions.csv.gz", config, seed)
            for seed in seeds
        ]
        patient_interaction_name = FINAL_NAMES["patient_interactions"].format(
            config=config
        )
        interaction_patient_qc, patient_interaction_location = emit_patient_csv(
            patient_interaction_sources,
            patient_interaction_name,
            interaction=True,
        )
        known_row_counts[patient_interaction_name] = int(
            interaction_patient_qc["rows"]
        )
        outputs[f"{config}_patient_interactions"] = patient_interaction_location
        ranges = [feature_patient_qc, branch_patient_qc, interaction_patient_qc]
        finite_min = [x["probability_min"] for x in ranges if pd.notna(x["probability_min"])]
        finite_max = [x["probability_max"] for x in ranges if pd.notna(x["probability_max"])]
        patient_qc[config] = {
            "rows": sum(int(x["rows"]) for x in ranges),
            "probability_min": min(finite_min) if finite_min else np.nan,
            "probability_max": max(finite_max) if finite_max else np.nan,
        }

    archive_member_info: dict[str, zipfile.ZipInfo] = {}
    if patient_archive_work is not None:
        assert patient_archive_path is not None
        expected_members = {
            FINAL_NAMES["patient_feature"].format(config=config)
            for config in CONFIGS
        } | {
            FINAL_NAMES["patient_interactions"].format(config=config)
            for config in CONFIGS
        } | {
            f"{config}_fixed_contrast_patient_level_branch_attribution.csv"
            for config in CONFIGS
        }
        if set(archive_member_names) != expected_members:
            raise RuntimeError(
                "Patient archive member mismatch: "
                f"expected={sorted(expected_members)}, "
                f"observed={sorted(archive_member_names)}"
            )
        with zipfile.ZipFile(patient_archive_work, mode="r") as archive:
            observed = archive.namelist()
            if len(observed) != len(set(observed)):
                raise RuntimeError("Patient archive contains duplicate member names")
            archive_member_info = {
                info.filename: info for info in archive.infolist()
            }
        patient_archive_work.rename(patient_archive_path)
        outputs["patient_archive"] = patient_archive_path

    quantile_frames = []
    calibration_frames = []
    support_frames = []
    for config in CONFIGS:
        quantile_frames.append(_load_seed_tables(
            partition_root, config, seeds, "quantile_audit.csv", FEATURE_ALIASES
        ))
        calibration_frames.append(_load_seed_tables(
            partition_root, config, seeds, "calibration_qc.csv", FEATURE_ALIASES
        ))
        support = _load_seed_tables(
            partition_root, config, seeds, "treatment_support.csv", INTERACTION_ALIASES
        )
        support_frames.append(_normalize_treatment_orientation(support))
    quantiles = pd.concat(quantile_frames, ignore_index=True)
    calibration = pd.concat(calibration_frames, ignore_index=True)
    support = pd.concat(support_frames, ignore_index=True)
    quantiles.to_csv(output_dir / "fixed_contrast_quantile_audit.csv", index=False)
    calibration.to_csv(output_dir / "fixed_contrast_calibration_qc.csv", index=False)
    support.to_csv(output_dir / "fixed_contrast_treatment_support.csv", index=False)

    qc_payloads = _read_json_payloads(partition_root, seeds)
    (output_dir / "fixed_contrast_seed_qc.json").write_text(
        json.dumps(qc_payloads, indent=2, default=str), encoding="utf-8"
    )
    qc_path = output_dir / "fixed_contrast_QC_report.txt"
    failures = _write_qc_report(
        qc_path, seeds, feature_summaries, branch_summaries, cross_summaries,
        interaction_summaries, quantiles, calibration, support, qc_payloads,
        patient_qc, missing,
    )
    outputs["qc_report"] = qc_path

    manifest_rows = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file():
            continue
        row_count = None
        if path.suffix == ".csv":
            if path.name in known_row_counts:
                row_count = known_row_counts[path.name]
            else:
                with path.open("rb") as handle:
                    row_count = max(sum(1 for _ in handle) - 1, 0)
        manifest_rows.append({
            "file": path.name,
            "bytes": path.stat().st_size,
            "rows": row_count,
            "container": "",
            "compressed_bytes": np.nan,
        })
    if patient_archive_path is not None:
        for member_name in sorted(archive_member_info):
            info = archive_member_info[member_name]
            manifest_rows.append({
                "file": member_name,
                "bytes": int(info.file_size),
                "rows": known_row_counts.get(member_name),
                "container": str(patient_archive_path),
                "compressed_bytes": int(info.compress_size),
            })
    pd.DataFrame(manifest_rows).to_csv(output_dir / "fixed_contrast_output_manifest.csv", index=False)

    print("NO DGP models were retrained.")
    print("Donor sampling was removed from the primary analyses.")
    print(f"Prediction scale: {PREDICTION_SCALE}")
    print(f"Horizons: {', '.join(map(str, HORIZONS))} months")
    print(f"Final output directory: {output_dir}")
    if patient_archive_path is not None:
        print(f"Patient-level CSV archive: {patient_archive_path}")
        print(
            "Patient-level archive members: "
            + ", ".join(sorted(archive_member_info))
        )
    print(f"QC report: {qc_path}")
    if failures:
        print(f"QC failures: {len(failures)}")
        for failure in failures:
            print(f"  - {failure}")
        if strict:
            raise RuntimeError(
                f"Finalization completed with {len(failures)} QC failure(s); see {qc_path}"
            )
    else:
        print("QC failures: 0")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--archive-patient-level",
        action="store_true",
        help=(
            "Stream the six large patient-level CSVs into "
            "<output-dir>/fixed_contrast_patient_level_outputs.zip instead of "
            "materializing uncompressed CSVs."
        ),
    )
    parser.add_argument(
        "--patient-archive",
        type=Path,
        help=(
            "Optional ZIP path for --archive-patient-level. Providing this option "
            "also enables archive mode. Relative paths are resolved under output-dir."
        ),
    )
    parser.add_argument(
        "--allow-qc-failures", action="store_true",
        help="Write all outputs and return success even if a post-finalization QC check fails.",
    )
    args = parser.parse_args()
    invalid = [seed for seed in args.seeds if seed not in DEFAULT_SEEDS]
    if invalid:
        parser.error(f"Only saved outer split seeds 42--51 are allowed; got {invalid}")
    patient_archive = args.patient_archive
    if args.archive_patient_level and patient_archive is None:
        patient_archive = Path("fixed_contrast_patient_level_outputs.zip")
    finalize(
        args.partition_root,
        args.output_dir,
        seeds=args.seeds,
        strict=not args.allow_qc_failures,
        patient_archive=patient_archive,
    )


if __name__ == "__main__":
    main()
