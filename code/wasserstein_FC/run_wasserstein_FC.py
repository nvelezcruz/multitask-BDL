#!/usr/bin/env python3
"""wasserstein_FC molecular, latent-branch, and treatment-interaction analysis.

This program evaluates saved PathNoUDP and PathClin DGP artifacts.  It never
fits or updates a DGP.  Molecular features are set to prespecified LOW/HIGH
states for the same held-out patient, all counterfactual scenarios are sampled
jointly through both GP layers, and final event probabilities are recalibrated
with cancer-specific, fixed-slope intercepts learned from outer-training OOF
predictions.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import itertools
import json
import math
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Some constrained batch environments cannot allocate OpenMP shared memory at
# import time. Respect a caller's setting, otherwise keep the preflight/import
# path single-threaded; ``--torch-threads`` controls actual sampling work.
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    from linear_operator.utils.warnings import NumericalWarning

    warnings.filterwarnings(
        "ignore",
        message=r"A not p\.d\., added jitter.*",
        category=NumericalWarning,
    )
except Exception:
    pass


try:  # Support both ``python file.py`` and ``python -m`` execution.
    from .wasserstein_FC_io import (
        CANCERS,
        GROUP_TO_CANCER,
        Config,
        artifact_path,
        build_units,
        cancer_codes,
        config_to_manifest,
        load_artifact,
        load_config,
        load_model_module,
        oof_path,
        patient_ids,
        stable_seed,
        treatment_columns,
    )
    from .wasserstein_FC_calibration import fit_all_horizons, get_gamma
    from .wasserstein_FC_sampling import (
        joint_latent_samples,
        joint_output_samples,
        scores_to_calibrated_event_probabilities,
    )
except ImportError:  # pragma: no cover - direct-script invocation
    from wasserstein_FC_io import (
        CANCERS,
        GROUP_TO_CANCER,
        Config,
        artifact_path,
        build_units,
        cancer_codes,
        config_to_manifest,
        load_artifact,
        load_config,
        load_model_module,
        oof_path,
        patient_ids,
        stable_seed,
        treatment_columns,
    )
    from wasserstein_FC_calibration import fit_all_horizons, get_gamma
    from wasserstein_FC_sampling import (
        joint_latent_samples,
        joint_output_samples,
        scores_to_calibrated_event_probabilities,
    )


HORIZONS = (18, 24, 36)
POSTERIOR_DRAWS = 256
PATIENT_CHUNK = 64
PREDICTION_SCALE = (
    "cancer_specific_fixed_slope_intercept_recalibrated_PFI_event_probability"
)
CALIBRATION_KIND = "cancer_specific_intercept_fixed_slope_1"
PROFILE_MIN_N = 12
PROFILE_MIN_EVENTS = 3
PROFILE_SUPPORT_RULE = "outer_train_n>=12_and_PFI_events>=3"
ACTIVE_PAIR_SUPPORT_RULE = "both_active_profiles_outer_train_n>=12_and_PFI_events>=3"

TX_NAMES = {
    "TX_CHEMO": "Chemotherapy",
    "TX_HORMONE": "Hormone therapy",
    "TX_RADIATION": "Radiotherapy",
    "TX_PHARMACEUTICAL": "Pharmaceutical therapy",
}
TX_CODES = {
    "TX_CHEMO": "C",
    "TX_HORMONE": "H",
    "TX_RADIATION": "R",
    "TX_PHARMACEUTICAL": "P",
}


PATIENT_FEATURE_COLUMNS = [
    "analysis_config", "seed", "cancer", "patient_id",
    "observed_treatment_profile_code", "observed_treatment_profile",
    "feature", "feature_type", "inclusion_type", "selection_origins",
    "source_feature_names", "source_columns", "model_input_scale",
    "contrast_definition", "feature_low_raw", "feature_high_raw", "horizon",
    "W_feature", "Delta_feature",
    "risk_low_mean", "risk_low_median", "risk_low_sd", "risk_low_p05", "risk_low_p95",
    "risk_high_mean", "risk_high_median", "risk_high_sd", "risk_high_p05", "risk_high_p95",
    "posterior_draws", "prediction_scale", "calibration_kind", "calibration_intercept",
]

PATIENT_BRANCH_COLUMNS = [
    "analysis_config", "seed", "cancer", "patient_id",
    "observed_treatment_profile_code", "observed_treatment_profile",
    "feature", "feature_type", "inclusion_type", "selection_origins",
    "source_feature_names", "source_columns", "model_input_scale",
    "contrast_definition", "feature_low_raw", "feature_high_raw", "horizon",
    "W_feature", "W_shared_low_anchor", "W_shared_high_anchor", "W_shared",
    "W_private_low_anchor", "W_private_high_anchor", "W_private", "branch_delta",
    "shared_signed_low_anchor", "shared_signed_high_anchor", "shared_signed_change",
    "private_signed_low_anchor", "private_signed_high_anchor", "private_signed_change",
    "nonadditivity_residual", "posterior_draws", "prediction_scale",
    "calibration_kind", "calibration_intercept",
]

PATIENT_INTERACTION_COLUMNS = [
    "analysis_config", "seed", "cancer", "patient_id",
    "observed_treatment_profile_code", "observed_treatment_profile",
    "treatment_A_code", "treatment_A", "treatment_A_vector",
    "treatment_B_code", "treatment_B", "treatment_B_vector",
    "pair_orientation_rule", "feature", "feature_type", "inclusion_type",
    "selection_origins", "source_feature_names", "source_columns", "model_input_scale",
    "contrast_definition", "feature_low_raw", "feature_high_raw", "horizon",
    "psi_mean", "psi_median", "psi_sd", "psi_p05", "psi_p95",
    "Pr_psi_gt_0", "Pr_psi_lt_0", "abs_mean_psi",
    "delta_low_mean", "delta_low_median", "delta_low_sd", "delta_low_p05", "delta_low_p95",
    "delta_high_mean", "delta_high_median", "delta_high_sd", "delta_high_p05", "delta_high_p95",
    "treatment_contrast_sign_reversal",
    "risk_A_low_mean", "risk_A_low_median", "risk_A_low_sd", "risk_A_low_p05", "risk_A_low_p95",
    "risk_A_high_mean", "risk_A_high_median", "risk_A_high_sd", "risk_A_high_p05", "risk_A_high_p95",
    "risk_B_low_mean", "risk_B_low_median", "risk_B_low_sd", "risk_B_low_p05", "risk_B_low_p95",
    "risk_B_high_mean", "risk_B_high_median", "risk_B_high_sd", "risk_B_high_p05", "risk_B_high_p95",
    "posterior_draws", "prediction_scale", "calibration_kind", "calibration_intercept",
]


class FrameAppender:
    """Append frames to a CSV or gzip CSV without retaining the full table."""

    def __init__(self, path: Path, columns: list[str]):
        self.path = path
        self.columns = list(columns)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".gz":
            self.handle = gzip.open(path, "wt", encoding="utf-8", newline="")
        else:
            self.handle = open(path, "w", encoding="utf-8", newline="")
        self.first = True

    def append(self, frame: pd.DataFrame) -> None:
        if frame is None or frame.empty:
            return
        missing = [c for c in self.columns if c not in frame.columns]
        if missing:
            raise KeyError(f"Missing output columns for {self.path.name}: {missing}")
        frame.loc[:, self.columns].to_csv(
            self.handle, index=False, header=self.first, float_format="%.8g"
        )
        self.first = False

    def close(self) -> None:
        if self.first:
            pd.DataFrame(columns=self.columns).to_csv(self.handle, index=False)
        self.handle.close()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def profile_metadata(
    values: Iterable[int], bases: Iterable[str]
) -> tuple[str, str, str]:
    values = tuple(int(x) for x in values)
    bases = tuple(str(x) for x in bases)
    active_bases = [b for b, value in zip(bases, values) if value]
    code = "+".join(TX_CODES[b] for b in active_bases) if active_bases else "None"
    label = " + ".join(TX_NAMES[b] for b in active_bases) if active_bases else "No recorded systemic or radiation treatment"
    vector = ";".join(f"{b}={value}" for b, value in zip(bases, values))
    return code, label, vector


def profile_is_supported(n_train: int, n_train_events: int) -> bool:
    """Apply the treatment-profile eligibility rule."""

    return int(n_train) >= PROFILE_MIN_N and int(n_train_events) >= PROFILE_MIN_EVENTS


def active_pair_indices(profiles: list[dict[str, object]]) -> list[tuple[int, int]]:
    """Return only active-versus-active profile pairs, never a none-treatment pair."""

    return [
        (left, right)
        for left, right in itertools.combinations(range(len(profiles)), 2)
        if bool(profiles[left]["active"]) and bool(profiles[right]["active"])
    ]


def profiles_from_matrix(
    raw: np.ndarray, tx_indices: list[int], tx_bases: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    values = (np.asarray(raw[:, tx_indices], dtype=float) > 0.5).astype(int)
    codes, labels, vectors = [], [], []
    mapping: dict[str, np.ndarray] = {}
    for row in values:
        code, label, vector = profile_metadata(row, tx_bases)
        codes.append(code)
        labels.append(label)
        vectors.append(vector)
        mapping[code] = np.asarray(row, dtype=np.int8)
    return (
        np.asarray(codes, dtype=object),
        np.asarray(labels, dtype=object),
        np.asarray(vectors, dtype=object),
        mapping,
    )


def parse_selection_origin(name: str) -> tuple[str, str | None]:
    prefix = str(name).split("::", 1)[0]
    if prefix.endswith("_shared"):
        return "shared_selection", None
    for cancer in CANCERS:
        if prefix.endswith(f"_specific_{cancer}"):
            return "cancer_specific_selection", cancer
    if prefix == "tmb":
        return "design_included", None
    return "other", None


def canonical_name(raw_name: str, feature_type: str) -> str:
    suffix = str(raw_name).split("::", 1)[-1].strip()
    if feature_type == "mutation":
        return suffix.upper()
    if feature_type == "tmb":
        return "TMB"
    return suffix


def canonical_units_for_cancer(
    units: pd.DataFrame, cancer: str, molecular_types: tuple[str, ...]
) -> pd.DataFrame:
    rows = []
    for row in units.itertuples(index=False):
        ftype = str(row.feature_type)
        if ftype not in molecular_types:
            continue
        origin, specific_cancer = parse_selection_origin(str(row.feature))
        if specific_cancer is not None and specific_cancer != cancer:
            continue
        rows.append({
            "feature": canonical_name(str(row.feature), ftype),
            "feature_type": ftype,
            "source_feature_name": str(row.feature),
            "source_columns": tuple(int(x) for x in row.columns),
            "selection_origin": origin,
            "specific_cancer": specific_cancer,
            "model_input_scale": "LOG1P_TMB" if ftype == "tmb" else "model_input_raw_scale",
            "inclusion_type": "design_included" if ftype == "tmb" else "screened",
        })
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    combined = []
    for (feature, ftype), group in raw.groupby(["feature", "feature_type"], sort=True):
        columns = tuple(sorted({i for value in group.source_columns for i in value}))
        combined.append({
            "feature": feature,
            "feature_type": ftype,
            "columns": columns,
            "source_feature_names": ";".join(sorted(group.source_feature_name.unique())),
            "source_columns": ";".join(map(str, columns)),
            "selection_origins": ";".join(sorted(group.selection_origin.unique())),
            "specific_cancers": ";".join(sorted(x for x in group.specific_cancer.dropna().unique())),
            "model_input_scale": str(group.model_input_scale.iloc[0]),
            "inclusion_type": str(group.inclusion_type.iloc[0]),
        })
    return pd.DataFrame(combined).sort_values(["feature_type", "feature"]).reset_index(drop=True)


def contrast_values(
    unit,
    train_raw: np.ndarray,
    train_cancer_mask: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, str, dict[str, object]]:
    columns = np.asarray(unit.columns, dtype=int)
    values = np.asarray(train_raw[np.asarray(train_cancer_mask), :][:, columns], dtype=float)
    qc: dict[str, object] = {
        "n_training_patients": int(values.shape[0]),
        "n_columns": int(values.shape[1]),
        "duplicate_column_max_abs_difference": 0.0,
        "evaluable": True,
        "non_evaluable_reason": "",
    }
    if values.size == 0 or not np.isfinite(values).all():
        qc.update(evaluable=False, non_evaluable_reason="missing_or_nonfinite_training_values")
        return None, None, "", qc
    if values.shape[1] > 1:
        diff = float(np.max(np.abs(values - values[:, [0]])))
        qc["duplicate_column_max_abs_difference"] = diff
        if diff > 1e-8:
            raise ValueError(
                f"Canonical feature {unit.feature} has non-identical duplicate model columns"
            )
    if unit.feature_type == "mutation":
        unique = np.unique(values)
        if not np.all(np.isin(unique, [0.0, 1.0])):
            raise ValueError(f"Mutation {unit.feature} is not exactly binary: {unique[:10]}")
        if not ({0.0, 1.0}.issubset(set(unique.tolist()))):
            qc.update(evaluable=False, non_evaluable_reason="mutation_lacks_both_states_in_outer_training")
            return None, None, "0_vs_1", qc
        low = np.zeros(len(columns), dtype=float)
        high = np.ones(len(columns), dtype=float)
        definition = "wild_type_0_vs_mutated_1"
    else:
        q25 = np.quantile(values, 0.25, axis=0)
        q75 = np.quantile(values, 0.75, axis=0)
        low, high = np.asarray(q25, float), np.asarray(q75, float)
        definition = "outer_training_cancer_specific_Q25_vs_Q75"
        if np.any(np.abs(high - low) <= 1e-12):
            qc.update(evaluable=False, non_evaluable_reason="degenerate_Q25_Q75")
            return low, high, definition, qc
    return low, high, definition, qc


def posterior_summary(x: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": np.mean(x, axis=0),
        "median": np.median(x, axis=0),
        "sd": np.std(x, axis=0, ddof=1),
        "p05": np.quantile(x, 0.05, axis=0),
        "p95": np.quantile(x, 0.95, axis=0),
    }


def empirical_w1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return np.mean(np.abs(np.sort(a, axis=0) - np.sort(b, axis=0)), axis=0)


def repeat_dict(base: dict[str, object], n: int) -> dict[str, np.ndarray]:
    return {key: np.repeat(value, n) for key, value in base.items()}


def set_columns_scaled(
    matrix: np.ndarray,
    columns: Iterable[int],
    raw_values: Iterable[float],
    mu: np.ndarray,
    sd: np.ndarray,
) -> None:
    columns = np.asarray(tuple(columns), dtype=int)
    values = np.asarray(tuple(raw_values), dtype=float)
    matrix[:, columns] = (values - mu[columns]) / sd[columns]


def build_joint_scenarios(
    base_scaled: np.ndarray,
    feature_columns: tuple[int, ...],
    feature_low: np.ndarray,
    feature_high: np.ndarray,
    supported_profiles: list[dict[str, object]],
    profile_tx_indices: list[int],
    excluded_tx_indices: list[int],
    mu: np.ndarray,
    sd: np.ndarray,
) -> np.ndarray:
    variants = []
    for feature_values in (feature_low, feature_high):
        x = base_scaled.copy()
        set_columns_scaled(x, feature_columns, feature_values, mu, sd)
        variants.append(x)
    for profile in supported_profiles:
        vector = np.asarray(profile["values"], dtype=float)
        for feature_values in (feature_low, feature_high):
            x = base_scaled.copy()
            set_columns_scaled(x, feature_columns, feature_values, mu, sd)
            if excluded_tx_indices:
                set_columns_scaled(
                    x, excluded_tx_indices, np.zeros(len(excluded_tx_indices)), mu, sd
                )
            set_columns_scaled(x, profile_tx_indices, vector, mu, sd)
            variants.append(x)
    return np.asarray(variants, dtype=np.float32)


def sample_feature_scenarios(
    model,
    x_variants: np.ndarray,
    group_index: int,
    draws: int,
    random_seed: int,
):
    import torch

    torch.manual_seed(int(random_seed))
    x = torch.as_tensor(x_variants, dtype=torch.float32)
    shared, private = joint_latent_samples(model, x, group_index, draws)
    sl, sh = shared[:, 0], shared[:, 1]
    pl, ph = private[:, 0], private[:, 1]
    output_latents = [
        torch.cat([sl, pl], dim=-1),
        torch.cat([sh, pl], dim=-1),
        torch.cat([sl, ph], dim=-1),
        torch.cat([sh, ph], dim=-1),
    ]
    if int(x.shape[0]) > 2:
        direct = torch.cat([shared[:, 2:], private[:, 2:]], dim=-1)
        output_latents.extend([direct[:, j] for j in range(int(direct.shape[1]))])
    latent_tensor = torch.stack(output_latents, dim=1)
    return joint_output_samples(model, latent_tensor, group_index)


def ranking_scopes(config_name: str, feature_type: str) -> list[str]:
    if config_name == "PathNoUDP":
        scopes = ["molecular_all"]
        if feature_type == "mutation":
            scopes.append("mutation_only")
        if feature_type == "pathway":
            scopes.append("pathway_only")
        return scopes
    return ["pathway_only"] if feature_type == "pathway" else []


def add_ranks(
    frame: pd.DataFrame,
    group_columns: list[str],
    score_column: str,
    rank_column: str,
    percentile_column: str,
) -> pd.DataFrame:
    if frame.empty:
        frame[rank_column] = pd.Series(dtype=float)
        frame[percentile_column] = pd.Series(dtype=float)
        return frame
    pieces = []
    for _, group in frame.groupby(group_columns, sort=False, dropna=False):
        value = group.copy()
        value[rank_column] = value[score_column].rank(method="average", ascending=False)
        if len(value) == 1:
            value[percentile_column] = 1.0
        else:
            value[percentile_column] = 1.0 - (value[rank_column] - 1.0) / (len(value) - 1.0)
        pieces.append(value)
    return pd.concat(pieces, ignore_index=True)


def run_seed(
    config: Config,
    seed: int,
    output_root: Path,
    draws: int,
    chunk_size: int,
    max_features: int | None = None,
    cancers: tuple[str, ...] = CANCERS,
    torch_threads: int = 4,
    write_patient_level: bool = False,
) -> dict[str, object]:
    import torch

    torch.set_num_threads(max(1, int(torch_threads)))
    config_name = config.name
    module = load_model_module(config, output_root / "_artifact_load_runtime" / config_name)
    artifact, artifact_path = load_artifact(config, seed, module)
    model = artifact["model"].eval()
    units = build_units(artifact, module)

    final_dir = output_root / config_name / f"seed{seed}"
    work_dir = output_root / config_name / f"seed{seed}.inprogress"
    if final_dir.exists():
        print(f"[{config_name} seed {seed}] already complete; skipping", flush=True)
        return {"status": "already_complete", "path": str(final_dir)}
    if work_dir.exists():
        raise FileExistsError(
            f"Incomplete partition exists: {work_dir}. Inspect it before rerunning."
        )
    work_dir.mkdir(parents=True, exist_ok=False)

    feature_writer = (
        FrameAppender(work_dir / "patient_feature.csv.gz", PATIENT_FEATURE_COLUMNS)
        if write_patient_level else None
    )
    branch_writer = (
        FrameAppender(work_dir / "patient_branch.csv.gz", PATIENT_BRANCH_COLUMNS)
        if write_patient_level else None
    )
    interaction_writer = (
        FrameAppender(work_dir / "patient_interactions.csv.gz", PATIENT_INTERACTION_COLUMNS)
        if write_patient_level else None
    )

    train_raw = np.asarray(artifact["Xtr_np_sub"], dtype=np.float32)
    test_raw = np.asarray(artifact["Xte_np_sub"], dtype=np.float32)
    test_scaled = np.asarray(artifact["Xte_t"].detach().cpu().numpy(), dtype=np.float32)
    mu = np.asarray(artifact["standardize_mu"], dtype=np.float32).reshape(-1)
    sd = np.asarray(artifact["standardize_sd"], dtype=np.float32).reshape(-1)
    if np.any(~np.isfinite(sd)) or np.any(sd <= 0):
        raise ValueError("Saved standardization SD contains nonpositive/nonfinite values")
    feature_names = [str(x) for x in artifact["feat_names"]]
    tx_indices, tx_bases = treatment_columns(feature_names)
    train_cancers = cancer_codes(artifact, "tr")
    test_cancers = cancer_codes(artifact, "te")
    profile_definition_by_cancer: dict[str, dict[str, object]] = {}
    for cancer in cancers:
        # The paper tables retain all four recorded treatment indicators in
        # every cancer.  We do not remove hormone therapy outside breast.
        profile_indices = list(tx_indices)
        profile_bases = list(tx_bases)
        excluded_indices: list[int] = []
        train_profiles = profiles_from_matrix(train_raw, profile_indices, profile_bases)
        test_profiles = profiles_from_matrix(test_raw, profile_indices, profile_bases)
        profile_definition_by_cancer[cancer] = {
            "indices": profile_indices,
            "bases": profile_bases,
            "excluded_indices": excluded_indices,
            "train_codes": train_profiles[0],
            "train_labels": train_profiles[1],
            "train_vectors": train_profiles[2],
            "test_codes": test_profiles[0],
            "test_labels": test_profiles[1],
            "test_vectors": test_profiles[2],
        }
    train_events = np.asarray(artifact["e_tr"], dtype=int)
    ids = patient_ids(artifact["meta_te"].reset_index(drop=True), len(test_raw))
    if set(patient_ids(artifact["meta_tr"].reset_index(drop=True), len(train_raw))).intersection(ids):
        raise ValueError("Outer-training and held-out patient IDs overlap")

    calibration_input_path = oof_path(config, seed)
    calibration_bundle = fit_all_horizons(
        calibration_input_path,
        artifact,
        horizons=HORIZONS,
        config=config_name,
        seed=seed,
        artifact_path=artifact_path,
    )
    calibration = {
        cancer: {
            horizon: get_gamma(calibration_bundle, horizon, cancer)
            for horizon in HORIZONS
        }
        for cancer in CANCERS
    }
    calibration_qc = pd.DataFrame(calibration_bundle["qc"]["rows"])
    calibration_qc["analysis_config"] = config_name
    calibration_qc["seed"] = seed
    calibration_qc["artifact_path"] = str(artifact_path)
    calibration_qc["artifact_sha256"] = sha256_file(artifact_path)
    calibration_qc["oof_path"] = str(calibration_input_path)
    calibration_qc["oof_sha256"] = sha256_file(calibration_input_path)
    calibration_qc.to_csv(work_dir / "calibration_qc.csv", index=False)

    profile_support_rows: list[dict[str, object]] = []
    support_by_cancer: dict[str, list[dict[str, object]]] = {}
    observed_profile_support: dict[tuple[str, str], dict[str, object]] = {}
    for cancer in cancers:
        mask = train_cancers == cancer
        definition = profile_definition_by_cancer[cancer]
        train_codes = np.asarray(definition["train_codes"], dtype=object)
        profile_tx_indices = list(definition["indices"])
        profile_tx_bases = list(definition["bases"])
        profiles = []
        for code in sorted(np.unique(train_codes[mask]).tolist()):
            cell = mask & (train_codes == code)
            n = int(cell.sum())
            events = int(train_events[cell].sum())
            idx = int(np.where(cell)[0][0])
            values = (train_raw[idx, profile_tx_indices] > 0.5).astype(np.int8)
            code2, label, vector = profile_metadata(values, profile_tx_bases)
            assert code2 == code
            supported = profile_is_supported(n, events)
            active = bool(np.sum(values) > 0)
            row = {
                "analysis_config": config_name, "seed": seed, "record_type": "profile",
                "cancer": cancer, "treatment_profile_code": code,
                "treatment_profile": label, "treatment_profile_vector": vector,
                "n_train": n, "n_train_events": events, "active_profile": active,
                "supported": supported,
                "support_rule": PROFILE_SUPPORT_RULE,
            }
            profile_support_rows.append(row)
            observed_profile_support[(cancer, code)] = row
            if supported:
                profiles.append({"code": code, "label": label, "vector": vector, "values": values,
                                 "n_train": n, "n_events": events, "active": active})
        profiles = sorted(profiles, key=lambda x: str(x["code"]))
        support_by_cancer[cancer] = profiles
        active_profiles = [profile for profile in profiles if bool(profile["active"])]
        for a, b in itertools.combinations(active_profiles, 2):
            profile_support_rows.append({
                "analysis_config": config_name, "seed": seed, "record_type": "active_pair",
                "cancer": cancer, "treatment_profile_code": f"{a['code']}__vs__{b['code']}",
                "treatment_profile": f"{a['label']} vs {b['label']}",
                "treatment_profile_vector": f"A:{a['vector']}|B:{b['vector']}",
                "n_train": int(a["n_train"]) + int(b["n_train"]),
                "n_train_events": int(a["n_events"]) + int(b["n_events"]),
                "active_profile": True,
                "supported": True,
                "support_rule": ACTIVE_PAIR_SUPPORT_RULE,
            })
    pd.DataFrame(profile_support_rows).to_csv(work_dir / "treatment_support.csv", index=False)

    mapping_rows: list[dict[str, object]] = []
    quantile_rows: list[dict[str, object]] = []
    seed_feature_rows: list[dict[str, object]] = []
    seed_branch_rows: list[dict[str, object]] = []
    seed_profile_rows: list[dict[str, object]] = []
    seed_interaction_rows: list[dict[str, object]] = []
    qc_checks: list[dict[str, object]] = []
    probability_min = math.inf
    probability_max = -math.inf
    nonfinite_probabilities = 0

    for cancer in cancers:
        cancer_units = canonical_units_for_cancer(units, cancer, config.molecular_types)
        if max_features is not None:
            cancer_units = cancer_units.iloc[: int(max_features)].copy()
        train_mask = train_cancers == cancer
        test_indices = np.where(test_cancers == cancer)[0]
        if not len(test_indices):
            continue
        group_names = [GROUP_TO_CANCER.get(str(x), str(x).upper()) for x in artifact["groups"]]
        group_index = group_names.index(cancer)
        cancer_ids = ids[test_indices]
        cancer_test_scaled = test_scaled[test_indices]
        definition = profile_definition_by_cancer[cancer]
        profile_tx_indices = list(definition["indices"])
        excluded_tx_indices = list(definition["excluded_indices"])
        test_codes = np.asarray(definition["test_codes"], dtype=object)
        test_labels = np.asarray(definition["test_labels"], dtype=object)
        observed_codes = test_codes[test_indices]
        observed_labels = test_labels[test_indices]
        supported_profiles = support_by_cancer[cancer]
        supported_pairs = active_pair_indices(supported_profiles)
        # Exact output-GP sampling builds one covariance matrix over every
        # scenario-patient point for each posterior draw.  Keep that event axis
        # near 256 points so active-profile-rich BRCA cells remain bounded in
        # memory while every scenario for a patient stays in the same call.
        output_scenario_count = 4 + 2 * len(supported_profiles)
        effective_chunk_size = min(
            int(chunk_size), max(4, 256 // max(output_scenario_count, 1))
        )
        qc_checks.append({
            "check": f"joint_chunk_{cancer}", "status": "pass",
            "detail": (
                f"observed_profiles={len(supported_profiles)};"
                f"output_scenarios={output_scenario_count};"
                f"effective_patient_chunk={effective_chunk_size}"
            ),
        })

        for feature_number, unit in enumerate(cancer_units.itertuples(index=False), start=1):
            low, high, contrast_definition, contrast_qc = contrast_values(
                unit, train_raw, train_mask
            )
            mapping = {
                "analysis_config": config_name, "seed": seed, "cancer": cancer,
                "feature": unit.feature, "feature_type": unit.feature_type,
                "inclusion_type": unit.inclusion_type, "selected": True,
                "evaluable": bool(contrast_qc["evaluable"]),
                "non_evaluable_reason": contrast_qc["non_evaluable_reason"],
                "source_feature_names": unit.source_feature_names,
                "source_columns": unit.source_columns,
                "selection_origins": unit.selection_origins,
                "specific_cancers": unit.specific_cancers,
                "model_input_scale": unit.model_input_scale,
                "selection_origin_is_not_latent_routing": True,
            }
            mapping_rows.append(mapping)
            low_text = "" if low is None else ";".join(f"{x:.12g}" for x in low)
            high_text = "" if high is None else ";".join(f"{x:.12g}" for x in high)
            quantile_rows.append({
                **mapping,
                "contrast_definition": contrast_definition,
                "feature_low_raw": low_text,
                "feature_high_raw": high_text,
                **contrast_qc,
            })
            if not bool(contrast_qc["evaluable"]):
                continue

            overall_collect = {
                h: defaultdict(list) for h in HORIZONS
            }
            branch_collect = {
                h: defaultdict(list) for h in HORIZONS
            }
            interaction_collect: dict[tuple[int, int, int], defaultdict[str, list]] = {
                (h, a, b): defaultdict(list)
                for h in HORIZONS for a, b in supported_pairs
            }
            profile_collect: dict[tuple[int, int], defaultdict[str, list]] = {
                (h, p): defaultdict(list)
                for h in HORIZONS for p in range(len(supported_profiles))
            }

            for chunk_start in range(0, len(test_indices), effective_chunk_size):
                chunk_stop = min(chunk_start + effective_chunk_size, len(test_indices))
                local = slice(chunk_start, chunk_stop)
                base = cancer_test_scaled[local]
                scenarios = build_joint_scenarios(
                    base_scaled=base,
                    feature_columns=unit.columns,
                    feature_low=low,
                    feature_high=high,
                    supported_profiles=supported_profiles,
                    profile_tx_indices=profile_tx_indices,
                    excluded_tx_indices=excluded_tx_indices,
                    mu=mu,
                    sd=sd,
                )
                allowed_feature = set(int(x) for x in unit.columns)
                outside = [i for i in range(base.shape[1]) if i not in allowed_feature]
                max_outside_feature = float(
                    max(np.max(np.abs(scenarios[0][:, outside] - base[:, outside])),
                        np.max(np.abs(scenarios[1][:, outside] - base[:, outside])))
                ) if outside else 0.0
                if max_outside_feature != 0.0:
                    raise AssertionError("Non-target input changed in feature-only contrast")
                allowed_tx = allowed_feature.union(tx_indices)
                outside_tx = [i for i in range(base.shape[1]) if i not in allowed_tx]
                if len(scenarios) > 2 and outside_tx:
                    max_outside_tx = float(
                        np.max(
                            np.abs(
                                scenarios[2:][:, :, outside_tx]
                                - base[None, :, outside_tx]
                            )
                        )
                    )
                    if max_outside_tx != 0.0:
                        raise AssertionError("Non-feature/non-treatment input changed in interaction scenario")

                scores = sample_feature_scenarios(
                    model=model,
                    x_variants=scenarios,
                    group_index=group_index,
                    draws=draws,
                    random_seed=stable_seed(
                        "wasserstein_FC", config_name, seed, cancer, unit.feature,
                        chunk_start, draws
                    ),
                )
                probability_by_horizon = scores_to_calibrated_event_probabilities(
                    scores,
                    artifact["alpha_dt"],
                    artifact["dt_edges"],
                    cancer=cancer,
                    calibration_intercepts=calibration,
                    horizons=HORIZONS,
                )
                for horizon in HORIZONS:
                    gamma = float(calibration[cancer][horizon])
                    probs_tensor = probability_by_horizon[horizon]
                    probs = np.asarray(
                        probs_tensor.detach().cpu().numpy()
                        if hasattr(probs_tensor, "detach") else probs_tensor,
                        dtype=np.float32,
                    )
                    probability_min = min(probability_min, float(np.min(probs)))
                    probability_max = max(probability_max, float(np.max(probs)))
                    nonfinite_probabilities += int(np.size(probs) - np.isfinite(probs).sum())

                    ll, hl, lh, hh = (probs[:, i, :] for i in range(4))
                    w_feature = empirical_w1(ll, hh)
                    delta_feature = np.mean(hh - ll, axis=0)

                    common = {
                        "analysis_config": config_name, "seed": seed, "cancer": cancer,
                        "feature": unit.feature, "feature_type": unit.feature_type,
                        "inclusion_type": unit.inclusion_type,
                        "selection_origins": unit.selection_origins,
                        "source_feature_names": unit.source_feature_names,
                        "source_columns": unit.source_columns,
                        "model_input_scale": unit.model_input_scale,
                        "contrast_definition": contrast_definition,
                        "feature_low_raw": low_text, "feature_high_raw": high_text,
                        "horizon": horizon, "posterior_draws": draws,
                        "prediction_scale": PREDICTION_SCALE,
                        "calibration_kind": CALIBRATION_KIND,
                        "calibration_intercept": gamma,
                    }
                    n_chunk = chunk_stop - chunk_start
                    if feature_writer is not None:
                        low_summary = posterior_summary(ll)
                        high_summary = posterior_summary(hh)
                        feature_frame = pd.DataFrame({
                            **repeat_dict(common, n_chunk),
                            "patient_id": cancer_ids[local],
                            "observed_treatment_profile_code": observed_codes[local],
                            "observed_treatment_profile": observed_labels[local],
                            "W_feature": w_feature, "Delta_feature": delta_feature,
                            **{f"risk_low_{key}": value for key, value in low_summary.items()},
                            **{f"risk_high_{key}": value for key, value in high_summary.items()},
                        })
                        feature_writer.append(feature_frame)
                    for key, value in {
                        "W_feature": w_feature,
                        "Delta_feature": delta_feature,
                    }.items():
                        overall_collect[horizon][key].append(np.asarray(value))

                    ws_low = empirical_w1(ll, hl)
                    ws_high = empirical_w1(lh, hh)
                    ws = 0.5 * (ws_low + ws_high)
                    wp_low = empirical_w1(ll, lh)
                    wp_high = empirical_w1(hl, hh)
                    wp = 0.5 * (wp_low + wp_high)
                    ss_low = np.mean(hl - ll, axis=0)
                    ss_high = np.mean(hh - lh, axis=0)
                    ss = 0.5 * (ss_low + ss_high)
                    ps_low = np.mean(lh - ll, axis=0)
                    ps_high = np.mean(hh - hl, axis=0)
                    ps = 0.5 * (ps_low + ps_high)
                    branch_delta = ws - wp
                    if branch_writer is not None:
                        branch_frame = pd.DataFrame({
                            **repeat_dict(common, n_chunk),
                            "patient_id": cancer_ids[local],
                            "observed_treatment_profile_code": observed_codes[local],
                            "observed_treatment_profile": observed_labels[local],
                            "W_feature": w_feature,
                            "W_shared_low_anchor": ws_low,
                            "W_shared_high_anchor": ws_high,
                            "W_shared": ws,
                            "W_private_low_anchor": wp_low,
                            "W_private_high_anchor": wp_high,
                            "W_private": wp,
                            "branch_delta": branch_delta,
                            "shared_signed_low_anchor": ss_low,
                            "shared_signed_high_anchor": ss_high,
                            "shared_signed_change": ss,
                            "private_signed_low_anchor": ps_low,
                            "private_signed_high_anchor": ps_high,
                            "private_signed_change": ps,
                            "nonadditivity_residual": w_feature - (ws + wp),
                        })
                        branch_writer.append(branch_frame)
                    for key, value in {
                        "W_feature": w_feature, "W_shared": ws, "W_private": wp,
                        "branch_delta": branch_delta, "shared_signed_change": ss,
                        "private_signed_change": ps,
                    }.items():
                        branch_collect[horizon][key].append(np.asarray(value))

                    if supported_profiles:
                        direct = probs[:, 4:, :]
                        for p in range(len(supported_profiles)):
                            p_low = direct[:, 2 * p]
                            p_high = direct[:, 2 * p + 1]
                            collect = profile_collect[(horizon, p)]
                            collect["W_feature"].append(empirical_w1(p_low, p_high))
                            collect["Delta_feature"].append(
                                np.mean(p_high - p_low, axis=0)
                            )
                        for a, b in supported_pairs:
                            a_low, a_high = direct[:, 2 * a], direct[:, 2 * a + 1]
                            b_low, b_high = direct[:, 2 * b], direct[:, 2 * b + 1]
                            delta_low = b_low - a_low
                            delta_high = b_high - a_high
                            psi = delta_high - delta_low
                            pa, pb = supported_profiles[a], supported_profiles[b]
                            psi_mean = np.mean(psi, axis=0)
                            delta_low_mean = np.mean(delta_low, axis=0)
                            delta_high_mean = np.mean(delta_high, axis=0)
                            pr_gt = np.mean(psi > 0.0, axis=0)
                            pr_lt = np.mean(psi < 0.0, axis=0)
                            reversal = (
                                np.sign(delta_low_mean) * np.sign(delta_high_mean) < 0
                            )
                            if interaction_writer is not None:
                                summaries = {
                                    "psi": posterior_summary(psi),
                                    "delta_low": posterior_summary(delta_low),
                                    "delta_high": posterior_summary(delta_high),
                                    "risk_A_low": posterior_summary(a_low),
                                    "risk_A_high": posterior_summary(a_high),
                                    "risk_B_low": posterior_summary(b_low),
                                    "risk_B_high": posterior_summary(b_high),
                                }
                                pair_common = {
                                    **common,
                                    "treatment_A_code": pa["code"], "treatment_A": pa["label"],
                                    "treatment_A_vector": pa["vector"],
                                    "treatment_B_code": pb["code"], "treatment_B": pb["label"],
                                    "treatment_B_vector": pb["vector"],
                                    "pair_orientation_rule": "lexicographic profile code; B minus A",
                                }
                                interaction_frame = pd.DataFrame({
                                    **repeat_dict(pair_common, n_chunk),
                                    "patient_id": cancer_ids[local],
                                    "observed_treatment_profile_code": observed_codes[local],
                                    "observed_treatment_profile": observed_labels[local],
                                    **{f"psi_{key}": value for key, value in summaries["psi"].items()},
                                    "Pr_psi_gt_0": pr_gt, "Pr_psi_lt_0": pr_lt,
                                    "abs_mean_psi": np.abs(psi_mean),
                                    **{f"delta_low_{key}": value for key, value in summaries["delta_low"].items()},
                                    **{f"delta_high_{key}": value for key, value in summaries["delta_high"].items()},
                                    "treatment_contrast_sign_reversal": reversal,
                                    **{
                                        f"{prefix}_{key}": value
                                        for prefix in ("risk_A_low", "risk_A_high", "risk_B_low", "risk_B_high")
                                        for key, value in summaries[prefix].items()
                                    },
                                })
                                interaction_writer.append(interaction_frame)
                            collect = interaction_collect[(horizon, a, b)]
                            for key, value in {
                                "psi_mean": psi_mean,
                                "abs_mean_psi": np.abs(psi_mean),
                                "delta_low_mean": delta_low_mean,
                                "delta_high_mean": delta_high_mean,
                                "Pr_psi_gt_0": pr_gt, "Pr_psi_lt_0": pr_lt,
                                "sign_reversal": reversal.astype(float),
                            }.items():
                                collect[key].append(np.asarray(value))

            for horizon in HORIZONS:
                w = np.concatenate(overall_collect[horizon]["W_feature"])
                d = np.concatenate(overall_collect[horizon]["Delta_feature"])
                base_summary = {
                    "analysis_config": config_name, "seed": seed, "cancer": cancer,
                    "horizon": horizon, "feature": unit.feature,
                    "feature_type": unit.feature_type, "inclusion_type": unit.inclusion_type,
                    "selection_origins": unit.selection_origins,
                    "n_test_patients": len(w),
                    "mean_W_feature": float(np.mean(w)),
                    "sd_W_feature": float(np.std(w, ddof=1)) if len(w) > 1 else 0.0,
                    "median_W_feature": float(np.median(w)),
                    "mean_Delta_feature": float(np.mean(d)),
                    "sd_Delta_feature": float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
                }
                for scope in ranking_scopes(config_name, unit.feature_type):
                    seed_feature_rows.append({**base_summary, "ranking_scope": scope})

                bc = branch_collect[horizon]
                ws = np.concatenate(bc["W_shared"])
                wp = np.concatenate(bc["W_private"])
                bd = np.concatenate(bc["branch_delta"])
                ss = np.concatenate(bc["shared_signed_change"])
                ps = np.concatenate(bc["private_signed_change"])
                seed_branch_rows.append({
                    "analysis_config": config_name, "seed": seed, "cancer": cancer,
                    "horizon": horizon, "feature": unit.feature,
                    "feature_type": unit.feature_type, "inclusion_type": unit.inclusion_type,
                    "n_test_patients": len(ws),
                    "mean_W_feature": float(np.mean(w)),
                    "mean_W_shared": float(np.mean(ws)),
                    "sd_W_shared": float(np.std(ws, ddof=1)) if len(ws) > 1 else 0.0,
                    "mean_W_private": float(np.mean(wp)),
                    "sd_W_private": float(np.std(wp, ddof=1)) if len(wp) > 1 else 0.0,
                    "mean_branch_delta": float(np.mean(bd)),
                    "sd_branch_delta": float(np.std(bd, ddof=1)) if len(bd) > 1 else 0.0,
                    "mean_shared_signed_change": float(np.mean(ss)),
                    "mean_private_signed_change": float(np.mean(ps)),
                    "branch_dominance": (
                        "shared" if float(np.mean(bd)) > 0 else
                        "private" if float(np.mean(bd)) < 0 else "tie"
                    ),
                })

                for p, profile in enumerate(supported_profiles):
                    support = observed_profile_support[(cancer, profile["code"])]
                    profile_w = np.concatenate(
                        profile_collect[(horizon, p)]["W_feature"]
                    )
                    profile_d = np.concatenate(
                        profile_collect[(horizon, p)]["Delta_feature"]
                    )
                    seed_profile_rows.append({
                        "analysis_config": config_name, "seed": seed, "cancer": cancer,
                        "horizon": horizon,
                        "treatment_profile_code": profile["code"],
                        "treatment_profile": profile["label"],
                        "treatment_profile_vector": profile["vector"],
                        "n_train_profile": support["n_train"],
                        "n_train_profile_events": support["n_train_events"],
                        "n_test_profile": len(profile_w), "profile_supported": True,
                        "profile_conditioning": "treatment indicators fixed to exact profile for all held-out cancer patients",
                        "feature": unit.feature, "feature_type": unit.feature_type,
                        "mean_W_feature": float(np.mean(profile_w)),
                        "sd_W_feature": float(np.std(profile_w, ddof=1)) if len(profile_w) > 1 else 0.0,
                        "median_W_feature": float(np.median(profile_w)),
                        "mean_Delta_feature": float(np.mean(profile_d)),
                    })

                for a, b in supported_pairs:
                    collect = interaction_collect[(horizon, a, b)]
                    psi_mean = np.concatenate(collect["psi_mean"])
                    abs_psi = np.concatenate(collect["abs_mean_psi"])
                    delta_low_mean = np.concatenate(collect["delta_low_mean"])
                    delta_high_mean = np.concatenate(collect["delta_high_mean"])
                    gt = np.concatenate(collect["Pr_psi_gt_0"])
                    lt = np.concatenate(collect["Pr_psi_lt_0"])
                    reversal = np.concatenate(collect["sign_reversal"])
                    pa, pb = supported_profiles[a], supported_profiles[b]
                    seed_interaction_rows.append({
                        "analysis_config": config_name, "seed": seed, "cancer": cancer,
                        "horizon": horizon,
                        "treatment_A_code": pa["code"], "treatment_A": pa["label"],
                        "treatment_A_vector": pa["vector"],
                        "treatment_B_code": pb["code"], "treatment_B": pb["label"],
                        "treatment_B_vector": pb["vector"],
                        "n_test_cancer": len(psi_mean),
                        "feature": unit.feature, "feature_type": unit.feature_type,
                        "mean_psi": float(np.mean(psi_mean)),
                        "sd_patient_psi": float(np.std(psi_mean, ddof=1)) if len(psi_mean) > 1 else 0.0,
                        "median_psi": float(np.median(psi_mean)),
                        "mean_abs_psi": float(np.mean(abs_psi)),
                        "mean_delta_low": float(np.mean(delta_low_mean)),
                        "mean_delta_high": float(np.mean(delta_high_mean)),
                        "mean_Pr_psi_gt_0": float(np.mean(gt)),
                        "mean_Pr_psi_lt_0": float(np.mean(lt)),
                        "n_sign_reversals": int(np.sum(reversal)),
                        "proportion_sign_reversal": float(np.mean(reversal)),
                        "direction": "positive" if np.mean(psi_mean) > 0 else "negative" if np.mean(psi_mean) < 0 else "zero",
                    })

            if feature_number % 10 == 0 or feature_number == len(cancer_units):
                print(
                    f"[{config_name} seed {seed}] {cancer}: "
                    f"{feature_number}/{len(cancer_units)} molecular features",
                    flush=True,
                )

    for writer in (feature_writer, branch_writer, interaction_writer):
        if writer is not None:
            writer.close()

    seed_feature = pd.DataFrame(seed_feature_rows)
    seed_feature = add_ranks(
        seed_feature,
        ["analysis_config", "seed", "cancer", "horizon", "ranking_scope"],
        "mean_W_feature", "within_seed_rank", "within_seed_importance_percentile",
    )
    seed_feature.to_csv(work_dir / "seed_feature_summary.csv", index=False)

    pd.DataFrame(seed_branch_rows).to_csv(work_dir / "seed_branch_summary.csv", index=False)
    seed_profile = add_ranks(
        pd.DataFrame(seed_profile_rows),
        ["analysis_config", "seed", "cancer", "horizon", "treatment_profile_code"],
        "mean_W_feature", "within_profile_rank", "within_profile_percentile",
    )
    seed_profile.to_csv(work_dir / "seed_profile_summary.csv", index=False)
    seed_interaction = add_ranks(
        pd.DataFrame(seed_interaction_rows),
        ["analysis_config", "seed", "cancer", "horizon", "treatment_A_code", "treatment_B_code"],
        "mean_abs_psi", "within_pair_magnitude_rank", "within_pair_magnitude_percentile",
    )
    seed_interaction.to_csv(work_dir / "seed_interaction_summary.csv", index=False)
    pd.DataFrame(mapping_rows).to_csv(work_dir / "feature_identity_mapping.csv", index=False)
    pd.DataFrame(quantile_rows).to_csv(work_dir / "quantile_audit.csv", index=False)

    probability_ok = (
        nonfinite_probabilities == 0
        and probability_min >= 0.0
        and probability_max <= 1.0
    )
    qc_checks.extend([
        {"check": "no_DGP_retraining", "status": "pass", "detail": "saved artifacts loaded in eval mode"},
        {"check": "donor_sampling_removed", "status": "pass", "detail": "fixed 0/1 or training Q25/Q75 contrasts only"},
        {"check": "joint_posterior_sampling", "status": "pass", "detail": "all paired scenarios sampled jointly at both GP layers"},
        {"check": "probability_range", "status": "pass" if probability_ok else "fail",
         "detail": f"min={probability_min};max={probability_max};nonfinite={nonfinite_probabilities}"},
        {"check": "horizons", "status": "pass", "detail": ",".join(map(str, HORIZONS))},
        {"check": "calibration_scale", "status": "pass", "detail": PREDICTION_SCALE},
        {"check": "latent_branch_routing", "status": "pass",
         "detail": "all selected inputs enter both branches; attribution uses output-GP hybrids"},
    ])
    qc = {
        "analysis_config": config_name,
        "seed": seed,
        "artifact": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "no_model_retraining": True,
        "donor_sampling_used": False,
        "horizons": list(HORIZONS),
        "posterior_draws": int(draws),
        "patient_level_csvs_written": bool(write_patient_level),
        "patient_chunk": int(chunk_size),
        "prediction_scale": PREDICTION_SCALE,
        "calibration_kind": CALIBRATION_KIND,
        "treatment_profile_support_rule": PROFILE_SUPPORT_RULE,
        "active_treatment_pair_support_rule": ACTIVE_PAIR_SUPPORT_RULE,
        "treatment_profile_minimum_n": PROFILE_MIN_N,
        "treatment_profile_minimum_events": PROFILE_MIN_EVENTS,
        "checks": qc_checks,
        "completed_at_unix": time.time(),
    }
    (work_dir / "qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    if not probability_ok:
        raise RuntimeError("Probability range QC failed")
    work_dir.rename(final_dir)
    print(f"[{config_name} seed {seed}] COMPLETE -> {final_dir}", flush=True)
    del artifact, model
    gc.collect()
    return {"status": "complete", "path": str(final_dir)}


def parse_seed_spec(value: str) -> list[int]:
    value = str(value).strip()
    if "-" in value and "," not in value:
        start, stop = map(int, value.split("-", 1))
        return list(range(start, stop + 1))
    return [int(x) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-file",
        type=Path,
        required=True,
        help="JSON file following input_manifest.example.json.",
    )
    parser.add_argument("--seeds", default="42-51")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=POSTERIOR_DRAWS)
    parser.add_argument("--chunk-size", type=int, default=PATIENT_CHUNK)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--max-features", type=int)
    parser.add_argument("--cancers", default=",".join(CANCERS))
    parser.add_argument(
        "--write-patient-level", action="store_true",
        help="Write large patient-level gzip CSVs in addition to seed summaries.",
    )
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="Print the exact required model/artifact/OOF files and exit without analysis.",
    )
    args = parser.parse_args()
    config = load_config(args.config_file)
    if config.name not in {"PathNoUDP", "PathClin"}:
        raise ValueError(
            "wasserstein_FC accepts PathNoUDP or PathClin; "
            f"got {config.name!r}"
        )
    seeds = parse_seed_spec(args.seeds)
    cancers = tuple(x.strip() for x in args.cancers.split(",") if x.strip())
    bad = sorted(set(cancers) - set(CANCERS))
    if bad:
        raise ValueError(f"Unsupported cancers: {bad}")
    required_inputs = [("model script", config.model_script)]
    for seed in seeds:
        required_inputs.extend(
            [
                (f"seed {seed} saved DGP artifact", artifact_path(config, seed)),
                (f"seed {seed} OOF calibration CSV", oof_path(config, seed)),
            ]
        )
    missing = [(label, path) for label, path in required_inputs if not path.is_file()]
    if args.check_inputs:
        for label, path in required_inputs:
            print(f"{'OK' if path.is_file() else 'MISSING'}\t{label}\t{path}")
        if missing:
            print(
                "Preflight failed: supply the missing restricted inputs listed above.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        return
    if missing:
        raise FileNotFoundError(
            "Run --check-inputs to inspect the required inputs. Missing: "
            + "; ".join(f"{label} ({path})" for label, path in missing)
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "analysis": "wasserstein_FC",
        "source_note": "rebuilt from saved outputs and code; the original September 16 runner was not retained",
        "config": config_to_manifest(config),
        "seeds": seeds,
        "horizons": list(HORIZONS),
        "posterior_draws": args.draws,
        "chunk_size": args.chunk_size,
        "no_DGP_retraining": True,
        "donor_sampling": False,
        "prediction_scale": PREDICTION_SCALE,
        "calibration_kind": CALIBRATION_KIND,
        "treatment_profile_support_rule": PROFILE_SUPPORT_RULE,
        "active_treatment_pair_support_rule": ACTIVE_PAIR_SUPPORT_RULE,
        "treatment_profile_minimum_n": PROFILE_MIN_N,
        "treatment_profile_minimum_events": PROFILE_MIN_EVENTS,
        "patient_level_csvs_written": bool(args.write_patient_level),
    }
    config_dir = args.output_root / config.name
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for seed in manifest["seeds"]:
        run_seed(
            config=config,
            seed=int(seed),
            output_root=args.output_root,
            draws=int(args.draws),
            chunk_size=int(args.chunk_size),
            max_features=args.max_features,
            cancers=cancers,
            torch_threads=int(args.torch_threads),
            write_patient_level=bool(args.write_patient_level),
        )


if __name__ == "__main__":
    main()
