from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch


APPROVED_HORIZONS = (12.0, 18.0, 24.0, 36.0)
DONOR_THRESHOLDS = (5, 10, 15)
PRIMARY_DONOR_THRESHOLD = 10
MAX_DISTINCT_DONORS = 20
BOOTSTRAP_SAMPLES = 1000
CANONICAL_CANCERS = ("BRCA", "COAD", "LUAD")
CANCER_LABELS = {"BRCA": "Breast", "COAD": "Colon", "LUAD": "Lung"}
LABEL_TO_CANCER = {v: k for k, v in CANCER_LABELS.items()}
WFC_IMPLEMENTATION_VERSION = "full_batch_crn_v2"
TX_BY_CANCER = {
    "BRCA": ("TX_CHEMO", "TX_HORMONE", "TX_RADIATION", "TX_PHARMACEUTICAL"),
    "COAD": ("TX_CHEMO", "TX_RADIATION", "TX_PHARMACEUTICAL"),
    "LUAD": ("TX_CHEMO", "TX_RADIATION", "TX_PHARMACEUTICAL"),
}
TX_LABELS = {
    "TX_CHEMO": "Chemo",
    "TX_HORMONE": "Hormone",
    "TX_RADIATION": "Radiation",
    "TX_PHARMACEUTICAL": "Pharmaceutical",
}


def _stable_seed(*parts) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _patient_ids(meta, expected_n: int, split_name: str) -> np.ndarray:
    if meta is None or len(meta) != int(expected_n):
        raise ValueError(f"{split_name} metadata is missing or has the wrong row count")
    for col in ("PATIENT_ID", "patient_id", "bcr_patient_barcode"):
        if col in meta.columns:
            ids = meta[col].astype("string").str.strip()
            break
    else:
        raise ValueError(f"{split_name} metadata has no stable TCGA patient identifier")
    bad = ids.isna() | ids.eq("") | ids.str.lower().isin({"nan", "none", "<na>"})
    if bool(bad.any()):
        raise ValueError(f"{split_name} contains {int(bad.sum())} missing patient identifiers")
    dup = ids.duplicated(keep=False)
    if bool(dup.any()):
        examples = sorted(ids[dup].astype(str).unique().tolist())[:10]
        raise ValueError(f"{split_name} contains duplicate patient identifiers: {examples}")
    return ids.astype(str).to_numpy()


def _cancer_codes(art, split: str) -> np.ndarray:
    group_index = np.asarray(art[f"g_{split}"], dtype=int)
    groups = [str(x) for x in art["groups"]]
    out = []
    for value in group_index:
        label = groups[int(value)]
        code = LABEL_TO_CANCER.get(label, label.upper())
        if code not in CANONICAL_CANCERS:
            raise ValueError(f"Unrecognized cancer group {label!r}")
        out.append(code)
    return np.asarray(out, dtype=object)


def _tx_value(meta: pd.DataFrame, row: int, tx_col: str) -> int:
    candidates = (tx_col, f"clinbin::{tx_col}")
    for col in candidates:
        if col in meta.columns:
            value = pd.to_numeric(pd.Series([meta.iloc[row][col]]), errors="coerce").iloc[0]
            return int(bool(np.isfinite(value) and float(value) > 0.5))
    raise ValueError(f"Required treatment indicator {tx_col} is absent from metadata")


def _treatment_profiles(meta: pd.DataFrame, cancer_codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = []
    vectors = []
    for row, cancer in enumerate(cancer_codes):
        tx_cols = TX_BY_CANCER[str(cancer)]
        values = tuple(_tx_value(meta, row, col) for col in tx_cols)
        active = [TX_LABELS[col] for col, value in zip(tx_cols, values) if value]
        labels.append("TX=" + ("+".join(active) if active else "None"))
        vectors.append(";".join(f"{col}={value}" for col, value in zip(tx_cols, values)))
    return np.asarray(labels, dtype=object), np.asarray(vectors, dtype=object)


def _is_treatment_feature(name: str, family: str) -> bool:
    base = str(name).split("::")[-1]
    return str(family) == "treatment" or base in {
        "TX_CHEMO", "TX_HORMONE", "TX_RADIATION", "TX_PHARMACEUTICAL",
        "TX_EXPLICIT_NO_TREATMENT",
    }


def _encoded_source(name: str) -> str | None:
    text = str(name)
    if "=" in text:
        return text.rsplit("=", 1)[0]
    if text.startswith("clinbin::") and "_" in text.split("::", 1)[1]:
        return text.rsplit("_", 1)[0]
    return None


def build_feature_units(feat_names, family_fn, scope_fn) -> pd.DataFrame:
    records = []
    scalar = []
    encoded = {}
    for index, raw_name in enumerate(feat_names):
        name = str(raw_name)
        family = str(family_fn(name))
        scope = str(scope_fn(name))
        if _is_treatment_feature(name, family):
            records.append({
                "feature": name,
                "feature_columns": name,
                "column_indices": str(index),
                "n_columns": 1,
                "feature_family": family,
                "feature_scope": scope,
                "ranked": False,
                "exclusion_reason": "treatment indicator used only for conditioning",
            })
            continue
        source = _encoded_source(name)
        if source is None:
            scalar.append((index, name, family, scope))
        else:
            encoded.setdefault(source, []).append((index, name, family, scope))

    for source, members in encoded.items():
        if len(members) == 1:
            scalar.append(members[0])
            continue
        families = sorted({x[2] for x in members})
        scopes = sorted({x[3] for x in members})
        if len(families) != 1 or len(scopes) != 1:
            raise ValueError(f"Encoded feature block {source!r} has inconsistent family/scope labels")
        records.append({
            "feature": source,
            "feature_columns": ";".join(x[1] for x in members),
            "column_indices": ";".join(str(x[0]) for x in members),
            "n_columns": len(members),
            "feature_family": families[0],
            "feature_scope": scopes[0],
            "ranked": True,
            "exclusion_reason": "",
        })
    for index, name, family, scope in scalar:
        records.append({
            "feature": name,
            "feature_columns": name,
            "column_indices": str(index),
            "n_columns": 1,
            "feature_family": family,
            "feature_scope": scope,
            "ranked": True,
            "exclusion_reason": "",
        })
    manifest = pd.DataFrame(records)
    if manifest.empty or not bool(manifest["ranked"].any()):
        raise ValueError("No non-treatment WFC features were identified")
    return manifest.sort_values(["ranked", "feature"], ascending=[False, True]).reset_index(drop=True)


def _feature_applies_to_cancer(feature: str, cancer: str) -> bool:
    text = str(feature)
    for code in CANONICAL_CANCERS:
        if f"specific_{code}::" in text or text.startswith(f"{code}_PC"):
            return code == cancer
    return True


def _truthy(values: pd.Series) -> pd.Series:
    """Parse CSV booleans without treating the string ``False`` as true."""
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _set_rng_state(seed: int):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _capture_torch_state():
    return (
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def _restore_torch_state(state):
    cpu_state, cuda_state = state
    torch.set_rng_state(cpu_state)
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def _rank_within(df: pd.DataFrame, value_col: str) -> pd.Series:
    return pd.to_numeric(df[value_col], errors="coerce").rank(method="average", ascending=False)


def _spearman_from_ranks(left: pd.Series, right: pd.Series) -> float:
    mask = left.notna() & right.notna()
    if int(mask.sum()) < 2:
        return np.nan
    return float(np.corrcoef(left[mask].to_numpy(float), right[mask].to_numpy(float))[0, 1])


def run_corrected_wfc_partition(
    art,
    outdir,
    configuration,
    cfg_tag,
    seed,
    horizon,
    risk_samples_fn,
    wasserstein_fn,
    family_fn,
    scope_fn,
    posterior_draws=256,
):
    horizon = float(horizon)
    if horizon not in APPROVED_HORIZONS:
        raise ValueError(f"WFC horizon {horizon} is not approved; expected {APPROVED_HORIZONS}")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    Xtr_raw = np.asarray(art["Xtr_np_sub"], dtype=float)
    Xte_raw = np.asarray(art["Xte_np_sub"], dtype=float)
    Xte_scaled = np.asarray(art["Xte_t"].detach().cpu().numpy(), dtype=float)
    mu = np.asarray(art["standardize_mu"], dtype=float)
    sd = np.asarray(art["standardize_sd"], dtype=float)
    mu_flat = mu.reshape(-1)
    sd_flat = sd.reshape(-1)
    feat_names = [str(x) for x in art["feat_names"]]
    if Xtr_raw.shape[1] != len(feat_names) or Xte_raw.shape[1] != len(feat_names):
        raise ValueError("Artifact feature names and matrices are misaligned")
    if not np.isfinite(Xtr_raw).all() or not np.isfinite(Xte_raw).all():
        raise ValueError("Rankable artifact matrices contain nonfinite values; common denominators cannot be preserved")

    meta_tr = art["meta_tr"].reset_index(drop=True).copy()
    meta_te = art["meta_te"].reset_index(drop=True).copy()
    train_ids = _patient_ids(meta_tr, len(Xtr_raw), "training split")
    test_ids = _patient_ids(meta_te, len(Xte_raw), "test split")
    overlap = sorted(set(train_ids).intersection(test_ids))
    if overlap:
        raise ValueError(f"Training and test patient identifiers overlap: {overlap[:10]}")

    cancer_tr = _cancer_codes(art, "tr")
    cancer_te = _cancer_codes(art, "te")
    profile_tr, vector_tr = _treatment_profiles(meta_tr, cancer_tr)
    profile_te, vector_te = _treatment_profiles(meta_te, cancer_te)
    manifest = build_feature_units(feat_names, family_fn, scope_fn)
    manifest.to_csv(outdir / "corrected_wfc_feature_block_manifest.csv", index=False)

    donor_map = {}
    for cancer in CANONICAL_CANCERS:
        for profile in sorted(pd.unique(profile_tr[cancer_tr == cancer]).tolist()):
            donor_map[(cancer, str(profile))] = np.where((cancer_tr == cancer) & (profile_tr == profile))[0]

    patient_support_rows = []
    donor_audit_rows = []
    donor_sequences = {}
    for test_index, patient_id in enumerate(test_ids):
        cancer = str(cancer_te[test_index])
        profile = str(profile_te[test_index])
        donors = np.asarray(donor_map.get((cancer, profile), np.zeros(0, dtype=int)), dtype=int)
        donor_count = int(len(donors))
        if donor_count <= MAX_DISTINCT_DONORS:
            selected = donors.copy()
        else:
            rng = np.random.default_rng(_stable_seed(configuration, seed, cancer, profile, patient_id, "donors"))
            selected = rng.choice(donors, size=MAX_DISTINCT_DONORS, replace=False)
        donor_sequences[test_index] = np.asarray(selected, dtype=int)
        if len(np.unique(selected)) != len(selected):
            raise RuntimeError(f"Donor sampling for {patient_id} was not without replacement")
        for donor_repeat, donor_index in enumerate(selected, start=1):
            donor_audit_rows.append({
                "configuration": configuration,
                "config_tag": cfg_tag,
                "seed": int(seed),
                "horizon_months": horizon,
                "patient_id": patient_id,
                "cancer": cancer,
                "treatment_profile": profile,
                "donor_repeat": int(donor_repeat),
                "donor_train_index": int(donor_index),
                "donor_patient_id": str(train_ids[int(donor_index)]),
                "donor_cancer": str(cancer_tr[int(donor_index)]),
                "donor_treatment_profile": str(profile_tr[int(donor_index)]),
                "training_donor_pool_size": donor_count,
                "actual_M": int(len(selected)),
            })
        patient_support_rows.append({
            "configuration": configuration,
            "config_tag": cfg_tag,
            "seed": int(seed),
            "horizon_months": horizon,
            "patient_id": patient_id,
            "cancer": cancer,
            "cancer_label": CANCER_LABELS[cancer],
            "treatment_profile": profile,
            "treatment_profile_vector": str(vector_te[test_index]),
            "training_donor_pool_size": donor_count,
            "actual_M": int(len(selected)),
            **{f"eligible_nmin_{threshold}": donor_count >= threshold for threshold in DONOR_THRESHOLDS},
        })
    support = pd.DataFrame(patient_support_rows)
    support.to_csv(outdir / "corrected_wfc_patient_donor_support.csv", index=False)
    pd.DataFrame(donor_audit_rows).to_csv(
        outdir / "corrected_wfc_donor_audit.csv.gz", index=False, compression="gzip"
    )

    integrity = pd.DataFrame([
        {"check": "training_patient_ids_unique", "status": "ok", "detail": f"n={len(train_ids)}"},
        {"check": "test_patient_ids_unique", "status": "ok", "detail": f"n={len(test_ids)}"},
        {"check": "train_test_patient_ids_disjoint", "status": "ok", "detail": "overlap=0"},
        {"check": "finite_feature_matrices", "status": "ok", "detail": f"features={len(feat_names)}"},
        {"check": "approved_horizon", "status": "ok", "detail": f"H={horizon:g}"},
        {"check": "training_donors_only", "status": "ok", "detail": "donors indexed only from Xtr_np_sub"},
        {"check": "distinct_donors_without_replacement", "status": "ok", "detail": "M=min(20,n_donor)"},
    ])
    integrity.insert(0, "configuration", configuration)
    integrity.insert(1, "seed", int(seed))
    integrity.insert(2, "horizon_months", horizon)
    integrity.to_csv(outdir / "corrected_wfc_integrity_checks.csv", index=False)

    patient_rows = []
    feature_failure_rows = []
    model = art["model"]
    max_M = min(MAX_DISTINCT_DONORS, int(support["training_donor_pool_size"].max())) if len(support) else 0
    crn_null_replay_max_abs_diff = 0.0

    for cancer in CANONICAL_CANCERS:
        cancer_indices = np.where(cancer_te == cancer)[0]
        if len(cancer_indices) == 0:
            raise ValueError(f"No held-out {cancer} patients are present")
        local_for_global = {int(global_index): local for local, global_index in enumerate(cancer_indices)}
        Xraw_cancer = Xte_raw[cancer_indices].copy()
        Xscaled_cancer = Xte_scaled[cancer_indices].copy()
        cancer_index_t = torch.as_tensor(
            cancer_indices, dtype=torch.long, device=art["g_te_t"].device
        )
        group_tensor = art["g_te_t"].index_select(0, cancer_index_t)

        baseline_draws = []
        baseline_states = []
        for donor_repeat in range(max_M):
            repeat_seed = _stable_seed(configuration, seed, horizon, cancer, donor_repeat, "posterior_crn")
            _set_rng_state(repeat_seed)
            baseline_state = _capture_torch_state()
            baseline = np.asarray(risk_samples_fn(
                model, Xscaled_cancer, group_tensor, horizon,
                art["alpha_dt"], art["dt_edges"], S=int(posterior_draws),
            ), dtype=float)
            _restore_torch_state(baseline_state)
            null_replay = np.asarray(risk_samples_fn(
                model, Xscaled_cancer, group_tensor, horizon,
                art["alpha_dt"], art["dt_edges"], S=int(posterior_draws),
            ), dtype=float)
            if baseline.shape != null_replay.shape:
                raise RuntimeError(
                    "Common-random-number null replay changed posterior sample shape: "
                    f"{baseline.shape} versus {null_replay.shape}"
                )
            replay_diff = float(np.max(np.abs(baseline - null_replay)))
            crn_null_replay_max_abs_diff = max(crn_null_replay_max_abs_diff, replay_diff)
            if not np.allclose(baseline, null_replay, rtol=1e-6, atol=1e-7):
                raise RuntimeError(
                    "Common-random-number null replay failed for identical inputs: "
                    f"cancer={cancer}, donor_repeat={donor_repeat}, max_abs_diff={replay_diff:.6g}"
                )
            baseline_states.append(baseline_state)
            baseline_draws.append(baseline)

        for unit in manifest[manifest["ranked"]].itertuples(index=False):
            if not _feature_applies_to_cancer(unit.feature, cancer):
                continue
            columns = np.asarray([int(x) for x in str(unit.column_indices).split(";")], dtype=int)
            per_patient = {int(index): [] for index in cancer_indices}
            per_original_mean = {int(index): [] for index in cancer_indices}
            per_perturbed_mean = {int(index): [] for index in cancer_indices}
            try:
                for donor_repeat in range(max_M):
                    active_global = [
                        int(index) for index in cancer_indices
                        if len(donor_sequences[int(index)]) > donor_repeat
                    ]
                    if not active_global:
                        continue
                    active_local = np.asarray([local_for_global[index] for index in active_global], dtype=int)
                    # Keep the complete cancer batch and its original patient order for
                    # both posterior calls. Inactive patients remain unchanged. This is
                    # required for the restored RNG state to provide patient-wise common
                    # random numbers when donor pools have different sizes.
                    perturbed_scaled = Xscaled_cancer.copy()
                    for global_index in active_global:
                        local_index = local_for_global[global_index]
                        donor_index = int(donor_sequences[global_index][donor_repeat])
                        perturbed_scaled[local_index, columns] = (
                            Xtr_raw[donor_index, columns] - mu_flat[columns]
                        ) / sd_flat[columns]
                    _restore_torch_state(baseline_states[donor_repeat])
                    perturbed_draws = np.asarray(risk_samples_fn(
                        model, perturbed_scaled, group_tensor, horizon,
                        art["alpha_dt"], art["dt_edges"], S=int(posterior_draws),
                    ), dtype=float)
                    original_draws = np.asarray(baseline_draws[donor_repeat], dtype=float)
                    if perturbed_draws.shape != original_draws.shape:
                        raise RuntimeError(
                            "Baseline and perturbed posterior draws used different full-batch shapes: "
                            f"{original_draws.shape} versus {perturbed_draws.shape}"
                        )
                    distances = np.asarray(wasserstein_fn(original_draws, perturbed_draws), dtype=float)
                    for global_index, local_index in zip(active_global, active_local):
                        per_patient[global_index].append(float(distances[local_index]))
                        per_original_mean[global_index].append(float(np.mean(original_draws[:, local_index])))
                        per_perturbed_mean[global_index].append(float(np.mean(perturbed_draws[:, local_index])))
            except Exception as exc:
                feature_failure_rows.append({
                    "configuration": configuration,
                    "seed": int(seed),
                    "horizon_months": horizon,
                    "cancer": cancer,
                    "feature": unit.feature,
                    "failure": f"{type(exc).__name__}: {exc}",
                })
                continue

            for global_index in cancer_indices:
                values = np.asarray(per_patient[int(global_index)], dtype=float)
                original_values = np.asarray(per_original_mean[int(global_index)], dtype=float)
                perturbed_values = np.asarray(per_perturbed_mean[int(global_index)], dtype=float)
                donor_count = int(support.loc[global_index, "training_donor_pool_size"])
                actual_M = int(support.loc[global_index, "actual_M"])
                if len(values) != actual_M:
                    raise RuntimeError(
                        f"Patient {test_ids[global_index]} feature {unit.feature} used {len(values)} perturbations; expected {actual_M}"
                    )
                patient_rows.append({
                    "configuration": configuration,
                    "config_tag": cfg_tag,
                    "seed": int(seed),
                    "horizon_months": horizon,
                    "cancer": cancer,
                    "cancer_label": CANCER_LABELS[cancer],
                    "patient_id": str(test_ids[global_index]),
                    "treatment_profile": str(profile_te[global_index]),
                    "treatment_profile_vector": str(vector_te[global_index]),
                    "feature": str(unit.feature),
                    "feature_columns": str(unit.feature_columns),
                    "feature_family": str(unit.feature_family),
                    "feature_scope": str(unit.feature_scope),
                    "training_donor_pool_size": donor_count,
                    "actual_M": actual_M,
                    "posterior_draw_count": int(posterior_draws),
                    "mean_W1": float(np.mean(values)) if len(values) else np.nan,
                    "sd_W1": float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if len(values) == 1 else np.nan),
                    "mc_se_W1": float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else (0.0 if len(values) == 1 else np.nan),
                    "mean_original_risk": float(np.mean(original_values)) if len(original_values) else np.nan,
                    "mean_perturbed_risk": float(np.mean(perturbed_values)) if len(perturbed_values) else np.nan,
                    "signed_mean_risk_change": float(np.mean(perturbed_values - original_values)) if len(values) else np.nan,
                    **{f"eligible_nmin_{threshold}": donor_count >= threshold for threshold in DONOR_THRESHOLDS},
                    "data_quality_status": "ok",
                    "nonestimable_reason": "" if donor_count >= min(DONOR_THRESHOLDS) else "insufficient matching training donors",
                })

    patients = pd.DataFrame(patient_rows)
    if patients.empty:
        raise RuntimeError("Corrected WFC produced no patient-level rows")
    patients.to_csv(outdir / "corrected_wfc_patient_level.csv.gz", index=False, compression="gzip")
    pd.DataFrame(feature_failure_rows).to_csv(outdir / "corrected_wfc_feature_failures.csv", index=False)
    if feature_failure_rows:
        raise RuntimeError(
            f"Corrected WFC failed for {len(feature_failure_rows)} cancer-feature computations; "
            "see corrected_wfc_feature_failures.csv"
        )
    for cancer in CANONICAL_CANCERS:
        expected_ids = set(test_ids[cancer_te == cancer])
        cancer_patients = patients[patients["cancer"].eq(cancer)]
        for feature, feature_rows in cancer_patients.groupby("feature", sort=False):
            observed_ids = set(feature_rows["patient_id"].astype(str))
            if observed_ids != expected_ids:
                raise RuntimeError(
                    f"Feature-specific patient denominator detected for {cancer}/{feature}: "
                    f"expected {len(expected_ids)}, observed {len(observed_ids)}"
                )
    integrity = pd.concat([integrity, pd.DataFrame([
        {
            "configuration": configuration,
            "seed": int(seed),
            "horizon_months": horizon,
            "check": "common_test_patient_denominator_within_cancer",
            "status": "ok",
            "detail": "same held-out patient IDs used for every applicable ranked feature",
        },
        {
            "configuration": configuration,
            "seed": int(seed),
            "horizon_months": horizon,
            "check": "common_random_numbers_null_replay",
            "status": "ok",
            "detail": (
                f"implementation={WFC_IMPLEMENTATION_VERSION}; full cancer batch and patient order; "
                f"max_abs_diff={crn_null_replay_max_abs_diff:.6g}"
            ),
        },
    ])], ignore_index=True)
    integrity.to_csv(outdir / "corrected_wfc_integrity_checks.csv", index=False)

    feature_summaries = []
    profile_summaries = []
    coverage_rows = []
    for threshold in DONOR_THRESHOLDS:
        eligible_col = f"eligible_nmin_{threshold}"
        for (cancer, feature), group in patients.groupby(["cancer", "feature"], sort=False):
            eligible = group[_truthy(group[eligible_col])]
            feature_summaries.append({
                "configuration": configuration,
                "config_tag": cfg_tag,
                "seed": int(seed),
                "horizon_months": horizon,
                "cancer": cancer,
                "feature": feature,
                "feature_family": group["feature_family"].iloc[0],
                "feature_scope": group["feature_scope"].iloc[0],
                "donor_threshold": int(threshold),
                "donor_threshold_role": "primary" if threshold == PRIMARY_DONOR_THRESHOLD else "sensitivity",
                "WFC": float(eligible["mean_W1"].mean()) if len(eligible) else np.nan,
                "eligible_test_patients": int(eligible["patient_id"].nunique()),
                "total_test_patients": int(group["patient_id"].nunique()),
                "eligible_coverage": float(eligible["patient_id"].nunique() / group["patient_id"].nunique()),
                "estimable": bool(len(eligible)),
            })
        for (cancer, profile, feature), group in patients.groupby(["cancer", "treatment_profile", "feature"], sort=False):
            eligible = group[_truthy(group[eligible_col])]
            profile_summaries.append({
                "configuration": configuration,
                "config_tag": cfg_tag,
                "seed": int(seed),
                "horizon_months": horizon,
                "cancer": cancer,
                "treatment_profile": profile,
                "feature": feature,
                "donor_threshold": int(threshold),
                "donor_threshold_role": "primary" if threshold == PRIMARY_DONOR_THRESHOLD else "sensitivity",
                "WFC": float(eligible["mean_W1"].mean()) if len(eligible) else np.nan,
                "eligible_test_patients": int(eligible["patient_id"].nunique()),
                "total_test_patients": int(group["patient_id"].nunique()),
                "training_donor_pool_size": int(group["training_donor_pool_size"].iloc[0]),
                "estimable": bool(len(eligible)),
            })
        for (cancer, profile), group in support.groupby(["cancer", "treatment_profile"], sort=False):
            eligible = group[_truthy(group[eligible_col])]
            coverage_rows.append({
                "configuration": configuration,
                "config_tag": cfg_tag,
                "seed": int(seed),
                "horizon_months": horizon,
                "cancer": cancer,
                "treatment_profile": profile,
                "donor_threshold": int(threshold),
                "donor_threshold_role": "primary" if threshold == PRIMARY_DONOR_THRESHOLD else "sensitivity",
                "training_donor_pool_size": int(group["training_donor_pool_size"].iloc[0]),
                "total_test_patients": int(group["patient_id"].nunique()),
                "eligible_test_patients": int(eligible["patient_id"].nunique()),
                "excluded_insufficient_training_donors": int(group["patient_id"].nunique() - eligible["patient_id"].nunique()),
                "eligible_percentage": float(100.0 * eligible["patient_id"].nunique() / group["patient_id"].nunique()),
            })

    feature_summary = pd.DataFrame(feature_summaries)
    feature_summary["seed_rank"] = feature_summary.groupby(
        ["configuration", "seed", "horizon_months", "cancer", "donor_threshold"], dropna=False
    )["WFC"].rank(method="average", ascending=False)
    feature_summary.to_csv(outdir / "corrected_wfc_seed_cancer_feature_summary.csv", index=False)
    profile_summary = pd.DataFrame(profile_summaries)
    profile_summary["seed_profile_rank"] = profile_summary.groupby(
        ["configuration", "seed", "horizon_months", "cancer", "treatment_profile", "donor_threshold"],
        dropna=False,
    )["WFC"].rank(method="average", ascending=False)
    profile_summary.to_csv(outdir / "corrected_wfc_seed_treatment_profile_summary.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(outdir / "corrected_wfc_seed_coverage.csv", index=False)

    sensitivity_rows = []
    for cancer in CANONICAL_CANCERS:
        subset = feature_summary[feature_summary["cancer"].eq(cancer)]
        for left_threshold, right_threshold in itertools.combinations(DONOR_THRESHOLDS, 2):
            left = subset[subset["donor_threshold"].eq(left_threshold)][["feature", "seed_rank"]].rename(columns={"seed_rank": "left_rank"})
            right = subset[subset["donor_threshold"].eq(right_threshold)][["feature", "seed_rank"]].rename(columns={"seed_rank": "right_rank"})
            joined = left.merge(right, on="feature", how="inner")
            left_top = set(joined.nsmallest(10, "left_rank")["feature"])
            right_top = set(joined.nsmallest(10, "right_rank")["feature"])
            sensitivity_rows.append({
                "configuration": configuration,
                "seed": int(seed),
                "horizon_months": horizon,
                "cancer": cancer,
                "left_donor_threshold": int(left_threshold),
                "right_donor_threshold": int(right_threshold),
                "features_compared": int(len(joined)),
                "spearman_rank_correlation": _spearman_from_ranks(joined["left_rank"], joined["right_rank"]),
                "top10_overlap_count": int(len(left_top.intersection(right_top))),
                "top10_jaccard": float(len(left_top.intersection(right_top)) / len(left_top.union(right_top))) if left_top.union(right_top) else np.nan,
            })
    pd.DataFrame(sensitivity_rows).to_csv(outdir / "corrected_wfc_seed_donor_threshold_sensitivity.csv", index=False)

    partition_status = pd.DataFrame([{
        "configuration": configuration,
        "config_tag": cfg_tag,
        "implementation_version": WFC_IMPLEMENTATION_VERSION,
        "seed": int(seed),
        "horizon_months": horizon,
        "status": "completed",
        "patient_feature_rows": int(len(patients)),
        "cancers": ";".join(CANONICAL_CANCERS),
        "risk_definition": "posterior_-log_S(H)",
        "donor_source": "training_same_cancer_exact_treatment_profile",
    }])
    partition_status.to_csv(outdir / "corrected_wfc_partition_status.csv", index=False)
    return {
        "patients": patients,
        "feature_summary": feature_summary,
        "profile_summary": profile_summary,
        "coverage": pd.DataFrame(coverage_rows),
    }


def _collect_pattern(roots, pattern: str) -> pd.DataFrame:
    frames = []
    for configuration, root in roots.items():
        for path in sorted(Path(root).glob(pattern)):
            try:
                frame = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                frame = pd.DataFrame()
            if len(frame):
                if "configuration" not in frame:
                    frame.insert(0, "configuration", configuration)
                frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _bootstrap_mean_draws(values, B: int, rng) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.full(int(B), np.nan, dtype=float)
    if len(values) == 1:
        return np.full(int(B), float(values[0]), dtype=float)
    indices = rng.integers(0, len(values), size=(int(B), len(values)))
    return values[indices].mean(axis=1)


def _bootstrap_mean_ci(values, B: int, rng) -> tuple[float, float]:
    means = _bootstrap_mean_draws(values, B, rng)
    if not np.isfinite(means).any():
        return np.nan, np.nan
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def collect_corrected_wfc(roots, expected_seeds, bootstrap_samples=BOOTSTRAP_SAMPLES):
    expected_seeds = [int(seed) for seed in expected_seeds]
    patient = _collect_pattern(roots, "seed*/**/corrected_wfc_patient_level.csv.gz")
    seed_feature = _collect_pattern(roots, "seed*/**/corrected_wfc_seed_cancer_feature_summary.csv")
    seed_profile = _collect_pattern(roots, "seed*/**/corrected_wfc_seed_treatment_profile_summary.csv")
    coverage = _collect_pattern(roots, "seed*/**/corrected_wfc_seed_coverage.csv")
    manifest = _collect_pattern(roots, "seed*/**/corrected_wfc_feature_block_manifest.csv")
    donor_support = _collect_pattern(roots, "seed*/**/corrected_wfc_patient_donor_support.csv")
    donor_audit = _collect_pattern(roots, "seed*/**/corrected_wfc_donor_audit.csv.gz")
    integrity = _collect_pattern(roots, "seed*/**/corrected_wfc_integrity_checks.csv")
    failures = _collect_pattern(roots, "seed*/**/corrected_wfc_feature_failures.csv")
    partition_status = _collect_pattern(roots, "seed*/**/corrected_wfc_partition_status.csv")
    seed_sensitivity = _collect_pattern(roots, "seed*/**/corrected_wfc_seed_donor_threshold_sensitivity.csv")
    if patient.empty or seed_feature.empty:
        raise RuntimeError("Corrected WFC patient-level or seed-level outputs are missing")

    unique_rows = []
    cancer_rows = []
    profile_unique_rows = []
    for threshold in DONOR_THRESHOLDS:
        eligible_col = f"eligible_nmin_{threshold}"
        eligible = patient[_truthy(patient[eligible_col])].copy()
        keys = [
            "configuration", "cancer", "patient_id", "treatment_profile", "feature",
            "feature_family", "feature_scope", "horizon_months",
        ]
        unique = eligible.groupby(keys, dropna=False, as_index=False).agg(
            unique_patient_WFC=("mean_W1", "mean"),
            eligible_heldout_appearances=("seed", "size"),
            eligible_seeds=("seed", "nunique"),
            mean_training_donor_pool_size=("training_donor_pool_size", "mean"),
        )
        unique["donor_threshold"] = int(threshold)
        unique["donor_threshold_role"] = "primary" if threshold == PRIMARY_DONOR_THRESHOLD else "sensitivity"
        unique_rows.append(unique)
        group_keys = [
            "configuration", "cancer", "feature", "feature_family", "feature_scope",
            "horizon_months",
        ]
        for group_values, group in unique.groupby(group_keys, dropna=False):
            record = dict(zip(group_keys, group_values))
            rng = np.random.default_rng(_stable_seed(*group_values, threshold, "unique_patient_bootstrap"))
            lower, upper = _bootstrap_mean_ci(group["unique_patient_WFC"], bootstrap_samples, rng)
            record.update({
                "donor_threshold": int(threshold),
                "donor_threshold_role": "primary" if threshold == PRIMARY_DONOR_THRESHOLD else "sensitivity",
                "unique_patient_WFC": float(group["unique_patient_WFC"].mean()),
                "bootstrap_95ci_lower": lower,
                "bootstrap_95ci_upper": upper,
                "bootstrap_samples": int(bootstrap_samples),
                "eligible_unique_patients": int(group["patient_id"].nunique()),
                "eligible_heldout_appearances": int(group["eligible_heldout_appearances"].sum()),
                "estimable_seeds": int(eligible[
                    eligible["configuration"].eq(record["configuration"])
                    & eligible["cancer"].eq(record["cancer"])
                    & eligible["feature"].eq(record["feature"])
                    & eligible["horizon_months"].eq(record["horizon_months"])
                ]["seed"].nunique()),
            })
            cancer_rows.append(record)
        profile_keys = ["configuration", "cancer", "treatment_profile", "feature", "horizon_months"]
        profile = unique.groupby(profile_keys, dropna=False, as_index=False).agg(
            unique_patient_WFC=("unique_patient_WFC", "mean"),
            eligible_unique_patients=("patient_id", "nunique"),
            eligible_heldout_appearances=("eligible_heldout_appearances", "sum"),
        )
        profile["donor_threshold"] = int(threshold)
        profile["donor_threshold_role"] = "primary" if threshold == PRIMARY_DONOR_THRESHOLD else "sensitivity"
        profile_unique_rows.append(profile)

    unique_patient = pd.concat(unique_rows, ignore_index=True, sort=False)
    cancer_final = pd.DataFrame(cancer_rows)
    cancer_final["final_cancer_rank"] = cancer_final.groupby(
        ["configuration", "cancer", "horizon_months", "donor_threshold"], dropna=False
    )["unique_patient_WFC"].rank(method="average", ascending=False)

    seed_variability = seed_feature.groupby(
        ["configuration", "cancer", "feature", "horizon_months", "donor_threshold"],
        dropna=False, as_index=False,
    ).agg(
        seed_mean_WFC=("WFC", "mean"),
        seed_sd_WFC=("WFC", "std"),
        seed_mean_rank=("seed_rank", "mean"),
        seed_sd_rank=("seed_rank", "std"),
        estimable_seed_count=("estimable", "sum"),
    )
    top10 = seed_feature.assign(in_top10=seed_feature["seed_rank"].le(10)).groupby(
        ["configuration", "cancer", "feature", "horizon_months", "donor_threshold"],
        dropna=False, as_index=False,
    )["in_top10"].mean().rename(columns={"in_top10": "top10_frequency"})
    seed_variability = seed_variability.merge(top10, on=[
        "configuration", "cancer", "feature", "horizon_months", "donor_threshold"
    ], how="left")
    cancer_final = cancer_final.merge(seed_variability, on=[
        "configuration", "cancer", "feature", "horizon_months", "donor_threshold"
    ], how="left")
    final_profile = pd.concat(profile_unique_rows, ignore_index=True, sort=False)
    final_profile["final_profile_rank"] = final_profile.groupby(
        ["configuration", "cancer", "treatment_profile", "horizon_months", "donor_threshold"],
        dropna=False,
    )["unique_patient_WFC"].rank(method="average", ascending=False)

    split_rank_rows = []
    rank_keys = ["configuration", "cancer", "horizon_months", "donor_threshold"]
    for group_values, group in seed_feature.groupby(rank_keys, dropna=False):
        pivot = group.pivot_table(index="feature", columns="seed", values="seed_rank", aggfunc="first")
        for left_seed, right_seed in itertools.combinations(sorted(pivot.columns), 2):
            split_rank_rows.append({
                **dict(zip(rank_keys, group_values)),
                "left_seed": int(left_seed),
                "right_seed": int(right_seed),
                "features_compared": int(pivot[[left_seed, right_seed]].dropna().shape[0]),
                "spearman_rank_correlation": _spearman_from_ranks(pivot[left_seed], pivot[right_seed]),
            })
    split_rank = pd.DataFrame(split_rank_rows)

    seed_shared_rows = []
    seed_cross_rank_rows = []
    # A feature enters a cross-cancer table only if the exact same feature is
    # actually present and estimable in BRCA, COAD, and LUAD. No numerical
    # consistency cutoff or shared/not-shared label is imposed.
    seed_shared_source = seed_feature[_truthy(seed_feature["estimable"])].copy()
    seed_shared_keys = [
        "configuration", "seed", "feature", "feature_family", "feature_scope",
        "horizon_months", "donor_threshold",
    ]
    for group_values, group in seed_shared_source.groupby(seed_shared_keys, dropna=False):
        by_cancer = group.drop_duplicates("cancer").set_index("cancer")
        if not all(cancer in by_cancer.index for cancer in CANONICAL_CANCERS):
            continue
        values = by_cancer.loc[list(CANONICAL_CANCERS), "WFC"].to_numpy(float)
        counts = by_cancer.loc[list(CANONICAL_CANCERS), "eligible_test_patients"].to_numpy(float)
        if not np.isfinite(values).all():
            continue
        seed_shared_rows.append({
            **dict(zip(seed_shared_keys, group_values)),
            "BRCA_WFC": float(values[0]),
            "COAD_WFC": float(values[1]),
            "LUAD_WFC": float(values[2]),
            "BRCA_rank": float(by_cancer.loc["BRCA", "seed_rank"]),
            "COAD_rank": float(by_cancer.loc["COAD", "seed_rank"]),
            "LUAD_rank": float(by_cancer.loc["LUAD", "seed_rank"]),
            "equal_cancer_seed_WFC": float(np.mean(values)),
            "patient_weighted_seed_WFC": float(np.average(values, weights=counts)) if counts.sum() else np.nan,
            "analysis_role": "seed_specific_stability_only_not_primary_shared_ranking",
            "cross_cancer_inclusion_rule": "same_feature_present_and_estimable_in_BRCA_COAD_LUAD",
            "numerical_consistency_cutoff_applied": False,
            "donor_threshold_role": "primary" if int(group_values[6]) == PRIMARY_DONOR_THRESHOLD else "sensitivity",
        })
    seed_shared = pd.DataFrame(seed_shared_rows)
    if len(seed_shared):
        seed_shared["seed_shared_rank"] = seed_shared.groupby(
            ["configuration", "seed", "horizon_months", "donor_threshold"], dropna=False
        )["equal_cancer_seed_WFC"].rank(method="average", ascending=False)
        for group_values, group in seed_shared.groupby(
            ["configuration", "seed", "horizon_months", "donor_threshold"], dropna=False
        ):
            for left, right in itertools.combinations(CANONICAL_CANCERS, 2):
                seed_cross_rank_rows.append({
                    "configuration": group_values[0],
                    "seed": int(group_values[1]),
                    "horizon_months": group_values[2],
                    "donor_threshold": group_values[3],
                    "left_cancer": left,
                    "right_cancer": right,
                    "features_compared": int(len(group)),
                    "spearman_rank_correlation": _spearman_from_ranks(
                        group[f"{left}_rank"], group[f"{right}_rank"]
                    ),
                })

    shared_rows = []
    cross_rank_rows = []
    shared_source = cancer_final.copy()
    shared_keys = ["configuration", "feature", "feature_family", "feature_scope", "horizon_months", "donor_threshold"]
    for group_values, group in shared_source.groupby(shared_keys, dropna=False):
        by_cancer = group.drop_duplicates("cancer").set_index("cancer")
        if not all(cancer in by_cancer.index for cancer in CANONICAL_CANCERS):
            continue
        if not np.isfinite(by_cancer.loc[list(CANONICAL_CANCERS), "unique_patient_WFC"].to_numpy(float)).all():
            continue
        values = by_cancer.loc[list(CANONICAL_CANCERS), "unique_patient_WFC"].to_numpy(float)
        counts = by_cancer.loc[list(CANONICAL_CANCERS), "eligible_unique_patients"].to_numpy(float)
        shared_bootstrap_parts = []
        for cancer in CANONICAL_CANCERS:
            source_values = unique_patient[
                unique_patient["configuration"].eq(group_values[0])
                & unique_patient["cancer"].eq(cancer)
                & unique_patient["feature"].eq(group_values[1])
                & unique_patient["horizon_months"].eq(group_values[4])
                & unique_patient["donor_threshold"].eq(group_values[5])
            ]["unique_patient_WFC"].to_numpy(float)
            shared_bootstrap_parts.append(_bootstrap_mean_draws(
                source_values,
                bootstrap_samples,
                np.random.default_rng(_stable_seed(*group_values, cancer, "shared_unique_patient_bootstrap")),
            ))
        shared_bootstrap = np.vstack(shared_bootstrap_parts)
        primary_bootstrap = shared_bootstrap.mean(axis=0)
        secondary_bootstrap = np.average(shared_bootstrap, axis=0, weights=counts)
        shared_rows.append({
            **dict(zip(shared_keys, group_values)),
            "BRCA_WFC": float(values[0]),
            "COAD_WFC": float(values[1]),
            "LUAD_WFC": float(values[2]),
            "BRCA_rank": float(by_cancer.loc["BRCA", "final_cancer_rank"]),
            "COAD_rank": float(by_cancer.loc["COAD", "final_cancer_rank"]),
            "LUAD_rank": float(by_cancer.loc["LUAD", "final_cancer_rank"]),
            "BRCA_unique_patients": int(counts[0]),
            "COAD_unique_patients": int(counts[1]),
            "LUAD_unique_patients": int(counts[2]),
            "primary_equal_cancer_shared_WFC": float(np.mean(values)),
            "primary_bootstrap_95ci_lower": float(np.quantile(primary_bootstrap, 0.025)),
            "primary_bootstrap_95ci_upper": float(np.quantile(primary_bootstrap, 0.975)),
            "secondary_patient_weighted_shared_WFC": float(np.average(values, weights=counts)) if counts.sum() else np.nan,
            "secondary_bootstrap_95ci_lower": float(np.quantile(secondary_bootstrap, 0.025)) if counts.sum() else np.nan,
            "secondary_bootstrap_95ci_upper": float(np.quantile(secondary_bootstrap, 0.975)) if counts.sum() else np.nan,
            "bootstrap_samples": int(bootstrap_samples),
            "primary_source": "final_unique_patient_cancer_WFCs",
            "cross_cancer_inclusion_rule": "same_feature_present_and_estimable_in_BRCA_COAD_LUAD",
            "numerical_consistency_cutoff_applied": False,
            "biological_sharedness_requires_interpretation_of_per_cancer_WFCs_and_ranks": True,
            "donor_threshold_role": "primary" if int(group_values[5]) == PRIMARY_DONOR_THRESHOLD else "sensitivity",
        })
    shared_final = pd.DataFrame(shared_rows)
    if len(shared_final):
        shared_final["shared_rank"] = shared_final.groupby(
            ["configuration", "horizon_months", "donor_threshold"], dropna=False
        )["primary_equal_cancer_shared_WFC"].rank(method="average", ascending=False)
        for group_values, group in shared_final.groupby(
            ["configuration", "horizon_months", "donor_threshold"], dropna=False
        ):
            for left, right in itertools.combinations(CANONICAL_CANCERS, 2):
                cross_rank_rows.append({
                    "configuration": group_values[0],
                    "horizon_months": group_values[1],
                    "donor_threshold": group_values[2],
                    "left_cancer": left,
                    "right_cancer": right,
                    "features_compared": int(len(group)),
                    "spearman_rank_correlation": _spearman_from_ranks(group[f"{left}_rank"], group[f"{right}_rank"]),
                })

    unique_coverage_rows = []
    patient_identity = patient[[
        "configuration", "cancer", "patient_id", "treatment_profile", "seed",
        *[f"eligible_nmin_{threshold}" for threshold in DONOR_THRESHOLDS],
    ]].drop_duplicates()
    for threshold in DONOR_THRESHOLDS:
        eligible_col = f"eligible_nmin_{threshold}"
        for group_values, group in patient_identity.groupby(
            ["configuration", "cancer", "treatment_profile"], dropna=False
        ):
            patient_eligibility = group.assign(
                _eligible=_truthy(group[eligible_col])
            ).groupby("patient_id", as_index=False)["_eligible"].max()
            unique_coverage_rows.append({
                "configuration": group_values[0],
                "cancer": group_values[1],
                "treatment_profile": group_values[2],
                "donor_threshold": int(threshold),
                "total_unique_test_patients": int(patient_eligibility["patient_id"].nunique()),
                "eligible_unique_patients": int(patient_eligibility["_eligible"].sum()),
                "eligible_unique_patient_percentage": float(
                    100.0 * patient_eligibility["_eligible"].mean()
                ),
            })
    unique_coverage = pd.DataFrame(unique_coverage_rows)

    final_sensitivity_rows = []
    for group_values, group in cancer_final.groupby(
        ["configuration", "cancer", "horizon_months"], dropna=False
    ):
        for left_threshold, right_threshold in itertools.combinations(DONOR_THRESHOLDS, 2):
            left = group[group["donor_threshold"].eq(left_threshold)][["feature", "final_cancer_rank"]].rename(
                columns={"final_cancer_rank": "left_rank"}
            )
            right = group[group["donor_threshold"].eq(right_threshold)][["feature", "final_cancer_rank"]].rename(
                columns={"final_cancer_rank": "right_rank"}
            )
            joined = left.merge(right, on="feature", how="inner")
            left_top = set(joined.nsmallest(10, "left_rank")["feature"])
            right_top = set(joined.nsmallest(10, "right_rank")["feature"])
            final_sensitivity_rows.append({
                "configuration": group_values[0],
                "analysis_scope": group_values[1],
                "horizon_months": group_values[2],
                "left_donor_threshold": int(left_threshold),
                "right_donor_threshold": int(right_threshold),
                "features_compared": int(len(joined)),
                "spearman_rank_correlation": _spearman_from_ranks(joined["left_rank"], joined["right_rank"]),
                "top10_overlap_count": int(len(left_top.intersection(right_top))),
                "top10_jaccard": float(len(left_top.intersection(right_top)) / len(left_top.union(right_top))) if left_top.union(right_top) else np.nan,
            })
    if len(shared_final):
        for group_values, group in shared_final.groupby(
            ["configuration", "horizon_months"], dropna=False
        ):
            for left_threshold, right_threshold in itertools.combinations(DONOR_THRESHOLDS, 2):
                left = group[group["donor_threshold"].eq(left_threshold)][["feature", "shared_rank"]].rename(columns={"shared_rank": "left_rank"})
                right = group[group["donor_threshold"].eq(right_threshold)][["feature", "shared_rank"]].rename(columns={"shared_rank": "right_rank"})
                joined = left.merge(right, on="feature", how="inner")
                left_top = set(joined.nsmallest(10, "left_rank")["feature"])
                right_top = set(joined.nsmallest(10, "right_rank")["feature"])
                final_sensitivity_rows.append({
                    "configuration": group_values[0],
                    "analysis_scope": "shared_across_cancers",
                    "horizon_months": group_values[1],
                    "left_donor_threshold": int(left_threshold),
                    "right_donor_threshold": int(right_threshold),
                    "features_compared": int(len(joined)),
                    "spearman_rank_correlation": _spearman_from_ranks(joined["left_rank"], joined["right_rank"]),
                    "top10_overlap_count": int(len(left_top.intersection(right_top))),
                    "top10_jaccard": float(len(left_top.intersection(right_top)) / len(left_top.union(right_top))) if left_top.union(right_top) else np.nan,
                })

    expected_partition_keys = {
        (str(configuration), int(seed), float(horizon))
        for configuration in roots
        for seed in expected_seeds
        for horizon in APPROVED_HORIZONS
    }
    expected_partitions = len(expected_partition_keys)
    unique_partitions = partition_status[["configuration", "seed", "horizon_months"]].drop_duplicates()
    observed_partition_keys = {
        (str(row.configuration), int(row.seed), float(row.horizon_months))
        for row in unique_partitions.itertuples(index=False)
    }
    implementation_version_ok = (
        "implementation_version" in partition_status.columns
        and set(partition_status["implementation_version"].astype(str)) == {WFC_IMPLEMENTATION_VERSION}
    )
    contract_ok = (
        observed_partition_keys == expected_partition_keys
        and partition_status["status"].astype(str).eq("completed").all()
        and implementation_version_ok
        and set(pd.to_numeric(partition_status["horizon_months"], errors="coerce").dropna().unique()) == set(APPROVED_HORIZONS)
        and integrity["status"].astype(str).eq("ok").all()
        and failures.empty
    )
    contract = pd.DataFrame([{
        "required_partitions": int(expected_partitions),
        "observed_partitions": int(len(unique_partitions)),
        "approved_horizons": ";".join(str(int(x)) for x in APPROVED_HORIZONS),
        "required_implementation_version": WFC_IMPLEMENTATION_VERSION,
        "status": "ok" if contract_ok else "failed",
    }])
    return {
        "master_corrected_wfc_patient_level.csv.gz": patient,
        "master_corrected_wfc_unique_patient.csv.gz": unique_patient,
        "master_corrected_wfc_seed_cancer_feature.csv": seed_feature,
        "master_corrected_wfc_seed_treatment_profile.csv": seed_profile,
        "master_corrected_wfc_seed_shared_across_cancers.csv": seed_shared,
        "master_corrected_wfc_seed_cross_cancer_rank_consistency.csv": pd.DataFrame(seed_cross_rank_rows),
        "master_corrected_wfc_final_cancer_unique_patient.csv": cancer_final,
        "master_corrected_wfc_final_treatment_profile_unique_patient.csv": final_profile,
        "master_corrected_wfc_final_shared_across_cancers.csv": shared_final,
        "master_corrected_wfc_cross_cancer_rank_consistency.csv": pd.DataFrame(cross_rank_rows),
        "master_corrected_wfc_cross_seed_rank_consistency.csv": split_rank,
        "master_corrected_wfc_seed_donor_threshold_sensitivity.csv": seed_sensitivity,
        "master_corrected_wfc_final_donor_threshold_sensitivity.csv": pd.DataFrame(final_sensitivity_rows),
        "master_corrected_wfc_coverage.csv": coverage,
        "master_corrected_wfc_unique_patient_coverage.csv": unique_coverage,
        "master_corrected_wfc_feature_block_manifest.csv": manifest,
        "master_corrected_wfc_patient_donor_support.csv": donor_support,
        "master_corrected_wfc_donor_audit.csv.gz": donor_audit,
        "master_corrected_wfc_integrity_checks.csv": integrity,
        "master_corrected_wfc_failures.csv": failures,
        "master_corrected_wfc_partition_status.csv": partition_status,
        "master_corrected_wfc_required_output_contract.csv": contract,
    }, bool(contract_ok)
