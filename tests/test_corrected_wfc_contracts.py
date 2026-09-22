#!/usr/bin/env python3
"""Focused synthetic checks for the corrected treatment-conditioned WFC workflow."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from TCGA_treatment_conditioned_WFC_NPJ_August import (
    APPROVED_HORIZONS,
    WFC_IMPLEMENTATION_VERSION,
    build_feature_units,
    collect_corrected_wfc,
    run_corrected_wfc_partition,
)


FEATURES = [
    "act_shared::P1",
    "act_specific_BRCA::P2",
    "clinbin::STAGE_A",
    "clinbin::STAGE_B",
    "clinbin::TX_CHEMO",
    "clinbin::TX_HORMONE",
    "clinbin::TX_RADIATION",
    "clinbin::TX_PHARMACEUTICAL",
]


def family(name):
    if str(name).startswith("act_"):
        return "pathway"
    if "TX_" in str(name):
        return "treatment"
    return "clinical_binary"


def scope(name):
    if str(name).startswith("act_shared::"):
        return "shared_across_cancers"
    if str(name).startswith("act_specific_"):
        return "cancer_specific"
    return "clinical_covariate"


def risk_samples(_model, X, _groups, _horizon, _alpha, _edges, S=32):
    X = np.asarray(X, dtype=float)
    noise = torch.randn(int(S), len(X)).cpu().numpy() * 0.01
    return X[:, 0][None, :] + 0.25 * X[:, 2][None, :] + 0.10 * X[:, 3][None, :] + noise


def wasserstein_rows(left, right):
    left = np.sort(np.asarray(left, dtype=float), axis=0)
    right = np.sort(np.asarray(right, dtype=float), axis=0)
    return np.mean(np.abs(left - right), axis=0)


def make_artifact():
    groups = ["Breast", "Colon", "Lung"]
    counts = {"BRCA": 25, "COAD": 12, "LUAD": 10}
    code_to_group = {"BRCA": 0, "COAD": 1, "LUAD": 2}
    train_rows = []
    g_train = []
    X_train = []
    cursor = 0
    for code, count in counts.items():
        default_tx = {
            "BRCA": (1, 1, 0, 0),
            "COAD": (1, 1, 1, 0),  # hormone is deliberately ignored outside BRCA
            "LUAD": (0, 1, 0, 1),  # hormone is deliberately ignored outside BRCA
        }[code]
        for within in range(count):
            # Give BRCA two treatment profiles with unequal donor-pool sizes.
            # This catches loss of patient-wise CRN alignment when the active
            # subset shrinks across donor repeats.
            tx = (1, 0, 1, 0) if code == "BRCA" and within >= 20 else default_tx
            train_rows.append({
                "PATIENT_ID": f"TRAIN-{code}-{within:03d}",
                "TX_CHEMO": tx[0],
                "TX_HORMONE": tx[1],
                "TX_RADIATION": tx[2],
                "TX_PHARMACEUTICAL": tx[3],
            })
            g_train.append(code_to_group[code])
            X_train.append([
                float(cursor + 1),
                float(within + 1) if code == "BRCA" else 0.0,
                float(within % 2),
                float((within + 1) % 2),
                *map(float, tx),
            ])
            cursor += 1

    test_rows = []
    g_test = []
    X_test = []
    test_specs = [
        ("BRCA", (1, 1, 0, 0)),
        ("BRCA", (1, 0, 1, 0)),
        ("COAD", (1, 1, 1, 0)),
        ("LUAD", (0, 1, 0, 1)),
    ]
    cancer_seen = {"BRCA": 0, "COAD": 0, "LUAD": 0}
    for code, tx in test_specs:
        cancer_seen[code] += 1
        test_rows.append({
            "PATIENT_ID": f"TEST-{code}-{cancer_seen[code]:03d}",
            "TX_CHEMO": tx[0],
            "TX_HORMONE": tx[1],
            "TX_RADIATION": tx[2],
            "TX_PHARMACEUTICAL": tx[3],
        })
        g_test.append(code_to_group[code])
        X_test.append([0.5, 0.2 if code == "BRCA" else 0.0, 1.0, 0.0, *map(float, tx)])

    Xtr = np.asarray(X_train, dtype=float)
    Xte = np.asarray(X_test, dtype=float)
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True)
    sd[~np.isfinite(sd) | (sd <= 0)] = 1.0
    return {
        "model": object(),
        "alpha_dt": torch.zeros(2),
        "dt_edges": np.asarray([0.0, 12.0, 36.0]),
        "Xtr_np_sub": Xtr,
        "Xte_np_sub": Xte,
        "Xte_t": torch.tensor((Xte - mu) / sd, dtype=torch.float32),
        "standardize_mu": mu,
        "standardize_sd": sd,
        "g_tr": np.asarray(g_train, dtype=int),
        "g_te": np.asarray(g_test, dtype=int),
        "g_te_t": torch.tensor(g_test, dtype=torch.long),
        "groups": groups,
        "feat_names": FEATURES,
        "meta_tr": pd.DataFrame(train_rows),
        "meta_te": pd.DataFrame(test_rows),
    }


def main():
    manifest = build_feature_units(FEATURES, family, scope)
    stage = manifest[manifest["feature"].eq("clinbin::STAGE")].iloc[0]
    assert int(stage["n_columns"]) == 2
    assert set(stage["feature_columns"].split(";")) == {"clinbin::STAGE_A", "clinbin::STAGE_B"}
    treatment = manifest[manifest["feature"].str.contains("TX_")]
    assert len(treatment) == 4 and not treatment["ranked"].astype(bool).any()

    # The encoded-block rule is generic and is not limited to stage/grade/tumor
    # category. Any additional one-hot predictor is copied as one donor block.
    generic = build_feature_units(
        ["clinbin::RACE_A", "clinbin::RACE_B", "clinbin::RACE_C", "clincont::AGE"],
        family,
        scope,
    )
    race = generic[generic["feature"].eq("clinbin::RACE")].iloc[0]
    assert int(race["n_columns"]) == 3
    assert set(race["feature_columns"].split(";")) == {
        "clinbin::RACE_A", "clinbin::RACE_B", "clinbin::RACE_C"
    }

    with tempfile.TemporaryDirectory(prefix="corrected_wfc_test_") as tmp:
        root = Path(tmp) / "config_root"
        art = make_artifact()
        for horizon in APPROVED_HORIZONS:
            out = root / "seed42" / "analysis" / f"corrected_wfc_parts/H{int(horizon)}_all_cancers"
            run_corrected_wfc_partition(
                art=art,
                outdir=out,
                configuration="PathClin",
                cfg_tag="cfg_test",
                seed=42,
                horizon=horizon,
                risk_samples_fn=risk_samples,
                wasserstein_fn=wasserstein_rows,
                family_fn=family,
                scope_fn=scope,
                posterior_draws=16,
            )
            support = pd.read_csv(out / "corrected_wfc_patient_donor_support.csv")
            actual_M = dict(zip(support["patient_id"], support["actual_M"]))
            assert actual_M == {
                "TEST-BRCA-001": 20,
                "TEST-BRCA-002": 5,
                "TEST-COAD-001": 12,
                "TEST-LUAD-001": 10,
            }
            profiles = dict(zip(support["patient_id"], support["treatment_profile"]))
            assert profiles == {
                "TEST-BRCA-001": "TX=Chemo+Hormone",
                "TEST-BRCA-002": "TX=Chemo+Radiation",
                "TEST-COAD-001": "TX=Chemo+Radiation",
                "TEST-LUAD-001": "TX=Pharmaceutical",
            }
            audit = pd.read_csv(out / "corrected_wfc_donor_audit.csv.gz")
            assert audit.groupby("patient_id")["donor_patient_id"].nunique().to_dict() == {
                "TEST-BRCA-001": 20,
                "TEST-BRCA-002": 5,
                "TEST-COAD-001": 12,
                "TEST-LUAD-001": 10,
            }
            assert audit["donor_patient_id"].str.startswith("TRAIN-").all()
            assert (audit["treatment_profile"] == audit["donor_treatment_profile"]).all()
            patient = pd.read_csv(out / "corrected_wfc_patient_level.csv.gz")
            assert not patient["feature"].str.contains("TX_").any()
            null_feature = patient[
                patient["cancer"].eq("BRCA")
                & patient["feature"].eq("act_specific_BRCA::P2")
            ]
            assert len(null_feature) == 2
            assert np.allclose(null_feature["mean_W1"].to_numpy(float), 0.0, atol=1e-12)
            integrity = pd.read_csv(out / "corrected_wfc_integrity_checks.csv")
            assert (
                integrity.loc[
                    integrity["check"].eq("common_random_numbers_null_replay"), "status"
                ].eq("ok").all()
            )
            partition_status = pd.read_csv(out / "corrected_wfc_partition_status.csv")
            assert partition_status["implementation_version"].eq(WFC_IMPLEMENTATION_VERSION).all()
            for cancer, group in patient.groupby("cancer"):
                expected_ids = set(
                    art["meta_te"].loc[
                        np.asarray(art["g_te"]) == {"BRCA": 0, "COAD": 1, "LUAD": 2}[cancer],
                        "PATIENT_ID",
                    ]
                )
                assert all(set(rows["patient_id"]) == expected_ids for _, rows in group.groupby("feature"))

            if horizon == 12.0:
                repeat_out = Path(tmp) / "repeat_seed42_H12"
                run_corrected_wfc_partition(
                    art=art,
                    outdir=repeat_out,
                    configuration="PathClin",
                    cfg_tag="cfg_test",
                    seed=42,
                    horizon=horizon,
                    risk_samples_fn=risk_samples,
                    wasserstein_fn=wasserstein_rows,
                    family_fn=family,
                    scope_fn=scope,
                    posterior_draws=16,
                )
                repeat_audit = pd.read_csv(repeat_out / "corrected_wfc_donor_audit.csv.gz")
                pd.testing.assert_frame_equal(audit, repeat_audit, check_dtype=False)
                repeat_patient = pd.read_csv(repeat_out / "corrected_wfc_patient_level.csv.gz")
                compare_cols = ["cancer", "patient_id", "feature", "mean_W1", "sd_W1", "actual_M"]
                pd.testing.assert_frame_equal(
                    patient[compare_cols], repeat_patient[compare_cols], check_dtype=False
                )

        try:
            run_corrected_wfc_partition(
                art=art,
                outdir=Path(tmp) / "invalid_horizon",
                configuration="PathClin",
                cfg_tag="cfg_test",
                seed=42,
                horizon=26.0,
                risk_samples_fn=risk_samples,
                wasserstein_fn=wasserstein_rows,
                family_fn=family,
                scope_fn=scope,
                posterior_draws=16,
            )
        except ValueError as exc:
            assert "not approved" in str(exc)
        else:
            raise AssertionError("A nonapproved WFC horizon was not rejected")

        tables, ok = collect_corrected_wfc(
            {"PathClin": root}, expected_seeds=[42], bootstrap_samples=50
        )
        assert ok
        shared = tables["master_corrected_wfc_final_shared_across_cancers.csv"]
        primary = shared[shared["donor_threshold"].eq(10)]
        assert set(primary["feature"]) == {"act_shared::P1", "clinbin::STAGE"}
        assert "act_specific_BRCA::P2" not in set(shared["feature"])
        expected = primary[["BRCA_WFC", "COAD_WFC", "LUAD_WFC"]].mean(axis=1)
        assert np.allclose(primary["primary_equal_cancer_shared_WFC"], expected)
        assert primary["numerical_consistency_cutoff_applied"].eq(False).all()
        assert set(primary.columns).issuperset({"BRCA_rank", "COAD_rank", "LUAD_rank", "shared_rank"})

        # A collector must reject stale partitions from the pre-hotfix WFC
        # implementation even when their filenames and completion status look valid.
        stale_status_path = (
            root / "seed42" / "analysis" / "corrected_wfc_parts/H12_all_cancers"
            / "corrected_wfc_partition_status.csv"
        )
        stale_status = pd.read_csv(stale_status_path)
        stale_status["implementation_version"] = "pre_full_batch_crn"
        stale_status.to_csv(stale_status_path, index=False)
        _, stale_ok = collect_corrected_wfc(
            {"PathClin": root}, expected_seeds=[42], bootstrap_samples=50
        )
        assert not stale_ok

    print("CORRECTED_WFC_SYNTHETIC_TEST_OK")


if __name__ == "__main__":
    main()
