"""Small synthetic checks for the release runner's output contracts."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from validate_seed_outputs import (
    BASELINE_MODELS,
    CFG_TAG,
    HORIZONS,
    WFC_FILES,
    check_baseline,
    check_dgp,
    check_wfc,
)


class ReleaseValidationTests(unittest.TestCase):
    def test_dgp_result_status(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / CFG_TAG
            cfg.mkdir()
            result = cfg / f"{CFG_TAG}_results.csv"
            pd.DataFrame([{"status": "ok"}]).to_csv(result, index=False)
            check_dgp(42, root)
            pd.DataFrame([{"status": "failed"}]).to_csv(result, index=False)
            with self.assertRaises(RuntimeError):
                check_dgp(42, root)

    def test_baseline_contract(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {"model": model, "eval_horizon_months": horizon, "error": ""}
                for model in BASELINE_MODELS for horizon in HORIZONS
            ]
            metrics = root / "tcga_baselines_npj_allmodels_pfi.csv"
            pd.DataFrame(rows).to_csv(metrics, index=False)
            pd.DataFrame([{"patient": "synthetic"}]).to_csv(
                root / "tcga_baselines_patient_predictions.csv.gz", index=False,
                compression="gzip",
            )
            check_baseline(42, root)
            self.assertTrue((root / "baseline_required_output_contract.csv").is_file())
            pd.DataFrame(rows[:-1]).to_csv(metrics, index=False)
            with self.assertRaises(RuntimeError):
                check_baseline(42, root)

    def test_wfc_contract(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in WFC_FILES:
                (root / name).write_text("synthetic\n")
            pd.DataFrame({"cancer": ["BRCA", "COAD", "LUAD"]}).to_csv(
                root / "corrected_wfc_seed_cancer_feature_summary.csv", index=False,
            )
            pd.DataFrame({"implementation_version": ["full_batch_crn_v2"]}).to_csv(
                root / "corrected_wfc_partition_status.csv", index=False,
            )
            pd.DataFrame({
                "check": ["common_random_numbers_null_replay"], "status": ["ok"],
            }).to_csv(root / "corrected_wfc_integrity_checks.csv", index=False)
            check_wfc(18, root)
            self.assertTrue((root / "corrected_wfc_required_output_contract.csv").is_file())


if __name__ == "__main__":
    unittest.main()
