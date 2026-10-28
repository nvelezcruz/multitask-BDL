"""Small public contracts for the fixed-support post-fit analysis."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
PACKAGE = REPOSITORY / "code" / "fixed_contrast"
sys.path.insert(0, str(PACKAGE))

import fixed_contrast_io  # noqa: E402
import run_fixed_contrast_wasserstein as runner  # noqa: E402


class FixedContrastPublicContractsTest(unittest.TestCase):
    def test_fixed_profile_support_boundaries(self) -> None:
        self.assertTrue(runner.profile_is_supported(12, 3))
        self.assertFalse(runner.profile_is_supported(11, 3))
        self.assertFalse(runner.profile_is_supported(12, 2))

    def test_untreated_profile_is_not_an_interaction_pair(self) -> None:
        profiles = [
            {"code": "None", "active": False},
            {"code": "C", "active": True},
            {"code": "C+R", "active": True},
        ]
        self.assertEqual(runner.active_pair_indices(profiles), [(1, 2)])

    def test_pathnoudp_example_resolves_to_included_model_script(self) -> None:
        config = fixed_contrast_io.load_config(
            PACKAGE / "input_manifest.PathNoUDP.example.json"
        )
        self.assertEqual(config.name, "PathNoUDP")
        self.assertTrue(config.model_script.is_file())
        self.assertEqual(
            fixed_contrast_io.artifact_path(config, 42).name,
            "august_fitted_artifacts_seed42.pt",
        )
        self.assertEqual(
            fixed_contrast_io.oof_path(config, 42).name,
            "posthoc_crossfit_oof_native_risks.csv",
        )

    def test_runner_is_separate_from_legacy_donor_driver(self) -> None:
        source = (PACKAGE / "run_fixed_contrast_wasserstein.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("run_revised_wasserstein", source)
        self.assertIn("PROFILE_MIN_N = 12", source)
        self.assertIn("PROFILE_MIN_EVENTS = 3", source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
