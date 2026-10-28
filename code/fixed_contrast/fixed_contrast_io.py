"""Portable input handling for the fixed-support Wasserstein analysis.

This module intentionally contains no donor selection or donor substitution.
It loads a saved DGP artifact, identifies molecular feature blocks, and resolves
the matched out-of-fold calibration table declared in a user-supplied JSON
configuration.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


CANCERS = ("BRCA", "COAD", "LUAD")
GROUP_TO_CANCER = {"Breast": "BRCA", "Colon": "COAD", "Lung": "LUAD"}
TX_BASES = ("TX_CHEMO", "TX_HORMONE", "TX_RADIATION", "TX_PHARMACEUTICAL")
TX_LABELS = {
    "TX_CHEMO": "Chemotherapy",
    "TX_HORMONE": "Hormone therapy",
    "TX_RADIATION": "Radiotherapy",
    "TX_PHARMACEUTICAL": "Pharmaceutical therapy",
}


@dataclass(frozen=True)
class Config:
    """Locations and feature families for one saved DGP configuration."""

    name: str
    model_script: Path
    artifact_root: Path
    artifact_pattern: str
    oof_root: Path
    oof_pattern: str
    molecular_types: tuple[str, ...]
    molecular_label: str


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_config(config_file: str | Path) -> Config:
    """Load one public configuration template without embedding local paths."""

    source = Path(config_file).expanduser().resolve()
    with source.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    required = {
        "name",
        "model_script",
        "artifact_root",
        "artifact_pattern",
        "oof_root",
        "oof_pattern",
        "molecular_types",
        "molecular_label",
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise ValueError(f"Configuration file is missing required keys: {missing}")
    molecular_types = tuple(str(value) for value in raw["molecular_types"])
    if not molecular_types:
        raise ValueError("molecular_types must be nonempty")
    return Config(
        name=str(raw["name"]),
        model_script=_resolve_path(raw["model_script"], source.parent),
        artifact_root=_resolve_path(raw["artifact_root"], source.parent),
        artifact_pattern=str(raw["artifact_pattern"]),
        oof_root=_resolve_path(raw["oof_root"], source.parent),
        oof_pattern=str(raw["oof_pattern"]),
        molecular_types=molecular_types,
        molecular_label=str(raw["molecular_label"]),
    )


def config_to_manifest(config: Config) -> dict[str, object]:
    """Return paths and settings written into a run manifest for auditing."""

    return {
        "name": config.name,
        "model_script": str(config.model_script),
        "artifact_root": str(config.artifact_root),
        "artifact_pattern": config.artifact_pattern,
        "oof_root": str(config.oof_root),
        "oof_pattern": config.oof_pattern,
        "molecular_types": list(config.molecular_types),
        "molecular_label": config.molecular_label,
    }


def artifact_path(config: Config, seed: int) -> Path:
    return config.artifact_root / config.artifact_pattern.format(seed=int(seed))


def oof_path(config: Config, seed: int) -> Path:
    return config.oof_root / config.oof_pattern.format(seed=int(seed))


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def load_model_module(config: Config, runtime_dir: str | Path) -> Any:
    """Import model classes solely so a matching serialized artifact can load.

    The DGP scripts have module-level paths used by their training workflow.
    During artifact-only analysis those paths are redirected to ``runtime_dir``;
    no preprocessing or model fitting is invoked here.
    """

    model_script = Path(config.model_script)
    if not model_script.is_file():
        raise FileNotFoundError(f"Model script not found: {model_script}")
    runtime = Path(runtime_dir).resolve()
    model_tag = "PathNoUDP_MutTMB" if config.name == "PathNoUDP" else config.name
    safe_root = runtime / f"fixed_contrast_August_{model_tag}"
    safe_root.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(safe_root / "mpl")
    os.environ["TCGA_ROOT"] = str(safe_root)
    os.environ["NPJ_SCRATCH_ROOT"] = str(safe_root)
    os.environ["OUTDIR"] = str(safe_root / "artifact_load")
    os.environ["PREP_OUTDIR"] = str(safe_root / "preprocessing")
    parent = str(model_script.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    module_spec = importlib.util.spec_from_file_location(
        f"fixed_contrast_model_{config.name.lower()}", model_script
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Cannot import model script {model_script}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def load_artifact(config: Config, seed: int, model_module: Any) -> tuple[Mapping[str, Any], Path]:
    """Load one saved artifact with its original model classes available."""

    import __main__
    import torch

    for name, value in vars(model_module).items():
        if isinstance(value, type) or callable(value):
            setattr(__main__, name, value)
    # Artifacts written under NumPy 2 may refer to these module aliases.
    sys.modules["numpy._core"] = np.core
    sys.modules["numpy._core.multiarray"] = np.core.multiarray
    sys.modules["numpy._core.numeric"] = np.core.numeric

    path = artifact_path(config, seed)
    if not path.is_file():
        raise FileNotFoundError(f"Saved DGP artifact not found: {path}")
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in artifact:
        raise KeyError(f"Artifact {path} has no model entry")
    artifact["model"].eval()
    return artifact, path


def patient_ids(meta: pd.DataFrame, n: int) -> np.ndarray:
    for column in ("PATIENT_ID", "patient_id", "bcr_patient_barcode"):
        if column in meta.columns:
            output = meta[column].astype(str).str.strip().to_numpy()
            break
    else:
        output = np.asarray([f"test_{index}" for index in range(n)], dtype=object)
    if len(output) != n or len(set(output.tolist())) != n:
        raise ValueError("Patient identifiers are absent, duplicated, or misaligned")
    return output


def cancer_codes(artifact: Mapping[str, Any], split: str) -> np.ndarray:
    index = np.asarray(artifact[f"g_{split}"], dtype=int)
    groups = [str(value) for value in artifact["groups"]]
    output = np.asarray(
        [GROUP_TO_CANCER.get(groups[value], groups[value].upper()) for value in index],
        dtype=object,
    )
    if not set(output).issubset(set(CANCERS)):
        raise ValueError(f"Unexpected cancer labels: {sorted(set(output))}")
    return output


def treatment_columns(feature_names: list[str]) -> tuple[list[int], list[str]]:
    indices: list[int] = []
    bases: list[str] = []
    for base in TX_BASES:
        hits = [
            index
            for index, name in enumerate(feature_names)
            if str(name).split("::")[-1] == base
        ]
        if len(hits) != 1:
            raise ValueError(f"Expected one model input for {base}, found {hits}")
        indices.append(hits[0])
        bases.append(base)
    return indices, bases


def feature_type_from_family(family: str) -> str:
    return "clinical_categorical" if str(family) == "clinical_binary" else str(family)


def _is_treatment_feature(name: str, family: str) -> bool:
    base = str(name).split("::")[-1]
    return str(family) == "treatment" or base in {
        "TX_CHEMO",
        "TX_HORMONE",
        "TX_RADIATION",
        "TX_PHARMACEUTICAL",
        "TX_EXPLICIT_NO_TREATMENT",
    }


def _encoded_source(name: str) -> str | None:
    text = str(name)
    if "=" in text:
        return text.rsplit("=", 1)[0]
    if text.startswith("clinbin::") and "_" in text.split("::", 1)[1]:
        return text.rsplit("_", 1)[0]
    return None


def build_feature_units(
    feature_names: Iterable[str], family_function: Any, scope_function: Any
) -> pd.DataFrame:
    """Build feature blocks without importing the legacy donor-WFC helper."""

    records: list[dict[str, object]] = []
    scalar: list[tuple[int, str, str, str]] = []
    encoded: dict[str, list[tuple[int, str, str, str]]] = {}
    for index, raw_name in enumerate(feature_names):
        name = str(raw_name)
        family = str(family_function(name))
        scope = str(scope_function(name))
        if _is_treatment_feature(name, family):
            records.append(
                {
                    "feature": name,
                    "feature_columns": name,
                    "column_indices": str(index),
                    "n_columns": 1,
                    "feature_family": family,
                    "feature_scope": scope,
                    "ranked": False,
                    "exclusion_reason": "treatment indicator used only for conditioning",
                }
            )
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
        families = sorted({member[2] for member in members})
        scopes = sorted({member[3] for member in members})
        if len(families) != 1 or len(scopes) != 1:
            raise ValueError(
                f"Encoded feature block {source!r} has inconsistent family/scope labels"
            )
        records.append(
            {
                "feature": source,
                "feature_columns": ";".join(member[1] for member in members),
                "column_indices": ";".join(str(member[0]) for member in members),
                "n_columns": len(members),
                "feature_family": families[0],
                "feature_scope": scopes[0],
                "ranked": True,
                "exclusion_reason": "",
            }
        )
    for index, name, family, scope in scalar:
        records.append(
            {
                "feature": name,
                "feature_columns": name,
                "column_indices": str(index),
                "n_columns": 1,
                "feature_family": family,
                "feature_scope": scope,
                "ranked": True,
                "exclusion_reason": "",
            }
        )
    manifest = pd.DataFrame(records)
    if manifest.empty or not bool(manifest["ranked"].any()):
        raise ValueError("No non-treatment feature blocks were identified")
    return manifest.sort_values(["ranked", "feature"], ascending=[False, True]).reset_index(
        drop=True
    )


def build_units(artifact: Mapping[str, Any], model_module: Any) -> pd.DataFrame:
    manifest = build_feature_units(
        [str(name) for name in artifact["feat_names"]],
        model_module._feature_family_for_mechanism,
        model_module._feature_scope_for_mechanism,
    )
    manifest = manifest[manifest["ranked"]].copy()
    manifest["feature_type"] = manifest["feature_family"].map(feature_type_from_family)
    manifest["columns"] = manifest["column_indices"].map(
        lambda text: tuple(int(value) for value in str(text).split(";") if value)
    )
    return manifest.reset_index(drop=True)
