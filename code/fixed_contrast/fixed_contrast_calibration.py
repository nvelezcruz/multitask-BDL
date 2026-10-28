#!/usr/bin/env python3
"""Cancer-specific fixed-slope calibration for fixed-contrast analyses.

This module fits the calibration model selected for the manuscript::

    logit(p_calibrated_i(H)) = gamma_{cancer(i), H} + logit(p_native_i(H))

The slope is exactly one.  Each ``gamma`` is estimated from outer-training
out-of-fold (OOF) predictions using fixed-horizon IPCW weights.  The censoring
Kaplan--Meier curve is always fitted to the *pooled* outer-training cohort and
then used for every cancer-specific intercept.  The saved DGP is never refit.

The implementation deliberately fails on incomplete OOF predictions,
artifact/OOF misalignment, insufficient cases or controls, a degenerate
censoring distribution, or failure to bracket the score root.  It never
silently substitutes the identity transform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq


HORIZONS = (18, 24, 36)
EPS = 1e-5
MIN_N = 20
MIN_EVENTS = 5
MIN_NONEVENTS = 5
TIME_ATOL = 1e-8
SCORE_XTOL = 1e-12
SCORE_RTOL = 1e-12
SCORE_BRACKET = (-60.0, 60.0)

CALIBRATION_KIND = "cancer_specific_intercept_fixed_slope_1"
PREDICTION_SCALE = (
    "cancer_specific_fixed_slope_intercept_recalibrated_PFI_event_probability"
)
KM_SCOPE = "pooled_outer_training_all_cancers"

GROUP_TO_CANCER = {"Breast": "BRCA", "Colon": "COAD", "Lung": "LUAD"}
CANCER_TO_GROUP = {value: key for key, value in GROUP_TO_CANCER.items()}


class CalibrationError(RuntimeError):
    """Raised when a valid manuscript-scale calibrator cannot be fitted."""


def _as_1d(values: Any, *, name: str, dtype: Any) -> np.ndarray:
    out = np.asarray(values, dtype=dtype).reshape(-1)
    if out.ndim != 1:
        raise CalibrationError(f"{name} could not be converted to one dimension")
    return out


def _expit_numpy(x: Any) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x_arr, dtype=np.float64)
    positive = x_arr >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-x_arr[positive]))
    ex = np.exp(x_arr[~positive])
    out[~positive] = ex / (1.0 + ex)
    return out


def _logit_numpy(p: Any, eps: float = EPS) -> np.ndarray:
    p_arr = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p_arr) - np.log1p(-p_arr)


def _array_hash(*arrays: Any) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        arr = np.ascontiguousarray(np.asarray(value))
        digest.update(str(arr.shape).encode("utf-8"))
        digest.update(str(arr.dtype).encode("utf-8"))
        digest.update(arr.tobytes())
    return digest.hexdigest()


def _text_hash(values: Iterable[Any]) -> str:
    payload = "\x1f".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity_hash(path: str | Path) -> str:
    source = Path(path).resolve()
    stat = source.stat()
    return _text_hash((source, stat.st_size, stat.st_mtime_ns))


def _km_predict_1d(kmf: Any, times: Any) -> np.ndarray:
    """Evaluate a lifelines KM curve and always return a one-dimensional array.

    ``KaplanMeierFitter.predict`` returns a scalar for a length-one vector in
    some lifelines releases.  That behavior previously triggered a caught
    indexing error and silently replaced IPCW weights by one.  Using
    ``survival_function_at_times`` plus the explicit size check avoids that
    failure mode.
    """

    query = np.atleast_1d(np.asarray(times, dtype=np.float64)).reshape(-1)
    predicted = np.asarray(
        kmf.survival_function_at_times(query), dtype=np.float64
    ).reshape(-1)
    if predicted.size != query.size:
        raise CalibrationError(
            "Censoring-KM prediction changed length: "
            f"requested {query.size}, received {predicted.size}"
        )
    return predicted


def fit_pooled_censoring_km(times: Any, events: Any) -> Any:
    """Fit ``G(t)=P(C>t)`` on all outer-training patients pooled by cancer."""

    try:
        from lifelines import KaplanMeierFitter
    except Exception as exc:  # pragma: no cover - dependency is present in production
        raise CalibrationError("lifelines is required for IPCW calibration") from exc

    t = _as_1d(times, name="times", dtype=np.float64)
    e = _as_1d(events, name="events", dtype=np.int64)
    if t.size != e.size or t.size == 0:
        raise CalibrationError("times/events are empty or have unequal lengths")
    if not np.all(np.isfinite(t)) or np.any(t < 0.0):
        raise CalibrationError("outer-training times must be finite and nonnegative")
    if not np.all(np.isin(e, (0, 1))):
        raise CalibrationError("outer-training event indicators must be binary")

    kmf = KaplanMeierFitter()
    kmf.fit(t, event_observed=(e == 0).astype(np.int64))
    return kmf


def ipcw_binary_targets(
    times: Any,
    events: Any,
    horizon: float,
    *,
    censoring_km: Any | None = None,
) -> dict[str, np.ndarray | float]:
    """Create fixed-horizon event labels and pooled-censoring IPCW weights.

    Cases are observed PFI events at or before ``horizon`` and receive
    ``1/G(T_i-)``.  Controls are patients observed beyond ``horizon`` and
    receive ``1/G(horizon)``.  Patients censored at or before the horizon have
    weight zero and are excluded from fitting.
    """

    t = _as_1d(times, name="times", dtype=np.float64)
    e = _as_1d(events, name="events", dtype=np.int64)
    H = float(horizon)
    if t.size != e.size or t.size == 0:
        raise CalibrationError("times/events are empty or have unequal lengths")
    if not np.isfinite(H) or H <= 0.0:
        raise CalibrationError(f"horizon must be positive and finite, got {horizon!r}")
    if not np.all(np.isfinite(t)) or np.any(t < 0.0):
        raise CalibrationError("times must be finite and nonnegative")
    if not np.all(np.isin(e, (0, 1))):
        raise CalibrationError("event indicators must be binary")

    kmf = censoring_km if censoring_km is not None else fit_pooled_censoring_km(t, e)
    cases = (e == 1) & (t <= H)
    controls = t > H
    early_censored = (e == 0) & (t <= H)
    usable = cases | controls
    y = cases.astype(np.int64)
    weights = np.zeros(t.size, dtype=np.float64)

    G_H = float(_km_predict_1d(kmf, np.asarray([H], dtype=np.float64))[0])
    if not np.isfinite(G_H) or G_H <= 0.0:
        raise CalibrationError(f"Pooled censoring KM is nonpositive at H={H}: {G_H}")

    G_tminus = np.full(t.size, np.nan, dtype=np.float64)
    if np.any(cases):
        case_times_left = np.maximum(np.nextafter(t[cases], -np.inf), 0.0)
        case_survival = _km_predict_1d(kmf, case_times_left)
        if np.any(~np.isfinite(case_survival)) or np.any(case_survival <= 0.0):
            raise CalibrationError(
                f"Pooled censoring KM is nonpositive immediately before a case at H={H}"
            )
        G_tminus[cases] = case_survival
        weights[cases] = 1.0 / case_survival
    weights[controls] = 1.0 / G_H

    if np.any(~np.isfinite(weights[usable])) or np.any(weights[usable] <= 0.0):
        raise CalibrationError(f"Non-finite or nonpositive IPCW weights at H={H}")

    return {
        "y": y,
        "weights": weights,
        "usable": usable,
        "cases": cases,
        "controls": controls,
        "early_censored": early_censored,
        "G_H": G_H,
        "G_tminus": G_tminus,
    }


def validate_oof_against_artifact(
    oof: pd.DataFrame,
    artifact: Mapping[str, Any],
    *,
    horizons: Sequence[int] = HORIZONS,
    time_atol: float = TIME_ATOL,
    oof_path: str | Path | None = None,
    artifact_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Align OOF rows by ``train_index`` and audit against artifact arrays."""

    required = {"train_index", "time", "event", "group_idx", "group"}
    required.update(f"p_native_oof_H{int(H)}" for H in horizons)
    missing = sorted(required.difference(oof.columns))
    if missing:
        raise CalibrationError(f"OOF file is missing required columns: {missing}")
    for key in ("t_tr", "e_tr", "g_tr", "groups"):
        if key not in artifact:
            raise CalibrationError(f"Artifact is missing authoritative field {key!r}")

    t_art = _as_1d(artifact["t_tr"], name="artifact t_tr", dtype=np.float64)
    e_art = _as_1d(artifact["e_tr"], name="artifact e_tr", dtype=np.int64)
    g_art = _as_1d(artifact["g_tr"], name="artifact g_tr", dtype=np.int64)
    n = int(t_art.size)
    if e_art.size != n or g_art.size != n:
        raise CalibrationError("Artifact t_tr/e_tr/g_tr lengths differ")

    index_numeric = pd.to_numeric(oof["train_index"], errors="coerce").to_numpy()
    if np.any(~np.isfinite(index_numeric)) or np.any(index_numeric != np.floor(index_numeric)):
        raise CalibrationError("OOF train_index contains non-integer or missing values")
    index = index_numeric.astype(np.int64)
    if np.unique(index).size != index.size:
        raise CalibrationError("OOF train_index contains duplicates")
    expected = np.arange(n, dtype=np.int64)
    if index.size != n or not np.array_equal(np.sort(index), expected):
        missing_index = np.setdiff1d(expected, index)
        extra_index = np.setdiff1d(index, expected)
        raise CalibrationError(
            "OOF train_index does not provide 100% outer-training coverage; "
            f"n_artifact={n}, n_oof={index.size}, missing={missing_index[:10].tolist()}, "
            f"extra={extra_index[:10].tolist()}"
        )

    aligned = oof.assign(train_index=index).sort_values("train_index").reset_index(drop=True)
    t_oof = pd.to_numeric(aligned["time"], errors="coerce").to_numpy(dtype=np.float64)
    e_oof = pd.to_numeric(aligned["event"], errors="coerce").to_numpy(dtype=np.float64)
    g_oof = pd.to_numeric(aligned["group_idx"], errors="coerce").to_numpy(dtype=np.float64)
    if np.any(~np.isfinite(t_oof)) or np.any(~np.isfinite(e_oof)) or np.any(~np.isfinite(g_oof)):
        raise CalibrationError("OOF time/event/group_idx contains non-finite values")

    max_time_abs_diff = float(np.max(np.abs(t_oof - t_art))) if n else 0.0
    if max_time_abs_diff > float(time_atol):
        raise CalibrationError(
            f"OOF time is misaligned with artifact t_tr (max abs diff={max_time_abs_diff:.3g})"
        )
    if not np.array_equal(e_oof.astype(np.int64), e_art):
        raise CalibrationError("OOF event is misaligned with artifact e_tr")
    if not np.array_equal(g_oof.astype(np.int64), g_art):
        raise CalibrationError("OOF group_idx is misaligned with artifact g_tr")

    group_names = [str(value) for value in artifact["groups"]]
    if not group_names:
        raise CalibrationError("Artifact group-name list is empty")
    if np.any(g_art < 0) or np.any(g_art >= len(group_names)):
        raise CalibrationError("Artifact g_tr contains an invalid group index")
    expected_labels = np.asarray([group_names[int(value)] for value in g_art], dtype=object)
    observed_labels = aligned["group"].astype(str).to_numpy(dtype=object)
    if not np.array_equal(expected_labels, observed_labels):
        raise CalibrationError("OOF group labels are misaligned with artifact groups/g_tr")

    probability_qc: dict[str, dict[str, Any]] = {}
    for H in horizons:
        column = f"p_native_oof_H{int(H)}"
        probability = pd.to_numeric(aligned[column], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(probability)
        if not np.all(finite):
            raise CalibrationError(
                f"OOF {column} is incomplete/non-finite: {int(np.sum(finite))}/{n} finite"
            )
        if np.any((probability < 0.0) | (probability > 1.0)):
            raise CalibrationError(f"OOF {column} contains values outside [0,1]")
        probability_qc[str(int(H))] = {
            "column": column,
            "n_finite": int(np.sum(finite)),
            "coverage": float(np.mean(finite)),
            "minimum": float(np.min(probability)),
            "maximum": float(np.max(probability)),
            "n_clipped_for_logit": int(np.sum((probability < EPS) | (probability > 1.0 - EPS))),
        }

    artifact_arrays_sha256 = _array_hash(t_art, e_art, g_art)
    qc: dict[str, Any] = {
        "status": "ok",
        "n_outer_train": n,
        "n_oof_rows": int(len(aligned)),
        "oof_coverage": float(len(aligned) / max(n, 1)),
        "train_index_min": int(index.min()) if index.size else None,
        "train_index_max": int(index.max()) if index.size else None,
        "train_index_unique": int(np.unique(index).size),
        "max_time_abs_diff": max_time_abs_diff,
        "time_atol": float(time_atol),
        "event_exact_match": True,
        "group_index_exact_match": True,
        "group_label_exact_match": True,
        "groups": group_names,
        "probability_qc": probability_qc,
        "artifact_arrays_sha256": artifact_arrays_sha256,
        "km_source_sha256": _array_hash(t_art, e_art),
        "km_scope": KM_SCOPE,
    }
    if oof_path is not None:
        qc["oof_path"] = str(Path(oof_path).resolve())
        qc["oof_file_sha256"] = file_sha256(oof_path)
    if artifact_path is not None:
        qc["artifact_path"] = str(Path(artifact_path).resolve())
        qc["artifact_file_identity_sha256"] = _file_identity_hash(artifact_path)
    return aligned, qc


def _fit_one_intercept(
    probability: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    group_mask: np.ndarray,
    group_name: str,
    group_idx: int,
    cancer_code: str,
    horizon: int,
    ipcw: Mapping[str, Any],
    alignment_qc: Mapping[str, Any],
    config: str | None,
    seed: int | None,
) -> tuple[float, dict[str, Any]]:
    usable = np.asarray(ipcw["usable"], dtype=bool)
    cases = np.asarray(ipcw["cases"], dtype=bool)
    controls = np.asarray(ipcw["controls"], dtype=bool)
    early_censored = np.asarray(ipcw["early_censored"], dtype=bool)
    fit_mask = np.asarray(group_mask, dtype=bool) & usable
    n_original = int(np.sum(group_mask))
    n_fit = int(np.sum(fit_mask))
    n_pos = int(np.sum(cases & group_mask))
    n_neg = int(np.sum(controls & group_mask))
    n_early_censored = int(np.sum(early_censored & group_mask))
    if n_fit < MIN_N or n_pos < MIN_EVENTS or n_neg < MIN_NONEVENTS:
        raise CalibrationError(
            "Insufficient OOF support for cancer-specific fixed-slope calibration: "
            f"group={group_name}, H={horizon}, n_fit={n_fit}, "
            f"events={n_pos}, nonevents={n_neg}; required "
            f"n>={MIN_N}, events>={MIN_EVENTS}, nonevents>={MIN_NONEVENTS}"
        )

    p_fit_raw = probability[fit_mask]
    p_fit = np.clip(p_fit_raw, EPS, 1.0 - EPS)
    z = _logit_numpy(p_fit)
    y_fit = y[fit_mask].astype(np.float64)
    w_fit = weights[fit_mask].astype(np.float64)
    if np.any(~np.isfinite(z)) or np.any(~np.isfinite(w_fit)) or np.any(w_fit <= 0.0):
        raise CalibrationError(
            f"Non-finite logit or IPCW weight for group={group_name}, H={horizon}"
        )

    def score(gamma: float) -> float:
        return float(np.sum(w_fit * (_expit_numpy(z + gamma) - y_fit)))

    bracket_low, bracket_high = SCORE_BRACKET
    score_low = score(bracket_low)
    score_high = score(bracket_high)
    if not (np.isfinite(score_low) and np.isfinite(score_high)):
        raise CalibrationError(
            f"Non-finite calibration score bracket for group={group_name}, H={horizon}"
        )
    if not (score_low < 0.0 < score_high):
        raise CalibrationError(
            "Calibration score root is not bracketed; refusing identity fallback: "
            f"group={group_name}, H={horizon}, score({bracket_low})={score_low}, "
            f"score({bracket_high})={score_high}"
        )

    gamma, result = brentq(
        score,
        bracket_low,
        bracket_high,
        xtol=SCORE_XTOL,
        rtol=SCORE_RTOL,
        maxiter=500,
        full_output=True,
        disp=False,
    )
    if not bool(result.converged) or not np.isfinite(gamma):
        raise CalibrationError(
            f"Intercept solver did not converge for group={group_name}, H={horizon}"
        )

    calibrated = _expit_numpy(z + gamma)
    score_residual = score(float(gamma))
    weight_sum = float(np.sum(w_fit))
    weighted_event_rate = float(np.sum(w_fit * y_fit) / weight_sum)
    weighted_predicted_rate = float(np.sum(w_fit * calibrated) / weight_sum)
    weighted_residual = weighted_predicted_rate - weighted_event_rate
    tolerance = max(1e-10, 1e-10 * weight_sum)
    if abs(score_residual) > tolerance or abs(weighted_residual) > 1e-10:
        raise CalibrationError(
            "Calibration score residual exceeds tolerance: "
            f"group={group_name}, H={horizon}, score={score_residual}, "
            f"weighted residual={weighted_residual}"
        )

    case_G = np.asarray(ipcw["G_tminus"], dtype=np.float64)[cases & group_mask]
    qc: dict[str, Any] = {
        "status": "ok",
        "config": config,
        "seed": int(seed) if seed is not None else None,
        "calibration_kind": CALIBRATION_KIND,
        "prediction_scale": PREDICTION_SCALE,
        "slope": 1.0,
        "horizon": int(horizon),
        "group": group_name,
        "cancer": cancer_code,
        "group_idx": int(group_idx),
        "oof_coverage": float(alignment_qc["oof_coverage"]),
        "n_original": n_original,
        "n_original_outer_train": int(alignment_qc["n_outer_train"]),
        "n_fit": n_fit,
        "n_early_censored": n_early_censored,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "G_H": float(ipcw["G_H"]),
        "min_G_tminus": float(np.min(case_G)) if case_G.size else np.nan,
        "ipcw_weight_sum": weight_sum,
        "weighted_event_rate": weighted_event_rate,
        "weighted_predicted_rate_after_calibration": weighted_predicted_rate,
        "gamma": float(gamma),
        "score_residual": float(score_residual),
        "weighted_predicted_observed_residual": float(weighted_residual),
        "solver": "scipy.optimize.brentq",
        "solver_converged": bool(result.converged),
        "solver_iterations": int(result.iterations),
        "solver_function_calls": int(result.function_calls),
        "solver_bracket_low": float(bracket_low),
        "solver_bracket_high": float(bracket_high),
        "score_at_bracket_low": float(score_low),
        "score_at_bracket_high": float(score_high),
        "n_probability_values_clipped": int(np.sum(p_fit_raw != p_fit)),
        "probability_clip_epsilon": EPS,
        "minimum_n_required": MIN_N,
        "minimum_events_required": MIN_EVENTS,
        "minimum_nonevents_required": MIN_NONEVENTS,
        "km_scope": alignment_qc["km_scope"],
        "km_source_sha256": alignment_qc["km_source_sha256"],
        "artifact_arrays_sha256": alignment_qc["artifact_arrays_sha256"],
        "oof_file_sha256": alignment_qc.get("oof_file_sha256"),
        "artifact_file_identity_sha256": alignment_qc.get(
            "artifact_file_identity_sha256"
        ),
    }
    return float(gamma), qc


def fit_all_horizons(
    oof: pd.DataFrame | str | Path,
    artifact: Mapping[str, Any],
    *,
    horizons: Sequence[int] = HORIZONS,
    config: str | None = None,
    seed: int | None = None,
    artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fit every cancer-by-horizon intercept and return a serializable bundle."""

    oof_path: Path | None = None
    if isinstance(oof, (str, Path)):
        oof_path = Path(oof)
        oof_df = pd.read_csv(oof_path)
    elif isinstance(oof, pd.DataFrame):
        oof_df = oof.copy()
    else:
        raise TypeError("oof must be a pandas DataFrame or CSV path")

    horizons_clean = tuple(int(H) for H in horizons)
    if len(horizons_clean) == 0 or len(set(horizons_clean)) != len(horizons_clean):
        raise CalibrationError("horizons must be a nonempty unique sequence")
    aligned, alignment_qc = validate_oof_against_artifact(
        oof_df,
        artifact,
        horizons=horizons_clean,
        oof_path=oof_path,
        artifact_path=artifact_path,
    )

    times = _as_1d(artifact["t_tr"], name="artifact t_tr", dtype=np.float64)
    events = _as_1d(artifact["e_tr"], name="artifact e_tr", dtype=np.int64)
    groups_idx = _as_1d(artifact["g_tr"], name="artifact g_tr", dtype=np.int64)
    group_names = [str(value) for value in artifact["groups"]]
    kmf = fit_pooled_censoring_km(times, events)

    gammas: dict[int, dict[str, float]] = {}
    qc_by_horizon: dict[int, dict[str, dict[str, Any]]] = {}
    qc_rows: list[dict[str, Any]] = []
    for H in horizons_clean:
        probability = aligned[f"p_native_oof_H{H}"].to_numpy(dtype=np.float64)
        ipcw = ipcw_binary_targets(times, events, H, censoring_km=kmf)
        y = np.asarray(ipcw["y"], dtype=np.int64)
        weights = np.asarray(ipcw["weights"], dtype=np.float64)
        gammas[H] = {}
        qc_by_horizon[H] = {}
        for group_idx, group_name in enumerate(group_names):
            cancer_code = GROUP_TO_CANCER.get(group_name, group_name.upper())
            gamma, qc = _fit_one_intercept(
                probability,
                y,
                weights,
                group_mask=(groups_idx == int(group_idx)),
                group_name=group_name,
                group_idx=group_idx,
                cancer_code=cancer_code,
                horizon=H,
                ipcw=ipcw,
                alignment_qc=alignment_qc,
                config=config,
                seed=seed,
            )
            gammas[H][group_name] = gamma
            if cancer_code != group_name:
                gammas[H][cancer_code] = gamma
            qc_by_horizon[H][group_name] = qc
            qc_rows.append(qc)

    return {
        "status": "ok",
        "config": config,
        "seed": int(seed) if seed is not None else None,
        "calibration_kind": CALIBRATION_KIND,
        "prediction_scale": PREDICTION_SCALE,
        "formula": "logit(p_cal) = gamma_{cancer,H} + logit(p_native)",
        "slope": 1.0,
        "horizons": list(horizons_clean),
        "groups": group_names,
        "group_to_cancer": {
            group: GROUP_TO_CANCER.get(group, group.upper()) for group in group_names
        },
        "gammas": gammas,
        "qc": {
            "alignment": alignment_qc,
            "by_horizon": qc_by_horizon,
            "rows": qc_rows,
        },
    }


fit_cancer_specific_fixed_slope_intercepts = fit_all_horizons


def resolve_group_name(bundle: Mapping[str, Any], group: str) -> str:
    candidate = str(group)
    groups = [str(value) for value in bundle["groups"]]
    if candidate in groups:
        return candidate
    alias = CANCER_TO_GROUP.get(candidate.upper())
    if alias in groups:
        return str(alias)
    for name in groups:
        if GROUP_TO_CANCER.get(name, name.upper()) == candidate.upper():
            return name
    raise CalibrationError(f"Unknown cancer/group label {group!r}; expected {groups}")


def get_gamma(bundle: Mapping[str, Any], horizon: int, group: str) -> float:
    H = int(horizon)
    if H not in bundle["gammas"]:
        # JSON round trips turn integer dictionary keys into strings.
        if str(H) not in bundle["gammas"]:
            raise CalibrationError(f"No calibrator was fitted for H={H}")
        horizon_map = bundle["gammas"][str(H)]
    else:
        horizon_map = bundle["gammas"][H]
    group_name = resolve_group_name(bundle, group)
    if group_name not in horizon_map:
        raise CalibrationError(f"No gamma for group={group_name}, H={H}")
    gamma = float(horizon_map[group_name])
    if not np.isfinite(gamma):
        raise CalibrationError(f"Non-finite gamma for group={group_name}, H={H}")
    return gamma


def apply_numpy(probability: Any, gamma: float, *, eps: float = EPS) -> np.ndarray:
    """Apply a fixed-slope intercept shift to any NumPy-compatible shape."""

    p = np.asarray(probability, dtype=np.float64)
    if np.any(~np.isfinite(p)):
        raise CalibrationError("Cannot calibrate non-finite probabilities")
    if not np.isfinite(float(gamma)):
        raise CalibrationError("Cannot apply a non-finite calibration intercept")
    return np.clip(
        _expit_numpy(_logit_numpy(p, eps=eps) + float(gamma)), eps, 1.0 - eps
    )


def apply_torch(probability: Any, gamma: float, *, eps: float = EPS) -> Any:
    """Apply the same transform without detaching a torch tensor."""

    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is a model dependency
        raise CalibrationError("torch is required for tensor calibration") from exc
    if not torch.is_tensor(probability):
        raise TypeError("probability must be a torch.Tensor")
    if not torch.isfinite(probability).all():
        raise CalibrationError("Cannot calibrate non-finite probabilities")
    gamma_value = float(gamma)
    if not np.isfinite(gamma_value):
        raise CalibrationError("Cannot apply a non-finite calibration intercept")
    p = torch.clamp(probability, min=float(eps), max=1.0 - float(eps))
    gamma_tensor = torch.as_tensor(gamma_value, dtype=p.dtype, device=p.device)
    return torch.clamp(
        torch.sigmoid(torch.logit(p, eps=None) + gamma_tensor),
        min=float(eps),
        max=1.0 - float(eps),
    )


def apply_bundle_numpy(
    probability: Any,
    bundle: Mapping[str, Any],
    *,
    horizon: int,
    group: str,
    eps: float = EPS,
) -> np.ndarray:
    return apply_numpy(probability, get_gamma(bundle, horizon, group), eps=eps)


def apply_bundle_torch(
    probability: Any,
    bundle: Mapping[str, Any],
    *,
    horizon: int,
    group: str,
    eps: float = EPS,
) -> Any:
    return apply_torch(probability, get_gamma(bundle, horizon, group), eps=eps)


def _parse_seeds(tokens: Sequence[str]) -> list[int]:
    seeds: list[int] = []
    for token in tokens:
        for part in str(token).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_text, end_text = part.split("-", 1)
                start, end = int(start_text), int(end_text)
                if end < start:
                    raise ValueError(f"Invalid descending seed range {part!r}")
                seeds.extend(range(start, end + 1))
            else:
                seeds.append(int(part))
    return sorted(set(seeds))


def run_self_test(
    *,
    config_file: str | Path,
    seeds: Sequence[int] = tuple(range(42, 52)),
    runtime_dir: str | Path,
) -> dict[str, Any]:
    """Fit/audit one configured analysis and exercise NumPy/Torch application."""

    import torch
    try:
        from .fixed_contrast_io import load_artifact, load_config, load_model_module, oof_path
    except ImportError:  # pragma: no cover - direct-script invocation
        from fixed_contrast_io import load_artifact, load_config, load_model_module, oof_path

    config = load_config(config_file)
    model_module = load_model_module(config, runtime_dir)
    report: dict[str, Any] = {
        "status": "running",
        "calibration_kind": CALIBRATION_KIND,
        "prediction_scale": PREDICTION_SCALE,
        "config": config.name,
        "seeds": {},
    }
    for seed in seeds:
        input_oof = oof_path(config, int(seed))
        if not input_oof.is_file():
            raise FileNotFoundError(input_oof)
        artifact, artifact_location = load_artifact(config, int(seed), model_module)
        bundle = fit_all_horizons(
            input_oof,
            artifact,
            config=config.name,
            seed=int(seed),
            artifact_path=artifact_location,
        )
        application_checks: list[dict[str, Any]] = []
        probe = np.asarray([0.01, 0.15, 0.5, 0.85, 0.99], dtype=np.float64)
        for horizon in HORIZONS:
            for group in bundle["groups"]:
                gamma = get_gamma(bundle, horizon, str(group))
                numpy_value = apply_numpy(probe, gamma)
                torch_value = (
                    apply_torch(torch.tensor(probe, dtype=torch.float64), gamma)
                    .detach()
                    .cpu()
                    .numpy()
                )
                max_abs_diff = float(np.max(np.abs(numpy_value - torch_value)))
                monotone = bool(np.all(np.diff(numpy_value) > 0.0))
                if max_abs_diff > 1e-12 or not monotone:
                    raise CalibrationError(
                        f"NumPy/Torch application self-test failed: {config.name}, "
                        f"seed={seed}, H={horizon}, group={group}"
                    )
                application_checks.append(
                    {
                        "horizon": horizon,
                        "group": group,
                        "gamma": gamma,
                        "numpy_torch_max_abs_diff": max_abs_diff,
                        "strictly_monotone": monotone,
                    }
                )
        report["seeds"][str(int(seed))] = {
            "status": "ok",
            "artifact": str(artifact_location),
            "oof": str(input_oof),
            "gammas": bundle["gammas"],
            "alignment_qc": bundle["qc"]["alignment"],
            "fit_qc": bundle["qc"]["rows"],
            "application_checks": application_checks,
        }
    report["status"] = "ok"
    return report


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=["42-51"],
        help="seed values, comma lists, or inclusive ranges (default: 42-51)",
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        required=True,
        help="Writable directory used only while importing the saved model classes.",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    report = run_self_test(
        config_file=args.config_file,
        seeds=_parse_seeds(args.seeds),
        runtime_dir=args.runtime_dir,
    )
    payload = json.dumps(report, indent=2, sort_keys=True, default=_json_default)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
