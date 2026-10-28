#!/usr/bin/env python3
"""Coherent posterior sampling for fixed-contrast DGP analyses.

The functions in this module evaluate all counterfactual scenarios for one
patient chunk in the same posterior draw.  In particular, the low/high feature
values, branch hybrids, and treatment profiles are *not* sampled from separate
marginal predictive distributions.  A single multivariate draw is used at
each GP layer so that covariance between counterfactual scenarios is retained.

The public probability helpers convert scalar output-GP scores to PFI-event
probabilities at 18, 24, and 36 months and then apply the prespecified
cancer-specific, fixed-slope logistic recalibration to every posterior draw.
They intentionally require a cancer-specific intercept for every requested
horizon; there is no silent uncalibrated fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import gpytorch
import numpy as np
import torch


MANUSCRIPT_HORIZONS: tuple[int, ...] = (18, 24, 36)

# Scenario order is part of the interface.  Downstream code should use these
# names instead of relying on undocumented integer positions.
FEATURE_BRANCH_SCENARIOS: tuple[str, ...] = (
    "shared_low__private_low",
    "shared_high__private_low",
    "shared_low__private_high",
    "shared_high__private_high",
)
FEATURE_TREATMENT_SCENARIOS: tuple[str, ...] = (
    "treatment_A__feature_low",
    "treatment_A__feature_high",
    "treatment_B__feature_low",
    "treatment_B__feature_high",
)


def _model_device_and_dtype(model) -> tuple[torch.device, torch.dtype]:
    """Return the device and floating dtype used by a fitted model."""
    try:
        parameter = next(model.parameters())
    except StopIteration as exc:  # pragma: no cover - fitted DGPs have parameters
        raise ValueError("The model has no parameters from which to infer dtype/device") from exc
    dtype = parameter.dtype if parameter.is_floating_point() else torch.float32
    return parameter.device, dtype


@contextmanager
def _exact_joint_gp_settings(jitter: float = 1e-3) -> Iterator[None]:
    """Disable marginal/LOVE shortcuts and request an exact joint GP draw."""
    if not math.isfinite(float(jitter)) or float(jitter) <= 0:
        raise ValueError(f"jitter must be a positive finite number, got {jitter!r}")
    with ExitStack() as stack:
        stack.enter_context(gpytorch.settings.fast_pred_var(False))
        stack.enter_context(gpytorch.settings.max_cholesky_size(1_000_000))
        stack.enter_context(
            gpytorch.settings.fast_computations(
                covar_root_decomposition=False,
                log_prob=False,
                solves=False,
            )
        )
        stack.enter_context(gpytorch.settings.cholesky_jitter(float(jitter)))
        yield


@contextmanager
def _local_random_seed(model, random_seed: int | None) -> Iterator[None]:
    """Use an isolated Torch RNG state when a deterministic seed is supplied."""
    if random_seed is None:
        with nullcontext():
            yield
        return

    device, _ = _model_device_and_dtype(model)
    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(int(random_seed))
        if cuda_devices:
            torch.cuda.manual_seed_all(int(random_seed))
        yield


def _validated_scenarios(
    model,
    x_scenarios: torch.Tensor | np.ndarray,
    *,
    minimum_scenarios: int = 1,
    max_joint_points: int | None = None,
) -> torch.Tensor:
    """Validate and move a [scenario, patient, feature] array to the model."""
    device, dtype = _model_device_and_dtype(model)
    x = torch.as_tensor(x_scenarios, device=device, dtype=dtype)
    if x.ndim != 3:
        raise ValueError(
            "x_scenarios must have shape [scenario, patient, feature]; "
            f"received {tuple(x.shape)}"
        )
    scenario_count, patient_count, feature_count = map(int, x.shape)
    if scenario_count < int(minimum_scenarios):
        raise ValueError(
            f"Expected at least {minimum_scenarios} scenarios, got {scenario_count}"
        )
    if patient_count < 1 or feature_count < 1:
        raise ValueError(f"Empty patient or feature axis in {tuple(x.shape)}")
    joint_points = scenario_count * patient_count
    if max_joint_points is not None and joint_points > int(max_joint_points):
        raise ValueError(
            f"The joint draw contains {joint_points} scenario-patient points, "
            f"exceeding max_joint_points={int(max_joint_points)}. Reduce the "
            "patient chunk size; do not split the scenarios for one patient."
        )
    if not torch.isfinite(x).all():
        raise ValueError("x_scenarios contains non-finite values")
    return x.contiguous()


def joint_latent_samples(
    model,
    x_scenarios: torch.Tensor | np.ndarray,
    group_index: int,
    draws: int,
    *,
    jitter: float = 1e-3,
    max_joint_points: int | None = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Jointly sample the shared and cancer-specific first-layer latents.

    Parameters
    ----------
    x_scenarios
        Standardized model inputs with shape ``[V, N, D]``.  Every scenario
        for the same patient must be present in this call.
    group_index
        Integer cancer-output index used by the fitted DGP.
    draws
        Number of posterior draws.

    Returns
    -------
    shared, private
        Tensors with shapes ``[S, V, N, shared_dim]`` and
        ``[S, V, N, private_dim]``.  Scenario covariance is preserved.
    """
    if int(draws) < 2:
        raise ValueError(f"draws must be at least 2, got {draws}")
    if int(group_index) < 0:
        raise ValueError(f"group_index must be nonnegative, got {group_index}")
    x = _validated_scenarios(
        model,
        x_scenarios,
        max_joint_points=max_joint_points,
    )
    scenario_count, patient_count, feature_count = map(int, x.shape)
    x_joint = x.reshape(scenario_count * patient_count, feature_count)

    with torch.inference_mode(), _exact_joint_gp_settings(jitter):
        shared_dimensions: list[torch.Tensor] = []
        for latent_gp in model.shared_latent_mt.dims:
            # Shape: [S, V*N, cancer].  This single rsample is the critical
            # operation: all scenario points share one multivariate GP draw.
            values = latent_gp(x_joint).rsample(torch.Size([int(draws)]))
            if values.ndim != 3 or int(values.shape[1]) != scenario_count * patient_count:
                raise RuntimeError(
                    "Unexpected shared-latent sample shape "
                    f"{tuple(values.shape)} for {scenario_count * patient_count} points"
                )
            if int(group_index) >= int(values.shape[-1]):
                raise IndexError(
                    f"group_index={group_index} is outside the shared-GP task axis "
                    f"of size {int(values.shape[-1])}"
                )
            shared_dimensions.append(values[..., int(group_index)])

        if not shared_dimensions:
            raise RuntimeError("The fitted model exposes no shared latent dimensions")
        shared = torch.stack(shared_dimensions, dim=-1).reshape(
            int(draws),
            scenario_count,
            patient_count,
            len(shared_dimensions),
        )

        if int(group_index) >= len(model.private):
            raise IndexError(
                f"group_index={group_index} is outside {len(model.private)} private GPs"
            )
        private_raw = model.private[int(group_index)](x_joint).rsample(
            torch.Size([int(draws)])
        )
        # PrivateLayer has batch shape [private_dim] and event shape [V*N].
        if private_raw.ndim != 3 or int(private_raw.shape[-1]) != scenario_count * patient_count:
            raise RuntimeError(
                "Unexpected private-latent sample shape "
                f"{tuple(private_raw.shape)} for {scenario_count * patient_count} points"
            )
        private = private_raw.transpose(-1, -2).reshape(
            int(draws),
            scenario_count,
            patient_count,
            int(private_raw.shape[1]),
        )

    return shared, private


def joint_output_samples(
    model,
    latent_scenarios: torch.Tensor,
    group_index: int,
    *,
    jitter: float = 1e-3,
    max_joint_points: int | None = 4096,
) -> torch.Tensor:
    """Jointly sample scalar output-GP scores at all latent scenarios.

    ``latent_scenarios`` must have shape ``[S, V, N, D]``.  The posterior-draw
    axis is used as a GP batch.  Within each draw, the ``V*N`` locations remain
    one multivariate event, retaining output-GP covariance between scenarios.
    The returned tensor has shape ``[S, V, N]``.
    """
    if latent_scenarios.ndim != 4:
        raise ValueError(
            "latent_scenarios must have shape [draw, scenario, patient, latent]; "
            f"received {tuple(latent_scenarios.shape)}"
        )
    draw_count, scenario_count, patient_count, latent_dim = map(
        int, latent_scenarios.shape
    )
    if draw_count < 2 or scenario_count < 1 or patient_count < 1 or latent_dim < 1:
        raise ValueError(f"Invalid latent scenario shape {tuple(latent_scenarios.shape)}")
    joint_points = scenario_count * patient_count
    if max_joint_points is not None and joint_points > int(max_joint_points):
        raise ValueError(
            f"The output draw contains {joint_points} scenario-patient points, "
            f"exceeding max_joint_points={int(max_joint_points)}"
        )
    if int(group_index) < 0 or int(group_index) >= len(model.output):
        raise IndexError(
            f"group_index={group_index} is outside {len(model.output)} output GPs"
        )
    device, dtype = _model_device_and_dtype(model)
    latent = latent_scenarios.to(device=device, dtype=dtype).contiguous()
    if not torch.isfinite(latent).all():
        raise ValueError("latent_scenarios contains non-finite values")
    latent_joint = latent.reshape(draw_count, joint_points, latent_dim)

    with torch.inference_mode(), _exact_joint_gp_settings(jitter):
        scores = model.output[int(group_index)](latent_joint).rsample()
    if tuple(scores.shape) != (draw_count, joint_points):
        raise RuntimeError(
            f"Unexpected output-GP sample shape {tuple(scores.shape)}; "
            f"expected {(draw_count, joint_points)}"
        )
    return scores.reshape(draw_count, scenario_count, patient_count)


def joint_scenario_scores(
    model,
    x_scenarios: torch.Tensor | np.ndarray,
    group_index: int,
    draws: int,
    *,
    random_seed: int | None = None,
    jitter: float = 1e-3,
    max_joint_points: int | None = 4096,
) -> torch.Tensor:
    """Sample complete-DGP scalar scores for arbitrary paired scenarios."""
    with _local_random_seed(model, random_seed):
        shared, private = joint_latent_samples(
            model,
            x_scenarios,
            group_index,
            draws,
            jitter=jitter,
            max_joint_points=max_joint_points,
        )
        return joint_output_samples(
            model,
            torch.cat([shared, private], dim=-1),
            group_index,
            jitter=jitter,
            max_joint_points=max_joint_points,
        )


def joint_feature_and_branch_scores(
    model,
    x_low_high: torch.Tensor | np.ndarray,
    group_index: int,
    draws: int,
    *,
    random_seed: int | None = None,
    jitter: float = 1e-3,
    max_joint_points: int | None = 4096,
) -> torch.Tensor:
    """Return coherent scores for the four shared/private branch hybrids.

    The required input order is feature LOW, feature HIGH.  The returned
    scenario order is ``FEATURE_BRANCH_SCENARIOS``: LL, HL, LH, HH, where the
    first letter refers to the shared branch and the second to the private
    branch.  Consequently, LL and HH also form the full feature contrast.
    """
    x = _validated_scenarios(
        model,
        x_low_high,
        minimum_scenarios=2,
        max_joint_points=max_joint_points,
    )
    if int(x.shape[0]) != 2:
        raise ValueError(
            "A feature/branch contrast requires exactly two input scenarios "
            "ordered LOW, HIGH"
        )
    with _local_random_seed(model, random_seed):
        shared, private = joint_latent_samples(
            model,
            x,
            group_index,
            draws,
            jitter=jitter,
            max_joint_points=max_joint_points,
        )
        shared_low, shared_high = shared[:, 0], shared[:, 1]
        private_low, private_high = private[:, 0], private[:, 1]
        hybrids = torch.stack(
            [
                torch.cat([shared_low, private_low], dim=-1),
                torch.cat([shared_high, private_low], dim=-1),
                torch.cat([shared_low, private_high], dim=-1),
                torch.cat([shared_high, private_high], dim=-1),
            ],
            dim=1,
        )
        return joint_output_samples(
            model,
            hybrids,
            group_index,
            jitter=jitter,
            max_joint_points=max_joint_points,
        )


def joint_feature_treatment_scores(
    model,
    x_a_low_a_high_b_low_b_high: torch.Tensor | np.ndarray,
    group_index: int,
    draws: int,
    *,
    random_seed: int | None = None,
    jitter: float = 1e-3,
    max_joint_points: int | None = 4096,
) -> torch.Tensor:
    """Return coherent scores for A-low, A-high, B-low, and B-high.

    Each patient must occur under all four scenarios in the exact order given
    by ``FEATURE_TREATMENT_SCENARIOS``.  The function samples both DGP layers
    jointly across the four counterfactual inputs.
    """
    x = _validated_scenarios(
        model,
        x_a_low_a_high_b_low_b_high,
        minimum_scenarios=4,
        max_joint_points=max_joint_points,
    )
    if int(x.shape[0]) != 4:
        raise ValueError(
            "A feature-by-treatment contrast requires exactly four scenarios "
            "ordered A-low, A-high, B-low, B-high"
        )
    return joint_scenario_scores(
        model,
        x,
        group_index,
        draws,
        random_seed=random_seed,
        jitter=jitter,
        max_joint_points=max_joint_points,
    )


# Backward-compatible name used by the validated experimental prototype.
joint_four_scenario_scores = joint_feature_treatment_scores


def _lookup_horizon(mapping: Mapping, horizon: int) -> float:
    """Resolve an integer horizon without silently choosing another value."""
    candidates: tuple[object, ...] = (
        int(horizon),
        float(horizon),
        str(int(horizon)),
        f"H{int(horizon)}",
    )
    for key in candidates:
        if key in mapping:
            value = float(mapping[key])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite calibration intercept for H={horizon}: {value}")
            return value
    raise KeyError(f"No calibration intercept was supplied for H={int(horizon)} months")


def cancer_specific_intercepts(
    calibration_intercepts: Mapping,
    cancer: str,
    horizons: Sequence[int] = MANUSCRIPT_HORIZONS,
) -> dict[int, float]:
    """Extract the fixed-slope intercepts for exactly one cancer.

    Accepted structures are ``{cancer: {horizon: gamma}}`` or
    ``{(cancer, horizon): gamma}``.  Cancer matching is exact first and then
    case-insensitive.  A missing cancer or horizon is an error because all
    manuscript-scale predictions must be recalibrated consistently.
    """
    if not isinstance(calibration_intercepts, Mapping):
        raise TypeError("calibration_intercepts must be a mapping")
    cancer_key: object | None = None
    if cancer in calibration_intercepts:
        cancer_key = cancer
    else:
        matches = [
            key
            for key in calibration_intercepts
            if isinstance(key, str) and key.casefold() == str(cancer).casefold()
        ]
        if len(matches) == 1:
            cancer_key = matches[0]
        elif len(matches) > 1:
            raise KeyError(f"Ambiguous case-insensitive calibration keys for {cancer!r}")

    if cancer_key is not None:
        per_horizon = calibration_intercepts[cancer_key]
        if not isinstance(per_horizon, Mapping):
            raise TypeError(
                f"Calibration entry for {cancer!r} must be a horizon mapping"
            )
        return {int(h): _lookup_horizon(per_horizon, int(h)) for h in horizons}

    result: dict[int, float] = {}
    for horizon in horizons:
        tuple_candidates = (
            (cancer, int(horizon)),
            (cancer, float(horizon)),
            (cancer, str(int(horizon))),
            (cancer, f"H{int(horizon)}"),
        )
        found = [key for key in tuple_candidates if key in calibration_intercepts]
        if not found:
            # Case-insensitive tuple match, while still requiring uniqueness.
            found = [
                key
                for key in calibration_intercepts
                if isinstance(key, tuple)
                and len(key) == 2
                and str(key[0]).casefold() == str(cancer).casefold()
                and str(key[1]).removeprefix("H") == str(int(horizon))
            ]
        if len(found) != 1:
            raise KeyError(
                f"Expected one calibration intercept for cancer={cancer!r}, "
                f"H={int(horizon)}; found {len(found)}"
            )
        value = float(calibration_intercepts[found[0]])
        if not math.isfinite(value):
            raise ValueError(
                f"Non-finite calibration intercept for {cancer}, H={horizon}: {value}"
            )
        result[int(horizon)] = value
    return result


def scores_to_calibrated_event_probabilities(
    scores: torch.Tensor | np.ndarray,
    alpha_dt: torch.Tensor | np.ndarray,
    dt_edges: Sequence[float] | np.ndarray,
    *,
    cancer: str,
    calibration_intercepts: Mapping,
    horizons: Sequence[int] = MANUSCRIPT_HORIZONS,
) -> dict[int, torch.Tensor]:
    """Convert scalar score draws to recalibrated PFI-event probabilities.

    The scalar score is used in every discrete-time hazard interval.  For each
    horizon, the native event probability is ``1 - product(1 - hazard_k)``.
    The cancer- and horizon-specific intercept is then applied draw by draw as

    ``sigmoid(logit(p_native) + gamma[cancer, horizon])``.

    The returned mapping contains one tensor per horizon, each with exactly the
    same shape as ``scores`` (normally ``[draw, scenario, patient]``).
    """
    # These are post-fit analyses.  Detaching here prevents an artifact's
    # trainable ``alpha_dt`` parameter from accidentally building a graph and
    # lets callers convert the returned probabilities directly to NumPy.
    score_tensor = torch.as_tensor(scores).detach()
    if not score_tensor.is_floating_point():
        score_tensor = score_tensor.to(dtype=torch.float32)
    if score_tensor.ndim < 1 or not torch.isfinite(score_tensor).all():
        raise ValueError("scores must be a finite floating tensor with at least one axis")

    alpha = torch.as_tensor(
        alpha_dt,
        device=score_tensor.device,
        dtype=score_tensor.dtype,
    ).detach().reshape(-1)
    edges = np.asarray(dt_edges, dtype=float).reshape(-1)
    if len(edges) != int(alpha.numel()) + 1:
        raise ValueError(
            f"dt_edges has length {len(edges)}, but alpha_dt has {int(alpha.numel())} bins"
        )
    if len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError("dt_edges must be finite and strictly increasing")
    if not torch.isfinite(alpha).all():
        raise ValueError("alpha_dt contains non-finite values")

    requested = tuple(int(h) for h in horizons)
    if len(set(requested)) != len(requested):
        raise ValueError(f"Duplicate horizons were requested: {requested}")
    gamma = cancer_specific_intercepts(calibration_intercepts, cancer, requested)
    right_edges = edges[1:]
    result: dict[int, torch.Tensor] = {}
    for horizon in requested:
        bin_indices = np.flatnonzero(right_edges <= float(horizon))
        if not len(bin_indices):
            raise ValueError(
                f"H={horizon} months precedes the first discrete-time bin ending at "
                f"{float(right_edges[0]):g} months"
            )
        index = torch.as_tensor(bin_indices, device=score_tensor.device, dtype=torch.long)
        logits = score_tensor.unsqueeze(-1) + alpha.index_select(0, index)
        # logsigmoid(-logit) is log(1 - sigmoid(logit)) but remains stable for
        # large positive or negative hazards.
        log_survival = torch.nn.functional.logsigmoid(-logits).sum(dim=-1)
        native_event_probability = -torch.expm1(log_survival)
        eps = torch.finfo(score_tensor.dtype).eps
        native_event_probability = native_event_probability.clamp(eps, 1.0 - eps)
        calibrated = torch.sigmoid(
            torch.logit(native_event_probability) + float(gamma[horizon])
        )
        result[horizon] = calibrated
    return result


def sample_feature_branch_probabilities(
    model,
    x_low_high: torch.Tensor | np.ndarray,
    group_index: int,
    draws: int,
    *,
    alpha_dt: torch.Tensor | np.ndarray,
    dt_edges: Sequence[float] | np.ndarray,
    cancer: str,
    calibration_intercepts: Mapping,
    horizons: Sequence[int] = MANUSCRIPT_HORIZONS,
    random_seed: int | None = None,
    jitter: float = 1e-3,
    max_joint_points: int | None = 4096,
) -> dict[int, torch.Tensor]:
    """Sample and recalibrate the LL/HL/LH/HH branch scenarios."""
    scores = joint_feature_and_branch_scores(
        model,
        x_low_high,
        group_index,
        draws,
        random_seed=random_seed,
        jitter=jitter,
        max_joint_points=max_joint_points,
    )
    return scores_to_calibrated_event_probabilities(
        scores,
        alpha_dt,
        dt_edges,
        cancer=cancer,
        calibration_intercepts=calibration_intercepts,
        horizons=horizons,
    )


def sample_feature_treatment_probabilities(
    model,
    x_a_low_a_high_b_low_b_high: torch.Tensor | np.ndarray,
    group_index: int,
    draws: int,
    *,
    alpha_dt: torch.Tensor | np.ndarray,
    dt_edges: Sequence[float] | np.ndarray,
    cancer: str,
    calibration_intercepts: Mapping,
    horizons: Sequence[int] = MANUSCRIPT_HORIZONS,
    random_seed: int | None = None,
    jitter: float = 1e-3,
    max_joint_points: int | None = 4096,
) -> dict[int, torch.Tensor]:
    """Sample and recalibrate A-low/A-high/B-low/B-high scenarios."""
    scores = joint_feature_treatment_scores(
        model,
        x_a_low_a_high_b_low_b_high,
        group_index,
        draws,
        random_seed=random_seed,
        jitter=jitter,
        max_joint_points=max_joint_points,
    )
    return scores_to_calibrated_event_probabilities(
        scores,
        alpha_dt,
        dt_edges,
        cancer=cancer,
        calibration_intercepts=calibration_intercepts,
        horizons=horizons,
    )


def treatment_modification_draws(
    probabilities: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Calculate paired B-A treatment contrasts and their feature modification.

    ``probabilities`` must have shape ``[S, 4, N]`` in
    ``FEATURE_TREATMENT_SCENARIOS`` order.  The signed convention is B minus A:
    ``psi = (B_high - A_high) - (B_low - A_low)``.
    """
    if probabilities.ndim != 3 or int(probabilities.shape[1]) != 4:
        raise ValueError(
            "probabilities must have shape [draw, 4, patient] ordered "
            "A-low, A-high, B-low, B-high"
        )
    a_low, a_high, b_low, b_high = probabilities.unbind(dim=1)
    delta_low = b_low - a_low
    delta_high = b_high - a_high
    return {
        "risk_A_low": a_low,
        "risk_A_high": a_high,
        "risk_B_low": b_low,
        "risk_B_high": b_high,
        "delta_low_B_minus_A": delta_low,
        "delta_high_B_minus_A": delta_high,
        "psi": delta_high - delta_low,
    }


def empirical_wasserstein_1(
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    draw_axis: int = 0,
) -> torch.Tensor:
    """Empirical one-dimensional W1 for equal-sized posterior samples."""
    if low.shape != high.shape:
        raise ValueError(f"W1 inputs must have equal shapes, got {low.shape} and {high.shape}")
    if low.ndim < 1:
        raise ValueError("W1 inputs must have at least one axis")
    return torch.mean(
        torch.abs(
            torch.sort(low, dim=int(draw_axis)).values
            - torch.sort(high, dim=int(draw_axis)).values
        ),
        dim=int(draw_axis),
    )


def _probability_self_test() -> dict[str, object]:
    """Fast deterministic test of horizon selection and draw-wise calibration."""
    scores = torch.tensor(
        [
            [[-0.7, 0.2], [-0.4, 0.5], [0.5, 1.0], [0.8, 1.3]],
            [[-0.3, 0.1], [0.0, 0.4], [0.8, 1.2], [1.1, 1.5]],
            [[-0.1, 0.4], [0.2, 0.7], [0.6, 1.5], [0.9, 1.8]],
        ],
        dtype=torch.float64,
    )
    alpha = torch.tensor([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0], dtype=torch.float64)
    edges = np.array([0.0, 6.0, 12.0, 18.0, 24.0, 36.0, 60.0])
    gamma = {"BRCA": {18: -0.4, 24: 0.2, 36: 0.7}}
    first = scores_to_calibrated_event_probabilities(
        scores,
        alpha,
        edges,
        cancer="BRCA",
        calibration_intercepts=gamma,
    )
    second = scores_to_calibrated_event_probabilities(
        scores,
        alpha,
        edges,
        cancer="BRCA",
        calibration_intercepts=gamma,
    )
    for horizon in MANUSCRIPT_HORIZONS:
        bins = np.flatnonzero(edges[1:] <= horizon)
        hazards = torch.sigmoid(scores.unsqueeze(-1) + alpha[bins])
        raw = 1.0 - torch.prod(1.0 - hazards, dim=-1)
        expected = torch.sigmoid(torch.logit(raw) + gamma["BRCA"][horizon])
        torch.testing.assert_close(first[horizon], expected, rtol=1e-12, atol=1e-12)
        if not torch.equal(first[horizon], second[horizon]):
            raise AssertionError("Probability conversion is not deterministic")
    psi = treatment_modification_draws(first[18])
    if tuple(psi["psi"].shape) != (3, 2):
        raise AssertionError(f"Unexpected psi shape: {tuple(psi['psi'].shape)}")
    w1 = empirical_wasserstein_1(first[18][:, 0], first[18][:, 1])
    if tuple(w1.shape) != (2,) or not torch.isfinite(w1).all():
        raise AssertionError("W1 self-test failed")
    return {
        "status": "passed",
        "horizons": list(MANUSCRIPT_HORIZONS),
        "probability_shape": list(first[18].shape),
        "psi_shape": list(psi["psi"].shape),
        "all_probabilities_in_unit_interval": bool(
            all(((value >= 0) & (value <= 1)).all() for value in first.values())
        ),
    }


def _prepare_artifact_smoke_scenarios(
    config_file: str | Path,
    seed: int,
    patient_count: int,
    runtime_dir: str | Path,
):
    """Load a fitted artifact and construct deterministic paired scenarios."""
    try:
        from .fixed_contrast_io import (
            load_artifact,
            load_config,
            load_model_module,
            treatment_columns,
        )
    except ImportError:  # pragma: no cover - direct-script invocation
        from fixed_contrast_io import (
            load_artifact,
            load_config,
            load_model_module,
            treatment_columns,
        )

    config = load_config(config_file)
    module = load_model_module(config, runtime_dir)
    artifact, artifact_path = load_artifact(config, int(seed), module)
    model = artifact["model"].eval()
    group_index = 0
    group_test = np.asarray(artifact["g_te"], dtype=int)
    selected = np.flatnonzero(group_test == group_index)[: int(patient_count)]
    if len(selected) != int(patient_count):
        raise RuntimeError(
            f"Need {patient_count} group-0 test patients; found {len(selected)}"
        )

    raw_train = np.asarray(artifact["Xtr_np_sub"], dtype=np.float32)
    scaled_test = np.asarray(
        artifact["Xte_t"].detach().cpu().numpy(), dtype=np.float32
    )[selected].copy()
    train_groups = np.asarray(artifact["g_tr"], dtype=int)
    train_mask = train_groups == group_index
    means = np.asarray(artifact["standardize_mu"], dtype=np.float32).reshape(-1)
    scales = np.asarray(artifact["standardize_sd"], dtype=np.float32).reshape(-1)
    feature_names = [str(value) for value in artifact["feat_names"]]
    feature_index = next(
        index for index, name in enumerate(feature_names) if name.startswith("act_")
    )
    q25, q75 = np.quantile(raw_train[train_mask, feature_index], [0.25, 0.75])

    low = scaled_test.copy()
    high = scaled_test.copy()
    low[:, feature_index] = (float(q25) - means[feature_index]) / scales[feature_index]
    high[:, feature_index] = (float(q75) - means[feature_index]) / scales[feature_index]

    treatment_indices, _ = treatment_columns(feature_names)
    observed_treatments = (raw_train[train_mask][:, treatment_indices] > 0.5).astype(int)
    active = observed_treatments[observed_treatments.sum(axis=1) > 0]
    profiles, counts = np.unique(active, axis=0, return_counts=True)
    order = np.argsort(-counts)
    if len(order) < 2:
        raise RuntimeError("The artifact does not contain two active treatment profiles")
    profile_a, profile_b = profiles[order[0]], profiles[order[1]]
    treatment_scenarios: list[np.ndarray] = []
    for profile, feature_value in (
        (profile_a, q25),
        (profile_a, q75),
        (profile_b, q25),
        (profile_b, q75),
    ):
        scenario = scaled_test.copy()
        scenario[:, feature_index] = (
            float(feature_value) - means[feature_index]
        ) / scales[feature_index]
        scenario[:, treatment_indices] = (
            profile.astype(np.float32) - means[treatment_indices]
        ) / scales[treatment_indices]
        treatment_scenarios.append(scenario)

    return {
        "config_name": config.name,
        "artifact": artifact,
        "artifact_path": str(artifact_path),
        "model": model,
        "group_index": group_index,
        "feature_index": feature_index,
        "feature_name": feature_names[feature_index],
        "x_low_high": np.stack([low, high]),
        "x_treatment": np.stack(treatment_scenarios),
        "profile_a": profile_a.tolist(),
        "profile_b": profile_b.tolist(),
    }


def run_deterministic_artifact_smoke(
    config_file: str | Path,
    *,
    seed: int = 42,
    draws: int = 16,
    patients: int = 2,
    runtime_dir: str | Path,
) -> dict[str, object]:
    """Exercise both joint samplers twice and require bitwise replay."""
    prepared = _prepare_artifact_smoke_scenarios(
        config_file,
        int(seed),
        int(patients),
        runtime_dir,
    )
    model = prepared["model"]
    group_index = int(prepared["group_index"])
    random_seed = 71_923

    started = time.perf_counter()
    first_branch = joint_feature_and_branch_scores(
        model,
        prepared["x_low_high"],
        group_index,
        int(draws),
        random_seed=random_seed,
    )
    second_branch = joint_feature_and_branch_scores(
        model,
        prepared["x_low_high"],
        group_index,
        int(draws),
        random_seed=random_seed,
    )
    first_treatment = joint_feature_treatment_scores(
        model,
        prepared["x_treatment"],
        group_index,
        int(draws),
        random_seed=random_seed + 1,
    )
    second_treatment = joint_feature_treatment_scores(
        model,
        prepared["x_treatment"],
        group_index,
        int(draws),
        random_seed=random_seed + 1,
    )
    elapsed = time.perf_counter() - started

    expected_shape = (int(draws), 4, int(patients))
    if tuple(first_branch.shape) != expected_shape:
        raise AssertionError(
            f"Branch smoke shape {tuple(first_branch.shape)} != {expected_shape}"
        )
    if tuple(first_treatment.shape) != expected_shape:
        raise AssertionError(
            f"Treatment smoke shape {tuple(first_treatment.shape)} != {expected_shape}"
        )
    branch_replay = bool(torch.equal(first_branch, second_branch))
    treatment_replay = bool(torch.equal(first_treatment, second_treatment))
    if not branch_replay or not treatment_replay:
        raise AssertionError(
            "Exact deterministic replay failed for joint posterior sampling"
        )
    if not torch.isfinite(first_branch).all() or not torch.isfinite(first_treatment).all():
        raise AssertionError("Joint posterior smoke produced non-finite scores")

    return {
        "status": "passed",
        "configuration": prepared["config_name"],
        "seed": int(seed),
        "artifact": prepared["artifact_path"],
        "feature": prepared["feature_name"],
        "treatment_profile_a": prepared["profile_a"],
        "treatment_profile_b": prepared["profile_b"],
        "draws": int(draws),
        "patients": int(patients),
        "branch_shape": list(first_branch.shape),
        "treatment_shape": list(first_treatment.shape),
        "branch_replay_bitwise_exact": branch_replay,
        "treatment_replay_bitwise_exact": treatment_replay,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic checks for coherent fixed-contrast DGP sampling."
    )
    parser.add_argument(
        "--artifact-smoke-config-file",
        action="append",
        type=Path,
        help="Optional JSON input manifest; may be supplied once per configuration.",
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        help="Writable artifact-load directory; required with --artifact-smoke-config-file.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--draws", type=int, default=16)
    parser.add_argument("--patients", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report: dict[str, object] = {"probability_self_test": _probability_self_test()}
    if args.artifact_smoke_config_file:
        if args.runtime_dir is None:
            parser.error("--runtime-dir is required with --artifact-smoke-config-file")
        report["artifact_smoke"] = [
            run_deterministic_artifact_smoke(
                config_file,
                seed=args.seed,
                draws=args.draws,
                patients=args.patients,
                runtime_dir=args.runtime_dir,
            )
            for config_file in args.artifact_smoke_config_file
        ]
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
