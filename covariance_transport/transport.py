"""Descriptor matching and covariance-corrected weighted risk recovery."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear, minimize

from selector.risk_recovery import pair_design


@lru_cache(maxsize=None)
def _augmented_recovery_structure(
    model_count: int,
    identity_block_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cache CSR indices/row pointers for pair design plus identity blocks."""

    design, _, _ = pair_design(model_count)
    pair_count = design.shape[0]
    indices = np.concatenate(
        [design.indices, *[np.arange(model_count)] * identity_block_count]
    )
    pair_indptr = design.indptr
    identity_indptr = np.arange(
        pair_indptr[-1] + 1,
        pair_indptr[-1] + identity_block_count * model_count + 1,
    )
    indptr = np.concatenate([pair_indptr, identity_indptr])
    indices.setflags(write=False)
    indptr.setflags(write=False)
    return indices, indptr


def _augmented_recovery_design(
    square_root_weight: np.ndarray,
    model_count: int,
    identity_scales: tuple[float, ...],
) -> sparse.csr_matrix:
    """Build a weighted pair design while reusing its immutable CSR structure."""

    indices, indptr = _augmented_recovery_structure(
        model_count,
        len(identity_scales),
    )
    pair_data_size = 2 * square_root_weight.size
    data = np.empty(
        pair_data_size + len(identity_scales) * model_count,
        dtype=np.float64,
    )
    data[:pair_data_size:2] = square_root_weight
    data[1:pair_data_size:2] = square_root_weight
    offset = pair_data_size
    for scale in identity_scales:
        data[offset : offset + model_count] = scale
        offset += model_count
    return sparse.csr_matrix(
        (data, indices, indptr),
        shape=(square_root_weight.size + len(identity_scales) * model_count, model_count),
        copy=False,
    )


def _augmented_recovery_observations(
    observations: np.ndarray,
    square_root_weight: np.ndarray,
    model_count: int,
    *,
    ridge: float,
    prior_strength: float,
    prior_risk: np.ndarray,
) -> np.ndarray:
    """Assemble augmented observations once for reuse by the robust refit."""

    identity_block_count = int(ridge > 0) + int(prior_strength > 0)
    pair_count = observations.size
    augmented = np.empty(
        pair_count + identity_block_count * model_count,
        dtype=np.float64,
    )
    np.multiply(observations, square_root_weight, out=augmented[:pair_count])
    offset = pair_count
    if ridge > 0:
        augmented[offset : offset + model_count] = 0.0
        offset += model_count
    if prior_strength > 0:
        np.multiply(
            prior_risk,
            np.sqrt(prior_strength),
            out=augmented[offset : offset + model_count],
        )
    return augmented


def _project_probability_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection onto ``{x >= 0, sum(x) = 1}``."""

    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("simplex projection expects a non-empty vector")
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    active = np.flatnonzero(ordered - cumulative / np.arange(1, vector.size + 1) > 0)
    if active.size == 0:
        return np.full(vector.shape, 1.0 / vector.size, dtype=np.float64)
    threshold = cumulative[active[-1]] / float(active[-1] + 1)
    projected = np.maximum(vector - threshold, 0.0)
    projected /= projected.sum()
    return projected


def _projected_gradient_simplex_quadratic(
    shifts: np.ndarray,
    target: np.ndarray,
    inverse_variance: np.ndarray,
    regularization: float,
    initial: np.ndarray,
    *,
    tolerance: float = 1e-9,
    max_iterations: int = 50_000,
) -> tuple[np.ndarray, int, bool]:
    """Solve the convex descriptor-matching quadratic on the simplex."""

    weighted_shifts = shifts * np.sqrt(inverse_variance)[None, :]
    weighted_target = target * np.sqrt(inverse_variance)
    gram = weighted_shifts @ weighted_shifts.T
    if regularization > 0:
        gram = gram + regularization * np.eye(shifts.shape[0], dtype=np.float64)
    linear = weighted_shifts @ weighted_target
    lipschitz = 2.0 * float(np.linalg.eigvalsh(gram).max(initial=0.0))
    if lipschitz <= np.finfo(np.float64).eps:
        return _project_probability_simplex(initial), 0, True

    solution = _project_probability_simplex(initial)
    step_size = 1.0 / lipschitz
    for iteration in range(1, max_iterations + 1):
        gradient = 2.0 * (gram @ solution - linear)
        updated = _project_probability_simplex(solution - step_size * gradient)
        difference = np.linalg.norm(updated - solution)
        solution = updated
        if difference <= tolerance * max(1.0, np.linalg.norm(solution)):
            return solution, iteration, True
    return solution, max_iterations, False


def match_shift_convex_combination(
    shift_deltas: np.ndarray,
    target_delta: np.ndarray,
    *,
    regularization: float,
    descriptor_floor: float = 0.05,
) -> tuple[np.ndarray, dict[str, float]]:
    """Match target shift by a regularized simplex combination of simulations."""

    shifts = np.asarray(shift_deltas, dtype=np.float64)
    target = np.asarray(target_delta, dtype=np.float64)
    if shifts.ndim != 2 or target.shape != (shifts.shape[1],):
        raise ValueError("shift and target descriptor shapes do not align")
    if regularization < 0 or descriptor_floor <= 0:
        raise ValueError("invalid transport regularization or descriptor floor")
    shift_count = shifts.shape[0]
    scale = np.maximum(np.std(shifts, axis=0), descriptor_floor)
    inverse_variance = 1.0 / (scale**2)

    def objective(alpha: np.ndarray) -> float:
        residual = alpha @ shifts - target
        return float(np.sum(inverse_variance * residual**2) + regularization * np.sum(alpha**2))

    def gradient(alpha: np.ndarray) -> np.ndarray:
        residual = alpha @ shifts - target
        return 2.0 * (shifts @ (inverse_variance * residual)) + 2.0 * regularization * alpha

    distances = np.sum(((shifts - target[None, :]) / scale[None, :]) ** 2, axis=1)
    initial = np.exp(-0.5 * (distances - distances.min()))
    initial /= initial.sum()
    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * shift_count,
        constraints=[{"type": "eq", "fun": lambda alpha: np.sum(alpha) - 1.0}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    solver = "slsqp"
    solver_iterations = int(result.nit)
    solver_converged = bool(result.success)
    if result.success and np.isfinite(result.x).all():
        alpha = _project_probability_simplex(result.x)
    else:
        alpha, solver_iterations, solver_converged = _projected_gradient_simplex_quadratic(
            shifts,
            target,
            inverse_variance,
            regularization,
            initial,
        )
        solver = "projected_gradient_fallback"
    if not solver_converged:
        raise RuntimeError(
            "shift matching failed to converge with both SLSQP and projected gradient"
        )
    matched = alpha @ shifts
    standardized_residual = (matched - target) / scale
    diagnostics = {
        "objective": objective(alpha),
        "descriptor_rmse": float(np.sqrt(np.mean(standardized_residual**2))),
        "descriptor_mae": float(np.mean(np.abs(standardized_residual))),
        "effective_shift_count": float(1.0 / np.sum(alpha**2)),
        "max_shift_weight": float(alpha.max()),
        "solver": solver,
        "solver_iterations": solver_iterations,
    }
    return alpha, diagnostics


def transport_statistics(
    alpha: np.ndarray,
    shift_band_risks: np.ndarray,
    shift_band_covariances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(alpha, dtype=np.float64)
    risks = np.asarray(shift_band_risks, dtype=np.float64)
    covariances = np.asarray(shift_band_covariances, dtype=np.float64)
    if risks.shape[0] != weights.shape[0] or covariances.shape[0] != weights.shape[0]:
        raise ValueError("transport weights and shift statistics do not align")
    transported_risks = np.tensordot(weights, risks, axes=(0, 0))
    transported_covariances = np.tensordot(weights, covariances, axes=(0, 0))
    return transported_risks, transported_covariances


def corrected_band_risk_recovery(
    disagreements: np.ndarray,
    transported_risks: np.ndarray,
    transported_covariances: np.ndarray,
    *,
    ridge: float,
    pair_weight_power: float,
    prior_strength: float = 0.0,
    robust: bool = True,
    correlation_epsilon: float = 0.05,
    max_pair_weight: float = 20.0,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Solve covariance-corrected weighted NNLS with an optional risk prior.

    The prior term is the theory-consistent penalty
    ``prior_strength * ||risk - transported_risk||_2^2``.  It uses only
    source-simulated transported risks and does not depend on candidate IDs.
    """

    disagreement = np.asarray(disagreements, dtype=np.float64)
    prior_risk = np.asarray(transported_risks, dtype=np.float64)
    covariance = np.asarray(transported_covariances, dtype=np.float64)
    if disagreement.ndim != 3:
        raise ValueError("disagreements must have shape [bands, models, models]")
    band_count, model_count, second_count = disagreement.shape
    if second_count != model_count:
        raise ValueError("disagreement matrices must be square")
    if prior_risk.shape != (model_count, band_count):
        raise ValueError("transported risk shape mismatch")
    if covariance.shape != (band_count, model_count, model_count):
        raise ValueError("transported covariance shape mismatch")
    if ridge < 0 or pair_weight_power < 0 or prior_strength < 0:
        raise ValueError("ridge, pair_weight_power, and prior_strength must be non-negative")

    _, first, second = pair_design(model_count)
    identity_scales = tuple(
        scale
        for scale in (np.sqrt(ridge), np.sqrt(prior_strength))
        if scale > 0
    )
    recovered = []
    diagnostics = []
    for band in range(band_count):
        observations = (
            disagreement[band, first, second]
            + 2.0 * covariance[band, first, second]
        )
        # Covariance correction can create a small number of implausible
        # pair observations.  Winsorize only the robust tails, using the
        # unlabeled pair-observation distribution itself; this keeps the
        # covariance-corrected signal while preventing a single extreme pair
        # from setting the NNLS scale.
        if robust:
            observation_center = float(np.median(observations))
            observation_scale = float(
                1.4826 * np.median(np.abs(observations - observation_center))
            )
            if observation_scale > 1e-10:
                observation_cutoff = 4.685 * observation_scale
                observations = np.clip(
                    observations,
                    max(0.0, observation_center - observation_cutoff),
                    observation_center + observation_cutoff,
                )
            else:
                observations = np.maximum(observations, 0.0)
        denominator = np.sqrt(
            np.maximum(prior_risk[first, band], 1e-12)
            * np.maximum(prior_risk[second, band], 1e-12)
        )
        correlation = covariance[band, first, second] / denominator
        if pair_weight_power == 0:
            pair_weights = np.ones_like(correlation)
        else:
            pair_weights = (
                1.0 / (np.abs(correlation) + correlation_epsilon)
            ) ** pair_weight_power
            pair_weights = np.minimum(pair_weights, max_pair_weight)
            pair_weights /= np.mean(pair_weights)
        square_root_weight = np.sqrt(pair_weights)
        weighted_design = _augmented_recovery_design(
            square_root_weight,
            model_count,
            identity_scales,
        )
        weighted_observations = _augmented_recovery_observations(
            observations,
            square_root_weight,
            model_count,
            ridge=ridge,
            prior_strength=prior_strength,
            prior_risk=prior_risk[:, band],
        )
        result = lsq_linear(
            weighted_design,
            weighted_observations,
            bounds=(0.0, np.inf),
            lsmr_tol="auto",
        )
        if not result.success:
            raise RuntimeError(f"corrected NNLS failed in band {band}: {result.message}")
        # One robust IRLS pass limits the influence of anomalous pair
        # identities while preserving the original covariance-corrected fit
        # when residuals are within the empirical scale.  The scale and
        # weights use only unlabeled target disagreements.
        initial_risks = result.x
        initial_residual = observations - (initial_risks[first] + initial_risks[second])
        residual_center = float(np.median(initial_residual))
        residual_scale = float(
            1.4826 * np.median(np.abs(initial_residual - residual_center))
        )
        if robust and residual_scale > 1e-10:
            standardized = np.abs(initial_residual - residual_center) / residual_scale
            # Tukey's bounded redescending M-estimator suppresses pair
            # identities that are incompatible with the consensus fit while
            # retaining full weight for in-scale observations.  Unlike a
            # candidate-specific heuristic this remains symmetric and uses
            # only unlabeled target pair residuals.
            cutoff = 4.685
            normalized = standardized / cutoff
            tukey_weights = np.where(
                normalized < 1.0,
                (1.0 - normalized**2) ** 2,
                0.0,
            )
            # Convert pair residual consistency into a symmetric candidate
            # reliability graph.  A candidate with anomalous residuals on
            # most incident edges is downweighted through both endpoints,
            # while isolated bad edges remain handled by Tukey itself.  This
            # is label-free and adds no solve: it reuses the existing robust
            # refinement pass.
            incident_sum = np.zeros(model_count, dtype=np.float64)
            incident_count = np.zeros(model_count, dtype=np.float64)
            incident_sum += np.bincount(
                first, weights=standardized, minlength=model_count
            )
            incident_sum += np.bincount(
                second, weights=standardized, minlength=model_count
            )
            incident_count += np.bincount(first, minlength=model_count)
            incident_count += np.bincount(second, minlength=model_count)
            node_residual = incident_sum / np.maximum(incident_count, 1.0)
            node_reliability = 1.0 / (1.0 + node_residual)
            endpoint_reliability = np.sqrt(
                node_reliability[first] * node_reliability[second]
            )
            robust_weights = pair_weights * tukey_weights * endpoint_reliability
            if not np.any(robust_weights > 0.0):
                robust_weights = pair_weights.copy()
            robust_weights /= np.mean(robust_weights)
            robust_sqrt = np.sqrt(robust_weights)
            pair_data_size = 2 * observations.size
            weighted_design.data[:pair_data_size:2] = robust_sqrt
            weighted_design.data[1:pair_data_size:2] = robust_sqrt
            np.multiply(
                observations,
                robust_sqrt,
                out=weighted_observations[: observations.size],
            )
            robust_result = lsq_linear(
                weighted_design,
                weighted_observations,
                bounds=(0.0, np.inf),
                lsmr_tol="auto",
            )
            if robust_result.success:
                result = robust_result
                pair_weights = robust_weights
        risks = result.x
        residual = observations - (risks[first] + risks[second])
        recovered.append(risks)
        diagnostics.append(
            {
                "band": band,
                "pair_rmse": float(np.sqrt(np.mean(residual**2))),
                "pair_mae": float(np.mean(np.abs(residual))),
                "weighted_pair_rmse": float(
                    np.sqrt(np.average(residual**2, weights=pair_weights))
                ),
                "mean_abs_transported_correlation": float(np.mean(np.abs(correlation))),
                "max_abs_transported_correlation": float(np.max(np.abs(correlation))),
                "min_pair_weight": float(pair_weights.min()),
                "max_pair_weight": float(pair_weights.max()),
            }
        )
    return np.stack(recovered, axis=1), diagnostics
