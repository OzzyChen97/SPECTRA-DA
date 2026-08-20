"""Pure metrics for target-label-free model-selection evaluation."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import f1_score


def macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(f1_score(labels, predictions, average="macro", zero_division=0))


def normalized_regret(selected_risk: float, risks: np.ndarray, epsilon: float = 1e-12) -> float:
    risks = np.asarray(risks, dtype=np.float64)
    best = float(risks.min())
    worst = float(risks.max())
    return float((selected_risk - best) / (worst - best + epsilon))


def rank_correlations(predicted_risks: np.ndarray, true_risks: np.ndarray) -> dict[str, float | None]:
    predicted_risks = np.asarray(predicted_risks, dtype=np.float64)
    true_risks = np.asarray(true_risks, dtype=np.float64)
    if predicted_risks.shape != true_risks.shape or predicted_risks.ndim != 1:
        raise ValueError("predicted and true risks must be aligned one-dimensional arrays")
    if predicted_risks.size < 2:
        return {"kendall_tau": None, "spearman_rho": None}
    tau = float(kendalltau(predicted_risks, true_risks, nan_policy="omit").statistic)
    rho = float(spearmanr(predicted_risks, true_risks, nan_policy="omit").statistic)
    return {
        "kendall_tau": tau if math.isfinite(tau) else None,
        "spearman_rho": rho if math.isfinite(rho) else None,
    }


def top_fraction_hit(selected_index: int, risks: np.ndarray, fraction: float = 0.05) -> bool:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    risks = np.asarray(risks, dtype=np.float64)
    cutoff = max(1, math.ceil(fraction * risks.size))
    best_indices = set(np.argsort(risks, kind="stable")[:cutoff].tolist())
    return selected_index in best_indices
