"""Deep Embedded Validation adapted to the frozen trajectory artifact schema."""

from __future__ import annotations

import hashlib

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


DEV_C_GRID = (1e-2, 10 ** -0.5, 10.0, 10 ** 2.5, 1e4)
DEV_MAX_DOMAIN_SAMPLES = 3000


def _stable_seed(identifier: str) -> int:
    return int.from_bytes(hashlib.sha256(identifier.encode("utf-8")).digest()[:4], "little")


def domain_importance_weights(
    source_features: np.ndarray,
    target_features: np.ndarray,
    validation_features: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Estimate p(target|z)/p(source|z) using a deterministic linear classifier."""

    source = np.asarray(source_features, dtype=np.float32)
    target = np.asarray(target_features, dtype=np.float32)
    validation = np.asarray(validation_features, dtype=np.float32)
    if source.ndim != 2 or target.ndim != 2 or validation.ndim != 2:
        raise ValueError("DEV features must be two-dimensional")
    if source.shape[1] != target.shape[1] or source.shape[1] != validation.shape[1]:
        raise ValueError("DEV feature dimensions do not match")

    rng = np.random.default_rng(seed)
    source = source[rng.permutation(source.shape[0])[:DEV_MAX_DOMAIN_SAMPLES]]
    target = target[rng.permutation(target.shape[0])[:DEV_MAX_DOMAIN_SAMPLES]]
    source_count = source.shape[0]
    target_count = target.shape[0]
    features = np.concatenate((source, target), axis=0)
    domain_labels = np.concatenate(
        (np.ones(source_count, dtype=np.int64), np.zeros(target_count, dtype=np.int64))
    )
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        domain_labels,
        train_size=0.8,
        random_state=seed,
        stratify=domain_labels,
    )
    best_accuracy = -np.inf
    best_classifier: LogisticRegression | None = None
    for regularization in DEV_C_GRID:
        classifier = LogisticRegression(
            C=regularization,
            solver="liblinear",
            max_iter=2000,
            random_state=seed,
        )
        classifier.fit(train_x, train_y)
        accuracy = float(classifier.score(test_x, test_y))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_classifier = classifier
    assert best_classifier is not None
    probabilities = best_classifier.predict_proba(validation)
    class_to_column = {int(label): index for index, label in enumerate(best_classifier.classes_)}
    target_probability = probabilities[:, class_to_column[0]]
    source_probability = probabilities[:, class_to_column[1]]
    ratio = target_probability / np.clip(source_probability, 1e-8, None)
    return (ratio * source_count / target_count).astype(np.float64, copy=False)


def dev_risk(weights: np.ndarray, errors: np.ndarray) -> float:
    """Return the control-variate DEV risk from importance weights and errors."""

    weight = np.asarray(weights, dtype=np.float64).reshape(-1)
    error = np.asarray(errors, dtype=np.float64).reshape(-1)
    if weight.shape != error.shape or weight.size < 2:
        raise ValueError("DEV weights and errors must be aligned non-trivial vectors")
    weighted_error = weight * error
    variance = float(np.var(weight, ddof=1))
    if variance <= 1e-12:
        return float(np.mean(weighted_error))
    covariance = float(np.cov(np.column_stack((weighted_error, weight)), rowvar=False)[0, 1])
    eta = -covariance / variance
    return float(np.mean(weighted_error) + eta * np.mean(weight) - eta)


def adapted_dev_score(
    source_validation_features: np.ndarray,
    target_features: np.ndarray,
    source_validation_errors: np.ndarray,
    *,
    candidate_id: str,
) -> float:
    """Compute DEV with the source-validation features available in schema v1."""

    source = np.asarray(source_validation_features, dtype=np.float32)
    target = np.asarray(target_features, dtype=np.float32)
    weights = domain_importance_weights(
        source,
        target,
        source,
        seed=_stable_seed(candidate_id),
    )
    return dev_risk(weights, source_validation_errors)
