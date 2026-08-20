#!/usr/bin/env python3
"""Reliability-aware rank fusion for target-label-free selector outputs.

The tool is a post-selector: it consumes already public selector JSON files and
produces a new selector JSON without reading candidate artifacts or target
labels.  It is intended for the next SPECTRA-DA iteration described in
``arxiv/notes/reliable_selection_next_steps.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
SHORTLIST_EXCLUSION_SCORE = 1.0e12
FUSION_MODES = (
    "rank_fusion",
    "transfer_shortlist_spectra_rerank",
    "spectra_shortlist_transfer_rerank",
    "support_adaptive",
)


def atomic_json(document: dict[str, Any], path: Path) -> None:
    """Write JSON atomically without importing trajectory code or torch."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def choose(scores: dict[str, float], direction: str) -> str:
    if not scores or not all(math.isfinite(value) for value in scores.values()):
        raise ValueError("selection scores must be non-empty and finite")
    optimum = min(scores.values()) if direction == "minimize" else max(scores.values())
    return min(candidate for candidate, value in scores.items() if value == optimum)


def load_selection(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    scores = document.get("candidate_scores")
    if not isinstance(scores, dict) or not scores:
        raise ValueError(f"selection has no candidate_scores: {path}")
    if int(document.get("label_access_count", 0)) != 0:
        raise ValueError(f"selection reports target-label access: {path}")
    if int(document.get("protocol_violation_count", 0)) != 0:
        raise ValueError(f"selection reports protocol violation: {path}")
    direction = document.get("score_direction")
    if direction not in {"minimize", "maximize"}:
        raise ValueError(f"selection has invalid score_direction: {path}")
    numeric = {str(candidate): float(score) for candidate, score in scores.items()}
    if not all(math.isfinite(value) for value in numeric.values()):
        raise ValueError(f"selection has non-finite scores: {path}")
    document["candidate_scores"] = numeric
    return document


def percentile_ranks(scores: dict[str, float], direction: str) -> dict[str, float]:
    """Return midranks in [0, 1], with lower always better.

    Equal scores receive the same average percentile rank.  This avoids letting
    candidate-id ordering create artificial ranking differences for selectors
    with plateaued or intentionally tied scores.
    """

    reverse = direction == "maximize"
    ordered = sorted(
        scores,
        key=lambda candidate: -scores[candidate] if reverse else scores[candidate],
    )
    denominator = max(1, len(ordered) - 1)
    ranks: dict[str, float] = {}
    start = 0
    while start < len(ordered):
        value = scores[ordered[start]]
        stop = start + 1
        while stop < len(ordered) and scores[ordered[stop]] == value:
            stop += 1
        midrank = 0.5 * (start + stop - 1) / denominator
        for candidate in ordered[start:stop]:
            ranks[candidate] = float(midrank)
        start = stop
    return ranks


def top_fraction_candidates(
    ranks: dict[str, float],
    *,
    fraction: float,
) -> set[str]:
    """Return the top fraction under lower-is-better percentile ranks."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("shortlist_fraction must lie in (0, 1]")
    ordered = sorted(ranks, key=lambda candidate: (ranks[candidate], candidate))
    cutoff = max(1, math.ceil(fraction * len(ordered)))
    cutoff_rank = ranks[ordered[cutoff - 1]]
    return {candidate for candidate, rank in ranks.items() if rank <= cutoff_rank}


def covariance_confidence(
    spectra: dict[str, Any],
    *,
    fallback_shrinkage: float,
) -> tuple[float, str]:
    """Extract label-free covariance confidence from a SPECTRA selector JSON."""

    if not 0.0 <= fallback_shrinkage <= 1.0:
        raise ValueError("fallback covariance shrinkage must lie in [0, 1]")
    diagnostics = spectra.get("transport_diagnostics")
    if isinstance(diagnostics, dict):
        shrinkage = diagnostics.get("covariance_shrinkage")
        if isinstance(shrinkage, dict) and "gamma" in shrinkage:
            gamma = float(shrinkage["gamma"])
            if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
                raise ValueError("SPECTRA covariance gamma must lie in [0, 1]")
            return gamma, "transport_diagnostics.covariance_shrinkage.gamma"
    return 1.0 - float(fallback_shrinkage), "1_minus_covariance_shrinkage"


def uncertainty_component(spectra: dict[str, Any], candidates: set[str]) -> dict[str, float]:
    raw = spectra.get("candidate_transport_uncertainty") or spectra.get("candidate_uncertainty")
    if raw is None:
        return {candidate: 0.0 for candidate in candidates}
    if not isinstance(raw, dict):
        raise ValueError("candidate uncertainty must be a dictionary when present")
    if set(raw) != candidates:
        raise ValueError("candidate uncertainty coverage does not match candidate scores")
    values = {str(candidate): float(value) for candidate, value in raw.items()}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("candidate uncertainty contains non-finite values")
    if np.allclose(list(values.values()), 0.0, rtol=0.0, atol=0.0):
        return {candidate: 0.0 for candidate in candidates}
    return percentile_ranks(values, "minimize")


def validate_pair(spectra: dict[str, Any], transfer: dict[str, Any]) -> set[str]:
    fields = ("task", "candidate_bank_sha256", "candidate_count")
    for field in fields:
        if spectra.get(field) != transfer.get(field):
            raise ValueError(f"selection metadata mismatch for {field}")
    candidates = set(spectra["candidate_scores"])
    if candidates != set(transfer["candidate_scores"]):
        raise ValueError("selection candidate coverage mismatch")
    if len(candidates) != int(spectra.get("candidate_count", -1)):
        raise ValueError("candidate_count does not match score coverage")
    return candidates


def reliable_rank_fusion(
    spectra: dict[str, Any],
    transfer: dict[str, Any],
    *,
    uncertainty_weight: float,
    transfer_score_weight: float,
    covariance_shrinkage: float,
    calibration_temperature: float,
    selector_name: str,
    fusion_mode: str = "rank_fusion",
    shortlist_fraction: float = 0.20,
) -> dict[str, Any]:
    if uncertainty_weight < 0.0 or transfer_score_weight < 0.0:
        raise ValueError("fusion weights must be non-negative")
    if not 0.0 <= covariance_shrinkage <= 1.0:
        raise ValueError("covariance_shrinkage must lie in [0, 1]")
    if calibration_temperature <= 0.0:
        raise ValueError("calibration_temperature must be positive")
    if fusion_mode not in FUSION_MODES:
        raise ValueError(f"unknown fusion_mode: {fusion_mode}")

    candidates = validate_pair(spectra, transfer)
    spectra_rank = percentile_ranks(spectra["candidate_scores"], spectra["score_direction"])
    transfer_rank = percentile_ranks(transfer["candidate_scores"], transfer["score_direction"])
    uncertainty_rank = uncertainty_component(spectra, candidates)
    selected_shortlist: set[str] | None = None
    covariance_gamma: float | None = None
    covariance_gamma_source = "not_used"

    if fusion_mode == "rank_fusion":
        spectra_weight = (1.0 - covariance_shrinkage) / calibration_temperature
        transfer_weight = transfer_score_weight
        fused_scores = {
            candidate: float(
                spectra_weight * spectra_rank[candidate]
                + transfer_weight * transfer_rank[candidate]
                + uncertainty_weight * uncertainty_rank[candidate]
            )
            for candidate in candidates
        }
    elif fusion_mode == "support_adaptive":
        covariance_gamma, covariance_gamma_source = covariance_confidence(
            spectra,
            fallback_shrinkage=covariance_shrinkage,
        )
        spectra_weight = covariance_gamma / calibration_temperature
        transfer_weight = (1.0 - covariance_gamma) * transfer_score_weight
        normalizer = spectra_weight + transfer_weight
        if normalizer > 0.0:
            spectra_weight /= normalizer
            transfer_weight /= normalizer
        fused_scores = {
            candidate: float(
                spectra_weight * spectra_rank[candidate]
                + transfer_weight * transfer_rank[candidate]
                + uncertainty_weight * uncertainty_rank[candidate]
            )
            for candidate in candidates
        }
    elif fusion_mode == "transfer_shortlist_spectra_rerank":
        selected_shortlist = top_fraction_candidates(
            transfer_rank,
            fraction=shortlist_fraction,
        )
        fused_scores = {
            candidate: float(
                spectra_rank[candidate]
                + uncertainty_weight * uncertainty_rank[candidate]
                if candidate in selected_shortlist
                else SHORTLIST_EXCLUSION_SCORE + transfer_rank[candidate]
            )
            for candidate in candidates
        }
    else:
        selected_shortlist = top_fraction_candidates(
            spectra_rank,
            fraction=shortlist_fraction,
        )
        fused_scores = {
            candidate: float(
                transfer_rank[candidate]
                + uncertainty_weight * uncertainty_rank[candidate]
                if candidate in selected_shortlist
                else SHORTLIST_EXCLUSION_SCORE + spectra_rank[candidate]
            )
            for candidate in candidates
        }
    selected = choose(fused_scores, "minimize")
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": spectra["task"],
        "selector": selector_name,
        "candidate_bank_sha256": spectra["candidate_bank_sha256"],
        "candidate_count": spectra["candidate_count"],
        "candidate_scores": fused_scores,
        "score_direction": "minimize",
        "score_semantics": "ranking",
        "selected_candidate_id": selected,
        "label_access_count": 0,
        "protocol_violation_count": 0,
        "fusion_config": {
            "spectra_selector": spectra.get("selector"),
            "transfer_selector": transfer.get("selector"),
            "uncertainty_source": (
                "candidate_transport_uncertainty"
                if "candidate_transport_uncertainty" in spectra
                else "candidate_uncertainty"
                if "candidate_uncertainty" in spectra
                else "none"
            ),
            "uncertainty_weight": float(uncertainty_weight),
            "transfer_score_weight": float(transfer_score_weight),
            "spectra_rank_shrinkage": float(covariance_shrinkage),
            "covariance_shrinkage": float(covariance_shrinkage),
            "covariance_shrinkage_note": (
                "post-hoc rank-fusion compatibility alias; actual covariance "
                "shrinkage must be produced by spectra_cal.py diagnostics"
            ),
            "calibration_temperature": float(calibration_temperature),
            "fusion_mode": fusion_mode,
            "shortlist_fraction": float(shortlist_fraction),
            "covariance_gamma": covariance_gamma,
            "covariance_gamma_source": covariance_gamma_source,
            "shortlist_size": len(selected_shortlist) if selected_shortlist is not None else None,
            "shortlist_exclusion_score": (
                SHORTLIST_EXCLUSION_SCORE if selected_shortlist is not None else None
            ),
            "shortlist_owner": (
                "transfer_score"
                if fusion_mode == "transfer_shortlist_spectra_rerank"
                else "spectra"
                if fusion_mode == "spectra_shortlist_transfer_rerank"
                else None
            ),
            "rank_scale": "percentile",
        },
        "component_selected_candidate_id": {
            "spectra": choose(spectra["candidate_scores"], spectra["score_direction"]),
            "transfer_score": choose(transfer["candidate_scores"], transfer["score_direction"]),
            "uncertainty": choose(uncertainty_rank, "minimize"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectra-selection", type=Path, required=True)
    parser.add_argument("--transfer-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selector-name", default="spectra_reliable_rank_fusion")
    parser.add_argument("--uncertainty-weight", type=float, default=0.25)
    parser.add_argument("--transfer-score-weight", type=float, default=0.50)
    parser.add_argument("--covariance-shrinkage", type=float, default=0.25)
    parser.add_argument("--calibration-temperature", type=float, default=1.0)
    parser.add_argument("--fusion-mode", choices=FUSION_MODES, default="rank_fusion")
    parser.add_argument("--shortlist-fraction", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = reliable_rank_fusion(
        load_selection(args.spectra_selection),
        load_selection(args.transfer_selection),
        uncertainty_weight=args.uncertainty_weight,
        transfer_score_weight=args.transfer_score_weight,
        covariance_shrinkage=args.covariance_shrinkage,
        calibration_temperature=args.calibration_temperature,
        selector_name=args.selector_name,
        fusion_mode=args.fusion_mode,
        shortlist_fraction=args.shortlist_fraction,
    )
    atomic_json(result, args.output)
    print(
        json.dumps(
            {
                "task": result["task"],
                "selector": result["selector"],
                "selected_candidate_id": result["selected_candidate_id"],
                "output": str(args.output),
                "fusion_config": result["fusion_config"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
