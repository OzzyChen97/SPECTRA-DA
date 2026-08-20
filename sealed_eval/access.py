"""Evaluator-only target-label access with append-only audit logging."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from protocol.tasks import TASKS

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parents[1]
SEALED_ROOT = WORKSPACE / ".sealed" / "spectra_da"
TASK_BY_ID = {task.id: task for task in TASKS}


def _append_audit(event: dict[str, object]) -> None:
    path = SEALED_ROOT / "evaluator_audit.jsonl"
    event = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, (json.dumps(event, sort_keys=True) + "\n").encode("utf-8"))
    finally:
        os.close(descriptor)


def load_target_labels(
    task: str,
    *,
    purpose: str,
    candidate_bank_sha256: str,
    expected_num_nodes: int,
) -> torch.Tensor:
    if task not in TASK_BY_ID:
        raise KeyError(f"unknown transfer task: {task}")
    target = TASK_BY_ID[task].target
    labels = torch.load(SEALED_ROOT / "labels" / f"{target}.pt", weights_only=True)
    if labels.ndim != 1 or labels.numel() != expected_num_nodes:
        _append_audit(
            {
                "event": "protocol_violation",
                "role": "evaluator",
                "purpose": purpose,
                "task": task,
                "domain": target,
                "candidate_bank_sha256": candidate_bank_sha256,
                "status": "blocked",
                "reason": "label_shape_mismatch",
            }
        )
        raise RuntimeError("sealed target-label shape does not match candidate predictions")
    _append_audit(
        {
            "event": "label_read",
            "role": "evaluator",
            "purpose": purpose,
            "task": task,
            "domain": target,
            "candidate_bank_sha256": candidate_bank_sha256,
            "status": "allowed",
        }
    )
    return labels.to(torch.long).cpu()
