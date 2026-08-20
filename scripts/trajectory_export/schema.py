"""Versioned trajectory artifact schema and validation helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCHEMA_VERSION = 1

SOURCE_VAL_KEYS = frozenset(
    {
        "indices",
        "labels",
        "logits",
        "probabilities",
        "hard_predictions",
        "embeddings",
    }
)
TARGET_PUBLIC_KEYS = frozenset(
    {
        "logits",
        "probabilities",
        "hard_predictions",
        "embeddings",
    }
)
FORBIDDEN_TARGET_KEYS = frozenset(
    {
        "label",
        "labels",
        "target_label",
        "target_labels",
        "target_y",
        "y",
        "ground_truth",
    }
)

REQUIRED_METADATA_KEYS = frozenset(
    {
        "artifact_sha256",
        "candidate_id",
        "config",
        "config_id",
        "cuda_visible_devices",
        "epoch",
        "family",
        "logical_device",
        "max_memory_bytes",
        "method",
        "physical_gpu",
        "seed",
        "source",
        "source_graph_sha256",
        "source_num_nodes",
        "source_split_sha256",
        "source_train_nodes",
        "source_val_macro_f1",
        "source_val_micro_f1",
        "source_val_nodes",
        "target",
        "target_entropy",
        "target_graph_sha256",
        "target_label_access_count",
        "target_num_nodes",
        "target_public_has_labels",
        "task",
        "train_alignment_loss",
        "train_source_loss",
        "train_total_loss",
        "trajectory_elapsed_seconds",
    }
)
FINITE_METADATA_FIELDS = frozenset(
    {
        "source_val_macro_f1",
        "source_val_micro_f1",
        "target_entropy",
        "train_alignment_loss",
        "train_source_loss",
        "train_total_loss",
        "trajectory_elapsed_seconds",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any, length: int = 12) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def atomic_json(value: Any, path: Path, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(temporary, mode)
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_source_val(path: Path) -> dict[str, tuple[int, ...]]:
    with np.load(path, allow_pickle=False) as artifact:
        keys = frozenset(artifact.files)
        _require(keys == SOURCE_VAL_KEYS, f"unexpected source_val keys: {sorted(keys)}")
        shapes = {key: tuple(artifact[key].shape) for key in artifact.files}
        rows = artifact["indices"].shape[0]
        _require(artifact["indices"].ndim == 1, "source indices must be one-dimensional")
        _require(artifact["labels"].shape == (rows,), "source labels shape mismatch")
        _require(artifact["hard_predictions"].shape == (rows,), "source predictions shape mismatch")
        _require(artifact["logits"].ndim == 2 and artifact["logits"].shape[0] == rows, "source logits shape mismatch")
        _require(artifact["probabilities"].shape == artifact["logits"].shape, "source probability shape mismatch")
        _require(artifact["embeddings"].ndim == 2 and artifact["embeddings"].shape[0] == rows, "source embeddings shape mismatch")
    return shapes


def _validate_target_public(path: Path) -> dict[str, tuple[int, ...]]:
    with np.load(path, allow_pickle=False) as artifact:
        keys = frozenset(artifact.files)
        lowered = {key.lower() for key in keys}
        _require(not lowered.intersection(FORBIDDEN_TARGET_KEYS), "target artifact contains a label-like field")
        _require(keys == TARGET_PUBLIC_KEYS, f"unexpected target_public keys: {sorted(keys)}")
        shapes = {key: tuple(artifact[key].shape) for key in artifact.files}
        rows = artifact["hard_predictions"].shape[0]
        _require(artifact["hard_predictions"].ndim == 1, "target predictions must be one-dimensional")
        _require(artifact["logits"].ndim == 2 and artifact["logits"].shape[0] == rows, "target logits shape mismatch")
        _require(artifact["probabilities"].shape == artifact["logits"].shape, "target probability shape mismatch")
        _require(artifact["embeddings"].ndim == 2 and artifact["embeddings"].shape[0] == rows, "target embeddings shape mismatch")
    return shapes


def _validate_metadata(metadata: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_METADATA_KEYS - set(metadata))
    _require(not missing, f"checkpoint metadata is missing required fields: {missing}")
    for name in FINITE_METADATA_FIELDS:
        value = metadata[name]
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value)),
            f"checkpoint metadata field must be finite: {name}",
        )
    for name in ("source_val_macro_f1", "source_val_micro_f1"):
        _require(0.0 <= float(metadata[name]) <= 1.0, f"invalid source validation score: {name}")
    _require(float(metadata["target_entropy"]) >= 0.0, "target entropy must be non-negative")
    _require(
        float(metadata["trajectory_elapsed_seconds"]) > 0.0,
        "trajectory runtime must be positive",
    )
    _require(
        isinstance(metadata["max_memory_bytes"], int)
        and not isinstance(metadata["max_memory_bytes"], bool)
        and metadata["max_memory_bytes"] >= 0,
        "peak GPU memory must be a non-negative integer",
    )
    for name in (
        "epoch",
        "source_num_nodes",
        "source_train_nodes",
        "source_val_nodes",
        "target_num_nodes",
    ):
        _require(
            isinstance(metadata[name], int)
            and not isinstance(metadata[name], bool)
            and metadata[name] > 0,
            f"checkpoint metadata field must be a positive integer: {name}",
        )
    _require(isinstance(metadata["config"], dict), "candidate config must be an object")
    for name in (
        "candidate_id",
        "config_id",
        "family",
        "method",
        "source",
        "source_graph_sha256",
        "source_split_sha256",
        "target",
        "target_graph_sha256",
        "task",
    ):
        _require(isinstance(metadata[name], str) and metadata[name], f"invalid metadata string: {name}")


def validate_checkpoint_dir(path: Path, verify_hashes: bool = True) -> dict[str, Any]:
    """Validate one immutable candidate checkpoint directory."""

    metadata_path = path / "metadata.json"
    source_path = path / "source_val.npz"
    target_path = path / "target_public.npz"
    state_path = path / "model_state.pt"
    for artifact in (metadata_path, source_path, target_path, state_path):
        _require(artifact.is_file(), f"missing trajectory artifact: {artifact}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _validate_metadata(metadata)
    _require(metadata.get("schema_version") == SCHEMA_VERSION, "unsupported schema version")
    _require(metadata.get("target_label_access_count") == 0, "candidate reports target-label access")
    _require(metadata.get("target_public_has_labels") is False, "candidate target artifact is not label-free")
    _require(metadata.get("logical_device") == "cuda:0", "candidate was not produced on logical cuda:0")
    _require(metadata.get("physical_gpu") == 7, "candidate was not produced on physical GPU 7")

    source_shapes = _validate_source_val(source_path)
    target_shapes = _validate_target_public(target_path)
    _require(
        target_shapes["hard_predictions"][0] == int(metadata["target_num_nodes"]),
        "target node count mismatch",
    )

    if verify_hashes:
        expected = metadata.get("artifact_sha256", {})
        for artifact in (source_path, target_path, state_path):
            _require(expected.get(artifact.name) == sha256_file(artifact), f"hash mismatch: {artifact.name}")

    return {
        "candidate_id": metadata["candidate_id"],
        "source_shapes": source_shapes,
        "target_shapes": target_shapes,
    }


def discover_candidate_records(
    output_root: Path,
    task: str,
    *,
    verify_hashes: bool = True,
) -> list[dict[str, Any]]:
    """Discover and validate all candidates for one transfer task."""

    task_root = output_root / task
    records: list[dict[str, Any]] = []
    if not task_root.is_dir():
        raise FileNotFoundError(f"candidate task directory does not exist: {task_root}")
    for metadata_path in sorted(task_root.glob("**/checkpoint_*/metadata.json")):
        directory = metadata_path.parent
        validate_checkpoint_dir(directory, verify_hashes=verify_hashes)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        _require(metadata.get("task") == task, f"candidate task mismatch: {directory}")
        records.append({"path": directory, "metadata": metadata})
    _require(bool(records), f"no candidate checkpoints found under {task_root}")
    identifiers = [record["metadata"]["candidate_id"] for record in records]
    _require(len(identifiers) == len(set(identifiers)), "duplicate candidate identifiers")
    return records


def candidate_bank_hash(records: list[dict[str, Any]]) -> str:
    """Return a path-independent digest for a validated candidate set."""

    entries = []
    for record in sorted(records, key=lambda item: item["metadata"]["candidate_id"]):
        metadata = record["metadata"]
        entries.append(
            {
                "candidate_id": metadata["candidate_id"],
                "task": metadata["task"],
                "method": metadata["method"],
                "config_id": metadata["config_id"],
                "seed": metadata["seed"],
                "epoch": metadata["epoch"],
                "source_graph_sha256": metadata["source_graph_sha256"],
                "target_graph_sha256": metadata["target_graph_sha256"],
                "source_split_sha256": metadata["source_split_sha256"],
                "artifact_sha256": metadata["artifact_sha256"],
            }
        )
    return canonical_hash(entries, length=64)
