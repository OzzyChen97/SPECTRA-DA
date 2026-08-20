"""Shared utilities for sealed multi-method trajectory exporters."""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score

from protocol.access import PUBLIC_ROOT
from protocol.tasks import domain_family, task_id
from scripts.trajectory_export.schema import (
    SCHEMA_VERSION,
    atomic_json,
    atomic_npz,
    atomic_torch_save,
    canonical_hash,
    sha256_file,
    validate_checkpoint_dir,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enforce_gpu7(device: str) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "7":
        raise RuntimeError(f"GPU policy violation: CUDA_VISIBLE_DEVICES must be 7, got {visible!r}")
    if device != "cuda:0":
        raise RuntimeError("GPU policy violation: use cuda:0 after masking physical GPU 7")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("GPU policy violation: exactly one CUDA device must be visible")


def to_numpy(tensor: torch.Tensor, dtype: np.dtype | None = None) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    return array.astype(dtype, copy=False) if dtype is not None else array


def load_graph_manifest() -> dict[str, Any]:
    return json.loads((PUBLIC_ROOT / "manifest.json").read_text(encoding="utf-8"))


def export_candidate(
    *,
    output_root: Path,
    method: str,
    source: str,
    target: str,
    seed: int,
    epoch: int,
    device: str,
    config: dict[str, Any],
    source_data,
    target_data,
    source_logits: torch.Tensor,
    source_embeddings: torch.Tensor,
    target_logits: torch.Tensor,
    target_embeddings: torch.Tensor,
    backbone_state_dict: dict[str, torch.Tensor],
    train_metrics: dict[str, float],
    elapsed_seconds: float,
    graph_manifest: dict[str, Any],
    state_extra: dict[str, Any] | None = None,
    inference_seed: int | None = None,
) -> dict[str, Any]:
    if "y" in target_data:
        raise RuntimeError("protocol violation: target graph contains labels")
    config_id = canonical_hash(config)
    task = task_id(source, target)
    final_destination = (
        output_root
        / task
        / method
        / config_id
        / f"seed_{seed}"
        / f"checkpoint_{epoch:04d}"
    )
    if final_destination.exists():
        raise FileExistsError(f"immutable checkpoint already exists: {final_destination}")
    final_destination.parent.mkdir(parents=True, exist_ok=True)
    destination = final_destination.with_name(
        f".{final_destination.name}.tmp-{os.getpid()}"
    )
    if destination.exists():
        raise FileExistsError(f"temporary checkpoint directory already exists: {destination}")
    destination.mkdir()

    source_probs = torch.softmax(source_logits, dim=1)
    target_probs = torch.softmax(target_logits, dim=1)
    source_predictions = source_probs.argmax(dim=1)
    target_predictions = target_probs.argmax(dim=1)
    val_mask = source_data.val_mask
    val_indices = torch.nonzero(val_mask, as_tuple=False).flatten()
    val_labels = source_data.y[val_mask]
    val_predictions = source_predictions[val_mask]
    source_val_micro = float((val_predictions == val_labels).float().mean().item())
    source_val_macro = float(
        f1_score(
            to_numpy(val_labels),
            to_numpy(val_predictions),
            average="macro",
            zero_division=0,
        )
    )
    target_entropy = float(
        (-(target_probs * target_probs.clamp_min(1e-12).log()).sum(dim=1)).mean().item()
    )

    source_path = destination / "source_val.npz"
    target_path = destination / "target_public.npz"
    state_path = destination / "model_state.pt"
    atomic_npz(
        source_path,
        indices=to_numpy(val_indices, np.int64),
        labels=to_numpy(val_labels, np.int64),
        logits=to_numpy(source_logits[val_mask], np.float32),
        probabilities=to_numpy(source_probs[val_mask], np.float32),
        hard_predictions=to_numpy(val_predictions, np.int64),
        embeddings=to_numpy(source_embeddings[val_mask], np.float16),
    )
    atomic_npz(
        target_path,
        logits=to_numpy(target_logits, np.float32),
        probabilities=to_numpy(target_probs, np.float32),
        hard_predictions=to_numpy(target_predictions, np.int64),
        embeddings=to_numpy(target_embeddings, np.float16),
    )
    state = {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "config": config,
        "epoch": epoch,
        "backbone_state_dict": {
            key: value.detach().cpu() for key, value in backbone_state_dict.items()
        },
    }
    if state_extra:
        state["method_state"] = state_extra
    atomic_torch_save(state, state_path)

    candidate_id = f"{task}__{method}__{config_id}__seed-{seed}__epoch-{epoch:04d}"
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "family": domain_family(source),
        "source": source,
        "target": target,
        "method": method,
        "config_id": config_id,
        "config": config,
        "seed": seed,
        "epoch": epoch,
        "source_num_nodes": source_data.num_nodes,
        "source_train_nodes": int(source_data.train_mask.sum().item()),
        "source_val_nodes": int(source_data.val_mask.sum().item()),
        "target_num_nodes": target_data.num_nodes,
        "source_val_micro_f1": source_val_micro,
        "source_val_macro_f1": source_val_macro,
        "target_entropy": target_entropy,
        "train_total_loss": train_metrics["total_loss"],
        "train_source_loss": train_metrics["source_loss"],
        "train_alignment_loss": train_metrics["alignment_loss"],
        "trajectory_elapsed_seconds": elapsed_seconds,
        "max_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "logical_device": device,
        "physical_gpu": 7,
        "cuda_visible_devices": "7",
        "target_label_access_count": 0,
        "target_public_has_labels": False,
        "source_graph_sha256": graph_manifest["domains"][source]["graph_sha256"],
        "target_graph_sha256": graph_manifest["domains"][target]["graph_sha256"],
        "source_split_sha256": graph_manifest["domains"][source]["split_sha256"],
        "artifact_sha256": {
            source_path.name: sha256_file(source_path),
            target_path.name: sha256_file(target_path),
            state_path.name: sha256_file(state_path),
        },
    }
    if inference_seed is not None:
        metadata["inference_seed"] = inference_seed
    atomic_json(metadata, destination / "metadata.json")
    validate_checkpoint_dir(destination)
    destination.replace(final_destination)
    return {
        "candidate_id": candidate_id,
        "path": str(final_destination),
        "metadata": metadata,
    }


def write_trajectory(
    *,
    output_root: Path,
    source: str,
    target: str,
    method: str,
    config: dict[str, Any],
    seed: int,
    epochs: int,
    checkpoint_interval: int,
    checkpoints: list[dict[str, str]],
) -> dict[str, Any]:
    config_id = canonical_hash(config)
    run_dir = (
        output_root
        / task_id(source, target)
        / method
        / config_id
        / f"seed_{seed}"
    )
    trajectory = {
        "schema_version": SCHEMA_VERSION,
        "task": task_id(source, target),
        "method": method,
        "config_id": config_id,
        "config": config,
        "seed": seed,
        "epochs": epochs,
        "checkpoint_interval": checkpoint_interval,
        "target_label_access_count": 0,
        "physical_gpu": 7,
        "checkpoints": checkpoints,
    }
    atomic_json(trajectory, run_dir / "trajectory.json")
    return trajectory


def should_export(epoch: int, epochs: int, interval: int) -> bool:
    return epoch == 1 or epoch == epochs or epoch % interval == 0


def synchronized_elapsed(start: float) -> float:
    torch.cuda.synchronize()
    return time.perf_counter() - start
