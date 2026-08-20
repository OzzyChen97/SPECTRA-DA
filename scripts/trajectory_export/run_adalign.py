#!/usr/bin/env python3
"""Export a sealed-label-compliant ADAlign training trajectory.

The target graph supplied by :mod:`protocol.access` has no ``y`` attribute.
This runner intentionally does not call ``NCFM.fit`` or ``NCFM.predict``, both
of which read target labels in the original research implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from configs import PAPER_DEFAULTS  # noqa: E402
from model.ncfm import NCFM  # noqa: E402
from protocol.access import PUBLIC_ROOT, load_training_task  # noqa: E402
from protocol.tasks import domain_family, task_id  # noqa: E402
from scripts.trajectory_export.schema import (  # noqa: E402
    SCHEMA_VERSION,
    atomic_json,
    atomic_npz,
    atomic_torch_save,
    canonical_hash,
    sha256_file,
    validate_checkpoint_dir,
)
from utils.loss import CF, SampleNet  # noqa: E402


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


def model_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "nhid": args.nhid,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "s_pnums": args.s_pnums,
        "t_pnums": args.t_pnums,
        "weight": args.weight,
        "weight_decay": args.weight_decay,
        "lr": args.lr,
        "t_batchsize": args.t_batchsize,
        "alpha": args.alpha,
        "training_epochs": args.epochs,
    }


def build_model(args: argparse.Namespace, source_data) -> NCFM:
    num_classes = int(source_data.y.max().item()) + 1
    model = NCFM(
        in_dim=source_data.x.size(1),
        hid_dim=args.nhid,
        num_classes=num_classes,
        device=args.device,
        num_layers=args.num_layers,
        dropout=args.dropout,
        s_pnums=args.s_pnums,
        t_pnums=args.t_pnums,
        weight=args.weight,
        weight_decay=args.weight_decay,
        lr=args.lr,
        epoch=args.epochs,
        t_batchsize=args.t_batchsize,
        alpha=args.alpha,
        verbose=0,
    )
    model.a2gnn = model.init_model(**model.kwargs)
    model.sample_net = SampleNet(
        feature_dim=args.nhid,
        t_batchsize=args.t_batchsize,
        t_var=1,
    ).to(args.device)
    return model


def forward_loss(model: NCFM, source_data, target_data):
    source_logits = model.a2gnn(source_data, model.s_pnums)
    source_loss = F.nll_loss(
        F.log_softmax(source_logits[source_data.train_mask], dim=1),
        source_data.y[source_data.train_mask],
    )
    source_features = model.a2gnn.feat_bottleneck(
        source_data.x, source_data.edge_index, None, model.s_pnums
    )
    target_features = model.a2gnn.feat_bottleneck(
        target_data.x, target_data.edge_index, None, model.t_pnums
    )
    sampled_t = model.sample_net(model.device)
    cf_loss = CF(
        model.alpha,
        1 - model.alpha,
        source_features,
        target_features,
        sampled_t,
    )
    total_loss = source_loss + model.weight * cf_loss

    # Preserve the original ADAlign forward path and random-number consumption.
    target_logits = model.a2gnn(target_data, model.t_pnums)
    return total_loss, source_loss, cf_loss, source_logits, target_logits


@torch.no_grad()
def infer(model: NCFM, data, prop_nums: int) -> tuple[torch.Tensor, torch.Tensor]:
    model.a2gnn.eval()
    embeddings = model.a2gnn.feat_bottleneck(data.x, data.edge_index, None, prop_nums)
    logits = model.a2gnn.feat_classifier(embeddings, data.edge_index, None, prop_nums=1)
    return logits, embeddings


def to_numpy(tensor: torch.Tensor, dtype: np.dtype | None = None) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    return array.astype(dtype, copy=False) if dtype is not None else array


def checkpoint_dir(output_root: Path, task: str, config_id: str, seed: int, epoch: int) -> Path:
    return output_root / task / "ADAlign" / config_id / f"seed_{seed}" / f"checkpoint_{epoch:04d}"


def export_checkpoint(
    *,
    model: NCFM,
    source_data,
    target_data,
    args: argparse.Namespace,
    epoch: int,
    epoch_metrics: dict[str, float],
    elapsed_seconds: float,
    output_root: Path,
    config: dict[str, Any],
    config_id: str,
    graph_manifest: dict[str, Any],
) -> dict[str, Any]:
    task = task_id(args.source, args.target)
    final_destination = checkpoint_dir(output_root, task, config_id, args.seed, epoch)
    if final_destination.exists():
        raise FileExistsError(f"immutable checkpoint already exists: {final_destination}")
    final_destination.parent.mkdir(parents=True, exist_ok=True)
    destination = final_destination.with_name(
        f".{final_destination.name}.tmp-{os.getpid()}"
    )
    if destination.exists():
        raise FileExistsError(f"temporary checkpoint directory already exists: {destination}")
    destination.mkdir()

    source_logits, source_embeddings = infer(model, source_data, model.s_pnums)
    target_logits, target_embeddings = infer(model, target_data, model.t_pnums)
    source_probs = torch.softmax(source_logits, dim=1)
    target_probs = torch.softmax(target_logits, dim=1)
    source_pred = source_probs.argmax(dim=1)
    target_pred = target_probs.argmax(dim=1)
    val_mask = source_data.val_mask
    val_idx = torch.nonzero(val_mask, as_tuple=False).flatten()
    val_labels = source_data.y[val_mask]
    val_pred = source_pred[val_mask]

    source_val_micro = float((val_pred == val_labels).float().mean().item())
    source_val_macro = float(
        f1_score(to_numpy(val_labels), to_numpy(val_pred), average="macro", zero_division=0)
    )
    target_entropy = float(
        (-(target_probs * target_probs.clamp_min(1e-12).log()).sum(dim=1)).mean().item()
    )

    source_path = destination / "source_val.npz"
    target_path = destination / "target_public.npz"
    state_path = destination / "model_state.pt"
    atomic_npz(
        source_path,
        indices=to_numpy(val_idx, np.int64),
        labels=to_numpy(val_labels, np.int64),
        logits=to_numpy(source_logits[val_mask], np.float32),
        probabilities=to_numpy(source_probs[val_mask], np.float32),
        hard_predictions=to_numpy(val_pred, np.int64),
        embeddings=to_numpy(source_embeddings[val_mask], np.float16),
    )
    atomic_npz(
        target_path,
        logits=to_numpy(target_logits, np.float32),
        probabilities=to_numpy(target_probs, np.float32),
        hard_predictions=to_numpy(target_pred, np.int64),
        embeddings=to_numpy(target_embeddings, np.float16),
    )
    atomic_torch_save(
        {
            "schema_version": SCHEMA_VERSION,
            "method": "ADAlign",
            "config": config,
            "epoch": epoch,
            "backbone_state_dict": {key: value.detach().cpu() for key, value in model.a2gnn.state_dict().items()},
        },
        state_path,
    )

    candidate_id = f"{task}__ADAlign__{config_id}__seed-{args.seed}__epoch-{epoch:04d}"
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "family": domain_family(args.source),
        "source": args.source,
        "target": args.target,
        "method": "ADAlign",
        "config_id": config_id,
        "config": config,
        "seed": args.seed,
        "epoch": epoch,
        "source_num_nodes": source_data.num_nodes,
        "source_train_nodes": int(source_data.train_mask.sum().item()),
        "source_val_nodes": int(source_data.val_mask.sum().item()),
        "target_num_nodes": target_data.num_nodes,
        "source_val_micro_f1": source_val_micro,
        "source_val_macro_f1": source_val_macro,
        "target_entropy": target_entropy,
        "train_total_loss": epoch_metrics["total_loss"],
        "train_source_loss": epoch_metrics["source_loss"],
        "train_alignment_loss": epoch_metrics["alignment_loss"],
        "trajectory_elapsed_seconds": elapsed_seconds,
        "max_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "logical_device": args.device,
        "physical_gpu": 7,
        "cuda_visible_devices": "7",
        "target_label_access_count": 0,
        "target_public_has_labels": False,
        "source_graph_sha256": graph_manifest["domains"][args.source]["graph_sha256"],
        "target_graph_sha256": graph_manifest["domains"][args.target]["graph_sha256"],
        "source_split_sha256": graph_manifest["domains"][args.source]["split_sha256"],
        "artifact_sha256": {
            source_path.name: sha256_file(source_path),
            target_path.name: sha256_file(target_path),
            state_path.name: sha256_file(state_path),
        },
    }
    atomic_json(metadata, destination / "metadata.json")
    validate_checkpoint_dir(destination)
    destination.replace(final_destination)
    return {
        "candidate_id": candidate_id,
        "path": str(final_destination),
        "metadata": metadata,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    enforce_gpu7(args.device)
    if args.source == args.target:
        raise ValueError("source and target must differ")
    if args.checkpoint_interval <= 0:
        raise ValueError("checkpoint interval must be positive")
    if args.t_batchsize < 16:
        raise ValueError("ADAlign SampleNet requires t_batchsize >= 16")

    set_seed(args.seed)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    source_data, target_data = load_training_task(args.source, args.target, device=args.device)
    if "y" in target_data:
        raise RuntimeError("protocol violation: target graph contains labels")

    config = model_config(args)
    config_id = canonical_hash(config)
    graph_manifest = json.loads((PUBLIC_ROOT / "manifest.json").read_text(encoding="utf-8"))
    model = build_model(args, source_data)
    optimizer_main = torch.optim.Adam(
        model.a2gnn.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    optimizer_sample = torch.optim.Adam(
        model.sample_net.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    start = time.perf_counter()
    exported: list[dict[str, Any]] = []
    for epoch_idx in range(args.epochs):
        epoch = epoch_idx + 1
        model.a2gnn.train()
        model.sample_net.train()

        for parameter in model.a2gnn.parameters():
            parameter.requires_grad = False
        optimizer_sample.zero_grad(set_to_none=True)
        _, _, cf_loss_max, _, _ = forward_loss(model, source_data, target_data)
        (-cf_loss_max).backward()
        optimizer_sample.step()
        for parameter in model.a2gnn.parameters():
            parameter.requires_grad = True

        optimizer_main.zero_grad(set_to_none=True)
        total_loss, source_loss, cf_loss, _, _ = forward_loss(model, source_data, target_data)
        total_loss.backward()
        optimizer_main.step()

        metrics = {
            "total_loss": float(total_loss.detach().item()),
            "source_loss": float(source_loss.detach().item()),
            "alignment_loss": float(cf_loss.detach().item()),
        }
        should_export = epoch == 1 or epoch == args.epochs or epoch % args.checkpoint_interval == 0
        if should_export:
            torch.cuda.synchronize()
            artifact = export_checkpoint(
                model=model,
                source_data=source_data,
                target_data=target_data,
                args=args,
                epoch=epoch,
                epoch_metrics=metrics,
                elapsed_seconds=time.perf_counter() - start,
                output_root=args.output_root.resolve(),
                config=config,
                config_id=config_id,
                graph_manifest=graph_manifest,
            )
            exported.append({"candidate_id": artifact["candidate_id"], "path": artifact["path"]})
            print(
                f"epoch={epoch:04d} loss={metrics['total_loss']:.6f} "
                f"source_val={artifact['metadata']['source_val_micro_f1']:.6f} "
                f"target_entropy={artifact['metadata']['target_entropy']:.6f}"
            )

    run_dir = args.output_root.resolve() / task_id(args.source, args.target) / "ADAlign" / config_id / f"seed_{args.seed}"
    trajectory = {
        "schema_version": SCHEMA_VERSION,
        "task": task_id(args.source, args.target),
        "method": "ADAlign",
        "config_id": config_id,
        "config": config,
        "seed": args.seed,
        "epochs": args.epochs,
        "checkpoint_interval": args.checkpoint_interval,
        "target_label_access_count": 0,
        "physical_gpu": 7,
        "checkpoints": exported,
    }
    atomic_json(trajectory, run_dir / "trajectory.json")
    return trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=PAPER_DEFAULTS["epochs"])
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--output-root", type=Path, default=REPO / "trajectory_bank" / "raw")
    parser.add_argument("--nhid", type=int, default=PAPER_DEFAULTS["nhid"])
    parser.add_argument("--num-layers", type=int, default=PAPER_DEFAULTS["num_layers"])
    parser.add_argument("--dropout", type=float, default=PAPER_DEFAULTS["dropout"])
    parser.add_argument("--s-pnums", type=int, default=PAPER_DEFAULTS["s_pnums"])
    parser.add_argument("--t-pnums", type=int, default=PAPER_DEFAULTS["t_pnums"])
    parser.add_argument("--weight", type=float, default=PAPER_DEFAULTS["weight"])
    parser.add_argument("--weight-decay", type=float, default=PAPER_DEFAULTS["weight_decay"])
    parser.add_argument("--lr", type=float, default=PAPER_DEFAULTS["lr"])
    parser.add_argument("--t-batchsize", type=int, default=PAPER_DEFAULTS["t_batchsize"])
    parser.add_argument("--alpha", type=float, default=PAPER_DEFAULTS["alpha"])
    return parser.parse_args()


if __name__ == "__main__":
    result = train(parse_args())
    print(json.dumps(result, indent=2))
