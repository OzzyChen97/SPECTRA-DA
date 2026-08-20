#!/usr/bin/env python3
"""Export sealed trajectories for SourceOnly, A2GNN, GRADE, or PairAlign."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from pygda.models import PairAlign
from pygda.nn import A2GNNBase, GRADEBase, GradReverse
from pygda.utils import MMD

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from configs import PAPER_DEFAULTS  # noqa: E402
from protocol.access import load_training_task  # noqa: E402
from protocol.tasks import TASKS, domain_family, task_id  # noqa: E402
from scripts.trajectory_export.common import (  # noqa: E402
    enforce_gpu7,
    export_candidate,
    load_graph_manifest,
    set_seed,
    should_export,
    synchronized_elapsed,
    write_trajectory,
)

METHOD_NAMES = {
    "source_only": "SourceOnly",
    "a2gnn": "A2GNN",
    "grade": "GRADE",
    "pairalign": "PairAlign",
}


def common_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "nhid": args.nhid,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "lr": args.lr,
        "training_epochs": args.epochs,
    }


def propagation_infer(backbone, data, prop_nums: int):
    backbone.eval()
    with torch.no_grad():
        embeddings = backbone.feat_bottleneck(data.x, data.edge_index, None, prop_nums)
        logits = backbone.feat_classifier(embeddings, data.edge_index, None, prop_nums=1)
    return logits, embeddings


def grade_infer(backbone, data):
    backbone.eval()
    with torch.no_grad():
        embeddings, _ = backbone.feat_bottleneck(data.x, data.edge_index, None)
        logits = backbone.feat_classifier(embeddings)
    return logits, embeddings


def pairalign_infer(backbone, source_data, target_data, inference_seed: int):
    backbone.eval()
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(inference_seed)
        torch.cuda.manual_seed_all(inference_seed)
        with torch.no_grad():
            source_embeddings, source_logits = backbone(source_data, source_data.x)
            target_embeddings, target_logits = backbone(target_data, target_data.x)
    return source_logits, source_embeddings, target_logits, target_embeddings


def train_source_or_a2gnn(args, source_data, target_data, graph_manifest):
    method = METHOD_NAMES[args.method]
    config = common_config(args)
    if args.method == "source_only":
        config.update({"prop_nums": args.source_only_pnums, "alignment": "none"})
        source_pnums = target_pnums = args.source_only_pnums
        alignment_weight = 0.0
    else:
        config.update(
            {
                "s_pnums": args.s_pnums,
                "t_pnums": args.t_pnums,
                "alignment": "MMD",
                "alignment_weight": args.weight,
                "mmd_sampling_num": args.mmd_sampling_num,
                "mmd_times": args.mmd_times,
            }
        )
        source_pnums, target_pnums = args.s_pnums, args.t_pnums
        alignment_weight = args.weight

    num_classes = int(source_data.y.max().item()) + 1
    backbone = A2GNNBase(
        in_dim=source_data.x.size(1),
        hid_dim=args.nhid,
        num_classes=num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        mode="node",
    ).to(args.device)
    optimizer = torch.optim.Adam(backbone.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start = time.perf_counter()
    exported = []
    for epoch_index in range(args.epochs):
        epoch = epoch_index + 1
        backbone.train()
        source_logits = backbone(source_data, source_pnums)
        source_loss = F.nll_loss(
            F.log_softmax(source_logits[source_data.train_mask], dim=1),
            source_data.y[source_data.train_mask],
        )
        if args.method == "a2gnn":
            source_features = backbone.feat_bottleneck(
                source_data.x, source_data.edge_index, None, source_pnums
            )
            target_features = backbone.feat_bottleneck(
                target_data.x, target_data.edge_index, None, target_pnums
            )
            alignment_loss = MMD(
                source_features,
                target_features,
                sampling_num=args.mmd_sampling_num,
                times=args.mmd_times,
            )
            _ = backbone(target_data, target_pnums)
        else:
            alignment_loss = source_loss.new_zeros(())
        total_loss = source_loss + alignment_weight * alignment_loss
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        if should_export(epoch, args.epochs, args.checkpoint_interval):
            source_eval_logits, source_embeddings = propagation_infer(
                backbone, source_data, source_pnums
            )
            target_eval_logits, target_embeddings = propagation_infer(
                backbone, target_data, target_pnums
            )
            artifact = export_candidate(
                output_root=args.output_root.resolve(),
                method=method,
                source=args.source,
                target=args.target,
                seed=args.seed,
                epoch=epoch,
                device=args.device,
                config=config,
                source_data=source_data,
                target_data=target_data,
                source_logits=source_eval_logits,
                source_embeddings=source_embeddings,
                target_logits=target_eval_logits,
                target_embeddings=target_embeddings,
                backbone_state_dict=backbone.state_dict(),
                train_metrics={
                    "total_loss": float(total_loss.detach().item()),
                    "source_loss": float(source_loss.detach().item()),
                    "alignment_loss": float(alignment_loss.detach().item()),
                },
                elapsed_seconds=synchronized_elapsed(start),
                graph_manifest=graph_manifest,
            )
            exported.append({"candidate_id": artifact["candidate_id"], "path": artifact["path"]})
            print(
                f"method={method} epoch={epoch:04d} loss={total_loss.item():.6f} "
                f"source_val={artifact['metadata']['source_val_micro_f1']:.6f}"
            )
    return method, config, exported


def train_grade(args, source_data, target_data, graph_manifest):
    if args.grade_disc not in {"JS", "MMD"}:
        raise ValueError("sealed GRADE adapter supports only JS or MMD discrimination")
    method = "GRADE"
    config = common_config(args)
    config.update({"disc": args.grade_disc, "alignment_weight": args.grade_weight})
    num_classes = int(source_data.y.max().item()) + 1
    backbone = GRADEBase(
        in_dim=source_data.x.size(1),
        hid_dim=args.nhid,
        num_classes=num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        disc=args.grade_disc,
        mode="node",
    ).to(args.device)
    optimizer = torch.optim.Adam(backbone.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start = time.perf_counter()
    exported = []
    for epoch_index in range(args.epochs):
        epoch = epoch_index + 1
        alpha = 2.0 / (1.0 + math.exp(-10.0 * epoch_index / args.epochs)) - 1.0
        backbone.train()
        source_logits, source_features = backbone(source_data)
        target_logits, target_features = backbone(target_data)
        source_loss = F.nll_loss(
            F.log_softmax(source_logits[source_data.train_mask], dim=1),
            source_data.y[source_data.train_mask],
        )
        if args.grade_disc == "JS":
            features = torch.cat([source_features, target_features], dim=0)
            domain_logits = backbone.discriminator(GradReverse.apply(features, alpha))
            domain_labels = torch.cat(
                [
                    torch.zeros(source_data.num_nodes, dtype=torch.long, device=args.device),
                    torch.ones(target_data.num_nodes, dtype=torch.long, device=args.device),
                ]
            )
            alignment_loss = backbone.criterion(domain_logits, domain_labels)
        else:
            alignment_loss = MMD(source_features, target_features)
        total_loss = source_loss + args.grade_weight * alignment_loss
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        if should_export(epoch, args.epochs, args.checkpoint_interval):
            source_eval_logits, source_embeddings = grade_infer(backbone, source_data)
            target_eval_logits, target_embeddings = grade_infer(backbone, target_data)
            artifact = export_candidate(
                output_root=args.output_root.resolve(),
                method=method,
                source=args.source,
                target=args.target,
                seed=args.seed,
                epoch=epoch,
                device=args.device,
                config=config,
                source_data=source_data,
                target_data=target_data,
                source_logits=source_eval_logits,
                source_embeddings=source_embeddings,
                target_logits=target_eval_logits,
                target_embeddings=target_embeddings,
                backbone_state_dict=backbone.state_dict(),
                train_metrics={
                    "total_loss": float(total_loss.detach().item()),
                    "source_loss": float(source_loss.detach().item()),
                    "alignment_loss": float(alignment_loss.detach().item()),
                },
                elapsed_seconds=synchronized_elapsed(start),
                graph_manifest=graph_manifest,
            )
            exported.append({"candidate_id": artifact["candidate_id"], "path": artifact["path"]})
            print(
                f"method={method} epoch={epoch:04d} loss={total_loss.item():.6f} "
                f"source_val={artifact['metadata']['source_val_micro_f1']:.6f}"
            )
    return method, config, exported


def initialize_masked_edge_classes(source_data, num_classes: int):
    row, col = source_data.edge_index
    valid = source_data.train_mask[row] & source_data.train_mask[col]
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    pair_index = source_data.y[row[valid]].to(torch.long) * num_classes + source_data.y[col[valid]].to(torch.long)
    edge_classes = F.one_hot(pair_index, num_classes=num_classes * num_classes).float()
    return valid_indices, edge_classes


def update_pairalign_edges(pair, source_data, target_data, source_logits, target_logits, valid_indices, edge_classes):
    num_classes = pair.num_classes
    source_rows = source_data.edge_index[0, valid_indices]
    source_cols = source_data.edge_index[1, valid_indices]
    target_rows, target_cols = target_data.edge_index
    source_edge_predictions = torch.einsum(
        "bi,bj->bij", source_logits[source_rows], source_logits[source_cols]
    ).reshape(-1, num_classes * num_classes)
    target_edge_predictions = torch.einsum(
        "bi,bj->bij", target_logits[target_rows], target_logits[target_cols]
    ).reshape(-1, num_classes * num_classes)
    covariance = source_edge_predictions.T @ edge_classes / max(1, edge_classes.shape[0])
    target_mean = target_edge_predictions.mean(dim=0).reshape(-1, 1)
    source_mean = edge_classes.mean(dim=0).reshape(-1, 1)
    beta = pair.LS_optimization(covariance, target_mean, source_mean, pair.ls_lambda)
    beta = np.asarray(beta).reshape(num_classes, num_classes)

    source_pair_probability = edge_classes.mean(dim=0).detach().cpu().numpy().reshape(
        num_classes, num_classes
    )
    target_pair_probability = source_pair_probability * beta
    source_reg = source_pair_probability + pair.gamma_reg
    target_reg = target_pair_probability + pair.gamma_reg
    source_row_mass = source_reg.sum(axis=1)
    target_row_mass = target_reg.sum(axis=1)
    gamma = (target_reg.T / target_row_mass).T / (source_reg.T / source_row_mass).T
    gamma = np.nan_to_num(gamma, nan=1.0, posinf=1.0, neginf=1.0)

    weights = torch.ones(source_data.num_edges, dtype=torch.float32, device=source_data.x.device)
    known_pairs = edge_classes.argmax(dim=1).detach().cpu().numpy()
    known_weights = gamma.reshape(-1)[known_pairs]
    weights[valid_indices] = torch.from_numpy(known_weights).to(weights.device, dtype=weights.dtype)
    source_data.edge_weight = weights
    return gamma


def update_pairalign_labels(pair, source_data, target_logits, source_logits):
    mask = source_data.train_mask
    labels = source_data.y[mask]
    source_onehot = F.one_hot(labels, num_classes=pair.num_classes).float()
    source_distribution = source_onehot.mean(dim=0)
    target_distribution = F.softmax(target_logits, dim=1).mean(dim=0)
    source_predictions = F.softmax(source_logits[mask], dim=1)
    covariance = source_predictions.T @ source_onehot / max(1, int(mask.sum().item()))
    return pair.calc_label_rw(
        source_distribution,
        target_distribution,
        covariance,
        pair.lw_lambda,
    )


def train_pairalign(args, source_data, target_data, graph_manifest):
    method = "PairAlign"
    config = common_config(args)
    config.update(
        {
            "backbone": args.pair_backbone,
            "pooling": args.pair_pooling,
            "cls_dim": args.pair_cls_dim,
            "cls_layers": args.pair_cls_layers,
            "rw_lmda": args.pair_rw_lmda,
            "label_rw": args.pair_label_rw,
            "edge_rw": args.pair_edge_rw,
            "ew_start": args.pair_ew_start,
            "ew_freq": args.pair_ew_freq,
            "lw_start": args.pair_lw_start,
            "lw_freq": args.pair_lw_freq,
            "ls_lambda": args.pair_ls_lambda,
            "lw_lambda": args.pair_lw_lambda,
            "gamma_reg": args.pair_gamma_reg,
            "masked_source_labels": True,
        }
    )
    num_classes = int(source_data.y.max().item()) + 1
    pair = PairAlign(
        in_dim=source_data.x.size(1),
        hid_dim=args.nhid,
        num_classes=num_classes,
        num_layers=args.num_layers,
        cls_dim=args.pair_cls_dim,
        cls_layers=args.pair_cls_layers,
        dropout=args.dropout,
        backbone=args.pair_backbone,
        pooling=args.pair_pooling,
        rw_lmda=args.pair_rw_lmda,
        ls_lambda=args.pair_ls_lambda,
        lw_lambda=args.pair_lw_lambda,
        label_rw=args.pair_label_rw,
        edge_rw=args.pair_edge_rw,
        ew_start=args.pair_ew_start,
        ew_freq=args.pair_ew_freq,
        lw_start=args.pair_lw_start,
        lw_freq=args.pair_lw_freq,
        gamma_reg=args.pair_gamma_reg,
        weight_decay=args.weight_decay,
        lr=args.lr,
        epoch=args.epochs,
        device=args.device,
        verbose=0,
    )
    pair.gnn = pair.init_model(**pair.kwargs)
    pair.opt = torch.optim.Adam(pair.gnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    source_data.edge_weight = torch.ones(source_data.num_edges, device=args.device)
    target_data.edge_weight = torch.ones(target_data.num_edges, device=args.device)
    source_data.lw = np.ones(num_classes, dtype=np.float64)
    valid_indices, edge_classes = initialize_masked_edge_classes(source_data, num_classes)
    if valid_indices.numel() == 0:
        raise RuntimeError("PairAlign adapter found no train-to-train source edges")

    start = time.perf_counter()
    exported = []
    current_gamma = np.ones((num_classes, num_classes), dtype=np.float64)
    for epoch_index in range(args.epochs):
        epoch = epoch_index + 1
        pair.gnn.train()
        _, source_logits = pair.gnn(source_data, source_data.x)
        _, target_logits = pair.gnn(target_data, target_data.x)
        labels = source_data.y[source_data.train_mask]
        train_logits = source_logits[source_data.train_mask]
        if args.pair_label_rw:
            source_loss = pair.CE_loss_weight(
                train_logits,
                labels,
                torch.from_numpy(source_data.lw).to(args.device),
                pair.weight_CE_src,
            )
        else:
            source_loss = pair.CE_loss(train_logits, labels)
        pair.opt.zero_grad(set_to_none=True)
        source_loss.backward()
        pair.opt.step()

        pair.gnn.train()
        _, source_updated = pair.gnn(source_data, source_data.x)
        _, target_updated = pair.gnn(target_data, target_data.x)
        if (
            args.pair_edge_rw
            and epoch >= args.pair_ew_start
            and epoch % args.pair_ew_freq == 0
        ):
            current_gamma = update_pairalign_edges(
                pair,
                source_data,
                target_data,
                source_updated,
                target_updated,
                valid_indices,
                edge_classes,
            )
        if (
            args.pair_label_rw
            and epoch >= args.pair_lw_start
            and epoch % args.pair_lw_freq == 0
        ):
            source_data.lw = update_pairalign_labels(
                pair,
                source_data,
                target_updated,
                source_updated,
            )

        if should_export(epoch, args.epochs, args.checkpoint_interval):
            inference_seed = args.seed * 1_000_003 + epoch
            source_eval_logits, source_embeddings, target_eval_logits, target_embeddings = pairalign_infer(
                pair.gnn, source_data, target_data, inference_seed
            )
            artifact = export_candidate(
                output_root=args.output_root.resolve(),
                method=method,
                source=args.source,
                target=args.target,
                seed=args.seed,
                epoch=epoch,
                device=args.device,
                config=config,
                source_data=source_data,
                target_data=target_data,
                source_logits=source_eval_logits,
                source_embeddings=source_embeddings,
                target_logits=target_eval_logits,
                target_embeddings=target_embeddings,
                backbone_state_dict=pair.gnn.state_dict(),
                train_metrics={
                    "total_loss": float(source_loss.detach().item()),
                    "source_loss": float(source_loss.detach().item()),
                    "alignment_loss": 0.0,
                },
                elapsed_seconds=synchronized_elapsed(start),
                graph_manifest=graph_manifest,
                state_extra={
                    "source_edge_weight": source_data.edge_weight.detach().cpu(),
                    "source_label_weight": np.asarray(source_data.lw),
                    "class_pair_weight": current_gamma,
                    "valid_source_edge_indices": valid_indices.detach().cpu(),
                },
                inference_seed=inference_seed,
            )
            exported.append({"candidate_id": artifact["candidate_id"], "path": artifact["path"]})
            print(
                f"method={method} epoch={epoch:04d} loss={source_loss.item():.6f} "
                f"source_val={artifact['metadata']['source_val_micro_f1']:.6f}"
            )
    return method, config, exported


def train(args: argparse.Namespace) -> dict[str, Any]:
    enforce_gpu7(args.device)
    if args.checkpoint_interval <= 0:
        raise ValueError("checkpoint interval must be positive")
    valid_tasks = {task.id for task in TASKS}
    if task_id(args.source, args.target) not in valid_tasks:
        raise ValueError("source/target pair is not a frozen GDA-Select task")
    if domain_family(args.source) != domain_family(args.target):
        raise ValueError("cross-family training is outside the frozen benchmark")

    set_seed(args.seed)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    source_data, target_data = load_training_task(args.source, args.target, device=args.device)
    if "y" in target_data:
        raise RuntimeError("protocol violation: target graph contains labels")
    graph_manifest = load_graph_manifest()

    if args.method in {"source_only", "a2gnn"}:
        method, config, exported = train_source_or_a2gnn(
            args, source_data, target_data, graph_manifest
        )
    elif args.method == "grade":
        method, config, exported = train_grade(args, source_data, target_data, graph_manifest)
    else:
        method, config, exported = train_pairalign(
            args, source_data, target_data, graph_manifest
        )
    return write_trajectory(
        output_root=args.output_root.resolve(),
        source=args.source,
        target=args.target,
        method=method,
        config=config,
        seed=args.seed,
        epochs=args.epochs,
        checkpoint_interval=args.checkpoint_interval,
        checkpoints=exported,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(METHOD_NAMES), required=True)
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
    parser.add_argument("--weight-decay", type=float, default=PAPER_DEFAULTS["weight_decay"])
    parser.add_argument("--lr", type=float, default=PAPER_DEFAULTS["lr"])
    parser.add_argument("--weight", type=float, default=5.0)
    parser.add_argument("--s-pnums", type=int, default=0)
    parser.add_argument("--t-pnums", type=int, default=30)
    parser.add_argument("--source-only-pnums", type=int, default=1)
    parser.add_argument("--mmd-sampling-num", type=int, default=1000)
    parser.add_argument("--mmd-times", type=int, default=5)
    parser.add_argument("--grade-disc", choices=("JS", "MMD"), default="JS")
    parser.add_argument("--grade-weight", type=float, default=0.01)
    parser.add_argument("--pair-backbone", choices=("GS", "GCN"), default="GS")
    parser.add_argument("--pair-pooling", default="mean")
    parser.add_argument("--pair-cls-dim", type=int, default=128)
    parser.add_argument("--pair-cls-layers", type=int, default=2)
    parser.add_argument("--pair-rw-lmda", type=float, default=1.0)
    parser.add_argument("--pair-label-rw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pair-edge-rw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pair-ew-start", type=int, default=10)
    parser.add_argument("--pair-ew-freq", type=int, default=10)
    parser.add_argument("--pair-lw-start", type=int, default=10)
    parser.add_argument("--pair-lw-freq", type=int, default=10)
    parser.add_argument("--pair-ls-lambda", type=float, default=1.0)
    parser.add_argument("--pair-lw-lambda", type=float, default=0.005)
    parser.add_argument("--pair-gamma-reg", type=float, default=1e-4)
    return parser.parse_args()


if __name__ == "__main__":
    result = train(parse_args())
    print(json.dumps(result, indent=2))
