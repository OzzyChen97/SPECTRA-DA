"""Rehydrate frozen candidate checkpoints for pseudo-target inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from pygda.nn import A2GNNBase, GRADEBase, ReweightGNN


def _load_state(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


@dataclass
class PreparedCandidate:
    """A frozen candidate loaded once and reusable across simulated shifts."""

    method: str
    config: dict[str, Any]
    backbone: torch.nn.Module
    device: torch.device

    def infer(
        self,
        graph,
        *,
        inference_seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        graph = graph.to(self.device)
        if self.method in {"SourceOnly", "A2GNN", "ADAlign"}:
            propagation = (
                self.config["prop_nums"]
                if self.method == "SourceOnly"
                else self.config["t_pnums"]
            )
            with torch.no_grad():
                embeddings = self.backbone.feat_bottleneck(
                    graph.x,
                    graph.edge_index,
                    None,
                    propagation,
                )
                logits = self.backbone.feat_classifier(
                    embeddings,
                    graph.edge_index,
                    None,
                    prop_nums=1,
                )
        elif self.method == "GRADE":
            with torch.no_grad():
                embeddings, _ = self.backbone.feat_bottleneck(
                    graph.x,
                    graph.edge_index,
                    None,
                )
                logits = self.backbone.feat_classifier(embeddings)
        elif self.method == "PairAlign":
            graph.edge_weight = torch.ones(
                graph.edge_index.shape[1],
                dtype=graph.x.dtype,
                device=self.device,
            )
            # PyGDA's ReweightGNN calls F.dropout without training=self.training;
            # fork and fix RNG so exported pseudo-target predictions are reproducible.
            cuda_devices = [self.device.index or 0] if self.device.type == "cuda" else []
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(inference_seed)
                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(inference_seed)
                with torch.no_grad():
                    embeddings, logits = self.backbone(graph, graph.x)
        else:  # pragma: no cover - construction rejects unsupported methods.
            raise ValueError(f"unsupported candidate method: {self.method}")
        return logits.detach().cpu(), embeddings.detach().cpu()


def prepare_candidate(
    record: dict,
    *,
    in_dim: int,
    num_classes: int,
    device: torch.device,
) -> PreparedCandidate:
    """Load a frozen checkpoint without reading any graph label field."""

    state = _load_state(record["path"] / "model_state.pt", device)
    method = state["method"]
    config = state["config"]

    if method in {"SourceOnly", "A2GNN", "ADAlign"}:
        backbone = A2GNNBase(
            in_dim=in_dim,
            hid_dim=config["nhid"],
            num_classes=num_classes,
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            mode="node",
        ).to(device)
        backbone.load_state_dict(state["backbone_state_dict"], strict=True)
    elif method == "GRADE":
        backbone = GRADEBase(
            in_dim=in_dim,
            hid_dim=config["nhid"],
            num_classes=num_classes,
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            disc=config["disc"],
            mode="node",
        ).to(device)
        backbone.load_state_dict(state["backbone_state_dict"], strict=True)
    elif method == "PairAlign":
        backbone = ReweightGNN(
            input_dim=in_dim,
            gnn_dim=config["nhid"],
            output_dim=num_classes,
            cls_dim=config["cls_dim"],
            gnn_layers=config["num_layers"],
            cls_layers=config["cls_layers"],
            backbone=config["backbone"],
            pooling=config["pooling"],
            dropout=config["dropout"],
            bn=False,
            rw_lmda=config["rw_lmda"],
        ).to(device)
        backbone.load_state_dict(state["backbone_state_dict"], strict=True)
    else:
        raise ValueError(f"unsupported candidate method: {method}")
    backbone.eval()
    return PreparedCandidate(
        method=method,
        config=config,
        backbone=backbone,
        device=device,
    )


def infer_candidate(
    record: dict,
    graph,
    *,
    num_classes: int,
    device: torch.device,
    inference_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility wrapper for one-off pseudo-target inference."""

    prepared = prepare_candidate(
        record,
        in_dim=int(graph.x.shape[1]),
        num_classes=num_classes,
        device=device,
    )
    return prepared.infer(graph, inference_seed=inference_seed)
