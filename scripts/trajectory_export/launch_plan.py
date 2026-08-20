#!/usr/bin/env python3
"""Run a frozen trajectory plan sequentially on physical GPU 7."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from protocol.tasks import PILOT_TASKS, TASKS, task_id  # noqa: E402
from scripts.trajectory_export.schema import (  # noqa: E402
    atomic_json,
    validate_checkpoint_dir,
)

SUPPORTED_METHODS = {"source_only", "a2gnn", "grade", "pairalign", "adalign"}
METHOD_NAMES = {
    "source_only": "SourceOnly",
    "a2gnn": "A2GNN",
    "grade": "GRADE",
    "pairalign": "PairAlign",
    "adalign": "ADAlign",
}


def plan_hash(plan: dict[str, Any]) -> str:
    payload = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != 1:
        raise ValueError("unsupported trajectory plan schema")
    if not isinstance(plan.get("epochs"), int) or plan["epochs"] <= 0:
        raise ValueError("plan epochs must be positive")
    if not isinstance(plan.get("checkpoint_interval"), int) or plan["checkpoint_interval"] <= 0:
        raise ValueError("plan checkpoint_interval must be positive")
    if not plan.get("seeds"):
        raise ValueError("plan must contain at least one seed")
    task_scope = plan.get("task_scope", "pilot")
    if task_scope not in {"pilot", "all"}:
        raise ValueError("plan task_scope must be 'pilot' or 'all'")
    allowed_tasks = PILOT_TASKS if task_scope == "pilot" else TASKS
    frozen_tasks = {task.id for task in allowed_tasks}
    listed_tasks = [task_id(source, target) for source, target in plan.get("tasks", [])]
    if len(listed_tasks) != len(set(listed_tasks)):
        raise ValueError("trajectory plan contains duplicate tasks")
    for source, target in plan.get("tasks", []):
        if task_id(source, target) not in frozen_tasks:
            raise ValueError(
                f"plan task is outside the frozen {task_scope} benchmark scope: {source}->{target}"
            )
    methods = set(plan.get("methods", {}))
    if not methods or not methods.issubset(SUPPORTED_METHODS):
        raise ValueError(f"unsupported methods in plan: {sorted(methods - SUPPORTED_METHODS)}")
    for method, configs in plan["methods"].items():
        names = [config["name"] for config in configs]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate config names for {method}")


def bool_flag(name: str, value: bool) -> str:
    flag = "--" + name.replace("_", "-")
    return flag if value else "--no-" + name.replace("_", "-")


def build_command(
    *,
    method: str,
    source: str,
    target: str,
    seed: int,
    epochs: int,
    checkpoint_interval: int,
    output_root: Path,
    params: dict[str, Any],
) -> list[str]:
    script = "run_adalign.py" if method == "adalign" else "run_methods.py"
    command = [
        sys.executable,
        str(REPO / "scripts" / "trajectory_export" / script),
    ]
    if method != "adalign":
        command.extend(["--method", method])
    command.extend(
        [
            "--source",
            source,
            "--target",
            target,
            "--seed",
            str(seed),
            "--epochs",
            str(epochs),
            "--checkpoint-interval",
            str(checkpoint_interval),
            "--output-root",
            str(output_root),
        ]
    )
    for name, value in params.items():
        if isinstance(value, bool):
            command.append(bool_flag(name, value))
        else:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    return command


def enumerate_runs(plan: dict[str, Any]) -> list[dict[str, Any]]:
    runs = []
    for source, target in plan["tasks"]:
        for method, configs in plan["methods"].items():
            for config in configs:
                for seed in plan["seeds"]:
                    key = f"{task_id(source, target)}__{method}__{config['name']}__seed-{seed}"
                    runs.append(
                        {
                            "key": key,
                            "source": source,
                            "target": target,
                            "method": method,
                            "config_name": config["name"],
                            "params": config["params"],
                            "seed": int(seed),
                        }
                    )
    return runs


def expected_checkpoint_epochs(epochs: int, checkpoint_interval: int) -> tuple[int, ...]:
    values = {1, epochs}
    values.update(range(checkpoint_interval, epochs + 1, checkpoint_interval))
    return tuple(sorted(values))


def locate_run_directory(root: Path) -> Path:
    trajectories = sorted(root.glob("**/trajectory.json"))
    if len(trajectories) != 1:
        raise ValueError(
            f"expected exactly one trajectory under {root}, found {len(trajectories)}"
        )
    return trajectories[0].parent


def validate_run_directory(
    run_directory: Path,
    run: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    trajectory_path = run_directory / "trajectory.json"
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"missing trajectory metadata: {trajectory_path}")
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    expected_task = task_id(run["source"], run["target"])
    expected_method = METHOD_NAMES[run["method"]]
    checks = {
        "task": expected_task,
        "method": expected_method,
        "seed": run["seed"],
        "epochs": plan["epochs"],
        "checkpoint_interval": plan["checkpoint_interval"],
        "target_label_access_count": 0,
        "physical_gpu": 7,
    }
    for name, expected in checks.items():
        if trajectory.get(name) != expected:
            raise ValueError(
                f"trajectory field mismatch for {run['key']}: "
                f"{name}={trajectory.get(name)!r}, expected {expected!r}"
            )

    expected_epochs = expected_checkpoint_epochs(
        plan["epochs"], plan["checkpoint_interval"]
    )
    expected_names = {f"checkpoint_{epoch:04d}" for epoch in expected_epochs}
    checkpoint_directories = sorted(
        path for path in run_directory.glob("checkpoint_*") if path.is_dir()
    )
    actual_names = {path.name for path in checkpoint_directories}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(
            f"checkpoint set mismatch for {run['key']}: missing={missing}, extra={extra}"
        )
    temporary_directories = sorted(run_directory.glob(".checkpoint_*.tmp-*"))
    if temporary_directories:
        raise ValueError(f"temporary checkpoint directories remain: {temporary_directories}")

    candidate_ids = []
    for directory in checkpoint_directories:
        validate_checkpoint_dir(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        epoch = int(directory.name.removeprefix("checkpoint_"))
        metadata_checks = {
            "task": expected_task,
            "source": run["source"],
            "target": run["target"],
            "method": expected_method,
            "seed": run["seed"],
            "epoch": epoch,
            "target_label_access_count": 0,
            "physical_gpu": 7,
        }
        for name, expected in metadata_checks.items():
            if metadata.get(name) != expected:
                raise ValueError(
                    f"checkpoint metadata mismatch in {directory}: "
                    f"{name}={metadata.get(name)!r}, expected {expected!r}"
                )
        candidate_ids.append(metadata["candidate_id"])
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(f"duplicate candidate identifiers for {run['key']}")
    if len(trajectory.get("checkpoints", [])) != len(expected_epochs):
        raise ValueError(f"trajectory checkpoint list is incomplete for {run['key']}")
    return trajectory


def quarantine_stage(stage_root: Path, output_root: Path, key: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = output_root / "_failed" / f"{key}__{timestamp}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage_root.replace(destination)
    return destination


def promote_stage(
    *,
    stage_root: Path,
    output_root: Path,
    run: dict[str, Any],
    plan: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    attempt: int,
    log_path: Path,
) -> Path:
    staged_run = locate_run_directory(stage_root)
    trajectory = validate_run_directory(staged_run, run, plan)
    relative = staged_run.relative_to(stage_root)
    final_run = output_root / relative
    if final_run.exists():
        raise FileExistsError(f"immutable final trajectory already exists: {final_run}")

    for checkpoint in trajectory["checkpoints"]:
        checkpoint_name = Path(checkpoint["path"]).name
        checkpoint["path"] = str(final_run / checkpoint_name)
    atomic_json(trajectory, staged_run / "trajectory.json")
    state["runs"][run["key"]] = {
        "status": "promoting",
        "attempt": attempt,
        "stage": str(stage_root),
        "artifact_dir": str(final_run),
        "log": str(log_path),
    }
    atomic_json(state, state_path)

    final_run.parent.mkdir(parents=True, exist_ok=True)
    staged_run.replace(final_run)
    shutil.rmtree(stage_root)
    validate_run_directory(final_run, run, plan)
    return final_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=REPO / "trajectory_bank" / "raw")
    parser.add_argument("--log-root", type=Path, default=REPO / "results" / "gda_select" / "logs")
    parser.add_argument("--only-task")
    parser.add_argument("--only-method", choices=sorted(SUPPORTED_METHODS))
    parser.add_argument("--max-runs", type=int)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="record failed runs and continue; a non-zero exit is returned after all selected runs",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "7":
        raise SystemExit("trajectory plans require CUDA_VISIBLE_DEVICES=7")
    plan = json.loads(args.plan.resolve().read_text(encoding="utf-8"))
    validate_plan(plan)
    digest = plan_hash(plan)
    output_root = args.output_root.resolve()
    log_root = (args.log_root.resolve() / plan["name"])
    state_path = output_root / "_plans" / f"{plan['name']}.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("plan_sha256") != digest:
            raise RuntimeError("existing plan state has a different plan hash")
    else:
        state = {
            "schema_version": 1,
            "plan": plan["name"],
            "plan_sha256": digest,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "runs": {},
        }

    runs = enumerate_runs(plan)
    if args.only_task:
        runs = [run for run in runs if task_id(run["source"], run["target"]) == args.only_task]
    if args.only_method:
        runs = [run for run in runs if run["method"] == args.only_method]
    if args.max_runs is not None:
        runs = runs[: args.max_runs]

    failures = []
    for run in runs:
        previous = state["runs"].get(run["key"], {})
        if previous.get("status") == "completed":
            artifact_dir = previous.get("artifact_dir")
            if artifact_dir is None:
                raise RuntimeError(
                    f"completed legacy state lacks artifact_dir for {run['key']}; "
                    "use a fresh output root or migrate the state explicitly"
                )
            validate_run_directory(Path(artifact_dir), run, plan)
            print(f"skip completed {run['key']}")
            continue
        stage_root = output_root / "_staging" / run["key"]
        attempt = int(previous.get("attempt", 0)) + 1

        if previous.get("status") == "promoting":
            artifact_dir = Path(previous["artifact_dir"])
            if artifact_dir.exists():
                validate_run_directory(artifact_dir, run, plan)
                state["runs"][run["key"]] = {
                    **previous,
                    "status": "completed",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "returncode": 0,
                }
                atomic_json(state, state_path)
                print(f"recovered promoted {run['key']}")
                continue

        if stage_root.exists():
            try:
                staged_log = Path(previous.get("log", log_root / f"{run['key']}.log"))
                final_run = promote_stage(
                    stage_root=stage_root,
                    output_root=output_root,
                    run=run,
                    plan=plan,
                    state=state,
                    state_path=state_path,
                    attempt=int(previous.get("attempt", 1)),
                    log_path=staged_log,
                )
                state["runs"][run["key"]] = {
                    "status": "completed",
                    "attempt": int(previous.get("attempt", 1)),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "returncode": 0,
                    "artifact_dir": str(final_run),
                    "log": str(staged_log),
                    "recovered_from_stage": True,
                }
                atomic_json(state, state_path)
                print(f"recovered completed stage {run['key']}")
                continue
            except (FileNotFoundError, ValueError):
                quarantined = quarantine_stage(stage_root, output_root, run["key"])
                print(f"quarantined incomplete stage {run['key']} -> {quarantined}")

        command = build_command(
            method=run["method"],
            source=run["source"],
            target=run["target"],
            seed=run["seed"],
            epochs=plan["epochs"],
            checkpoint_interval=plan["checkpoint_interval"],
            output_root=stage_root,
            params=run["params"],
        )
        print("run", run["key"])
        if args.dry_run:
            print(" ".join(command))
            continue
        log_root.mkdir(parents=True, exist_ok=True)
        stage_root.mkdir(parents=True, exist_ok=False)
        log_path = log_root / f"{run['key']}.attempt-{attempt:03d}.log"
        state["runs"][run["key"]] = {
            "status": "running",
            "attempt": attempt,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "stage": str(stage_root),
            "log": str(log_path),
        }
        atomic_json(state, state_path)
        with log_path.open("w", encoding="utf-8") as log_handle:
            result = subprocess.run(
                command,
                cwd=REPO,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode != 0:
            state["runs"][run["key"]] = {
                "status": "failed",
                "attempt": attempt,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "returncode": result.returncode,
                "stage": str(stage_root),
                "log": str(log_path),
            }
            atomic_json(state, state_path)
            failures.append(run["key"])
            if args.continue_on_error:
                print(f"failed {run['key']}; inspect {log_path}")
                continue
            raise RuntimeError(f"trajectory run failed; inspect {log_path}")

        final_run = promote_stage(
            stage_root=stage_root,
            output_root=output_root,
            run=run,
            plan=plan,
            state=state,
            state_path=state_path,
            attempt=attempt,
            log_path=log_path,
        )
        state["runs"][run["key"]] = {
            "status": "completed",
            "attempt": attempt,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "returncode": 0,
            "artifact_dir": str(final_run),
            "log": str(log_path),
        }
        atomic_json(state, state_path)

    completed = sum(1 for run in state["runs"].values() if run.get("status") == "completed")
    failed = sum(1 for run in state["runs"].values() if run.get("status") == "failed")
    print(
        json.dumps(
            {
                "plan": plan["name"],
                "selected_runs": len(runs),
                "completed_total": completed,
                "failed_total": failed,
            },
            indent=2,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
