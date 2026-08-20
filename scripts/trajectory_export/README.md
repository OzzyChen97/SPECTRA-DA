# GDA-Select trajectory artifacts

The trusted exporters in this directory consume `protocol.access` tasks. A
training task contains source labels and fixed source train/validation masks,
while its target graph has no `y` field.

Each immutable checkpoint is written under:

```text
trajectory_bank/raw/<task>/<method>/<config_id>/seed_<seed>/checkpoint_<epoch>/
  metadata.json
  source_val.npz
  target_public.npz
  model_state.pt
```

`source_val.npz` contains source-validation indices, labels, predictions, and
embeddings. `target_public.npz` contains only logits, probabilities, hard
predictions, and embeddings. `schema.validate_checkpoint_dir` rejects target
label-like keys, unexpected arrays, hash mismatches, target-label access, or a
GPU mapping other than physical GPU 7 / logical `cuda:0`.

The original `model/ncfm.py` is not used for orchestration because its `fit`
and `predict` paths read target labels. The exporter reuses the unchanged
ADAlign backbone, characteristic-function loss, and SampleNet while enforcing
the sealed protocol.

`run_methods.py` provides the same boundary for SourceOnly, A2GNN, GRADE, and
PairAlign. The public PairAlign implementation calls `calc_true_ew`,
`calc_true_lw`, and `calc_true_beta`, all of which read target labels. The
sealed adapter skips these truth-only calls, retains pseudo-label covariance
matching, and estimates source class-pair statistics only on train-to-train
source edges. Edges touching held-out source-validation nodes keep unit
weights, so source validation remains genuinely held out.

## Full frozen benchmark

The pre-registered 16-task plan expands to 720 training trajectories and
10,800 immutable checkpoint candidates:

```bash
CUDA_VISIBLE_DEVICES=7 .venv/bin/python scripts/trajectory_export/launch_plan.py \
  --plan trajectory_plans/gda_select_v1.json \
  --output-root trajectory_bank/candidates/gda_select_v1 \
  --log-root results/gda_select/logs_full \
  --continue-on-error
```

Each run is first written below `OUTPUT_ROOT/_staging/`. The launcher validates
all expected checkpoint epochs, artifact hashes, task/method/seed metadata,
physical GPU 7, and zero target-label access before atomically promoting the
trajectory into the candidate bank. A complete staging run is promoted after a
restart; an incomplete run is moved to `OUTPUT_ROOT/_failed/` and retrained.
Completed runs are schema-validated before they are skipped, so rerunning the
same command is a safe resumable operation. Per-attempt logs are retained.

After all trajectories complete, build one immutable, resumable calibration
artifact per task without reading target labels:

```bash
CUDA_VISIBLE_DEVICES=7 .venv/bin/python shift_simulator/build_calibration_suite.py \
  --candidate-root trajectory_bank/candidates/gda_select_v1 \
  --output-root trajectory_bank/calibration/gda_select_v1 \
  --continue-on-error
```

Existing calibration artifacts are accepted only after their candidate-bank
hash, candidate ordering, array hash, frame error, GPU metadata, and zero
target-label-access declaration are revalidated.
