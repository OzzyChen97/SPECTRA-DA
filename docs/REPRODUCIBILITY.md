# Reproducibility guide

## Artifact boundary

Candidate checkpoints are immutable directories containing:

```text
metadata.json
source_val.npz
target_public.npz
model_state.pt
```

`source_val.npz` may contain source-validation labels. `target_public.npz`
must contain only logits, probabilities, hard predictions, and embeddings.
The schema validator rejects label-like target keys and binds artifacts to
SHA-256 hashes.

## Sidecar calibration

The final controlled variant augments the base source-simulated calibration
bank with pre-frozen source-labelled sidecars. These artifacts are not included
because they are large and machine-specific. Generate them locally, copy
`configs/refinement_sidecars.example.json` to
`configs/refinement_sidecars_v1.json`, then replace each path and hash.

The manifest must:

- cover all four frozen development tasks;
- report zero target-label access and zero protocol violations;
- match the candidate-bank ordering and spectral configuration;
- pass the source-node rank gate for feature-mask grids.

## Frozen refinement command

```bash
CUDA_VISIBLE_DEVICES=7 python selector/refinement_objective.py \
  --calibration-root trajectory_bank/calibration/formal \
  --sidecar-manifest configs/refinement_sidecars_v1.json \
  --reference-results results/reference/final24.json \
  --bootstrap-workers 4 \
  --output results/autosota_refinement.json
```

The reference result contains development-fold regrets only; it is used to
compute localized-gain diagnostics and must itself report zero label access.

## Expected aggregate output

The exact aggregate values are stored in `results/final_metrics.json`. Runtime
varies by host; the frozen rerun was `327.394s`, with a hard guardrail of
`480s`. All non-runtime values should match the release metrics when the same
candidate bank, calibration arrays, sidecars, package versions, and random
seeds are used.

## Hidden evaluation

Do not run hidden evaluation inside the optimization process. The sealed
evaluator should run once after the selector and protocol are frozen, ideally
under a separate account or service that exposes neither target labels nor
candidate-level target scores.
