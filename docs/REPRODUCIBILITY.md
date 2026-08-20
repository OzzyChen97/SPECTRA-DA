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

The four-task refinement manifest must:

- cover all four frozen development tasks;
- report zero target-label access and zero protocol violations;
- match the candidate-bank ordering and spectral configuration;
- pass the source-node rank gate for feature-mask grids.

For frozen deployment, combine the development sidecars with the remaining
task sidecars and validate them against all 16 base calibration artifacts:

```bash
python scripts/build_sidecar_manifest.py \
  --calibration-root trajectory_bank/calibration/gda_select_v1 \
  --base-manifest configs/refinement_sidecars_v1.json \
  --sidecar-root /path/to/remaining/combined_sidecars \
  --sidecar-root /path/to/remaining/lifted_sidecars \
  --output configs/spectra_sidecars_16.json
```

The builder rejects incomplete task coverage, stale candidate ordering,
spectral-config or parent-calibration mismatches, artifact-hash mismatches,
target-label access, and protocol violations.

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

## Frozen 16-task selector generation

After freezing the sidecar manifest and selector configuration, generate a new
immutable selection root. This step uses public target predictions and
source-only calibration artifacts, but never target labels:

```bash
CUDA_VISIBLE_DEVICES=7 python selector/run_spectra_suite.py \
  --candidate-root trajectory_bank/candidates/gda_select_v1 \
  --calibration-root trajectory_bank/calibration/gda_select_v1 \
  --sidecar-manifest configs/spectra_sidecars_16.json \
  --config configs/search_space.yaml \
  --output-root results/gda_select/selections/spectra_frozen_v2 \
  --device cuda:0
```

If public graph artifacts live outside the release checkout, set
`SPECTRA_PUBLIC_ROOT=/absolute/path/to/trajectory_bank/public`. The loader still
rejects any public graph containing a label field.

Do not overwrite earlier selector outputs. Hash and submit the complete frozen
selection root to a separately operated evaluator; the local development
boundary is not sufficient evidence for a final sealed result.

## Hidden evaluation

Do not run hidden evaluation inside the optimization process. The sealed
evaluator should run once after the selector and protocol are frozen, ideally
under a separate account or service that exposes neither target labels nor
candidate-level target scores.
