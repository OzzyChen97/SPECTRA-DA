# SPECTRA-DA

**Target-label-free model selection for graph domain adaptation via spectral agreement**

SPECTRA-DA addresses a deployment problem that is usually hidden by oracle
evaluation: how to select an adaptation algorithm, hyperparameter setting,
random seed, and checkpoint without reading any target-domain label.

The selector decomposes pairwise prediction disagreement over a tight graph
spectral frame, recovers candidate risks from the resulting pair-sum system,
and transports source-simulated error covariance to correct correlated model
errors. The repository also provides a sealed-label protocol, trajectory
artifact schema, model-selection baselines, shift simulation, and audit tests.

## Final controlled result

| Metric | Value |
|---|---:|
| Mean normalized regret | **0.0141458660** |
| CVaR-20% regret | 0.0479275556 |
| Worst-fold regret | 0.0838775345 |
| Median Kendall tau | 0.9331080050 |
| Top-weighted Kendall | 0.9946191239 |
| Top-5% hit rate | 0.9545454545 |
| Localized-gain share | 0.4465211863 |
| Final runtime | 327.394 s |
| Target-label accesses | **0** |
| Protocol violations | **0** |

Relative to the original mean-regret baseline `0.0227846777`, the final value
is **37.915% lower**. The stricter internal target `0.0134` was not reached.
The controlled eight-iteration refinement preserved the scientific output and
reduced the reproduced runtime from 451.999 s to 320.695 s in the fastest
accepted run; the independent frozen rerun took 327.394 s.

Detailed results and the iteration audit are in [docs/RESULTS.md](docs/RESULTS.md).

## What is included

- `selector/`: SPECTRA-Static, calibrated SPECTRA, label-free baselines, and
  evaluation objectives.
- `spectral_filters/`: tight spectral-frame construction and filtering.
- `covariance_transport/`: descriptor matching and robust covariance-corrected
  risk recovery, including the exact runtime optimizations.
- `shift_simulator/`: source-labelled feature, topology, homophily, and
  label-prior simulations.
- `protocol/` and `sealed_eval/`: auditable separation between public artifacts
  and hidden target labels.
- `scripts/trajectory_export/`: immutable, label-free trajectory schema and
  exporters.
- `tests/`: theory, protocol, schema, and numerical-equivalence tests.
- `results/`: aggregate metrics only; no target labels, checkpoints, datasets,
  or private calibration artifacts.

The public repository intentionally excludes datasets, trajectory banks,
candidate checkpoints, hidden labels, private AutoSOTA state, and machine-local
sidecar paths.

## Installation

Python 3.10 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 \
  -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install -r requirements.txt
```

Use matching PyTorch/PyG wheels for a different CUDA version.

## Quick verification

The core selector and exact workspace-reuse changes can be checked without the
private benchmark artifacts:

```bash
pytest -q \
  tests/selector/test_recovery_workspace_reuse.py \
  tests/selector/test_recovery_structure_cache.py \
  tests/selector/test_spectra_theory_guarantees.py \
  tests/test_spectra_prior.py
```

The frozen release passes all 17 tests in this shard.

## Sealed-label workflow

1. Download the public graph datasets described in [data/README.md](data/README.md).
2. Materialize label-free public graphs and move labels to an external sealed
   directory:

```bash
python -m protocol.materialize \
  --public-root trajectory_bank/public \
  --sealed-root /secure/external/path/spectra_da
```

3. Export immutable candidate trajectories. The ADAlign adapter intentionally
   does not call the upstream `fit` or `predict` paths because those paths read
   target labels:

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/trajectory_export/launch_plan.py \
  --plan trajectory_plans/gda_select_v1.json \
  --output-root trajectory_bank/candidates/gda_select_v1 \
  --log-root results/logs \
  --continue-on-error
```

4. Build source-simulated calibration artifacts:

```bash
CUDA_VISIBLE_DEVICES=7 python shift_simulator/build_calibration_suite.py \
  --candidate-root trajectory_bank/candidates/gda_select_v1 \
  --output-root trajectory_bank/calibration/gda_select_v1 \
  --continue-on-error
```

5. Run the label-free selectors:

```bash
CUDA_VISIBLE_DEVICES=7 python selector/run_spectra_suite.py \
  --candidate-root trajectory_bank/candidates/gda_select_v1 \
  --calibration-root trajectory_bank/calibration/gda_select_v1 \
  --sidecar-manifest configs/spectra_sidecars_16.json \
  --output-root results/gda_select/selections/spectra_frozen_v2 \
  --device cuda:0
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for artifact contracts,
sidecar preparation, and the frozen refinement command.

## Important protocol note

`model/ncfm.py` retains the upstream ADAlign implementation for constructing
the unchanged backbone used by the safe trajectory exporter. Its original
`fit` and `predict` methods read target labels and are **not** part of the
SPECTRA-DA workflow. Do not use them for target-label-free experiments.

The selector code never imports `sealed_eval`, never reads target labels, and
binds every score vector to an immutable candidate-bank hash. A final hidden
evaluation should run in a separate account, container, or external service.

## Citation

```bibtex
@inproceedings{spectrada2026,
  title     = {SPECTRA-DA: Target-Label-Free Model Selection for Graph Domain
               Adaptation via Spectral Agreement},
  booktitle = {International Conference on Learning Representations},
  year      = {2026}
}
```

SPECTRA-DA was developed using ADAlign as one candidate adaptation method.
Please also cite the corresponding adaptation methods and datasets used to
build a trajectory bank.

## Licensing

No dataset, pretrained weight, or third-party candidate checkpoint is
redistributed here. Third-party dependencies and upstream algorithm components
retain their own terms. A project-wide source license has not yet been declared;
contact the maintainers before redistributing modified source files.
