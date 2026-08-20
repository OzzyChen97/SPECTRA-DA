# GDA-Select sealed-label protocol

This directory is protocol-owned and must not be modified by AutoSOTA.

1. `materialize.py` is a trusted setup command. It creates label-free graphs in
   `trajectory_bank/public/` and writes labels outside the optimization repo to
   `<workspace>/.sealed/spectra_da/`.
2. Candidate exporters call `load_training_task()`. The returned source graph
   has labels and a frozen source train/validation split. The target graph has
   no `y` field.
3. Every permitted source-label read is appended to the external audit log.
4. Only `sealed_eval/` may read target labels, and evaluator calls are audited.
5. AutoSOTA operates on frozen candidate artifacts, never on label files.

The current filesystem owner is trusted infrastructure. This protocol provides
artifact separation and auditable access, not a host-level security boundary
against a root process. Final sealed evaluation should run in a separate
container/account or external evaluator service.
