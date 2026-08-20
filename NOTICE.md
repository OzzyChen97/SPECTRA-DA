# Publication notice

This is a sanitized research release of the frozen SPECTRA-DA selector.

The release does not contain target labels, datasets, trajectory banks,
candidate checkpoints, private calibration sidecars, AutoSOTA state, access
tokens, or machine-local paths. Aggregate development metrics are included for
auditability.

`model/ncfm.py`, `utils/loss.py`, and the ADAlign trajectory adapter retain
upstream implementation structure solely to recreate candidate trajectories
under the sealed protocol. The original ADAlign `fit`/`predict` label-reading
paths are not invoked by SPECTRA-DA. Third-party libraries, algorithms, data,
and model weights remain subject to their respective terms.
