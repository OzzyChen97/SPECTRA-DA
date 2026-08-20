from __future__ import annotations

import numpy as np

from selector.spectra_cal import collapse_to_global


def test_global_collapse_preserves_total_tight_frame_statistics() -> None:
    generator = np.random.default_rng(17)
    shifts = 5
    models = 4
    bands = 3
    band_risks = generator.uniform(0.0, 1.0, size=(shifts, models, bands))
    band_covariances = generator.normal(
        0.0,
        0.1,
        size=(shifts, bands, models, models),
    )
    disagreements = generator.uniform(0.0, 1.0, size=(bands, models, models))
    arrays = {
        "shift_deltas": generator.normal(size=(shifts, 6)),
        "target_delta": generator.normal(size=6),
        "band_risks": band_risks,
        "band_covariances": band_covariances,
    }

    collapsed, global_disagreements = collapse_to_global(arrays, disagreements)

    assert collapsed["band_risks"].shape == (shifts, models, 1)
    assert collapsed["band_covariances"].shape == (shifts, 1, models, models)
    assert global_disagreements.shape == (1, models, models)
    np.testing.assert_allclose(collapsed["band_risks"][:, :, 0], band_risks.sum(axis=2))
    np.testing.assert_allclose(
        collapsed["band_covariances"][:, 0],
        band_covariances.sum(axis=1),
    )
    np.testing.assert_allclose(global_disagreements[0], disagreements.sum(axis=0))
    np.testing.assert_array_equal(collapsed["shift_deltas"], arrays["shift_deltas"])
    np.testing.assert_array_equal(collapsed["target_delta"], arrays["target_delta"])


def test_global_collapse_rejects_misaligned_band_shapes() -> None:
    arrays = {
        "band_risks": np.zeros((2, 3, 2)),
        "band_covariances": np.zeros((2, 3, 3, 3)),
    }
    disagreements = np.zeros((2, 3, 3))

    try:
        collapse_to_global(arrays, disagreements)
    except ValueError as error:
        assert "covariance" in str(error)
    else:
        raise AssertionError("misaligned band covariances must be rejected")
