"""Graph spectral tight-frame filters used by SPECTRA-DA."""

from .frame import (
    apply_tight_frame,
    chebyshev_coefficients,
    frame_approximation_diagnostics,
    tight_window_values,
)

__all__ = [
    "apply_tight_frame",
    "chebyshev_coefficients",
    "frame_approximation_diagnostics",
    "tight_window_values",
]
