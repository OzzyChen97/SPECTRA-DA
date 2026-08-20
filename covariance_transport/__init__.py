"""Shift-conditioned covariance transport for SPECTRA-DA."""

from .transport import (
    corrected_band_risk_recovery,
    match_shift_convex_combination,
    transport_statistics,
)

__all__ = [
    "corrected_band_risk_recovery",
    "match_shift_convex_combination",
    "transport_statistics",
]
