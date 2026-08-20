from __future__ import annotations

import pytest

from selector.shift_type_diagnostic import SHIFT_FAMILIES, shift_family
from shift_simulator.transforms import default_shift_specs


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("feature_mask_015", "feature"),
        ("feature_noise_030", "feature"),
        ("edge_dropout_035", "structure"),
        ("homophily_lower", "homophily"),
        ("label_prior_majority", "label_prior"),
        ("conditional_structure", "conditional_structure"),
    ],
)
def test_shift_family_uses_frozen_mechanism_groups(name: str, expected: str) -> None:
    assert shift_family(name) == expected


def test_default_shift_bank_matches_the_five_frozen_families() -> None:
    specs = default_shift_specs()
    observed = {spec.family for spec in specs}

    assert observed == set(SHIFT_FAMILIES)
    assert all(shift_family(spec.name) == spec.family for spec in specs)


def test_shift_family_rejects_unregistered_mechanisms() -> None:
    with pytest.raises(ValueError, match="unregistered shift mechanism"):
        shift_family("unknown_shift")
