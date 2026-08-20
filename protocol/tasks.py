"""Authoritative task definitions for GDA-Select."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransferTask:
    source: str
    target: str
    family: str

    @property
    def id(self) -> str:
        return task_id(self.source, self.target)


FAMILY_DOMAINS = {
    "Citation": ("ACMv9", "Citationv1", "DBLPv7"),
    "Airport": ("USA", "BRAZIL", "EUROPE"),
    "Blog": ("Blog1", "Blog2"),
    "Twitch": ("DE", "EN"),
}

DOMAINS = tuple(domain for domains in FAMILY_DOMAINS.values() for domain in domains)


def domain_family(domain: str) -> str:
    for family, domains in FAMILY_DOMAINS.items():
        if domain in domains:
            return family
    raise KeyError(f"unknown GDA-Select domain: {domain}")


def task_id(source: str, target: str) -> str:
    return f"{source}_to_{target}"


TASKS = tuple(
    TransferTask(source, target, family)
    for family, domains in FAMILY_DOMAINS.items()
    for source in domains
    for target in domains
    if source != target
)

# Gate-1 pilot: two reciprocal Citation transfers and two reciprocal Airport
# transfers.  The list is frozen before candidate generation.
PILOT_TASKS = (
    TransferTask("ACMv9", "Citationv1", "Citation"),
    TransferTask("Citationv1", "ACMv9", "Citation"),
    TransferTask("USA", "BRAZIL", "Airport"),
    TransferTask("BRAZIL", "USA", "Airport"),
)
