"""Simple client energy estimation from compute and communication time."""

from __future__ import annotations


def estimate_energy_joules(
    *,
    train_seconds: float,
    upload_seconds: float,
    download_seconds: float,
    compute_watts: float = 2.5,
    radio_watts: float = 1.2,
) -> float:
    """
    Estimate energy use with constant-power approximations.

    This is a lightweight Module-2 placeholder; Module 8 can refine it.
    """
    compute_energy = max(train_seconds, 0.0) * compute_watts
    radio_energy = (max(upload_seconds, 0.0) + max(download_seconds, 0.0)) * radio_watts
    return float(compute_energy + radio_energy)
