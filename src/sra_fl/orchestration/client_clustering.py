"""Cluster clients into specialist groups."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def cluster_clients(records: list[dict[str, Any]]) -> dict[str, list[int]]:
    """
    Group clients by device/network tier for specialist agents.

    Groups: fast_compute, slow_compute, weak_network, high_reputation, low_reputation.
    A client may appear in multiple groups.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for rec in records:
        cid = int(rec["client_id"])
        device = str(rec.get("device_tier", "medium"))
        network = str(rec.get("network_tier", "medium"))
        reputation = float(rec.get("reputation", 1.0))

        if device == "fast":
            groups["fast_compute"].append(cid)
        elif device == "slow":
            groups["slow_compute"].append(cid)
        else:
            groups["medium_compute"].append(cid)

        if network == "slow":
            groups["weak_network"].append(cid)
        elif network == "fast":
            groups["strong_network"].append(cid)

        if reputation >= 0.8:
            groups["high_reputation"].append(cid)
        elif reputation < 0.5:
            groups["low_reputation"].append(cid)

    return {k: sorted(set(v)) for k, v in groups.items() if v}
