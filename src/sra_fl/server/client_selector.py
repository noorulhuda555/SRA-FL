"""Client selection helpers.

Flower strategies already sample clients using ``fraction_fit`` /
``fraction_evaluate``. This module records selection decisions for monitoring
and will be extended in later modules (reputation-aware selection).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ClientSelector:
    """Track which clients were eligible/selected across rounds."""

    history: list[dict[str, object]] = field(default_factory=list)

    def record(
        self,
        server_round: int,
        *,
        phase: str,
        selected_ids: list[int],
        available_ids: list[int] | None = None,
    ) -> None:
        self.history.append(
            {
                "round": server_round,
                "phase": phase,
                "selected_ids": selected_ids,
                "available_ids": available_ids if available_ids is not None else selected_ids,
                "num_selected": len(selected_ids),
            }
        )
