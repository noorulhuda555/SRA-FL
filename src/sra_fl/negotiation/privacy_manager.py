"""Privacy budget tracking and constraints (Module 7)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sra_fl.negotiation.negotiation_protocol import TrainingProposal
from sra_fl.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PrivacyAccount:
    client_id: int
    epsilon_total: float
    epsilon_spent: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def remaining(self) -> float:
        return max(0.0, self.epsilon_total - self.epsilon_spent)

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "epsilon_total": self.epsilon_total,
            "epsilon_spent": self.epsilon_spent,
            "remaining": self.remaining,
            "history": self.history,
        }


class PrivacyManager:
    """
    Assign and consume per-client privacy budgets.

    Approximate cost model: epsilon_cost ~= base * local_epochs * sqrt(num_samples / batch).
    """

    def __init__(
        self,
        client_ids: list[int],
        *,
        default_epsilon: float = 8.0,
        base_cost: float = 0.35,
        seed: int = 42,
    ):
        import numpy as np

        rng = np.random.default_rng(seed)
        self.base_cost = float(base_cost)
        self.accounts: dict[int, PrivacyAccount] = {}
        for cid in client_ids:
            # Mild heterogeneity in privacy budgets
            eps = float(default_epsilon * rng.uniform(0.75, 1.25))
            self.accounts[int(cid)] = PrivacyAccount(client_id=int(cid), epsilon_total=eps)

    def remaining_map(self) -> dict[int, float]:
        return {cid: acc.remaining for cid, acc in self.accounts.items()}

    def estimate_cost(
        self,
        *,
        local_epochs: int,
        num_samples: int,
        batch_size: int,
    ) -> float:
        import math

        samples_factor = math.sqrt(max(num_samples, 1) / max(batch_size, 1))
        return float(self.base_cost * max(local_epochs, 1) * max(samples_factor / 10.0, 0.2))

    def can_afford(self, client_id: int, proposal: TrainingProposal, *, num_samples: int) -> bool:
        cost = self.estimate_cost(
            local_epochs=int(proposal.local_epochs),
            num_samples=num_samples,
            batch_size=int(proposal.batch_size or 32),
        )
        return self.accounts[int(client_id)].remaining >= cost

    def adjust_proposal(
        self,
        client_id: int,
        proposal: TrainingProposal,
        *,
        num_samples: int,
        min_epochs: int = 1,
    ) -> TrainingProposal:
        """Reduce epochs until the proposal fits the remaining privacy budget."""
        adjusted = TrainingProposal(
            local_epochs=int(proposal.local_epochs),
            compression_level=int(proposal.compression_level),
            batch_size=proposal.batch_size,
            learning_rate=proposal.learning_rate,
            notes=proposal.notes,
        )
        while adjusted.local_epochs > min_epochs and not self.can_afford(
            client_id, adjusted, num_samples=num_samples
        ):
            adjusted.local_epochs -= 1
            adjusted.compression_level = max(adjusted.compression_level, 1)
            adjusted.notes = "privacy_adjusted"

        if not self.can_afford(client_id, adjusted, num_samples=num_samples):
            adjusted.local_epochs = min_epochs
            adjusted.compression_level = max(adjusted.compression_level, 2)
            adjusted.notes = "privacy_min_config"
        return adjusted

    def consume(
        self,
        client_id: int,
        proposal: TrainingProposal,
        *,
        num_samples: int,
        round_id: int,
    ) -> float:
        cost = self.estimate_cost(
            local_epochs=int(proposal.local_epochs),
            num_samples=num_samples,
            batch_size=int(proposal.batch_size or 32),
        )
        acc = self.accounts[int(client_id)]
        spent = min(cost, acc.remaining)
        acc.epsilon_spent += spent
        acc.history.append(
            {
                "round": round_id,
                "cost": spent,
                "local_epochs": proposal.local_epochs,
                "remaining": acc.remaining,
            }
        )
        logger.info(
            "Privacy client %s spent=%.3f remaining=%.3f",
            client_id,
            spent,
            acc.remaining,
        )
        return spent

    def summary(self) -> dict[str, Any]:
        return {
            "clients": {str(cid): acc.to_dict() for cid, acc in self.accounts.items()},
            "mean_remaining": float(
                sum(a.remaining for a in self.accounts.values()) / max(len(self.accounts), 1)
            ),
        }
