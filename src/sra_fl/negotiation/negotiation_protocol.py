"""Negotiation message types and protocol loop (Module 4)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Decision(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    COUNTER = "counter"


@dataclass
class TrainingProposal:
    local_epochs: int
    compression_level: int
    batch_size: int | None = None
    learning_rate: float | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NegotiationMessage:
    sender: str  # client | server
    decision: Decision
    proposal: TrainingProposal
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sender": self.sender,
            "decision": self.decision.value,
            "proposal": self.proposal.to_dict(),
            "reason": self.reason,
        }


@dataclass
class NegotiationResult:
    client_id: int
    agreed: TrainingProposal
    transcript: list[NegotiationMessage] = field(default_factory=list)
    success: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "agreed": self.agreed.to_dict(),
            "success": self.success,
            "transcript": [m.to_dict() for m in self.transcript],
        }
