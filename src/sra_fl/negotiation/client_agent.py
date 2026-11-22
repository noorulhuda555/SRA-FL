"""Client negotiation agent wrapping the Guardian."""

from __future__ import annotations

from sra_fl.client.client_guardian import ClientGuardian
from sra_fl.client.client_state import ClientState
from sra_fl.negotiation.negotiation_protocol import (
    Decision,
    NegotiationMessage,
    TrainingProposal,
)


class ClientNegotiationAgent:
    """Client-side negotiator that can accept or counter server offers."""

    def __init__(self, guardian: ClientGuardian | None = None):
        self.guardian = guardian or ClientGuardian()

    def initial_request(self, state: ClientState) -> TrainingProposal:
        return self.guardian.build_request(state)

    def respond(
        self,
        state: ClientState,
        server_proposal: TrainingProposal,
    ) -> NegotiationMessage:
        desired = self.initial_request(state)
        # Accept if server is not asking for heavier work than the client can handle.
        if (
            server_proposal.local_epochs <= desired.local_epochs + 1
            and server_proposal.compression_level >= desired.compression_level - 1
        ):
            # Prefer client's safer compression if network is weak.
            agreed = TrainingProposal(
                local_epochs=min(server_proposal.local_epochs, max(desired.local_epochs, 1)),
                compression_level=max(
                    server_proposal.compression_level, desired.compression_level
                ),
                batch_size=server_proposal.batch_size or desired.batch_size,
                learning_rate=server_proposal.learning_rate,
                notes="client_accepted",
            )
            return NegotiationMessage(
                sender="client",
                decision=Decision.ACCEPT,
                proposal=agreed,
                reason="within_client_limits",
            )

        counter = TrainingProposal(
            local_epochs=desired.local_epochs,
            compression_level=desired.compression_level,
            batch_size=desired.batch_size,
            learning_rate=server_proposal.learning_rate,
            notes="client_counter",
        )
        return NegotiationMessage(
            sender="client",
            decision=Decision.COUNTER,
            proposal=counter,
            reason="server_request_too_demanding",
        )
