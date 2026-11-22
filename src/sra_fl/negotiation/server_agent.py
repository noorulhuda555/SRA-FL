"""Server negotiation agent."""

from __future__ import annotations

from sra_fl.client.client_state import ClientState, ResourceReport
from sra_fl.negotiation.negotiation_protocol import (
    Decision,
    NegotiationMessage,
    TrainingProposal,
)


class ServerNegotiationAgent:
    """Propose training conditions from reported resources and reputation."""

    def __init__(
        self,
        *,
        server_preferred_epochs: int = 2,
        min_local_epochs: int = 1,
        max_local_epochs: int = 3,
        default_batch_size: int = 32,
        default_learning_rate: float = 0.01,
    ):
        self.server_preferred_epochs = server_preferred_epochs
        self.min_local_epochs = min_local_epochs
        self.max_local_epochs = max_local_epochs
        self.default_batch_size = default_batch_size
        self.default_learning_rate = default_learning_rate

    def propose(
        self,
        report: ResourceReport,
        client_request: TrainingProposal,
        *,
        reputation: float = 1.0,
    ) -> TrainingProposal:
        # Low reputation => ask for less aggressive participation / more compression.
        if reputation < 0.4:
            epochs = self.min_local_epochs
            compression = max(client_request.compression_level, 2)
        elif report.speed_tier == "slow" or report.network_tier == "slow":
            epochs = min(client_request.local_epochs, self.min_local_epochs + 0)
            epochs = max(self.min_local_epochs, epochs)
            compression = max(client_request.compression_level, 1)
        else:
            epochs = int(
                round(
                    (client_request.local_epochs + self.server_preferred_epochs) / 2
                )
            )
            epochs = int(np_clip(epochs, self.min_local_epochs, self.max_local_epochs))
            compression = client_request.compression_level

        return TrainingProposal(
            local_epochs=epochs,
            compression_level=int(compression),
            batch_size=client_request.batch_size or self.default_batch_size,
            learning_rate=self.default_learning_rate,
            notes="server_proposal",
        )

    def respond(
        self,
        report: ResourceReport,
        client_message: NegotiationMessage,
        *,
        reputation: float = 1.0,
    ) -> NegotiationMessage:
        if client_message.decision == Decision.ACCEPT:
            return NegotiationMessage(
                sender="server",
                decision=Decision.ACCEPT,
                proposal=client_message.proposal,
                reason="confirm_accept",
            )

        # Client countered — meet halfway once.
        client_prop = client_message.proposal
        server_prop = self.propose(report, client_prop, reputation=reputation)
        midway = TrainingProposal(
            local_epochs=max(
                self.min_local_epochs,
                int(round((client_prop.local_epochs + server_prop.local_epochs) / 2)),
            ),
            compression_level=max(client_prop.compression_level, server_prop.compression_level),
            batch_size=client_prop.batch_size or self.default_batch_size,
            learning_rate=self.default_learning_rate,
            notes="server_compromise",
        )
        return NegotiationMessage(
            sender="server",
            decision=Decision.ACCEPT,
            proposal=midway,
            reason="compromise",
        )


def np_clip(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def propose_for_state(
    agent: ServerNegotiationAgent,
    state: ClientState,
    client_request: TrainingProposal,
    reputation: float,
) -> TrainingProposal:
    report = state.reported or state.build_honest_report(available=True)
    return agent.propose(report, client_request, reputation=reputation)
