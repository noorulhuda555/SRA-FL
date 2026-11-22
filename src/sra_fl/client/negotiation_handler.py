"""Client-side negotiation handler used by the FL loop."""

from __future__ import annotations

from sra_fl.client.client_state import ClientState
from sra_fl.negotiation.client_agent import ClientNegotiationAgent
from sra_fl.negotiation.negotiation_protocol import NegotiationMessage, TrainingProposal
from sra_fl.negotiation.privacy_manager import PrivacyManager
from sra_fl.negotiation.server_agent import ServerNegotiationAgent


class NegotiationHandler:
    """Run a short client↔server negotiation for one client."""

    def __init__(
        self,
        client_agent: ClientNegotiationAgent | None = None,
        server_agent: ServerNegotiationAgent | None = None,
        *,
        max_rounds: int = 3,
        privacy_manager: PrivacyManager | None = None,
    ):
        self.client_agent = client_agent or ClientNegotiationAgent()
        self.server_agent = server_agent or ServerNegotiationAgent()
        self.max_rounds = max_rounds
        self.privacy_manager = privacy_manager

    def negotiate(
        self,
        state: ClientState,
        *,
        reputation: float = 1.0,
        round_id: int = 0,
    ) -> tuple[TrainingProposal, list[NegotiationMessage]]:
        from sra_fl.negotiation.negotiation_protocol import Decision, NegotiationResult

        transcript: list[NegotiationMessage] = []
        request = self.client_agent.initial_request(state)
        report = state.reported or state.build_honest_report(available=True)

        # Privacy-aware request shaping (Module 7)
        if self.privacy_manager is not None:
            request = self.privacy_manager.adjust_proposal(
                state.client_id,
                request,
                num_samples=state.num_samples,
            )

        server_offer = self.server_agent.propose(report, request, reputation=reputation)
        if self.privacy_manager is not None:
            server_offer = self.privacy_manager.adjust_proposal(
                state.client_id,
                server_offer,
                num_samples=state.num_samples,
            )
        transcript.append(
            NegotiationMessage(
                sender="server",
                decision=Decision.COUNTER,
                proposal=server_offer,
                reason="initial_server_offer",
            )
        )

        agreed: TrainingProposal | None = None
        for _ in range(self.max_rounds):
            client_msg = self.client_agent.respond(state, server_offer)
            if self.privacy_manager is not None:
                client_msg.proposal = self.privacy_manager.adjust_proposal(
                    state.client_id,
                    client_msg.proposal,
                    num_samples=state.num_samples,
                )
            transcript.append(client_msg)
            if client_msg.decision == Decision.ACCEPT:
                agreed = client_msg.proposal
                break

            server_msg = self.server_agent.respond(
                report, client_msg, reputation=reputation
            )
            if self.privacy_manager is not None:
                server_msg.proposal = self.privacy_manager.adjust_proposal(
                    state.client_id,
                    server_msg.proposal,
                    num_samples=state.num_samples,
                )
            transcript.append(server_msg)
            if server_msg.decision == Decision.ACCEPT:
                agreed = server_msg.proposal
                break
            server_offer = server_msg.proposal

        if agreed is None:
            result = NegotiationResult(
                client_id=state.client_id,
                agreed=request,
                transcript=transcript,
                success=False,
            )
            agreed = result.agreed

        if self.privacy_manager is not None:
            self.privacy_manager.consume(
                state.client_id,
                agreed,
                num_samples=state.num_samples,
                round_id=round_id,
            )

        return agreed, transcript
