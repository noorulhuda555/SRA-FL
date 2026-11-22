"""DS-ANE negotiation package (Module 4)."""

from .negotiation_protocol import NegotiationResult, TrainingProposal
from .quantization import compress_parameters

__all__ = [
    "ClientNegotiationAgent",
    "NegotiationResult",
    "ServerNegotiationAgent",
    "TrainingProposal",
    "compress_parameters",
]


def __getattr__(name: str):
    # Lazy exports to avoid circular imports with client_guardian.
    if name == "ClientNegotiationAgent":
        from .client_agent import ClientNegotiationAgent

        return ClientNegotiationAgent
    if name == "ServerNegotiationAgent":
        from .server_agent import ServerNegotiationAgent

        return ServerNegotiationAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
