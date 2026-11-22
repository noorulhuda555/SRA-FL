"""Client Guardian Agent — monitors resources and forms training requests."""

from __future__ import annotations

from sra_fl.client.client_state import ClientState
from sra_fl.negotiation.negotiation_protocol import TrainingProposal


class ClientGuardian:
    """Prepare a client-side training request from local resource limits."""

    def __init__(
        self,
        *,
        weak_device_epochs: int = 1,
        strong_device_epochs: int = 2,
        weak_device_compression: int = 2,
        medium_device_compression: int = 1,
        strong_device_compression: int = 0,
        default_batch_size: int = 32,
    ):
        self.weak_device_epochs = weak_device_epochs
        self.strong_device_epochs = strong_device_epochs
        self.weak_device_compression = weak_device_compression
        self.medium_device_compression = medium_device_compression
        self.strong_device_compression = strong_device_compression
        self.default_batch_size = default_batch_size

    def assess(self, state: ClientState) -> dict[str, str | float]:
        return {
            "device_tier": state.device.speed_tier,
            "network_tier": state.network.speed_tier,
            "computation": state.actual_computation,
            "bandwidth_kbps": state.actual_bandwidth_kbps,
            "num_samples": state.num_samples,
        }

    def build_request(self, state: ClientState) -> TrainingProposal:
        device_tier = state.device.speed_tier
        net_tier = state.network.speed_tier

        if device_tier == "slow" or net_tier == "slow":
            epochs = self.weak_device_epochs
            compression = self.weak_device_compression
            note = "limited_resources"
        elif device_tier == "fast" and net_tier == "fast":
            epochs = self.strong_device_epochs
            compression = self.strong_device_compression
            note = "strong_resources"
        else:
            epochs = max(self.weak_device_epochs, self.strong_device_epochs - 1)
            compression = self.medium_device_compression
            note = "moderate_resources"

        return TrainingProposal(
            local_epochs=int(epochs),
            compression_level=int(compression),
            batch_size=self.default_batch_size,
            notes=note,
        )
