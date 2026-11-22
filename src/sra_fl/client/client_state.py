"""Client state with actual vs reported resources (Module 2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sra_fl.simulation.availability_simulator import AvailabilityTrace
from sra_fl.simulation.device_simulator import DeviceProfile
from sra_fl.simulation.network_simulator import NetworkProfile


@dataclass
class ResourceReport:
    """Resource values a client claims to the server."""

    computation: float
    communication: float
    bandwidth_kbps: float
    available: bool
    speed_tier: str
    network_tier: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "computation": self.computation,
            "communication": self.communication,
            "bandwidth_kbps": self.bandwidth_kbps,
            "available": self.available,
            "speed_tier": self.speed_tier,
            "network_tier": self.network_tier,
        }


@dataclass
class ClientState:
    """
    Full per-client simulation state.

    ``actual_*`` fields are ground truth used by simulators.
    ``reported`` is what the client tells the server (equals actual in Module 2;
    Module 3 may inject lies).
    """

    client_id: int
    num_samples: int
    device: DeviceProfile
    network: NetworkProfile
    availability: AvailabilityTrace
    time_offset: int = 0
    reported: ResourceReport | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reported is None:
            self.reported = self.build_honest_report(available=True)

    def build_honest_report(self, *, available: bool) -> ResourceReport:
        return ResourceReport(
            computation=float(self.device.computation),
            communication=float(self.device.communication),
            bandwidth_kbps=float(self.network.bandwidth_kbps),
            available=bool(available),
            speed_tier=self.device.speed_tier,
            network_tier=self.network.speed_tier,
        )

    def sync_reported_availability(self, available: bool) -> None:
        """Update the reported availability bit (honest by default)."""
        self.reported = self.build_honest_report(available=available)

    @property
    def actual_computation(self) -> float:
        return float(self.device.computation)

    @property
    def actual_bandwidth_kbps(self) -> float:
        return float(self.network.bandwidth_kbps)

    def is_available(self, timestamp: float) -> bool:
        return self.availability.is_available_at(float(timestamp) + float(self.time_offset))

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "num_samples": self.num_samples,
            "actual": {
                "computation": self.device.computation,
                "communication": self.device.communication,
                "bandwidth_kbps": self.network.bandwidth_kbps,
                "device_tier": self.device.speed_tier,
                "network_tier": self.network.speed_tier,
                "device_model": self.availability.model,
                "availability_ratio": self.availability.availability_ratio(),
                "time_offset": self.time_offset,
            },
            "reported": None if self.reported is None else self.reported.to_dict(),
            **self.metadata,
        }
