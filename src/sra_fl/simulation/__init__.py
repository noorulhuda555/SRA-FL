"""Heterogeneous client simulation (Module 2)."""

from .availability_simulator import AvailabilityTrace, load_availability_traces
from .device_simulator import DeviceProfile, load_device_profiles
from .network_simulator import NetworkProfile, load_network_profiles

__all__ = [
    "AvailabilityTrace",
    "DeviceProfile",
    "NetworkProfile",
    "load_availability_traces",
    "load_device_profiles",
    "load_network_profiles",
]


def __getattr__(name: str):
    # Lazy export to avoid circular imports with client_state.
    if name == "ClientEnvironment":
        from .client_environment import ClientEnvironment

        return ClientEnvironment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
