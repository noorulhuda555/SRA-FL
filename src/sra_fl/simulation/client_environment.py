"""Combined heterogeneous client environment for Module 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sra_fl.client.client_state import ClientState
from sra_fl.simulation.availability_simulator import (
    assign_availability_traces,
    round_timestamp,
)
from sra_fl.simulation.device_simulator import (
    assign_device_profiles,
    estimate_training_seconds,
)
from sra_fl.simulation.energy_simulator import estimate_energy_joules
from sra_fl.simulation.network_simulator import (
    assign_network_profiles,
    estimate_model_nbytes,
    estimate_transfer_seconds,
)
from sra_fl.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RoundParticipation:
    """Result of selecting clients under availability constraints."""

    available_ids: list[int]
    selected_ids: list[int]
    unavailable_ids: list[int]


class ClientEnvironment:
    """Owns per-client device, network, and availability state."""

    def __init__(
        self,
        states: list[ClientState],
        *,
        seconds_per_round: float = 3600.0,
        dropout_probability: float = 0.05,
        seed: int = 42,
        lying_injector: Any | None = None,
    ):
        self.states = {state.client_id: state for state in states}
        self.seconds_per_round = float(seconds_per_round)
        self.dropout_probability = float(dropout_probability)
        self.rng = np.random.default_rng(seed)
        self.lying_injector = lying_injector

    @classmethod
    def create(
        cls,
        *,
        num_clients: int,
        partition_sizes: list[int],
        seed: int = 42,
        seconds_per_round: float = 3600.0,
        dropout_probability: float = 0.05,
    ) -> ClientEnvironment:
        if len(partition_sizes) != num_clients:
            raise ValueError("partition_sizes length must equal num_clients")

        devices = assign_device_profiles(num_clients, seed=seed)
        networks = assign_network_profiles(num_clients, seed=seed)
        availabilities = assign_availability_traces(num_clients, seed=seed)
        offset_rng = np.random.default_rng(seed + 101)

        states = []
        for i in range(num_clients):
            finish = max(int(availabilities[i].finish_time), 1)
            offset = int(offset_rng.integers(0, finish))
            states.append(
                ClientState(
                    client_id=i,
                    num_samples=int(partition_sizes[i]),
                    device=devices[i],
                    network=networks[i],
                    availability=availabilities[i],
                    time_offset=offset,
                )
            )
        env = cls(
            states,
            seconds_per_round=seconds_per_round,
            dropout_probability=dropout_probability,
            seed=seed,
        )
        env.log_summary()
        return env

    def log_summary(self) -> None:
        tiers = {"fast": 0, "medium": 0, "slow": 0}
        net_tiers = {"fast": 0, "medium": 0, "slow": 0}
        for state in self.states.values():
            tiers[state.device.speed_tier] = tiers.get(state.device.speed_tier, 0) + 1
            net_tiers[state.network.speed_tier] = (
                net_tiers.get(state.network.speed_tier, 0) + 1
            )
        logger.info("Device tiers: %s", tiers)
        logger.info("Network tiers: %s", net_tiers)

    def get_state(self, client_id: int) -> ClientState:
        return self.states[client_id]

    def _refresh_report(self, state: ClientState, actually_available: bool) -> None:
        if self.lying_injector is not None:
            self.lying_injector.apply_to_state(state, actually_available=actually_available)
        else:
            state.sync_reported_availability(actually_available)

    def available_clients(self, server_round: int) -> list[int]:
        ts = round_timestamp(server_round, self.seconds_per_round)
        available: list[int] = []
        for client_id, state in self.states.items():
            is_up = state.is_available(ts)
            self._refresh_report(state, is_up)
            # Selection uses ground-truth availability; fake reports are audited later.
            if is_up:
                available.append(client_id)
        return sorted(available)

    def select_participants(
        self,
        server_round: int,
        *,
        fraction: float,
        min_clients: int,
        reputation_scores: dict[int, float] | None = None,
    ) -> RoundParticipation:
        available_ids = self.available_clients(server_round)
        unavailable_ids = sorted(set(self.states) - set(available_ids))

        # Fallback: if the virtual clock hits an all-offline slice, pick the
        # most frequently available clients so training can continue.
        if len(available_ids) < min_clients:
            ranked = sorted(
                self.states.values(),
                key=lambda s: s.availability.availability_ratio(),
                reverse=True,
            )
            needed = min_clients - len(available_ids)
            for state in ranked:
                if state.client_id in available_ids:
                    continue
                available_ids.append(state.client_id)
                self._refresh_report(state, True)
                needed -= 1
                if needed <= 0:
                    break
            available_ids = sorted(available_ids)
            unavailable_ids = sorted(set(self.states) - set(available_ids))

        if not available_ids:
            return RoundParticipation([], [], unavailable_ids)

        sample_size = max(min_clients, int(np.ceil(fraction * len(self.states))))
        sample_size = min(sample_size, len(available_ids))

        if reputation_scores:
            weights = np.array(
                [max(reputation_scores.get(cid, 0.05), 0.05) for cid in available_ids],
                dtype=np.float64,
            )
            weights = weights / weights.sum()
            selected = self.rng.choice(
                available_ids, size=sample_size, replace=False, p=weights
            )
        else:
            selected = self.rng.choice(available_ids, size=sample_size, replace=False)
        selected_ids = sorted(int(x) for x in selected)
        return RoundParticipation(available_ids, selected_ids, unavailable_ids)

    def should_dropout(self, client_id: int, server_round: int) -> bool:
        """Simulate mid-round disconnect using availability + random dropout."""
        state = self.states[client_id]
        # Check availability near the end of the round window.
        end_ts = round_timestamp(server_round, self.seconds_per_round) + (
            self.seconds_per_round * 0.75
        )
        if not state.is_available(end_ts):
            return True
        return bool(self.rng.random() < self.dropout_probability)

    def simulate_round_costs(
        self,
        client_id: int,
        *,
        local_epochs: int,
        parameters: list[np.ndarray],
    ) -> dict[str, float]:
        """Estimate train/upload/download time and energy for one client round."""
        state = self.states[client_id]
        model_bytes = estimate_model_nbytes(parameters)
        # Prefer MobiPerf bandwidth for transfers; fall back to device communication.
        bandwidth = max(state.network.bandwidth_kbps, state.device.communication, 1.0)
        train_s = estimate_training_seconds(
            num_samples=state.num_samples,
            local_epochs=local_epochs,
            computation=state.device.computation,
        )
        upload_s = estimate_transfer_seconds(
            num_bytes=model_bytes, bandwidth_kbps=bandwidth
        )
        download_s = estimate_transfer_seconds(
            num_bytes=model_bytes, bandwidth_kbps=bandwidth
        )
        energy = estimate_energy_joules(
            train_seconds=train_s,
            upload_seconds=upload_s,
            download_seconds=download_s,
        )
        return {
            "train_seconds": float(train_s),
            "upload_seconds": float(upload_s),
            "download_seconds": float(download_s),
            "comm_seconds": float(upload_s + download_s),
            "energy_joules": float(energy),
            "model_bytes": float(model_bytes),
            "bandwidth_kbps": float(bandwidth),
            "computation": float(state.device.computation),
        }

    def states_as_dicts(self) -> list[dict[str, Any]]:
        return [state.to_dict() for state in self.states.values()]
