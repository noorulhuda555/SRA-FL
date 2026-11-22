"""Apply communication decisions and track communication state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sra_fl.communication.reward_function import communication_reward
from sra_fl.communication.trigger_policy import Action, TriggerPolicy, discretize_state
from sra_fl.negotiation.quantization import compress_parameters
from sra_fl.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ClientCommState:
    bandwidth_kbps: float = 1000.0
    last_upload_seconds: float = 0.0
    last_train_loss: float = 1.0
    uploads: int = 0
    skips: int = 0
    waits: int = 0
    compressions: int = 0


@dataclass
class UploadManager:
    """Coordinate upload / wait / compress / skip decisions."""

    policy: TriggerPolicy = field(default_factory=TriggerPolicy)
    states: dict[int, ClientCommState] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    def update_bandwidth(self, client_id: int, bandwidth_kbps: float) -> None:
        st = self.states.setdefault(client_id, ClientCommState())
        st.bandwidth_kbps = float(bandwidth_kbps)

    def decide(
        self,
        client_id: int,
        *,
        train_loss: float,
        bandwidth_kbps: float,
    ) -> Action:
        st = self.states.setdefault(client_id, ClientCommState())
        st.bandwidth_kbps = float(bandwidth_kbps)
        st.last_train_loss = float(train_loss)
        state = discretize_state(
            bandwidth_kbps=bandwidth_kbps,
            train_loss=train_loss,
            recent_delay=st.last_upload_seconds,
        )
        action = self.policy.select_action(state)
        logger.info("Comm decision client %s | state=%s action=%s", client_id, state, action)
        return action

    def apply(
        self,
        client_id: int,
        action: Action,
        parameters: list[np.ndarray],
        *,
        compression_level: int,
        upload_seconds: float,
        accuracy_delta: float = 0.0,
    ) -> tuple[list[np.ndarray] | None, dict[str, Any]]:
        """
        Returns (parameters_or_None if skipped/wait, metrics).
        wait/skip => None (no upload this round).
        compress => stronger compression.
        upload => keep parameters (maybe already negotiated compression).
        """
        st = self.states.setdefault(client_id, ClientCommState())
        state = discretize_state(
            bandwidth_kbps=st.bandwidth_kbps,
            train_loss=st.last_train_loss,
            recent_delay=st.last_upload_seconds,
        )
        out_params: list[np.ndarray] | None
        level = int(compression_level)

        if action in {"wait", "skip"}:
            out_params = None
            if action == "wait":
                st.waits += 1
            else:
                st.skips += 1
        elif action == "compress":
            level = max(level, 2)
            out_params = compress_parameters(parameters, level)
            st.compressions += 1
            st.uploads += 1
            st.last_upload_seconds = float(upload_seconds) * 0.5
        else:
            out_params = parameters
            st.uploads += 1
            st.last_upload_seconds = float(upload_seconds)

        reward = communication_reward(
            action=action,
            accuracy_delta=accuracy_delta,
            upload_seconds=st.last_upload_seconds if out_params is not None else 0.0,
            bandwidth_kbps=st.bandwidth_kbps,
        )
        next_state = discretize_state(
            bandwidth_kbps=st.bandwidth_kbps,
            train_loss=st.last_train_loss,
            recent_delay=st.last_upload_seconds,
        )
        self.policy.update(state, action, reward, next_state)

        metrics = {
            "comm_action": action,
            "comm_reward": float(reward),
            "compression_level_applied": float(level if out_params is not None else -1),
            "uploaded": out_params is not None,
        }
        self.history.append({"client_id": client_id, **metrics, "state": state})
        return out_params, metrics
