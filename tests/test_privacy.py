"""Tests for Module 7 privacy-aware negotiation."""

from __future__ import annotations

from sra_fl.client.negotiation_handler import NegotiationHandler
from sra_fl.negotiation.negotiation_protocol import TrainingProposal
from sra_fl.negotiation.privacy_manager import PrivacyManager
from sra_fl.simulation.client_environment import ClientEnvironment


def test_privacy_budget_adjust_and_consume():
    pm = PrivacyManager([0, 1], default_epsilon=1.0, base_cost=0.8, seed=0)
    proposal = TrainingProposal(local_epochs=3, compression_level=0, batch_size=32)
    adjusted = pm.adjust_proposal(0, proposal, num_samples=5000)
    assert adjusted.local_epochs <= 3
    spent = pm.consume(0, adjusted, num_samples=5000, round_id=1)
    assert spent > 0
    assert pm.accounts[0].remaining < pm.accounts[0].epsilon_total


def test_negotiation_with_privacy():
    env = ClientEnvironment.create(
        num_clients=3,
        partition_sizes=[200] * 3,
        seed=4,
        dropout_probability=0.0,
    )
    pm = PrivacyManager([0, 1, 2], default_epsilon=2.0, base_cost=0.5, seed=4)
    handler = NegotiationHandler(privacy_manager=pm, max_rounds=2)
    agreed, transcript = handler.negotiate(env.get_state(0), reputation=1.0, round_id=1)
    assert agreed.local_epochs >= 1
    assert len(transcript) >= 1
    assert pm.accounts[0].epsilon_spent > 0
