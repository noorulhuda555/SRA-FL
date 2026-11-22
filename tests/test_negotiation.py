"""Tests for Module 4 DS-ANE negotiation."""

from __future__ import annotations

import numpy as np

from sra_fl.client.negotiation_handler import NegotiationHandler
from sra_fl.negotiation.quantization import compress_parameters, compression_ratio
from sra_fl.simulation.client_environment import ClientEnvironment


def test_negotiation_agrees_on_config():
    env = ClientEnvironment.create(
        num_clients=4,
        partition_sizes=[150] * 4,
        seed=3,
        dropout_probability=0.0,
    )
    handler = NegotiationHandler(max_rounds=3)
    state = env.get_state(0)
    agreed, transcript = handler.negotiate(state, reputation=1.0)
    assert agreed.local_epochs >= 1
    assert agreed.compression_level >= 0
    assert len(transcript) >= 1


def test_compress_parameters_levels():
    params = [np.ones((10, 10), dtype=np.float32), np.arange(20, dtype=np.float32)]
    full = compress_parameters(params, 0)
    half = compress_parameters(params, 1)
    sparse = compress_parameters(params, 2)
    assert full[0].dtype == np.float32
    assert half[0].shape == params[0].shape
    assert np.count_nonzero(sparse[0]) <= params[0].size
    assert compression_ratio(2) < compression_ratio(0)
