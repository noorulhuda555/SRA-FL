"""Tests for Module 5 hierarchical RAG orchestration."""

from __future__ import annotations

from sra_fl.orchestration.client_clustering import cluster_clients
from sra_fl.orchestration.global_orchestrator import GlobalOrchestrator
from sra_fl.orchestration.rag_engine import RAGEngine
from sra_fl.simulation.client_environment import ClientEnvironment
from sra_fl.utils.llm_client import LLMClient


def test_rag_index_and_retrieve():
    rag = RAGEngine()
    rag.store.upsert(0, {"device_tier": "fast", "network_tier": "slow", "reputation": 0.9})
    rag.store.upsert(1, {"device_tier": "slow", "network_tier": "fast", "reputation": 0.4})
    rag.reindex()
    hits = rag.retrieve("fast compute high reputation", top_k=2)
    assert len(hits) >= 1
    assert "client_id" in hits[0]


def test_cluster_and_orchestrator_without_llm():
    env = ClientEnvironment.create(
        num_clients=6,
        partition_sizes=[100] * 6,
        seed=2,
        dropout_probability=0.0,
    )
    groups = cluster_clients(
        [
            {
                "client_id": cid,
                "device_tier": state.device.speed_tier,
                "network_tier": state.network.speed_tier,
                "reputation": 1.0,
            }
            for cid, state in env.states.items()
        ]
    )
    assert groups

    orch = GlobalOrchestrator(llm=LLMClient(enabled=False), enable_llm=False)
    orch.sync_from_environment(env.states, available_ids=list(range(6)))
    decision = orch.decide(available_ids=list(range(6)), target_count=3)
    assert len(decision.recommended_client_ids) == 3
    assert decision.explanations
