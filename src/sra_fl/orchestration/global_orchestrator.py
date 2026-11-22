"""Central orchestrator combining specialist summaries and RAG retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sra_fl.orchestration.client_clustering import cluster_clients
from sra_fl.orchestration.rag_engine import ClientInfoStore, RAGEngine
from sra_fl.orchestration.specialist_agents import GroupSummary, SpecialistAgent
from sra_fl.utils.llm_client import LLMClient
from sra_fl.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OrchestratorDecision:
    recommended_client_ids: list[int]
    explanations: list[str]
    group_summaries: list[GroupSummary] = field(default_factory=list)
    retrieved_context: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommended_client_ids": self.recommended_client_ids,
            "explanations": self.explanations,
            "group_summaries": [g.to_dict() for g in self.group_summaries],
            "retrieved_context": self.retrieved_context,
        }


class GlobalOrchestrator:
    """Hierarchical / flat RAG orchestrator for client-selection recommendations."""

    def __init__(
        self,
        *,
        llm: LLMClient | None = None,
        enable_llm: bool = True,
        mode: str = "hierarchical",
    ):
        self.llm = llm or LLMClient(enabled=enable_llm)
        self.rag = RAGEngine()
        self.store: ClientInfoStore = self.rag.store
        self.last_decision: OrchestratorDecision | None = None
        mode_norm = str(mode).lower()
        if mode_norm not in {"hierarchical", "flat", "none"}:
            raise ValueError(f"Unsupported RAG mode: {mode}")
        self.mode = mode_norm

    def sync_from_environment(
        self,
        states: dict[int, Any],
        *,
        reputations: dict[int, float] | None = None,
        privacy_remaining: dict[int, float] | None = None,
        available_ids: list[int] | None = None,
    ) -> None:
        available = set(available_ids or list(states.keys()))
        reputations = reputations or {}
        privacy_remaining = privacy_remaining or {}
        for cid, state in states.items():
            self.store.upsert(
                cid,
                {
                    "device_tier": state.device.speed_tier,
                    "network_tier": state.network.speed_tier,
                    "computation": state.actual_computation,
                    "bandwidth_kbps": state.actual_bandwidth_kbps,
                    "reputation": float(reputations.get(cid, 1.0)),
                    "available": cid in available,
                    "privacy_remaining": float(privacy_remaining.get(cid, 1.0)),
                    "num_samples": state.num_samples,
                    "history": state.metadata.get("lie_type") or "honest",
                },
            )
        self.rag.reindex()

    def decide(
        self,
        *,
        available_ids: list[int],
        target_count: int,
        query: str = "select reliable available clients for this training round",
    ) -> OrchestratorDecision:
        records = [self.store.get(cid) for cid in available_ids]
        summaries: list[GroupSummary] = []
        groups: dict[str, list[int]] = {}
        retrieved: list[dict[str, Any]] = []

        if self.mode == "hierarchical":
            groups = cluster_clients(records)
            for group_name, members in groups.items():
                agent = SpecialistAgent(group_name, llm=self.llm)
                summaries.append(agent.summarize(members, self.store))
            retrieved = self.rag.retrieve(query, top_k=min(5, max(1, len(available_ids))))
        elif self.mode == "flat":
            retrieved = self.rag.retrieve(query, top_k=min(8, max(1, len(available_ids))))
        # mode == "none": no retrieval / specialist summaries; score from raw store only

        # Score available clients from reputation + specialist preference hints.
        scores: dict[int, float] = {cid: 0.0 for cid in available_ids}
        for cid in available_ids:
            rec = self.store.get(cid)
            scores[cid] += float(rec.get("reputation", 1.0))
            if rec.get("device_tier") == "fast":
                scores[cid] += 0.2
            if rec.get("network_tier") == "slow":
                scores[cid] -= 0.1
            if float(rec.get("privacy_remaining", 1.0)) <= 0:
                scores[cid] -= 10.0

        for hit in retrieved:
            cid = hit.get("client_id")
            if cid in scores:
                scores[int(cid)] += 0.15 * float(hit.get("score", 0.0))

        for summary in summaries:
            boost = 0.3 if "Prefer" in summary.recommendation or "prefer" in summary.recommendation else 0.0
            penalty = 0.3 if "Prefer other" in summary.recommendation else 0.0
            for cid in summary.client_ids:
                if cid in scores:
                    scores[cid] += boost - penalty

        ranked = sorted(available_ids, key=lambda c: scores.get(c, 0.0), reverse=True)
        recommended = ranked[: max(1, min(target_count, len(ranked)))]

        if self.mode == "hierarchical":
            central_prompt = (
                f"Available={available_ids}\n"
                f"Group summaries={[s.to_dict() for s in summaries]}\n"
                f"Retrieved={retrieved}\n"
                f"Proposed selection={recommended}\n"
                "Explain in 2 short sentences why this selection is reasonable."
            )
        elif self.mode == "flat":
            central_prompt = (
                f"Available={available_ids}\n"
                f"Retrieved={retrieved}\n"
                f"Proposed selection={recommended}\n"
                "Explain in 2 short sentences why this selection is reasonable."
            )
        else:
            central_prompt = (
                f"Available={available_ids}\n"
                f"Proposed selection={recommended}\n"
                "Explain in 2 short sentences why this selection is reasonable."
            )
        explanation = self.llm.complete(
            system="You are the central federated learning orchestrator.",
            user=central_prompt,
        )
        explanations = [
            explanation
            if explanation
            else (
                f"Selected clients {recommended} using reputation, device/network tiers, "
                f"and {self.mode} orchestration"
                + (f" from groups {list(groups.keys())}" if groups else "")
                + "."
            )
        ]

        decision = OrchestratorDecision(
            recommended_client_ids=recommended,
            explanations=explanations,
            group_summaries=summaries,
            retrieved_context=retrieved,
        )
        self.last_decision = decision
        logger.info(
            "Orchestrator recommended %s | mode=%s groups=%s",
            recommended,
            self.mode,
            list(groups.keys()),
        )
        return decision
