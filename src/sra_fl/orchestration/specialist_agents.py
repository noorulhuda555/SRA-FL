"""Specialist agents that summarize client groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sra_fl.orchestration.rag_engine import ClientInfoStore
from sra_fl.utils.llm_client import LLMClient


@dataclass
class GroupSummary:
    group_name: str
    client_ids: list[int]
    summary: str
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_name": self.group_name,
            "client_ids": self.client_ids,
            "summary": self.summary,
            "recommendation": self.recommendation,
        }


class SpecialistAgent:
    """Analyze one client group and produce a compact summary."""

    def __init__(self, group_name: str, llm: LLMClient | None = None):
        self.group_name = group_name
        self.llm = llm or LLMClient(enabled=False)

    def summarize(
        self,
        client_ids: list[int],
        store: ClientInfoStore,
    ) -> GroupSummary:
        records = [store.get(cid) for cid in client_ids]
        heuristic = self._heuristic(records)
        prompt_user = (
            f"Group={self.group_name}\nClients={client_ids}\n"
            f"Records={records}\n"
            "Write 2 short sentences: (1) group status (2) selection advice."
        )
        llm_text = self.llm.complete(
            system=(
                "You are a federated learning specialist agent. "
                "Be concise. Do not invent clients."
            ),
            user=prompt_user,
        )
        summary = llm_text if llm_text else heuristic["summary"]
        recommendation = heuristic["recommendation"]
        if llm_text and "recommend" in llm_text.lower():
            recommendation = llm_text
        return GroupSummary(
            group_name=self.group_name,
            client_ids=client_ids,
            summary=summary,
            recommendation=recommendation,
        )

    def _heuristic(self, records: list[dict[str, Any]]) -> dict[str, str]:
        if not records:
            return {
                "summary": f"Group {self.group_name} is empty.",
                "recommendation": "Skip this group.",
            }
        mean_rep = sum(float(r.get("reputation", 1.0)) for r in records) / len(records)
        online = sum(1 for r in records if r.get("available", True))
        summary = (
            f"Group {self.group_name} has {len(records)} clients "
            f"({online} available), mean reputation {mean_rep:.2f}."
        )
        if "low_reputation" in self.group_name:
            recommendation = "Prefer other groups; audit these clients carefully."
        elif "weak_network" in self.group_name:
            recommendation = "Allow participation with high compression / fewer epochs."
        elif "fast_compute" in self.group_name or "high_reputation" in self.group_name:
            recommendation = "Prefer these clients for training this round."
        else:
            recommendation = "Include a balanced subset if available."
        return {"summary": summary, "recommendation": recommendation}
