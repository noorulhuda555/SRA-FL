"""Groq LLM client for Module 5 agents (Qwen)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

from sra_fl.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = "qwen/qwen3.8-27b"


@dataclass
class LLMUsageRecord:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    model: str = DEFAULT_MODEL
    success: bool = False


@dataclass
class LLMClient:
    """Thin Groq chat wrapper with usage accounting and safe fallback."""

    model: str = DEFAULT_MODEL
    enabled: bool = True
    temperature: float = 0.2
    max_tokens: int = 400
    history: list[LLMUsageRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        load_dotenv()
        self.model = os.getenv("GROQ_MODEL", self.model)
        api_key = os.getenv("GROQ_API_KEY", "")
        self._client = None
        if self.enabled and api_key:
            try:
                from groq import Groq

                self._client = Groq(api_key=api_key)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Groq client init failed: %s", exc)
                self._client = None
        elif self.enabled:
            logger.warning("GROQ_API_KEY missing; LLM calls will use heuristic fallback")

    def complete(self, system: str, user: str) -> str:
        if self._client is None:
            record = LLMUsageRecord(model=self.model, success=False)
            self.history.append(record)
            return ""

        start = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            latency = time.perf_counter() - start
            usage = getattr(response, "usage", None)
            record = LLMUsageRecord(
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                latency_seconds=float(latency),
                model=self.model,
                success=True,
            )
            self.history.append(record)
            content = response.choices[0].message.content or ""
            return content.strip()
        except Exception as exc:  # noqa: BLE001
            latency = time.perf_counter() - start
            self.history.append(
                LLMUsageRecord(
                    latency_seconds=float(latency),
                    model=self.model,
                    success=False,
                )
            )
            logger.warning("LLM call failed: %s", exc)
            return ""

    def usage_summary(self) -> dict[str, Any]:
        if not self.history:
            return {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "mean_latency_seconds": 0.0,
                "success_rate": 0.0,
            }
        prompt = sum(r.prompt_tokens for r in self.history)
        completion = sum(r.completion_tokens for r in self.history)
        latency = [r.latency_seconds for r in self.history]
        success = sum(1 for r in self.history if r.success)
        return {
            "calls": len(self.history),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "mean_latency_seconds": float(sum(latency) / len(latency)),
            "success_rate": success / len(self.history),
            "model": self.model,
        }
