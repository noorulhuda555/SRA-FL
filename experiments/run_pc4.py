"""
PC-4 (Core i5 CPU) — Hierarchical RAG + privacy-aware negotiation (experiments_2.txt).

Runs:
  RAG offline probe (flat vs hierarchical retrieval quality)
  RAG FL: no / flat / hierarchical
  Privacy: resource-only vs privacy-aware (+ epsilon levels)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments_2.common import off_modules, out_dir, run_one
from sra_fl.orchestration.rag_engine import RAGEngine

PC = "pc4"


def run_rag_probe(*, dry_run: bool = False, num_clients: int = 50) -> None:
    dest = out_dir(PC, "R0_retrieval_probe")
    if dry_run:
        print(f"[dry-run] {PC}/R0_retrieval_probe")
        return

    engine = RAGEngine()
    for cid in range(num_clients):
        engine.store.upsert(
            cid,
            {
                "device_tier": "slow" if cid % 3 == 0 else "medium" if cid % 3 == 1 else "fast",
                "network_tier": "slow" if cid % 4 == 0 else "fast",
                "computation": 10 + cid,
                "bandwidth_kbps": 50 if cid % 4 == 0 else 2000,
                "reputation": 0.2 if cid % 5 == 0 else 1.0,
                "available": cid % 3 != 0,
                "privacy_remaining": 2.0 if cid % 3 == 0 else 8.0,
                "history": "dropout" if cid % 6 == 0 else "honest",
            },
        )
    engine.reindex()

    queries = [
        ("low bandwidth", lambda r: r.get("network_tier") == "slow"),
        ("repeatedly dropped out", lambda r: r.get("history") == "dropout"),
        ("low reputation", lambda r: float(r.get("reputation", 1.0)) < 0.5),
        ("can participate next round", lambda r: bool(r.get("available"))),
        ("strict privacy remaining", lambda r: float(r.get("privacy_remaining", 9)) < 3),
    ]
    # Expand to ~20 query variants for experiments_2.txt
    extra = [
        ("slow compute clients", lambda r: r.get("device_tier") == "slow"),
        ("fast network clients", lambda r: r.get("network_tier") == "fast"),
        ("high reputation", lambda r: float(r.get("reputation", 0)) >= 0.9),
        ("unavailable clients", lambda r: not bool(r.get("available"))),
        ("medium device tier", lambda r: r.get("device_tier") == "medium"),
    ]
    queries = (queries + extra) * 2  # 20 queries

    rows = []
    for query, pred in queries:
        relevant = {r["client_id"] for r in engine.store.all_records() if pred(r)}
        t0 = time.perf_counter()
        hits = engine.retrieve(query, top_k=5)
        latency = time.perf_counter() - t0
        retrieved = [int(h["client_id"]) for h in hits if "client_id" in h]
        tp = sum(1 for cid in retrieved if cid in relevant)
        precision = tp / len(retrieved) if retrieved else 0.0
        recall = tp / len(relevant) if relevant else 0.0
        rows.append(
            {
                "query": query,
                "precision_at_5": precision,
                "recall_at_5": recall,
                "latency_seconds": latency,
                "num_relevant": len(relevant),
            }
        )

    with (dest / "retrieval_quality.json").open("w", encoding="utf-8") as handle:
        json.dump({"k": 5, "num_clients": num_clients, "queries": rows}, handle, indent=2)
    with (dest / "retrieval_quality.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "experiment_id": "R0_retrieval_probe",
        "pc": PC,
        "mean_precision_at_5": sum(r["precision_at_5"] for r in rows) / len(rows),
        "mean_latency_seconds": sum(r["latency_seconds"] for r in rows) / len(rows),
        "num_queries": len(rows),
    }
    with (dest / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"DONE  {PC}/R0_retrieval_probe -> {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PC-4 RAG + privacy")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only",
        default="all",
        help="all | rag | privacy | probe | no_rag | flat | hier | res_only | priv | eps_high | eps_low",
    )
    args = parser.parse_args()
    only = args.only.lower()
    kw = dict(pc=PC, device=args.device, pilot=args.pilot, dry_run=args.dry_run)

    if only in {"all", "rag", "probe"}:
        run_rag_probe(dry_run=args.dry_run, num_clients=20 if args.pilot else 50)

    if only in {"all", "rag", "no_rag"}:
        run_one(
            experiment_id="R1_no_rag",
            strategy="fedavg",
            **off_modules(
                enable_audit=True,
                enable_negotiation=True,
                dishonest_fraction=0.2,
            ),
            **kw,
        )
    if only in {"all", "rag", "flat"}:
        run_one(
            experiment_id="R2_flat_rag",
            strategy="fedavg",
            **off_modules(
                enable_audit=True,
                enable_negotiation=True,
                enable_rag=True,
                enable_llm=False,
                dishonest_fraction=0.2,
            ),
            rag_mode="flat",
            **kw,
        )
    if only in {"all", "rag", "hier"}:
        run_one(
            experiment_id="R3_hierarchical_rag",
            strategy="fedavg",
            **off_modules(
                enable_audit=True,
                enable_negotiation=True,
                enable_rag=True,
                enable_llm=False,
                dishonest_fraction=0.2,
            ),
            rag_mode="hierarchical",
            **kw,
        )

    if only in {"all", "privacy", "res_only"}:
        run_one(
            experiment_id="P1_resource_only",
            strategy="fedavg",
            **off_modules(
                enable_audit=True,
                enable_negotiation=True,
                enable_privacy=False,
                dishonest_fraction=0.2,
            ),
            **kw,
        )
    if only in {"all", "privacy", "priv"}:
        run_one(
            experiment_id="P2_privacy_aware",
            strategy="fedavg",
            **off_modules(
                enable_audit=True,
                enable_negotiation=True,
                enable_privacy=True,
                dishonest_fraction=0.2,
            ),
            privacy_epsilon_default=8.0,
            **kw,
        )
    if only in {"all", "privacy", "eps_high"}:
        run_one(
            experiment_id="P3_epsilon_high",
            strategy="fedavg",
            **off_modules(
                enable_audit=True,
                enable_negotiation=True,
                enable_privacy=True,
                dishonest_fraction=0.2,
            ),
            privacy_epsilon_default=12.0,
            **kw,
        )
    if only in {"all", "privacy", "eps_low"}:
        run_one(
            experiment_id="P4_epsilon_low",
            strategy="fedavg",
            **off_modules(
                enable_audit=True,
                enable_negotiation=True,
                enable_privacy=True,
                dishonest_fraction=0.2,
            ),
            privacy_epsilon_default=3.0,
            **kw,
        )

    print("PC-4 finished -> experiments_2/results/pc4/")


if __name__ == "__main__":
    main()
