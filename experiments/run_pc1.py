"""
PC-1 (Core i7 CPU) — Baselines + final integration (experiments_2.txt).

Stronger CPU: FedAvg/FedProx + integration stack (no GPU required).

Runs:
  1) FedAvg baseline
  2) FedProx baseline
  3) Integration: FedAvg + AAVL + DS-ANE
  4) Integration: Complete SRA-FL (modules on, LLM off)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments_2.common import off_modules, run_one

PC = "pc1"


def main() -> None:
    parser = argparse.ArgumentParser(description="PC-1 (Core i7) baselines + integration")
    parser.add_argument("--device", default="cpu", help="cpu (default) | cuda | auto")
    parser.add_argument("--pilot", action="store_true", help="3 rounds / 5 clients smoke")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only",
        default="all",
        help="all | baselines | integration | fedavg | fedprox | integ_aavl_dsane | integ_complete",
    )
    args = parser.parse_args()
    only = args.only.lower()
    kw = dict(pc=PC, device=args.device, pilot=args.pilot, dry_run=args.dry_run)

    if only in {"all", "baselines", "fedavg"}:
        run_one(
            experiment_id="B1_fedavg",
            strategy="fedavg",
            **off_modules(),
            **kw,
        )
    if only in {"all", "baselines", "fedprox"}:
        run_one(
            experiment_id="B2_fedprox",
            strategy="fedprox",
            proximal_mu=0.01,
            **off_modules(),
            **kw,
        )

    # Integration (skip plain FedAvg clone of B1 to save CPU time)
    if only in {"all", "integration", "integ_aavl_dsane"}:
        run_one(
            experiment_id="I2_fedavg_aavl_dsane",
            strategy="fedavg",
            **off_modules(
                enable_audit=True,
                enable_negotiation=True,
                dishonest_fraction=0.2,
            ),
            **kw,
        )
    if only in {"all", "integration", "integ_complete"}:
        run_one(
            experiment_id="I3_complete_sra_fl",
            strategy="fedavg",
            **off_modules(
                enable_audit=True,
                enable_negotiation=True,
                enable_rag=True,
                enable_communication=True,
                enable_privacy=True,
                enable_llm=False,
                dishonest_fraction=0.2,
                rag_mode="hierarchical",
                comm_policy_mode="learning",
            ),
            **kw,
        )

    print("PC-1 (Core i7) finished -> experiments_2/results/pc1/")


if __name__ == "__main__":
    main()
