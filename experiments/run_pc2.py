"""
PC-2 (Core i5 CPU) — DS-ANE negotiation (experiments_2.txt).

Lighter CPU: fixed / rule / LLM negotiation. Optional complete+LLM via --only.

Runs (default `all`):
  1) Fixed configuration (no negotiation)
  2) Rule-based negotiation
  3) DS-ANE LLM-assisted negotiation

Optional:
  4) Complete SRA-FL with LLM  (--only integ_complete_llm)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common import off_modules, run_one

PC = "pc2"


def main() -> None:
    parser = argparse.ArgumentParser(description="PC-2 (Core i5) DS-ANE negotiation")
    parser.add_argument("--device", default="cpu", help="cpu (default) | cuda | auto")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only",
        default="all",
        help="all | dsane | fixed | rule | llm | integ_complete_llm",
    )
    args = parser.parse_args()
    only = args.only.lower()
    kw = dict(pc=PC, device=args.device, pilot=args.pilot, dry_run=args.dry_run)

    if only in {"all", "dsane", "fixed"}:
        run_one(
            experiment_id="N1_fixed_config",
            strategy="fedavg",
            **off_modules(
                enable_audit=True,
                dishonest_fraction=0.2,
            ),
            **kw,
        )
    if only in {"all", "dsane", "rule"}:
        run_one(
            experiment_id="N2_rule_based",
            strategy="fedavg",
            **off_modules(
                enable_audit=True,
                enable_negotiation=True,
                enable_llm=False,
                dishonest_fraction=0.2,
            ),
            **kw,
        )
    if only in {"all", "dsane", "llm"}:
        run_one(
            experiment_id="N3_dsane_llm",
            strategy="fedavg",
            **off_modules(
                enable_audit=True,
                enable_negotiation=True,
                enable_llm=True,
                dishonest_fraction=0.2,
            ),
            **kw,
        )

    # Heavy for i5 — only when explicitly requested (not part of default `all`)
    if only in {"integ_complete_llm"}:
        run_one(
            experiment_id="I3_complete_sra_fl_llm",
            strategy="fedavg",
            **off_modules(
                enable_audit=True,
                enable_negotiation=True,
                enable_rag=True,
                enable_communication=True,
                enable_privacy=True,
                enable_llm=True,
                dishonest_fraction=0.2,
                rag_mode="hierarchical",
                comm_policy_mode="learning",
            ),
            **kw,
        )

    print("PC-2 (Core i5) finished -> experiments_2/results/pc2/")


if __name__ == "__main__":
    main()
