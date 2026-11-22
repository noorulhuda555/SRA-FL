"""
PC-3 (Core i7 CPU) — AAVL + learning-based communication (experiments_2.txt).

Runs:
  AAVL: no verification / threshold (basic) / historical (AAVL)
  Comm: fixed / threshold / learning
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments_2.common import off_modules, run_one

PC = "pc3"


def main() -> None:
    parser = argparse.ArgumentParser(description="PC-3 AAVL + communication")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only",
        default="all",
        help="all | aavl | comm | no_verify | threshold | aavl_hist | fixed | thr_comm | learning",
    )
    args = parser.parse_args()
    only = args.only.lower()
    kw = dict(pc=PC, device=args.device, pilot=args.pilot, dry_run=args.dry_run)

    # Part A — AAVL (2 dishonest of 10 => 0.2)
    if only in {"all", "aavl", "no_verify"}:
        # Same planted liars as other AAVL runs, but threshold so high that nothing is flagged.
        run_one(
            experiment_id="A1_no_verification",
            strategy="fedavg",
            **off_modules(
                enable_audit=True,
                dishonest_fraction=0.2,
            ),
            suspicion_threshold=999.0,
            use_reputation_selection=False,
            **kw,
        )
    if only in {"all", "aavl", "threshold"}:
        run_one(
            experiment_id="A2_threshold_verification",
            strategy="fedavg",
            **off_modules(
                enable_audit=True,
                dishonest_fraction=0.2,
            ),
            audit_mode="basic",
            use_reputation_selection=False,
            **kw,
        )
    if only in {"all", "aavl", "aavl_hist"}:
        run_one(
            experiment_id="A3_aavl_historical",
            strategy="fedavg",
            **off_modules(
                enable_audit=True,
                dishonest_fraction=0.2,
            ),
            audit_mode="historical",
            use_reputation_selection=True,
            **kw,
        )

    # Part B — communication policies
    if only in {"all", "comm", "fixed"}:
        run_one(
            experiment_id="C1_comm_fixed",
            strategy="fedavg",
            **off_modules(
                enable_audit=True,
                enable_negotiation=True,
                enable_communication=True,
                dishonest_fraction=0.2,
            ),
            comm_policy_mode="fixed",
            **kw,
        )
    if only in {"all", "comm", "thr_comm"}:
        run_one(
            experiment_id="C2_comm_threshold",
            strategy="fedavg",
            **off_modules(
                enable_audit=True,
                enable_negotiation=True,
                enable_communication=True,
                dishonest_fraction=0.2,
            ),
            comm_policy_mode="threshold",
            **kw,
        )
    if only in {"all", "comm", "learning"}:
        run_one(
            experiment_id="C3_comm_learning",
            strategy="fedavg",
            **off_modules(
                enable_audit=True,
                enable_negotiation=True,
                enable_communication=True,
                dishonest_fraction=0.2,
            ),
            comm_policy_mode="learning",
            **kw,
        )

    print("PC-3 finished -> experiments_2/results/pc3/")


if __name__ == "__main__":
    main()
