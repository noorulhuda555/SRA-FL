"""Merge results copied from multiple PCs into one index + master CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
OUT_CSV = RESULTS / "all_summaries.csv"
OUT_JSON = RESULTS / "all_summaries.json"


def main() -> None:
    rows: list[dict] = []
    for summary in RESULTS.rglob("summary.json"):
        with summary.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        data["_path"] = str(summary)
        rows.append(data)

    rows.sort(key=lambda r: (str(r.get("pc", "")), str(r.get("experiment_id", ""))))
    with OUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, default=str)

    if not rows:
        print("No summary.json files found under experiments_2/results/")
        return

    # Union of flat keys
    keys: list[str] = []
    for row in rows:
        for k, v in row.items():
            if isinstance(v, (dict, list)):
                continue
            if k not in keys:
                keys.append(k)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})

    print(f"Merged {len(rows)} summaries -> {OUT_CSV}")
    by_pc: dict[str, int] = {}
    for row in rows:
        pc = str(row.get("pc", "?"))
        by_pc[pc] = by_pc.get(pc, 0) + 1
    for pc, n in sorted(by_pc.items()):
        print(f"  {pc}: {n}")


if __name__ == "__main__":
    main()
