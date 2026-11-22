"""Shared runner for experiments_2 (4-PC plan in experiments_2.txt)."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sra_fl.server.orchestration import run_simulation
from sra_fl.utils.config_loader import load_config

RESULTS = ROOT / "experiments" / "results"

# Shared FL settings from experiments_2.txt §PC-1 table
BASE_FL: dict[str, Any] = {
    "dataset": "fashion_mnist",
    "num_clients": 10,
    "num_rounds": 10,
    "local_epochs": 1,
    "batch_size": 32,
    "partition": "non_iid",
    "dirichlet_alpha": 0.5,
    "seed": 42,
    "fraction_fit": 0.5,
    "fraction_evaluate": 0.5,
    "min_fit_clients": 1,
    "min_evaluate_clients": 1,
    "min_available_clients": 1,
    "dropout_probability": 0.05,
    "enable_heterogeneity": True,
}


def resolve_device(preferred: str = "auto") -> str:
    pref = preferred.lower()
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref.startswith("cuda") and torch.cuda.is_available():
        return pref
    return "cpu"


def out_dir(pc: str, experiment_id: str) -> Path:
    path = RESULTS / pc / experiment_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "metrics").mkdir(parents=True, exist_ok=True)
    (path / "logs").mkdir(parents=True, exist_ok=True)
    return path


def history_to_csv(history: list[dict[str, Any]], csv_path: Path) -> None:
    if not history:
        return
    keys: list[str] = []
    for row in history:
        for k in row:
            if k not in keys and k not in {"fit_metrics", "reputation", "rag_explanation", "llm_usage"}:
                keys.append(k)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in history:
            flat = {k: row.get(k) for k in keys}
            # Flatten nested audit briefly
            audit = row.get("audit")
            if isinstance(audit, dict):
                flat["audit"] = json.dumps(audit)
            writer.writerow(flat)


def summarize(history: list[dict[str, Any]], participation: list[dict[str, Any]]) -> dict[str, Any]:
    acc = [h["accuracy"] for h in history if h.get("accuracy") is not None]
    loss = [h["loss"] for h in history if h.get("loss") is not None]
    f1 = [h["audit"]["f1"] for h in history if isinstance(h.get("audit"), dict)]
    dropouts = sum(len(h.get("dropouts") or []) for h in history)
    return {
        "final_accuracy": acc[-1] if acc else None,
        "best_accuracy": max(acc) if acc else None,
        "final_loss": loss[-1] if loss else None,
        "mean_audit_f1": (sum(f1) / len(f1)) if f1 else None,
        "total_dropouts": dropouts,
        "num_rounds_logged": len(history),
        "fit_participations": sum(
            1 for p in participation if p.get("phase") == "fit"
        ),
    }


def run_one(
    *,
    pc: str,
    experiment_id: str,
    device: str = "auto",
    pilot: bool = False,
    dry_run: bool = False,
    **overrides: Any,
) -> dict[str, Any]:
    """Run one independent FL experiment; writes JSON + CSV under results/<pc>/."""
    dest = out_dir(pc, experiment_id)
    cfg = dict(BASE_FL)
    cfg.update(overrides)
    cfg["device"] = resolve_device(device)
    cfg["results_dir"] = str(dest)
    cfg["experiment_name"] = experiment_id
    if pilot:
        cfg["num_clients"] = min(int(cfg["num_clients"]), 5)
        cfg["num_rounds"] = min(int(cfg["num_rounds"]), 3)

    meta = {
        "pc": pc,
        "experiment_id": experiment_id,
        "device": cfg["device"],
        "pilot": pilot,
        "config": {k: v for k, v in cfg.items()},
    }
    with (dest / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, default=str)

    if dry_run:
        print(f"[dry-run] {pc}/{experiment_id} device={cfg['device']}")
        return meta

    config = load_config(dataset=str(cfg["dataset"]), overrides=cfg)
    print(
        f"START {pc}/{experiment_id} | device={config.get('device')} "
        f"clients={config['num_clients']} rounds={config['num_rounds']}"
    )
    t0 = time.perf_counter()
    result = run_simulation(config)
    elapsed = time.perf_counter() - t0

    metrics_path = Path(result["metrics_path"])
    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    history = payload.get("history", [])
    participation = payload.get("participation", [])

    summary = summarize(history, participation)
    summary.update(
        {
            "pc": pc,
            "experiment_id": experiment_id,
            "elapsed_seconds": elapsed,
            "device": config.get("device"),
            "metrics_path": str(metrics_path),
        }
    )
    if result.get("negotiation_logs"):
        nego = result["negotiation_logs"]
        summary["negotiation_count"] = len(nego)
        summary["negotiation_accept_rate"] = (
            sum(1 for n in nego if n.get("agreed") is not None) / len(nego) if nego else None
        )
    if result.get("upload_manager") is not None:
        hist = result["upload_manager"].history
        summary["comm_decisions"] = len(hist)
        summary["comm_uploads"] = sum(1 for h in hist if h.get("uploaded"))
        summary["comm_skips"] = sum(
            1 for h in hist if h.get("comm_action") in {"skip", "wait"}
        )
    if result.get("orchestrator") is not None:
        summary["llm_usage"] = result["orchestrator"].llm.usage_summary()
    if result.get("privacy_manager") is not None:
        summary["privacy"] = result["privacy_manager"].summary()

    with (dest / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    history_to_csv(history, dest / "history.csv")
    with (dest / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        flat = {
            k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
            for k, v in summary.items()
        }
        writer = csv.DictWriter(handle, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)

    # Suite index
    index_path = RESULTS / pc / "index.json"
    index: list[dict[str, Any]] = []
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
    index = [r for r in index if r.get("experiment_id") != experiment_id]
    index.append(
        {
            "experiment_id": experiment_id,
            "final_accuracy": summary.get("final_accuracy"),
            "mean_audit_f1": summary.get("mean_audit_f1"),
            "elapsed_seconds": elapsed,
            "summary_path": str(dest / "summary.json"),
        }
    )
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2)

    print(
        f"DONE  {pc}/{experiment_id} | acc={summary.get('final_accuracy')} "
        f"elapsed={elapsed:.1f}s -> {dest}"
    )
    return summary


def off_modules(**on: bool) -> dict[str, Any]:
    """Default all modules off, then turn selected ones on."""
    flags = {
        "enable_audit": False,
        "enable_negotiation": False,
        "enable_rag": False,
        "enable_communication": False,
        "enable_privacy": False,
        "enable_llm": False,
        "dishonest_fraction": 0.0,
    }
    flags.update(on)
    return flags
