"""One single-panel figure per file, drawn from experiments/results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
OUT = ROOT / "assets" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def history(experiment_id: str) -> list[dict]:
    matches = list(RESULTS.glob(f"pc*/{experiment_id}/metrics/{experiment_id}.json"))
    if not matches:
        raise FileNotFoundError(experiment_id)
    data = json.loads(matches[0].read_text(encoding="utf-8"))
    return data["history"]


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name)
    plt.close(fig)
    print(name)


def line_chart(series: dict[str, list[dict]], key: str, ylabel: str, name: str, scale: float = 1.0) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for label, rows in series.items():
        xs = [row["round"] for row in rows if key in row and row[key] is not None]
        ys = [row[key] * scale for row in rows if key in row and row[key] is not None]
        ax.plot(xs, ys, marker="o", linewidth=2.2, label=label)
    ax.set_xlabel("Server round")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(0, 11))
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    save(fig, name)


def bars(labels: list[str], values: list[float], ylabel: str, name: str, ylim: tuple[float, float] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    xpos = range(len(labels))
    ax.bar(xpos, values, color="#2F6F8F", width=0.72)
    ax.set_xticks(list(xpos), labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    for x, value in zip(xpos, values):
        ax.text(x, value, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    save(fig, name)


def main() -> None:
    b1 = history("B1_fedavg")
    b2 = history("B2_fedprox")
    i3 = history("I3_complete_sra_fl")
    a3 = history("A3_aavl_historical")

    line_chart(
        {"FedAvg (B1)": b1, "FedProx (B2)": b2, "Complete stack (I3)": i3},
        "accuracy",
        "Test accuracy (%)",
        "paper_accuracy_convergence.png",
        scale=100.0,
    )
    line_chart(
        {"FedAvg (B1)": b1, "FedProx (B2)": b2, "Complete stack (I3)": i3},
        "loss",
        "Test loss",
        "paper_loss_convergence.png",
    )

    finals = [
        ("FedAvg", 72.65),
        ("FedProx", 71.94),
        ("AAVL", 74.08),
        ("Fixed nego.", 74.08),
        ("Rule nego.", 76.16),
        ("Complete", 76.15),
        ("Fixed comm.", 76.16),
        ("Learn comm.", 77.13),
        ("No RAG", 76.16),
        ("H-RAG", 77.08),
    ]
    bars(
        [name for name, _ in finals],
        [value for _, value in finals],
        "Final test accuracy (%)",
        "paper_final_accuracy.png",
        ylim=(68, 82),
    )

    audit_rounds = [row["round"] for row in a3 if "audit" in row]
    f1 = [row["audit"]["f1"] for row in a3 if "audit" in row]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(audit_rounds, f1, marker="o", color="#B85C38", linewidth=2.2)
    ax.set_xlabel("Server round")
    ax.set_ylabel("Audit F1")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(range(0, 11))
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    save(fig, "paper_audit_f1.png")

    rep_rounds = [row["round"] for row in a3 if "reputation" in row]
    rep_min = [row["reputation"]["min"] for row in a3 if "reputation" in row]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(rep_rounds, rep_min, marker="o", color="#3D6B4F", linewidth=2.2)
    ax.set_xlabel("Server round")
    ax.set_ylabel("Lowest client reputation")
    ax.set_ylim(0.5, 1.05)
    ax.set_xticks(range(0, 11))
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    save(fig, "paper_reputation_min.png")

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    labels = ["Fixed (C1)", "Threshold (C2)", "Learning (C3)"]
    uploads = [13, 13, 12]
    skips = [0, 0, 1]
    x = range(len(labels))
    ax.bar([i - 0.18 for i in x], uploads, width=0.36, label="Uploads", color="#2F6F8F")
    ax.bar([i + 0.18 for i in x], skips, width=0.36, label="Skips", color="#C47B2B")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Decisions")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    save(fig, "paper_comm_actions.png")

    bars(
        ["No RAG", "Flat RAG", "Hierarchical RAG"],
        [76.16, 77.08, 77.08],
        "Final test accuracy (%)",
        "paper_rag_accuracy.png",
        ylim=(74, 80),
    )
    bars(
        ["Resource only", "Budget 8", "Budget 12", "Budget 3"],
        [76.16, 76.16, 76.16, 76.16],
        "Final test accuracy (%)",
        "paper_privacy_accuracy.png",
        ylim=(74, 80),
    )

    retrieval_quality_figure()
    audit_reputation_figure()
    research_gap_figures()


def audit_reputation_figure() -> None:
    """One chart: the flagged client's reputation stays down after the round-1 detection."""
    path = RESULTS / "pc3" / "A3_aavl_historical" / "metrics" / "A3_aavl_historical.json"
    hist = json.loads(path.read_text(encoding="utf-8"))["history"]
    rows = [row for row in hist if "reputation" in row]
    rounds = [row["round"] for row in rows]
    client0 = [row["reputation"]["scores"]["0"] for row in rows]
    honest = [1.0 for _ in rows]
    f1_round1 = next(row["audit"]["f1"] for row in hist if row.get("audit") and row["round"] == 1)

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.plot(rounds, honest, marker="o", color="#2E7D4F", linewidth=2.4, label="Other clients (reputation 1.00)")
    ax.plot(rounds, client0, marker="s", color="#B85C38", linewidth=2.4, label="Client 0 (flagged)")
    ax.annotate(
        f"Round 1: 1 client flagged, F1 = {f1_round1:.1f}",
        xy=(1, client0[0]),
        xytext=(2.2, 0.91),
        fontsize=12,
        color="#1B2838",
        arrowprops={"arrowstyle": "->", "color": "#1B2838"},
    )
    ax.set_title("AAVL reputation after the audit", fontsize=16, pad=10)
    ax.set_xlabel("Server round")
    ax.set_ylabel("Reputation")
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.82, 1.04)
    ax.set_xticks(rounds)
    ax.legend(frameon=False, loc="center right")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()

    targets = [
        OUT / "paper_audit_reputation.png",
        ROOT / "results" / "figures" / "fig3_aavl_audit_reputation.png",
        ROOT / "assets" / "fig3_aavl_audit_reputation.png",
    ]
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target)
        print(target)
    plt.close(fig)


def final_accuracy(experiment_id: str) -> float:
    """Read one final test accuracy, in percent, from the merged summary table."""
    path = RESULTS / "all_summaries.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["experiment_id"] == experiment_id and row["final_accuracy"]:
                return float(row["final_accuracy"]) * 100.0
    raise KeyError(experiment_id)


def gap_bar_figure(title: str, runs: list[tuple[str, str]], filename: str, highlight: str) -> None:
    """One bar chart for one research gap. The highlighted bar is that gap's proposed variant."""
    labels = [label for label, _ in runs]
    values = [final_accuracy(experiment_id) for _, experiment_id in runs]
    colors = ["#B85C38" if label == highlight else "#2F6F8F" for label in labels]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    xpos = list(range(len(labels)))
    ax.bar(xpos, values, color=colors, width=0.62, edgecolor="#1B2838", linewidth=0.6)
    ax.set_xticks(xpos, labels)
    ax.set_ylabel("Final test accuracy (%)")
    ax.set_title(title, fontsize=16, pad=10)
    ax.set_ylim(70, 81)
    for x, value in zip(xpos, values):
        ax.text(x, value + 0.25, f"{value:.2f}%", ha="center", va="bottom", fontsize=12, color="#1B2838")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()

    targets = [
        OUT / filename,
        ROOT / "results" / "figures" / filename,
        ROOT / "assets" / filename,
    ]
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target)
        print(target)
    plt.close(fig)


def research_gap_figures() -> None:
    """Five separate gap charts. The old overall-performance panel is not drawn."""
    gap_bar_figure(
        "Gap 1: dishonest reporting",
        [
            ("No audit", "A1_no_verification"),
            ("Threshold", "A2_threshold_verification"),
            ("Historical AAVL", "A3_aavl_historical"),
        ],
        "fig2_gap1_honesty.png",
        "Historical AAVL",
    )
    gap_bar_figure(
        "Gap 2: server mandate",
        [
            ("Fixed", "N1_fixed_config"),
            ("Rule-based", "N2_rule_based"),
            ("LLM negotiation", "N3_dsane_llm"),
        ],
        "fig2_gap2_negotiation.png",
        "Rule-based",
    )
    gap_bar_figure(
        "Gap 3: fixed uploads",
        [
            ("Fixed", "C1_comm_fixed"),
            ("Threshold", "C2_comm_threshold"),
            ("Learning", "C3_comm_learning"),
        ],
        "fig2_gap3_communication.png",
        "Learning",
    )
    gap_bar_figure(
        "Gap 4: central scalability",
        [
            ("No RAG", "R1_no_rag"),
            ("Flat RAG", "R2_flat_rag"),
            ("Hierarchical RAG", "R3_hierarchical_rag"),
        ],
        "fig2_gap4_rag.png",
        "Hierarchical RAG",
    )
    gap_bar_figure(
        "Gap 5: privacy conflict",
        [
            ("Resource only", "P1_resource_only"),
            ("Budget 8", "P2_privacy_aware"),
            ("Budget 12", "P3_epsilon_high"),
            ("Budget 3", "P4_epsilon_low"),
        ],
        "fig2_gap5_privacy.png",
        "",
    )


def retrieval_quality_figure() -> None:
    """One chart: precision at 5 per probe query. Latency and token counts stay in the paper text."""
    probe = json.loads((RESULTS / "pc4" / "R0_retrieval_probe" / "retrieval_quality.json").read_text(encoding="utf-8"))
    seen: dict[str, float] = {}
    for row in probe["queries"]:
        seen.setdefault(row["query"], float(row["precision_at_5"]))
    labels = list(seen)
    values = [seen[name] for name in labels]
    mean_precision = sum(float(row["precision_at_5"]) for row in probe["queries"]) / len(probe["queries"])
    colors = ["#1F4E79", "#2E7D4F", "#C47B2B", "#B85C38", "#5C4B8A", "#1F6F8B", "#8B3A3A", "#3D6B4F", "#7A5B2E", "#3E4C6B"]

    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    y = list(range(len(labels)))
    ax.barh(y, values, color=colors[: len(values)], height=0.72, edgecolor="#1B2838", linewidth=0.6)
    ax.set_yticks(y, labels)
    for pos, value in zip(y, values):
        ax.text(value + 0.02, pos, f"{value:.2f}", va="center", ha="left", fontsize=12, color="#1B2838")
    ax.axvline(mean_precision, color="#8B1E3F", linestyle="--", linewidth=1.8, label=f"Mean {mean_precision:.2f}")
    ax.set_title("Retrieval quality", fontsize=16, pad=10)
    ax.set_xlabel("Precision at 5")
    ax.set_ylabel("Query")
    ax.set_xlim(0, 1.2)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    fig.tight_layout()

    targets = [
        OUT / "paper_retrieval_precision.png",
        ROOT / "results" / "figures" / "fig6_rag_orchestration_llm_profile.png",
        ROOT / "assets" / "fig6_rag_orchestration_llm_profile.png",
    ]
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
