"""Generate the eight figures used in the paper."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
OUT = ROOT / "results" / "figures"

BLUE = "#3D5A80"
TEAL = "#2A9D8F"
SAND = "#E9C46A"
CORAL = "#E76F51"
PLUM = "#6D597A"
INK = "#1D3557"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#4A4A4A",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def save(fig: plt.Figure, filename: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(filename)


def history(experiment_id: str) -> list[dict]:
    matches = list(RESULTS.glob(f"pc*/{experiment_id}/metrics/{experiment_id}.json"))
    if not matches:
        raise FileNotFoundError(f"Missing metrics for {experiment_id}")
    return json.loads(matches[0].read_text(encoding="utf-8"))["history"]


def final_accuracy(experiment_id: str) -> float:
    with (RESULTS / "all_summaries.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["experiment_id"] == experiment_id and row["final_accuracy"]:
                return float(row["final_accuracy"]) * 100.0
    raise KeyError(f"Missing final_accuracy for {experiment_id}")


def accuracy_bars(
    labels: list[str],
    values: list[float],
    colors: list[str],
    title: str,
    filename: str,
    horizontal: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    positions = range(len(labels))

    if horizontal:
        bars = ax.barh(list(positions), values, color=colors, height=0.62, zorder=2)
        ax.set_yticks(list(positions), labels)
        ax.set_xlabel("Final test accuracy (%)")
        ax.set_xlim(70, 80)
        ax.grid(axis="x", color="#E6E6E6", zorder=0)
        for bar, value in zip(bars, values):
            ax.text(value + 0.12, bar.get_y() + bar.get_height() / 2,
                    f"{value:.2f}%", va="center", ha="left", color=INK, fontsize=9)
    else:
        bars = ax.bar(list(positions), values, color=colors, width=0.62, zorder=2)
        ax.set_xticks(list(positions), labels)
        ax.set_ylabel("Final test accuracy (%)")
        ax.set_ylim(70, 80)
        ax.grid(axis="y", color="#E6E6E6", zorder=0)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.18,
                    f"{value:.2f}%", ha="center", va="bottom", color=INK, fontsize=9)

    ax.set_title(title)
    fig.tight_layout()
    save(fig, filename)


def figure_1_architecture() -> None:
    """Fig. 1: SRA-FL system architecture."""
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, color: str) -> None:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=1.4,
            edgecolor=color,
            facecolor="white",
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=9, fontweight="bold", color=INK)

    def arrow(x1: float, y1: float, x2: float, y2: float, color: str = "#555555") -> None:
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color=color,
        ))

    box(0.2, 4.6, 2.0, 1.4, "Simulation inputs\nFedScale, Fashion-MNIST,\nMobiPerf", "#6D597A")
    box(2.7, 4.6, 2.0, 1.4, "Client Guardian\nstate and resource checks", BLUE)
    box(5.2, 4.6, 1.8, 1.4, "AAVL audit\nreputation scoring", CORAL)
    box(7.1, 4.6, 1.8, 1.4, "Hierarchical RAG\nclient records", TEAL)
    box(9.0, 4.6, 1.8, 1.4, "DS-ANE negotiation\nand privacy", "#C47B2B")

    box(2.7, 2.2, 2.0, 1.4, "Local trainer\nCNN and FedAvg/FedProx", BLUE)
    box(5.2, 2.2, 2.0, 1.4, "SRA-FL orchestrator\nselection and rounds", "#6B8E23")
    box(7.7, 2.2, 2.0, 1.4, "Communication policy\nupload/compress/skip", "#4A7C59")
    box(10.2, 2.2, 1.6, 1.4, "Aggregation\nFedAvg/FedProx", "#3D6B4F")

    arrow(2.2, 5.3, 2.7, 5.3)
    arrow(4.7, 5.3, 5.2, 5.3)
    arrow(7.0, 5.3, 7.1, 5.3)
    arrow(8.9, 5.3, 9.0, 5.3)
    arrow(3.7, 4.6, 3.7, 3.6)
    arrow(3.7, 2.2, 5.2, 2.9)
    arrow(7.2, 2.9, 7.7, 2.9)
    arrow(9.7, 2.9, 10.2, 2.9)
    arrow(11.0, 3.6, 11.0, 4.6)

    ax.text(6.0, 6.45, "SRA-FL System Architecture", ha="center", va="center",
            fontsize=16, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "architecture_horizontal.png")


def figure_2_baselines() -> None:
    """Fig. 2: FedAvg, FedProx, and complete SRA-FL."""
    accuracy_bars(
        ["FedAvg", "FedProx", "Complete SRA-FL"],
        [final_accuracy("B1_fedavg"), final_accuracy("B2_fedprox"), final_accuracy("I3_complete_sra_fl")],
        [BLUE, SAND, TEAL],
        "Final test accuracy of the baselines and complete SRA-FL",
        "fig_baseline_final_accuracy.png",
    )


def figure_3_audit() -> None:
    """Fig. 3: audit settings."""
    accuracy_bars(
        ["No audit", "Single-round audit", "Historical audit"],
        [
            final_accuracy("A1_no_verification"),
            final_accuracy("A2_threshold_verification"),
            final_accuracy("A3_aavl_historical"),
        ],
        [BLUE, SAND, TEAL],
        "Final test accuracy under three audit settings",
        "fig_gap1_audit_accuracy.png",
    )


def figure_4_negotiation() -> None:
    """Fig. 4: negotiation settings."""
    accuracy_bars(
        ["Fixed configuration", "Rule-based negotiation", "Language-model flag on"],
        [
            final_accuracy("N1_fixed_config"),
            final_accuracy("N2_rule_based"),
            final_accuracy("N3_dsane_llm"),
        ],
        [BLUE, TEAL, PLUM],
        "Final test accuracy with and without negotiation",
        "fig_gap2_negotiation_accuracy.png",
        horizontal=True,
    )


def figure_5_communication() -> None:
    """Fig. 5: communication decisions."""
    labels = ["Always upload", "Bandwidth threshold", "Q-learning"]
    uploads = [13, 13, 11]
    compress = [0, 0, 1]
    skips = [0, 0, 1]

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    x = range(len(labels))
    ax.bar(x, uploads, color=TEAL, width=0.62, label="Upload", zorder=2)
    ax.bar(x, compress, bottom=uploads, color=SAND, width=0.62, label="Compress", zorder=2)
    bottom = [a + b for a, b in zip(uploads, compress)]
    ax.bar(x, skips, bottom=bottom, color=CORAL, width=0.62, label="Skip", zorder=2)
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Communication decisions")
    ax.set_ylim(0, 16)
    ax.set_title("Communication decisions after local training")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", color="#E6E6E6", zorder=0)
    fig.tight_layout()
    save(fig, "fig_gap3_comm_actions.png")


def figure_6_rag() -> None:
    """Fig. 6: RAG accuracy."""
    labels = ["No retrieval", "Flat retrieval", "Grouped retrieval"]
    values = [
        final_accuracy("R1_no_rag"),
        final_accuracy("R2_flat_rag"),
        final_accuracy("R3_hierarchical_rag"),
    ]
    colors = [BLUE, TEAL, PLUM]

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    y = range(len(labels))
    ax.hlines(list(y), 75.6, values, color="#C5C5C5", linewidth=2, zorder=1)
    ax.scatter(values, list(y), s=90, color=colors, zorder=2)
    for yi, value in zip(y, values):
        ax.text(value + 0.08, yi, f"{value:.2f}%", va="center", ha="left",
                color=INK, fontsize=9)
    ax.set_yticks(list(y), labels)
    ax.set_xlabel("Final test accuracy (%)")
    ax.set_xlim(75.6, 77.8)
    ax.set_title("Final test accuracy with and without client-record retrieval")
    ax.grid(axis="x", color="#E6E6E6", zorder=0)
    fig.tight_layout()
    save(fig, "fig_gap4_retrieval_accuracy.png")


def figure_7_privacy() -> None:
    """Fig. 7: privacy-budget settings."""
    accuracy_bars(
        ["No budget", "Budget 8", "Budget 12", "Budget 3"],
        [
            final_accuracy("P1_resource_only"),
            final_accuracy("P2_privacy_aware"),
            final_accuracy("P3_epsilon_high"),
            final_accuracy("P4_epsilon_low"),
        ],
        [BLUE, TEAL, SAND, CORAL],
        "Final test accuracy across privacy-budget settings",
        "fig_gap5_privacy_accuracy.png",
    )


def figure_8_convergence() -> None:
    """Fig. 8: test accuracy after each communication round."""
    series = [
        ("FedAvg", history("B1_fedavg"), BLUE),
        ("FedProx", history("B2_fedprox"), SAND),
        ("Complete SRA-FL", history("I3_complete_sra_fl"), TEAL),
        ("Q-learning communication", history("C3_comm_learning"), CORAL),
    ]

    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    for name, rows, color in series:
        ax.plot(
            [row["round"] for row in rows],
            [row["accuracy"] * 100 for row in rows],
            marker="o",
            markersize=4.5,
            linewidth=2,
            color=color,
            label=name,
        )
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xticks(range(0, 11))
    ax.set_ylim(0, 100)
    ax.set_title("Test accuracy after each communication round")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(color="#E6E6E6")
    fig.tight_layout()
    save(fig, "fig_convergence_accuracy.png")


def main() -> None:
    figure_1_architecture()
    figure_2_baselines()
    figure_3_audit()
    figure_4_negotiation()
    figure_5_communication()
    figure_6_rag()
    figure_7_privacy()
    figure_8_convergence()
    print(f"Wrote 8 figures to {OUT}")


if __name__ == "__main__":
    main()
