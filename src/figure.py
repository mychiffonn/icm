"""Generate comparison figure for ICM experiment results."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from src import (
    FEW_SHOT_GOLDEN_RESULT_PATH,
    FEW_SHOT_ICM_RESULT_PATH,
    MAIN_FIG_PATH,
    RESULTS_DIR,
    ZERO_SHOT_BASE_RESULT_PATH,
    ZERO_SHOT_CHAT_RESULT_PATH,
)
from src.utils import read_json

logger = logging.getLogger(__name__)


def load_results(results_dir: Path) -> dict:
    """Load evaluation results from multiple JSON files.

    Args:
        results_dir: Directory containing results files.

    Returns:
        Dictionary with evaluation results.
    """
    result_files = {
        "zero_shot_base": ZERO_SHOT_BASE_RESULT_PATH,
        "zero_shot_chat": ZERO_SHOT_CHAT_RESULT_PATH,
        "few_shot_oracle": FEW_SHOT_GOLDEN_RESULT_PATH,
        "few_shot_icm": FEW_SHOT_ICM_RESULT_PATH,
    }

    results = {}
    for name, file_path in result_files.items():
        if file_path.exists():
            data = read_json(file_path)
            results[name] = data
        else:
            logger.warning(f"Result file not found: {file_path}. Skipping {name}.")

    return results


def plot_comparison(results: dict, save_path: Path | str):
    """Create bar plot comparing all approaches.

    Args:
        results: Dictionary with evaluation results.
        save_path: Path to save the figure.
    """
    # Define approach labels and colors
    approaches = {
        "zero_shot_base": ("Zero-Shot\n(Base Model)", "#3498db"),
        "zero_shot_chat": ("Zero-Shot\n(Chat Model)", "#9b59b6"),
        "few_shot_oracle": ("Few-Shot\n(Ground Truth)", "#2ecc71"),
        "few_shot_icm": ("Few-Shot\n(ICM Labels)", "#e74c3c"),
    }

    # Extract accuracies
    labels = []
    accuracies = []
    colors = []

    for key, (label, color) in approaches.items():
        if key in results:
            labels.append(label)
            accuracies.append(results[key]["accuracy"] * 100)  # Convert to percentage
            colors.append(color)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars
    x = np.arange(len(labels))
    bars = ax.bar(
        x, accuracies, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5
    )

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies, strict=True):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Customize plot
    ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight="bold")
    ax.set_title(
        "TruthfulQA Accuracy Comparison Across Methods",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add horizontal line at 50% (random baseline)
    ax.axhline(
        y=50,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="Random Baseline (50%)",
    )
    ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()

    # Save figure
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f" Saved figure to: {save_path}")

    # Also save as PDF
    pdf_path = save_path.with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f" Saved PDF to: {pdf_path}")

    plt.close()


def plot_convergence(results_dir: Path, save_path: Path):
    """Plot ICM utility convergence over iterations.

    Args:
        results_dir: Directory containing results files.
        save_path: Path to save the figure.
    """
    history_file = results_dir / "icm_history.json"

    if not history_file.exists():
        print(f"History file not found: {history_file}")
        print("  Skipping convergence plot.")
        return

    with history_file.open() as f:
        history = json.load(f)

    utility_history = history["utility_history"]
    acceptance_history = history["acceptance_history"]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot utility
    iterations = range(1, len(utility_history) + 1)
    ax1.plot(iterations, utility_history, linewidth=2, color="#3498db")
    ax1.axhline(
        y=history["best_utility"],
        color="#e74c3c",
        linestyle="--",
        linewidth=1.5,
        label=f"Best: {history['best_utility']:.4f}",
    )
    ax1.set_ylabel("Utility Score", fontsize=12, fontweight="bold")
    ax1.set_title("ICM Search Convergence", fontsize=14, fontweight="bold")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower right")

    # Plot acceptance rate (rolling average)
    window = 10
    acceptance_rate = [
        sum(acceptance_history[max(0, i - window) : i + 1]) / min(window, i + 1)
        for i in range(len(acceptance_history))
    ]
    ax2.plot(iterations, acceptance_rate, linewidth=2, color="#2ecc71")
    ax2.set_xlabel("Iteration", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Acceptance Rate", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    # Save
    convergence_path = save_path.parent / "convergence.png"
    plt.savefig(convergence_path, dpi=300, bbox_inches="tight")
    print(f" Saved convergence plot to: {convergence_path}")

    plt.close()


def main():
    """Generate all figures from experiment results."""
    results = load_results(RESULTS_DIR)

    # Print summary
    print("\nResults Summary:")
    print("-" * 40)
    for name, res in results.items():
        print(f"{name:25s}: {res['accuracy']:.2%} ({res['correct']}/{res['total']})")

    # Generate comparison plot
    print("\nGenerating comparison plot...")
    plot_comparison(results, MAIN_FIG_PATH)

    # Generate convergence plot
    print("\nGenerating convergence plot...")
    plot_convergence(RESULTS_DIR, RESULTS_DIR / "convergence.png")


if __name__ == "__main__":
    main()
