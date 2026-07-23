"""
CSC111 Project 2: Tree-Based Sentence Complexity Analyzer
Module: visualizations.py
Description: Contains functions for generating charts and a results table
summarizing syntactic complexity across CEFR proficiency levels.
Authors: Tugra Canbaz, Elif Cakici, Alsade Brianna Daley
"""
from __future__ import annotations
import doctest
import python_ta
import matplotlib.pyplot as plt

_COLORS = ["#4CAF50", "#8BC34A", "#FFC107", "#FF9800", "#F44336", "#9C27B0"]


def plot_complexity_score(cefr_data: dict) -> None:
    """Bar chart showing average complexity score per CEFR level."""
    levels = list(cefr_data.keys())
    scores: list[float] = [cefr_data[lvl]["avg_score"] for lvl in levels]
    _, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(levels, scores, color=_COLORS[:len(levels)], edgecolor="black", width=0.5)
    for score_bar, score in zip(bars, scores):
        ax.text(score_bar.get_x() + score_bar.get_width() / 2,
                score_bar.get_height() + 0.1,
                f"{score:.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("CEFR Level vs Average Complexity Score", fontsize=14, fontweight="bold")
    ax.set_xlabel("CEFR Level", fontsize=12)
    ax.set_ylabel("Average Complexity Score", fontsize=12)
    max_score: float = max(scores)
    ax.set_ylim(0, max_score + 1)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("graph1_complexity_score.png", dpi=150)
    plt.show()


def plot_depth(cefr_data: dict) -> None:
    """A bar chart showing average tree depth per CEFR level."""
    levels = list(cefr_data.keys())
    depths: list[float] = [cefr_data[lvl]["avg_depth"] for lvl in levels]
    _, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(levels, depths, color=_COLORS[:len(levels)], edgecolor="black", width=0.5)
    for depth_bar, depth in zip(bars, depths):
        ax.text(depth_bar.get_x() + depth_bar.get_width() / 2,
                depth_bar.get_height() + 0.1,
                f"{depth:.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("CEFR Level vs Average Parse Tree Depth", fontsize=14, fontweight="bold")
    ax.set_xlabel("CEFR Level", fontsize=12)
    ax.set_ylabel("Average Depth", fontsize=12)
    max_depth: float = max(depths)
    ax.set_ylim(0, max_depth + 1.5)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("graph2_depth.png", dpi=150)
    plt.show()


def plot_clause_count(cefr_data: dict) -> None:
    """A bar chart showing average clause count per CEFR level."""
    levels = list(cefr_data.keys())
    clauses: list[float] = [cefr_data[lvl]["avg_clauses"] for lvl in levels]
    _, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(levels, clauses, color=_COLORS[:len(levels)], edgecolor="black", width=0.5)
    for clause_bar, clause in zip(bars, clauses):
        ax.text(clause_bar.get_x() + clause_bar.get_width() / 2,
                clause_bar.get_height() + 0.05,
                f"{clause:.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("CEFR Level vs Average Clause Count", fontsize=14, fontweight="bold")
    ax.set_xlabel("CEFR Level", fontsize=12)
    ax.set_ylabel("Average Clause Count", fontsize=12)
    max_clause: float = max(clauses)
    ax.set_ylim(0, max_clause + 1)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("graph3_clause_count.png", dpi=150)
    plt.show()


def plot_score_distribution(cefr_data: dict) -> None:
    """A histogram showing score distribution across all CEFR levels."""
    levels = list(cefr_data.keys())
    _, ax = plt.subplots(figsize=(10, 6))
    for i, level in enumerate(levels):
        scores = cefr_data[level]["scores"]
        ax.hist(scores, bins=5, alpha=0.6, label=level,
                color=_COLORS[i], edgecolor="black")
    ax.set_title("Score Distribution Across CEFR Levels", fontsize=14, fontweight="bold")
    ax.set_xlabel("Complexity Score", fontsize=12)
    ax.set_ylabel("Number of Sentences", fontsize=12)
    ax.legend(title="CEFR Level", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("graph4_distribution.png", dpi=150)
    plt.show()


def print_results_table(cefr_data: dict) -> None:
    """Print a formatted summary table of all CEFR level metrics."""
    levels = list(cefr_data.keys())
    print("\n" + "═" * 62)
    print(f"  {'Level':<8} {'Avg Depth':<14} {'Avg Clauses':<16} {'Avg Score':<10}")
    print("─" * 62)
    for level in levels:
        d = cefr_data[level]
        print(f"  {level:<8} {d['avg_depth']:<14} {d['avg_clauses']:<16} {d['avg_score']:<10}")
    print("═" * 62 + "\n")


if __name__ == "__main__":
    from data_analysis import compute_averages
    cefr_data_cleaned = compute_averages("small_data_cleaned.csv")
    print_results_table(cefr_data_cleaned)
    plot_complexity_score(cefr_data_cleaned)
    plot_depth(cefr_data_cleaned)
    plot_clause_count(cefr_data_cleaned)
    plot_score_distribution(cefr_data_cleaned)
    print("All graphs are now saved as PNG files!!")

    doctest.testmod(verbose=True)

    python_ta.check_all(config={
        'extra-imports': ['matplotlib.pyplot', 'data_analysis'],
        'allowed-io': ['print_results_table'],
        'max-line-length': 120
    })
