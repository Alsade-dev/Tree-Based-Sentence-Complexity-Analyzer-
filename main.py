"""
CSC111 Project 2: Tree-Based Sentence Complexity Analyzer
Module: main.py
Description: Entry point for the program. Runs the sentence analysis pipeline
and displays visualizations of syntactic complexity across CEFR levels.
Authors: Tugra Canbaz, Elif Cakici, Alsade Brianna Daley
"""
from __future__ import annotations
from feedback import analyze_sentence
from data_analysis import compute_averages
from visualization import (print_results_table, plot_complexity_score,plot_depth, plot_clause_count, plot_score_distribution)


def run_test_cases() -> None:
    """Run the analyzer on a set of sample sentences across difficulty levels."""
    test_sentences = [
        "I like school.",
        "She reads every night.",
        "Although it was late, she finished her homework.",
        "The results, which were published last year, suggest a strong correlation.",
    ]
    print("\n[ RUNNING TEST CASES ]\n")
    for sentence in test_sentences:
        analyze_sentence(sentence)


def run_interactive() -> None:
    """Prompt the user to enter a sentence and display its analysis."""
    print("─" * 55)
    print("  INTERACTIVE MODE — Enter your own sentence")
    print("─" * 55)
    user_sentence = input("  Enter a sentence (or press Enter to skip): ").strip()
    if user_sentence:
        analyze_sentence(user_sentence)
    else:
        print("  Skipping interactive mode.\n")


def run_visualizations(cefr_data: dict) -> None:
    """Display all CEFR complexity visualizations and print the results table."""
    print_results_table(cefr_data)
    plot_complexity_score(cefr_data)
    plot_depth(cefr_data)
    plot_clause_count(cefr_data)
    plot_score_distribution(cefr_data)
    print("All graphs saved as PNG files!")


if __name__ == "__main__":
    print("\n" + "█" * 55)
    print("  CSC111 — Sentence Complexity Analyzer")
    print("█" * 55)

    run_test_cases()
    run_interactive()

    data = compute_averages("small_data_cleaned.csv")
    run_visualizations(data)
