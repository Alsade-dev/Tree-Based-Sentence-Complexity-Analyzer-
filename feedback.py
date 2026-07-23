"""
CSC111 Project 2: Tree-Based Sentence Complexity Analyzer
Module: feedback.py
Description: Contains functions for computing complexity scores, classifying CEFR levels,
generating personalized feedback, and analyzing sentence structure.
Authors: Tugra Canbaz, Elif Cakici, Alsade Brianna Daley
"""

from __future__ import annotations
import doctest
import nltk
import python_ta
from parse_tree import get_depth, count_clauses, branching_factor, sentence_to_parsetree


def compute_score(depth: int, clause_count: int, branch: float) -> float:
    """Computes a weighted complexity score from three syntactic features.

    Preconditions:
        - depth >= 0
        - clause_count >= 0
        - branch >= 0.0

    >>> compute_score(2, 1, 1.0)
    1.4
    >>> compute_score(5, 2, 2.1)
    3.23
    """
    score = (depth * 0.4) + (clause_count * 0.3) + (branch * 0.3)
    return round(score, 2)


def classify_level(score: float) -> str:
    """Maps a complexity score to an approximate CEFR proficiency level.

    Preconditions:
        - score >= 0.0

    >>> classify_level(1.4)
    'A1 – Beginner'
    >>> classify_level(3.5)
    'B1 – Intermediate'
    >>> classify_level(6.0)
    'C1/C2 – Advanced'
    """
    if score < 2:
        return "A1 – Beginner"
    elif score < 3:
        return "A2 – Elementary"
    elif score < 4:
        return "B1 – Intermediate"
    elif score < 5:
        return "B2 – Upper Intermediate"
    else:
        return "C1/C2 – Advanced"


# Personalized feedback functions
def generate_feedback(sentence: str, depth: int, clause_count: int, branch: float) -> tuple:
    """Generates specific, metric-tied feedback and suggestions personalized to the input sentence."""
    feedback = []
    suggestions = []
    base = sentence.rstrip(".").rstrip("!").rstrip("?")

    # Depth checks
    if depth <= 2:
        feedback.append(f"Your sentence has a very shallow structure (depth: {depth}).")
        suggestions.append(f'→ Add descriptive phrases: "{base}, especially the interesting parts."')
    elif depth <= 4:
        feedback.append(f"Your sentence has moderate depth (depth: {depth}) — room to grow.")
        suggestions.append(f'→ Try a relative clause: "{base}, which is something I truly enjoy."')

    # Clause count checks
    if clause_count <= 1:
        feedback.append("Your sentence has only 1 clause — it expresses just one idea.")
        suggestions.append(f'→ Add a reason:    "{base} because it is meaningful to me."')
        suggestions.append(f'→ Add a contrast:  "{base}, although it can be challenging."')
        suggestions.append(f'→ Add a condition: "{base} when I get the chance."')
    elif clause_count == 2:
        feedback.append(f"Good — your sentence has {clause_count} clauses, showing some complexity.")
        suggestions.append(f'→ Push further: "{base}, which shows how much I have grown."')
    else:
        feedback.append(f"Great clause variety! Your sentence uses {clause_count} clauses.")

    # Branching factor checks
    if branch < 1.5:
        feedback.append(f"Low branching ({branch:.1f}) — your sentence lacks descriptive detail.")
        suggestions.append(f'→ Add an adjective:          "{base} every single day."')
        suggestions.append(f'→ Add a prepositional phrase: "{base} in a meaningful way."')
    elif branch < 2.5:
        feedback.append(f"Moderate branching ({branch:.1f}) — decent phrase variety.")
        suggestions.append(f'→ Add a modifier: "{base} in ways I never expected."')
    else:
        feedback.append(f"Strong branching ({branch:.1f}) — your sentence is structurally rich.")

    return feedback, suggestions


def suggest_upgrade(sentence: str) -> list:
    """Generates upgraded versions of the input sentence by adding
    subordinate clauses that are grammatically neutral and context-safe."""
    base = sentence.rstrip(".").rstrip("!").rstrip("?")
    return [
        f"{base}, even though it was not easy.",
        f"{base}, which is something I have been thinking about.",
        f"{base}, although the situation could have been different.",
        f"Since {base.lower()}, I have learned a lot from it.",
    ]


def _download_nltk_resources() -> None:
    """Downloads required NLTK resources if not already present."""
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)


def analyze_sentence(sentence: str) -> None:
    """Full pipeline: raw sentence to ParseTree to metrics to score to level to feedback."""
    _download_nltk_resources()
    sentence = sentence.strip().capitalize()
    tree = sentence_to_parsetree(sentence)

    depth = get_depth(tree)
    clause_count = count_clauses(tree)
    branch = branching_factor(tree)

    score = compute_score(depth, clause_count, branch)
    level = classify_level(score)

    feedback, suggestions = generate_feedback(sentence, depth, clause_count, branch)
    upgrades = suggest_upgrade(sentence)

    print("\n" + "═" * 55)
    print("  SENTENCE ANALYSIS")
    print("═" * 55)
    print(f"  Input    : {sentence}")
    print(f"  Score    : {score}")
    print(f"  Level    : {level}")
    print(f"  Depth    : {depth}  |  Clauses: {clause_count}  |  Branching: {branch:.2f}")
    print("─" * 55)
    print("  FEEDBACK:")
    for point in feedback:
        print(f"    • {point}")
    print("─" * 55)
    print("  SUGGESTIONS:")
    for tip in suggestions:
        print(f"    {tip}")
    print("─" * 55)
    print("  UPGRADED VERSIONS:")
    for upgrade in upgrades[:2]:
        print(f'    → "{upgrade}"')
    print("═" * 55 + "\n")


if __name__ == "__main__":
    print("\n" + "█" * 55)
    print("  CSC111 — Sentence Complexity Analyzer")
    print("█" * 55)

    test_sentences = [
        "I like school.",
        "She reads every night.",
        "Although it was late, she finished her homework.",
        "The results, which were published last year, suggest a strong correlation.",
    ]

    print("\n[ RUNNING TEST CASES ]\n")
    for s in test_sentences:
        analyze_sentence(s)

    print("─" * 55)
    print("  INTERACTIVE MODE — Enter your own sentence")
    print("─" * 55)
    user_sentence = input("  Enter a sentence (or press Enter to skip): ").strip()
    if user_sentence:
        analyze_sentence(user_sentence)
    else:
        print("  Skipping interactive mode.\n")

    doctest.testmod(verbose=True)

    python_ta.check_all(config={
        'extra-imports': ['nltk', 'parse_tree'],
        'allowed-io': ['analyze_sentence', '_download_nltk_resources'],
        'max-line-length': 120
    })
