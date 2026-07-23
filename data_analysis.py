"""
CSC111 Project 2: Tree-Based Sentence Complexity Analyzer
Module: data_analysis.py
Description: Loads and analyzes the EFCAMDAT learner corpus, computing syntactic
complexity metrics grouped by CEFR proficiency level.
Authors: Tugra Canbaz, Elif Cakici, Alsade Brianna Daley
"""

from __future__ import annotations
import doctest
import nltk
import pandas as pd
import python_ta
from nltk.tokenize import sent_tokenize
from parse_tree import get_depth, count_clauses, branching_factor, sentence_to_parsetree
from feedback import compute_score


def _download_nltk_resources() -> None:
    """Downloads required NLTK resources if not already present."""
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)


def _analyze_database(file: str) -> dict:
    """Analyze a dataset of learner texts and compute syntactic metrics grouped by CEFR level.

    Reads a CSV file containing learner-written texts and their associated CEFR proficiency
    levels. Each text is split into sentences and parsed into a ParseTree. For each sentence,
    syntactic metrics are computed including tree depth, clause count, branching factor,
    and a combined complexity score. Results are grouped by CEFR level.

    Preconditions:
        - file is a valid path to a readable CSV file
        - The CSV contains columns named 'text' and 'cefr'
    """
    _download_nltk_resources()
    df = pd.read_csv(file)
    results: dict = {}

    for _, row in df.iterrows():
        cefr_level = row["cefr"]
        text = row["text"]
        split_sentence_list = sent_tokenize(str(text))

        for sentence in split_sentence_list:
            parse_tree = sentence_to_parsetree(sentence)
            depth = get_depth(parse_tree)
            clause_count = count_clauses(parse_tree)
            branch = branching_factor(parse_tree)
            complexity_score = compute_score(depth, clause_count, branch)

            if cefr_level not in results:
                results[cefr_level] = {
                    "depth": [],
                    "clauses": [],
                    "branching": [],
                    "scores": []
                }

            results[cefr_level]["depth"].append(depth)
            results[cefr_level]["clauses"].append(clause_count)
            results[cefr_level]["branching"].append(branch)
            results[cefr_level]["scores"].append(complexity_score)

    return results


def compute_averages(file: str) -> dict:
    """Compute average syntactic metrics for each CEFR level from a CSV dataset.

    Returns a dictionary mapping each CEFR level to average depth, clauses,
    branching factor, complexity score, and the full list of scores.

    Preconditions:
        - file is a valid path to a readable CSV file
        - The CSV contains columns named 'text' and 'cefr'

    >>> isinstance(compute_averages("small_data_cleaned.csv"), dict)
    True
    """
    results = _analyze_database(file)
    averages_dict: dict = {}

    for level in results:
        level_data = results[level]

        if len(level_data["scores"]) != 0:
            averages_dict[level] = {
                "avg_depth": round(sum(level_data["depth"]) / len(level_data["depth"]), 1),
                "avg_clauses": round(sum(level_data["clauses"]) / len(level_data["clauses"]), 1),
                "avg_branching": round(sum(level_data["branching"]) / len(level_data["branching"]), 1),
                "avg_score": round(sum(level_data["scores"]) / len(level_data["scores"]), 1),
                "scores": level_data["scores"]
            }
        else:
            averages_dict[level] = {
                "avg_depth": 0.0,
                "avg_clauses": 0.0,
                "avg_branching": 0.0,
                "avg_score": 0.0,
                "scores": []
            }

    return averages_dict


if __name__ == "__main__":
    doctest.testmod(verbose=True)

    python_ta.check_all(config={
        'extra-imports': ['nltk', 'nltk.tokenize', 'pandas', 'parse_tree', 'feedback'],
        'allowed-io': ['_download_nltk_resources'],
        'max-line-length': 120
    })
