import os
from typing import Optional

import pandas as pd
import pytest

from classifier.constants import map_question_class_to_label
from classifier.semantic_router.topic_classifier import Classifier


MINIMUM_QUALITY_PERCENTAGE = 95


def load_csv(file_name: str, sep: str = ",") -> pd.DataFrame:
    """Helper function to load a CSV file."""
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, file_name)
    return pd.read_csv(file_path, sep=sep, encoding="utf-8")


def save_csv(df: pd.DataFrame, result_file_name: str) -> None:
    """Helper function to save a DataFrame to a CSV file."""
    current_dir = os.path.dirname(__file__)
    result_file_path = os.path.join(current_dir, result_file_name)
    df.to_csv(result_file_path, sep=",", encoding="utf-8", index=False)


def calculate_and_print_results(total: int, correct: int, incorrect: int) -> float:
    """Helper function to calculate and print the results."""
    print(f"Total: {total}, Correct: {correct}, Incorrect: {incorrect}")
    correct_answers_percentage = correct / total * 100
    print(f"Correct answers percentage: {correct_answers_percentage:.2f}%")
    return correct_answers_percentage


@pytest.mark.asyncio
async def test_topic_classifier(skip_assertion: bool = False):
    """Test of semantic router topic classifier"""
    df = load_csv("./topics.csv")
    total, correct, incorrect = 0, 0, 0
    classifier = Classifier()

    for idx, (question, topic) in enumerate(df.itertuples(index=False)):
        try:
            question_class: Optional[str] = map_question_class_to_label(
                classifier.classify_text(question)
            )
        except Exception:
            question_class = None

        print(f"{idx + 1}. {question} -> Expected {topic} -> Actual {question_class}")

        total += 1
        if str(topic) == question_class:
            correct += 1
        else:
            incorrect += 1
        df.at[idx, "Результат"] = question_class or "-"

    save_csv(df, "./topics_result.csv")
    correct_answers_percentage = calculate_and_print_results(total, correct, incorrect)

    if not skip_assertion:
        assert correct_answers_percentage >= MINIMUM_QUALITY_PERCENTAGE
