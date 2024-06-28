import logging

from classifier.constants.question_class import QuestionClass


MAP_LABEL_TO_QUESTION_CLASSES = {
    "1": QuestionClass.COMMON,
    "2": QuestionClass.CAR_AVAILABILITY,
}

_logger = logging.getLogger(__name__)


def map_label_to_question_class(label: str) -> QuestionClass:
    try:
        return MAP_LABEL_TO_QUESTION_CLASSES[label]
    except KeyError:
        _logger.error("Label '%s' is not a valid question class.", label)
        return QuestionClass.SWITCH_TO_OPERATOR


def map_question_class_to_label(question_class: QuestionClass) -> str:
    for label, qc in MAP_LABEL_TO_QUESTION_CLASSES.items():
        if qc == question_class:
            return label
    raise ValueError(f"Question class '{question_class}' is not a valid label.")
