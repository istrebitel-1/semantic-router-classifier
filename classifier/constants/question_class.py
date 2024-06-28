from classifier.constants.base import AppStringEnum


class QuestionClass(AppStringEnum):
    """Enum class for question classes"""

    COMMON = "common-issues"
    SWITCH_TO_OPERATOR = "switch-to-operator"
    CAR_AVAILABILITY = "car-availability"
