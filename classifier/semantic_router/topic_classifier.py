import logging

from semantic_router import Route
from semantic_router.encoders import OpenAIEncoder
from semantic_router.layer import RouteLayer

from classifier.constants import QuestionClass, map_label_to_question_class
from classifier.semantic_router.topics import Topic, get_topics_data
from classifier.settings import get_settings


logger = logging.getLogger(__name__)


class Classifier:
    """
    Classifier class that uses semantic routing to classify text into topics.

    The classifier is built using a RouteLayer from the semantic_router package,
    which is initialized with routes based on topic data and an encoder.
    """

    def __init__(self):
        """
        Initializes the Classifier instance by loading settings, creating routes,
        and initializing the RouteLayer classifier.
        """
        self._settings = get_settings()
        self._classifier = self._create_classifier()

    def _create_routes(self) -> list[Route]:
        """
        Creates routes for the classifier based on topic data.

        Returns:
            list[Route]: A list of Route objects for the classifier.
        """
        topics_data: list[Topic] = get_topics_data()
        routes: list[Route] = []
        for topic_data in topics_data:
            topic: str = topic_data.topic
            utterances: list[str] = topic_data.clean_texts

            # Limit the number of utterances per class based on settings
            original_count = len(utterances)
            utterances = utterances[: self._settings.NUMBER_OF_UTTERANCES_PER_CLASS]
            if original_count > self._settings.NUMBER_OF_UTTERANCES_PER_CLASS:
                logger.info(
                    "Topic '%s' has %d utterances, limiting to %d.",
                    topic,
                    original_count,
                    self._settings.NUMBER_OF_UTTERANCES_PER_CLASS,
                )

            route = Route(name=topic, utterances=utterances)
            routes.append(route)
        return routes

    def _create_classifier(self) -> RouteLayer:
        """
        Creates the RouteLayer classifier with routes and an encoder.

        Returns:
            RouteLayer: The initialized RouteLayer classifier.
        """
        routes = self._create_routes()
        encoder = OpenAIEncoder(
            openai_api_key=self._settings.OPEN_AI_API_KEY,
            name=self._settings.OPEN_AI_MODEL_EMBEDDINGS,
            score_threshold=self._settings.CLASSIFIER_SCORE_THRESHOLD,
            dimensions=self._settings.CLASSIFIER_EMBEDDINGS_DIMENSIONS,
        )
        return RouteLayer(encoder=encoder, routes=routes)

    def classify_text(self, text: str) -> QuestionClass:
        """
        Classifies a given text into a topic.

        Args:
            text (str): The text to classify.

        Returns:
            Optional[str]: The name of the topic the text is classified into, or None if no match is found.
        """
        label = self._classifier(text.lower()).name
        # Map the label to the QuestionClass enum
        question_class = map_label_to_question_class(str(label))
        return question_class
