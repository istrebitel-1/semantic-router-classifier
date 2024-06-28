from semantic_router.encoders import OpenAIEncoder
from semantic_router.utils.logger import logger

from classifier.settings import Settings, get_settings


class OpenAIBatchEncoder(OpenAIEncoder):
    """
    Encoder class that extends OpenAIEncoder to process documents in batches.

    This class is responsible for encoding a list of documents using OpenAI's API
    and handling them in batches as defined by the ENCODER_BATCH_SIZE setting.
    """

    _settings: Settings = get_settings()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, docs: list[str], truncate: bool = True) -> list[list[float]]:
        """
        Encodes a list of documents into embeddings.

        Args:
            docs: A list of documents to be encoded.

        Returns:
            A list of embeddings, where each embedding is a list of floats.

        Raises:
            ValueError: If the OpenAI client is not initialized.
        """
        if self.client is None:
            raise ValueError("OpenAI client is not initialized.")
        result: list[list[float]] = []

        # Create batches of documents based on the ENCODER_BATCH_SIZE setting
        batches: list[list[str]] = [
            docs[i : i + self._settings.ENCODER_BATCH_SIZE]
            for i in range(0, len(docs), self._settings.ENCODER_BATCH_SIZE)
        ]

        logger.info("Processing %d documents in %d batches.", len(docs), len(batches))

        # Process each batch
        for batch in batches:
            try:
                batch_embeddings = super().__call__(batch, truncate)
            except Exception as e:
                logger.error(
                    "OpenAI API call failed. Error: %s. Failed embeddings: %s",
                    e,
                    batch,
                )
                batch_embeddings = []
            result.extend(batch_embeddings)
            logger.info(
                "Processed %d documents. %d remaining.",
                len(result),
                len(docs) - len(result),
            )

        return result
