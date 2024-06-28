import logging
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """App settings"""

    model_config = SettingsConfigDict(env_file=".env")

    OPEN_AI_API_KEY: str
    OPEN_AI_MODEL_EMBEDDINGS: str = "text-embedding-3-large"

    ENCODER_BATCH_SIZE: int = 1000
    NUMBER_OF_UTTERANCES_PER_CLASS: int = 300
    CLASSIFIER_SCORE_THRESHOLD: float = 0.1
    CLASSIFIER_EMBEDDINGS_DIMENSIONS: int = 3072


@lru_cache()
def get_settings() -> Settings:
    return Settings()  # type: ignore
