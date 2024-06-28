import re
from pathlib import Path

import pandas as pd
from pydantic import BaseModel


FILE_NAME = "files/text_topics_optimized.xlsx"


class Topic(BaseModel):
    """
    Model representing a topic with associated clean texts.

    Attributes:
        topic (str): The name of the topic.
        clean_texts (list[str]): A list of cleaned text strings associated with the topic.
    """

    topic: str
    clean_texts: list[str]


def get_topics_data() -> list[Topic]:
    """
    Reads an Excel file containing text topics and returns a list of Topic instances.

    The Excel file is expected to have columns for clean text and topic names.
    This function processes the file, cleans up the text data, and groups texts by topic.

    Returns:
        list[Topic]: A list of Topic instances with clean texts grouped by topic.
    """
    file_path = Path(__file__).parent.resolve() / FILE_NAME
    topics_df = pd.read_excel(file_path)

    topics_dict: dict[str, Topic] = {}

    for clean_text, _, topic in topics_df.itertuples(index=False):
        if topic not in topics_dict:
            topics_dict[topic] = Topic(topic=str(topic), clean_texts=[])

        clean_text = re.sub(r"\s+", " ", str(clean_text)).strip().lower()
        topics_dict[topic].clean_texts.append(clean_text)

    return list(topics_dict.values())
