import asyncio
import os
import re

import numpy as np
import openai
import pandas as pd
from scipy.spatial.distance import cdist  # type: ignore
from sklearn.cluster import KMeans  # type: ignore

from classifier.settings import get_settings


_settings = get_settings()
_openai_client = openai.OpenAI(api_key=_settings.OPEN_AI_API_KEY)


def get_embeddings(texts: list[str]) -> np.ndarray:
    """
    Calculate embeddings for a list of texts using OpenAI's embedding model.

    Args:
        texts (list[str]): A list of texts to calculate embeddings for.

    Returns:
        np.ndarray: An array of embeddings.
    """
    embeddings_input = [re.sub(r"\s+", " ", s).strip().lower() for s in texts]
    response = _openai_client.embeddings.create(
        input=embeddings_input,
        model=_settings.OPEN_AI_MODEL_EMBEDDINGS,
        dimensions=_settings.CLASSIFIER_EMBEDDINGS_DIMENSIONS,
    )
    return np.array([embedding.embedding for embedding in response.data])


def get_representative_samples(
    df: pd.DataFrame,
    topic_number: int,
    n_samples: int = _settings.NUMBER_OF_UTTERANCES_PER_CLASS,
):
    """
    Find the most representative samples for a given topic using K-means clustering.

    Args:
        df (pd.DataFrame): The DataFrame containing the text data and embeddings.
        topic_number (int): The topic number to find representative samples for.
        n_samples (int, optional): The number of representative samples to find. Defaults to settings.NUMBER_OF_UTTERANCES_PER_CLASS.

    Returns:
        pd.DataFrame: A DataFrame containing the most representative samples for the topic.
    """
    topic_df = df[df["topic"] == topic_number]
    embeddings = np.array(topic_df["embeddings"].tolist())

    # Use KMeans to find cluster centers
    kmeans = KMeans(n_clusters=min(n_samples, len(embeddings)), random_state=0).fit(
        embeddings
    )
    centers = kmeans.cluster_centers_

    # Find the closest point in the dataset to each center
    representative_indices = np.argmin(cdist(embeddings, centers), axis=0)
    representative_samples = topic_df.iloc[representative_indices]

    return representative_samples


async def run():
    """
    Main function to process text data, calculate embeddings, and find representative samples.
    """
    file_name = "files/text_topics.xlsx"
    embeddings_file_name = "files/embeddings.npy"
    semantic_router_dir = os.path.join("backend", "semantic_router")
    file_path = os.path.join(semantic_router_dir, file_name)
    embeddings_file_path = os.path.join(semantic_router_dir, embeddings_file_name)

    df = pd.read_excel(file_path)
    # Check if the embeddings file already exists
    if os.path.exists(embeddings_file_path):
        df["embeddings"] = list(np.load(embeddings_file_path))
    else:
        # Process the DataFrame in batches and store embeddings
        batch_size = _settings.ENCODER_BATCH_SIZE
        embeddings_list = []

        for i in range(0, len(df), batch_size):
            batch_texts = df["clean_text"][i : i + batch_size].tolist()
            batch_embeddings = get_embeddings(batch_texts)
            embeddings_list.append(batch_embeddings)

        # Concatenate all batch embeddings and add them to the DataFrame
        all_embeddings = np.vstack(embeddings_list)
        df["embeddings"] = list(all_embeddings)

        # Save the embeddings for future use
        np.save(os.path.join(embeddings_file_path), all_embeddings)

    # Create a new DataFrame for the updated data
    updated_df = pd.DataFrame()

    for topic in df["topic"].unique():
        representative_samples = get_representative_samples(df, topic)
        updated_df = pd.concat([updated_df, representative_samples])

    # Save the updated data to a new Excel file without embeddings
    final_df = updated_df.drop(columns=["embeddings"])
    final_df.to_excel(
        os.path.join(semantic_router_dir, "files/text_topics_optimized.xlsx"),
        index=False,
    )


if __name__ == "__main__":
    asyncio.run(run())
