"""
Embedding Generator

This module loads a pre-trained SentenceTransformer model to convert text data into embeddings.
"""

from sentence_transformers import SentenceTransformer

# Load the SentenceTransformer model
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

def generate_embeddings(text_list):
    """
    Convert a list of text strings into embeddings using SentenceTransformer.

    Args:
        text_list (list): List of text strings.

    Returns:
        np.array: Numpy array of text embeddings.
    """
    return model.encode(text_list, convert_to_numpy=True)
