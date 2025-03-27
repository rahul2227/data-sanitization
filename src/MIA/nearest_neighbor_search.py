"""
Nearest Neighbor Search

This module implements a nearest neighbor search to find similar embeddings.
"""

from sklearn.neighbors import NearestNeighbors
import numpy as np

def build_nearest_neighbors(embeddings, n_neighbors=5):
    """
    Build a nearest neighbor model for similarity search.

    Args:
        embeddings (np.array): Array of text embeddings.
        n_neighbors (int): Number of nearest neighbors to retrieve.

    Returns:
        NearestNeighbors: Trained nearest neighbors model.
    """
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn_model.fit(embeddings)
    return nn_model

def find_nearest_neighbors(nn_model, query_embedding):
    """
    Find the nearest neighbors for a given query embedding.

    Args:
        nn_model (NearestNeighbors): Trained nearest neighbors model.
        query_embedding (np.array): Embedding of the query text.

    Returns:
        distances, indices: Distances and indices of nearest neighbors.
    """
    distances, indices = nn_model.kneighbors([query_embedding])
    return distances[0], indices[0]
