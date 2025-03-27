"""
Membership Inference Attack

This module performs a membership inference attack using similarity scores.
"""

import numpy as np

def is_member(similarity_score, threshold=0.85):
    """
    Determines if a text sample was likely part of the training data.

    Args:
        similarity_score (float): Cosine similarity score.
        threshold (float): Decision threshold for membership.

    Returns:
        bool: True if the sample is inferred as a member, False otherwise.
    """
    return similarity_score >= threshold

def membership_inference(nn_model, query_embedding):
    """
    Conducts a membership inference attack by checking similarity scores.

    Args:
        nn_model (NearestNeighbors): Trained nearest neighbors model.
        query_embedding (np.array): Embedding of the query text.

    Returns:
        bool: Membership inference result.
    """
    distances, _ = find_nearest_neighbors(nn_model, query_embedding)
    similarity_score = 1 - distances[0]  # Cosine similarity (1 - distance)
    return is_member(similarity_score)
