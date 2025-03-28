import numpy as np
from sklearn.neighbors import NearestNeighbors
import logging

def compute_neighborhood_similarity(embeddings, n_neighbors=6):
    """
    Compute the cosine similarity for each embedding with its nearest neighbors.
    
    Args:
        embeddings (np.ndarray): Array of embeddings.
        n_neighbors (int): Number of neighbors (default: 6).
        
    Returns:
        np.ndarray: Array of maximum neighbor similarity for each segment.
    """
    logging.info("Fitting NearestNeighbors model with %d neighbors.", n_neighbors)
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    nn_model.fit(embeddings)
    distances, _ = nn_model.kneighbors(embeddings)
    # Convert cosine distance to similarity
    similarities = 1 - distances
    max_neighbor_sim = []
    for sim in similarities:
        # Exclude self (first neighbor with similarity 1)
        max_sim = np.max(sim[1:]) if len(sim) > 1 else sim[0]
        max_neighbor_sim.append(max_sim)
    return np.array(max_neighbor_sim)

def flag_membership(embeddings, high_sim_threshold=0.95, low_sim_threshold=0.3, n_neighbors=6):
    """
    Flag segments based on neighborhood similarity. A segment is flagged as a duplicate
    if its maximum similarity (excluding self) is >= high_sim_threshold, and flagged as an outlier
    if its maximum similarity is below low_sim_threshold.
    
    Args:
        embeddings (np.ndarray): Array of embeddings.
        high_sim_threshold (float): Threshold for duplicate flag (default: 0.95).
        low_sim_threshold (float): Threshold for outlier flag (default: 0.3).
        n_neighbors (int): Number of nearest neighbors (default: 6).
        
    Returns:
        tuple: (duplicate_flags, outlier_flags, max_neighbor_sim)
    """
    max_neighbor_sim = compute_neighborhood_similarity(embeddings, n_neighbors=n_neighbors)
    duplicate_flags = max_neighbor_sim >= high_sim_threshold
    outlier_flags = max_neighbor_sim < low_sim_threshold
    return duplicate_flags, outlier_flags, max_neighbor_sim