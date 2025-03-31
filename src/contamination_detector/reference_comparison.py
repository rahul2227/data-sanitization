import logging
from sentence_transformers import SentenceTransformer, util

DEFAULT_REF_MODEL_NAME = 'all-MiniLM-L6-v2'


def load_reference_data(reference_texts, model_name=DEFAULT_REF_MODEL_NAME):
    """
    Load a SentenceTransformer model and compute embeddings for a list of reference texts.

    Args:
        reference_texts (list): List of reference text strings.
        model_name (str): Model name for SentenceTransformer (default: all-MiniLM-L6-v2).

    Returns:
        (model, embeddings): Tuple containing the loaded model and computed embeddings.
    """
    logging.info("Loading reference model: %s", model_name)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(reference_texts, convert_to_tensor=True)
    return model, embeddings


def check_reference_similarity(segment, ref_embeddings, threshold=0.9, ref_model=None):
    """
    Compute the maximum cosine similarity between a segment and reference embeddings.

    Args:
        segment (str): Text segment to check.
        ref_embeddings (tensor): Precomputed reference embeddings.
        threshold (float): Similarity threshold (default: 0.9).
        ref_model: SentenceTransformer model to encode the segment.

    Returns:
        (max_sim, flag): Maximum similarity value and a flag indicating if it exceeds threshold.
    """
    if ref_model is None:
        raise ValueError("A reference model is required to encode the segment.")
    segment_embedding = ref_model.encode(segment, convert_to_tensor=True)
    cos_scores = util.cos_sim(segment_embedding, ref_embeddings)
    max_sim = cos_scores.max().item()
    flag = max_sim >= threshold
    return max_sim, flag