"""
Main Script for Membership Inference Attack

This script loads preprocessed data, generates embeddings, builds a nearest neighbor model,
and performs membership inference attacks.
"""

import os
from data_loader import load_preprocessed_data
from embedding_generator import generate_embeddings
from nearest_neighbor_search import build_nearest_neighbors
from membership_inference import membership_inference

# Define data path
DATA_PATH = "../data/preprocessed_wikitext103_subset.csv"

# Load dataset
df = load_preprocessed_data(DATA_PATH)

# Generate embeddings
print("Generating embeddings...")
embeddings = generate_embeddings(df['cleaned_text'].tolist())

# Build nearest neighbor search model
print("Building nearest neighbor model...")
nn_model = build_nearest_neighbors(embeddings)

# Perform membership inference on a sample query
query_text = "This is a sample sentence to check membership."
query_embedding = generate_embeddings([query_text])[0]

is_member = membership_inference(nn_model, query_embedding)
print(f"Membership Inference Result: {'Member' if is_member else 'Non-member'}")
