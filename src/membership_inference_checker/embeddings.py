import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import logging
from tqdm import tqdm

def load_preprocessed_data(preprocessed_file, preprocess_if_missing=True):
    """
    Load preprocessed data from a CSV file. If the file does not exist and
    preprocess_if_missing is True, call the preprocessor module to generate the data.
    
    Args:
        preprocessed_file (str): Path to the preprocessed CSV file.
        preprocess_if_missing (bool): Whether to run preprocessing if file is missing.
        
    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    if os.path.exists(preprocessed_file):
        logging.info("Loading preprocessed data from: %s", preprocessed_file)
        df = pd.read_csv(preprocessed_file, on_bad_lines='skip', engine='python')
        return df
    else:
        if preprocess_if_missing:
            logging.info("Preprocessed data not found. Running preprocessor module...")
            # Import the preprocessing function from the preprocessor module
            from ..preprocessor.preprocessor_main import preprocess_dataset
            # Create a dummy arguments object with default values
            class DummyArgs:
                output_dir = "../../data"
                max_bytes = 25 * 1024 * 1024 * 1024  # 25GB
                segment_mode = "sentence"
                segment_limit = None
                remove_stopwords = False
            dummy_args = DummyArgs()
            df = preprocess_dataset(dummy_args)
            os.makedirs(os.path.dirname(preprocessed_file), exist_ok=True)
            df.to_csv(preprocessed_file, index=False)
            logging.info("Preprocessed data generated and saved to: %s", preprocessed_file)
            return df
        else:
            raise FileNotFoundError("Preprocessed data not found. Please run the preprocessor module first.")

def compute_embeddings_for_segments(df, text_column='segments', model_name="all-MiniLM-L6-v2", batch_size=32, embeddings_file=None):
    """
    Compute or load embeddings for each text segment using SentenceTransformer.
    
    Args:
        df (pd.DataFrame): DataFrame containing text segments.
        text_column (str): Column name for segments (default: 'segments').
        model_name (str): Model name for SentenceTransformer.
        batch_size (int): Batch size for encoding.
        embeddings_file (str): Optional file path to load/save embeddings.
        
    Returns:
        np.ndarray: Array of embeddings.
    """
    if embeddings_file is not None and os.path.exists(embeddings_file):
        logging.info("Loading precomputed embeddings from: %s", embeddings_file)
        embeddings = np.load(embeddings_file)
        return embeddings
    else:
        logging.info("Computing embeddings using model: %s", model_name)
        embedding_model = SentenceTransformer(model_name)
        texts = df[text_column].tolist()
        embeddings = embedding_model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=False)
        embeddings = np.array(embeddings)
        if embeddings_file is not None:
            os.makedirs(os.path.dirname(embeddings_file), exist_ok=True)
            np.save(embeddings_file, embeddings)
            logging.info("Embeddings computed and saved to: %s", embeddings_file)
        return embeddings