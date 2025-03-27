"""
Data Loader

This module loads preprocessed text data from a CSV file.
"""

import os
import pandas as pd

def load_preprocessed_data(file_path):
    """
    Load preprocessed text data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the preprocessed text.
    """
    if os.path.exists(file_path):
        print(f"Loading preprocessed data from {file_path}...")
        return pd.read_csv(file_path, on_bad_lines='skip', usecols=['cleaned_text'])
    else:
        raise FileNotFoundError(f"Preprocessed data not found at {file_path}")
