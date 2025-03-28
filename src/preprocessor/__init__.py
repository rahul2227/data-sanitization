# __init__.py

"""
Preprocessor module for Data Sanitization Pipeline

This package contains modules for cleaning, tokenizing, deduplicating,
and segmenting raw text data into a preprocessed form suitable for further analysis.
Modules:
    - cleaning: Text cleaning and normalization functions.
    - tokenization: Functions for tokenizing text using Hugging Face's tokenizer.
    - deduplication: Functions for removing duplicate text entries.
    - segmentation: Functions for segmenting text into sentences or fixed token chunks.

The main executable script is located in `preprocessor_main.py`, which orchestrates
the complete data preprocessing workflow.

I have decided to implement this module in a modular manner, so that if we decide to change any module
it all doesn't go haywire.
"""

__version__ = "0.1.0"