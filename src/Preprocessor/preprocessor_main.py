#!/usr/bin/env python3
"""
Data Preprocessing Module

This script performs data preprocessing on a raw text dataset by:
  - Loading a dataset (default: wikitext-103-raw-v1 from Hugging Face).
  - Capping the dataset size based on a maximum number of bytes.
  - Cleaning and normalizing text.
  - Tokenizing the cleaned text.
  - Removing duplicate entries.
  - Segmenting text into smaller units (sentences or fixed-length chunks).
  - Saving the preprocessed data as a CSV file.

Usage:
    python main.py [--output-dir PATH] [--max-bytes BYTES]
                   [--segment-mode MODE] [--segment-limit N]
                   [--remove-stopwords]
"""

import os
import argparse
import pandas as pd
from datasets import load_dataset

from cleaning import normalize_text
from tokenization import tokenize_text
from deduplication import remove_duplicates
from segmentation import segment_dataframe

# Default configuration parameters.
DEFAULT_DATASET_NAME = "iohadrubin/wikitext-103-raw-v1"
DEFAULT_SPLIT = "train"
DEFAULT_MAX_BYTES = 25 * 1024 * 1024 * 1024  # 25GB in bytes
DEFAULT_SEGMENT_MODE = "sentence"
DEFAULT_OUTPUT_FILENAME = "preprocessed_wikitext103_subset.csv"


def cap_dataset_by_bytes(df, max_bytes):
    """
    Cap the dataset so that its cumulative byte size does not exceed max_bytes.
    """
    df['text_size'] = df['text'].apply(lambda x: len(x.encode('utf-8')))
    df['cum_size'] = df['text_size'].cumsum()
    df_capped = df[df['cum_size'] <= max_bytes].copy()
    df_capped.drop(columns=['text_size', 'cum_size'], inplace=True)
    return df_capped


def preprocess_dataset(args):
    print("Loading dataset...")
    dataset = load_dataset(DEFAULT_DATASET_NAME, split=DEFAULT_SPLIT)
    df = pd.DataFrame(dataset)
    print(f"Original dataset rows: {len(df)}")

    print("Capping dataset size...")
    df = cap_dataset_by_bytes(df, args.max_bytes)
    print(f"Rows after capping: {len(df)}")

    print("Normalizing text...")
    df['cleaned_text'] = df['text'].apply(lambda x: normalize_text(x, remove_stopwords=args.remove_stopwords))

    print("Tokenizing text...")
    df['tokens'] = df['cleaned_text'].apply(tokenize_text)

    print("Removing duplicate entries...")
    df = remove_duplicates(df, text_column='cleaned_text')
    print(f"Rows after deduplication: {len(df)}")

    print(f"Segmenting text using mode: {args.segment_mode}")
    df_segmented = segment_dataframe(df, text_column='cleaned_text', mode=args.segment_mode)

    if args.segment_limit:
        df_segmented = df_segmented.iloc[:args.segment_limit].copy()
        print(f"Segmented rows limited to first {args.segment_limit} rows.")
    else:
        print(f"Total segmented rows: {len(df_segmented)}")

    return df_segmented


def main():
    parser = argparse.ArgumentParser(description="Data Preprocessing Module")
    parser.add_argument("--output-dir", type=str, default="../data",
                        help="Directory to save the preprocessed CSV file (default: ../data)")
    parser.add_argument("--output-filename", type=str, default=DEFAULT_OUTPUT_FILENAME,
                        help=f"Output CSV filename (default: {DEFAULT_OUTPUT_FILENAME})")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES,
                        help="Maximum dataset size in bytes (default: 25GB)")
    parser.add_argument("--segment-mode", type=str, choices=["sentence", "fixed", "none"],
                        default=DEFAULT_SEGMENT_MODE, help="Segmentation mode (default: sentence)")
    parser.add_argument("--segment-limit", type=int, default=None,
                        help="Optional limit on the number of segmented rows to keep (default: all)")
    parser.add_argument("--remove-stopwords", action="store_true", default=False,
                        help="Optionally remove stopwords during normalization")
    args = parser.parse_args()

    df_processed = preprocess_dataset(args)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_filename)
    df_processed.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")


if __name__ == "__main__":
    main()