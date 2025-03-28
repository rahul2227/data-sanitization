#!/usr/bin/env python3

import argparse
import logging
import os
import time
import pandas as pd
import logging
import subprocess
import sys

def run_command(command):
    logging.info(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"Error running {' '.join(command)}: {result.stderr}")
        sys.exit(result.returncode)
    else:
        logging.info(result.stdout)

def run_full_pipeline(args):
    """
    Runs the entire pipeline sequentially by invoking the other modules via subprocess.
    """

    logging.info("Starting Full Data Sanitization Pipeline")

    # Step 1: Preprocessing
    if args.use_default_raw_data:
        run_command([
            sys.executable, "src/preprocessor/preprocessor_main.py",
            "--output-dir", "data",
            "--remove-stopwords",
        ])
    elif args.raw_data_path is not None:
        run_command([
            sys.executable, "src/preprocessor/preprocessor_main.py",
            "--output-dir", "data",
            "--input-path", args.raw_data_path,
            "--remove-stopwords",
        ])



def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Full Data Sanitization Engine")
    parser.add_argument("--full-pipeline", action="store_true",
                        help="Run entire preprocessing, contamination detection, and membership checking pipeline before sanitization.")
    parser.add_argument("--raw-data-path", type=str, default=None,
                        help="Raw input data for preprocessing.")
    parser.add_argument("--use-default-raw-data", action="store_true",
                        help="Use the default raw data for preprocessing.")
    parser.add_argument("--sanitization-action", choices=["remove", "anonymize", "rewrite"],
                        default="remove", help="Sanitization action to perform.")
    parser.add_argument("--sanitized-output", type=str, default="../data/sanitized_dataset.csv",
                        help="Path to save the sanitized dataset.")
    parser.add_argument("--sanitization-log", type=str, default="../data/sanitization_log.csv",
                        help="Path to save detailed sanitization logs.")
    args = parser.parse_args()

    start_time = time.perf_counter()

    if args.full_pipeline:
        run_full_pipeline(args)

    logging.info("Starting Data Sanitization step.")

    # Load necessary data
    preprocessed_path = "../data/preprocessed_wikitext103_subset.csv"
    contamination_path = "../data/contamination_flags.csv"
    membership_path = "../data/membership_inference_flags.csv"

    df_preprocessed = pd.read_csv(preprocessed_path)
    df_contamination = pd.read_csv(contamination_path)
    df_membership = pd.read_csv(membership_path)



    end_time = time.perf_counter()
    logging.info(f"Sanitization pipeline completed in {end_time - start_time:.2f}s")

if __name__ == "__main__":
    main()