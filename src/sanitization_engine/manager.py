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
    logging.info("Starting Data Preprocessing Pipeline")
    # Step 1: Preprocessing
    if args.use_default_raw_data:
        run_command([
            sys.executable, "src/preprocessor/preprocessor_main.py",
            "--output-dir", "data",
            # "--segment-limit", "189700",
            "--segment-limit", "3414", # for testing purposes
            # "--output-filename", "preprocessed_wikitext103_subset.csv",
            "--output-filename", "preprocessed_wikitext103_subset_3414.csv",
            "--remove-stopwords"
        ])
    elif args.raw_data_path is not None:
        run_command([
            sys.executable, "src/preprocessor/preprocessor_main.py",
            "--output-dir", "data",
            "--input-path", args.raw_data_path,
            # "--segment-limit", "189700",
            "--segment-limit", "3414", # for testing purposes
            # "--output-filename", "preprocessed_wikitext103_subset.csv",
            "--output-filename", "preprocessed_wikitext103_subset_3414.csv",
            "--remove-stopwords"
        ])
    logging.info("Starting Contamination Detection Pipeline")
    # Step 2: Contamination Detection
    run_command([
        sys.executable, "src/contamination_detector/detector.py",
        # "--input-file", "data/preprocessed_wikitext103_subset.csv",
        "--input-file", "data/preprocessed_wikitext103_subset_3414.csv", # for testing
        # "--output-file", "data/contamination_flags.csv"
        "--output-file", "data/contamination_flags_3414.csv",
        "--ref_similarity_threshold", "0.9",
        "--perplexity_ratio_threshold", "0.8"
    ])

    logging.info("Starting Membership Inference Pipeline")
    # Step 3: Membership Inference Checker
    run_command([
        sys.executable, "src/membership_inference_checker/main.py",
        # "--input-file", "data/preprocessed_wikitext103_subset.csv",
        # "--output-file", "data/membership_inference_flags.csv"
        "--input-file", "data/preprocessed_wikitext103_subset_3414.csv",
        "--output-file", "data/membership_inference_flags_3414.csv",
        "--high-sim-threshold", "0.95",
        "--low-sim-threshold", "0.3"
    ])

    logging.info("Pipeline Modules Completed Successfully.")