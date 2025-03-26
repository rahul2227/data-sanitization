# data-sanitization
Project on the data contamination seminar - Goes deeper into data preprocessing pipeline so that LLM's are much more unbiased.

## Data Sanitization Pipeline

### Overview
The **Data Sanitization Pipeline** is a modular software project designed for cleaning training data for language models. It integrates four key modules:
- **Data Preprocessor:** Cleans and normalizes raw text data.
- **Contamination Detector:** Identifies unwanted or benchmark-contaminated data.
- **Membership Inference Checker:** Flags data at risk of memorization.
- **Sanitization Engine:** Aggregates flags and applies cleaning actions (removal, anonymization, rewriting).

### Directory Structure
```tree
data sanitization/
├── data/
├── documentation/
├── exploratory notebooks/
└── src/
    ├── preprocessor/
    │   └── __init__.py
    ├── contamination/
    │   ├── __init__.py
    │   └── utils/
    │       └── __init__.py
    ├── membership Inference checker/
    │   └── __init__.py
    ├── sanitization engine/
    │   └── __init__.py
    └── main.py
```

### Installation
1. Clone the repository
```bash
git clone https://github.com/rahul2227/data-sanitization.git
cd data-sanitization
```

2. Install the packages
```bash
conda env create -f environment.yml 
```

## Data Preprocessor module

This module downloads and works on the opensource data opensource [wikitext-103-raw-v1 dataset from huggingface](https://huggingface.co/datasets/iohadrubin/wikitext-103-raw-v1).

The general idea is that this script performs data preprocessing on a raw text dataset by:
  - Loading a dataset (default: wikitext-103-raw-v1 from Hugging Face).
  - Capping the dataset size based on a maximum number of bytes.
  - Cleaning and normalizing text.
  - Tokenizing the cleaned text.
  - Removing duplicate entries.
  - Segmenting text into smaller units (sentences or fixed-length chunks).
  - Saving the preprocessed data as a CSV file.

To preprocess the data you need to go to the following directory
```bash
cd src/Preprocessor/
```

and run the following command for the execution of the preprocessing module. 

> **Note:** If you are following the same folder structure as above you just need to run the following command, else you need to enter the aruguements for data directories to not have 
> further problems

```bash
 python3 preprocessor_main.py --remove-stopwords
```
