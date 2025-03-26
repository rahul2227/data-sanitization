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