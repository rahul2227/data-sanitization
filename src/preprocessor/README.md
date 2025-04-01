## Preprocessor module for Data Sanitization Pipeline

**This module prepares raw data for further analysis and model training**

The module downloads and works on the opensource data:- [wikitext-103-raw-v1 dataset from huggingface](https://huggingface.co/datasets/iohadrubin/wikitext-103-raw-v1).

### Module Functionality

#### **Text cleaning and Normalization:** 
Remove noise like HTML tags, non-UTF characters and extra whitespace. Normalize text by lowercasing and stripping access 
and apply standard Unicode normalization for consistency

1. **Contamination Simulation:** We simulate the contamination of the dataset artificially.

2. **Tokenization:** Produces tokenized representation for each segment.

3. **Deduplication:** Identify and remove duplicate or near duplicate entries via fuzzy matching (e.g., hashing or embedding similarity)

4. **Data Segmentation:** Split the cleaned corpus into logical segments for analysis(e.g., sentences, paragraphs, or document chunks).

    **_In our case we mostly work with sentence mode_**