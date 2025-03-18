The Starting of the project worked with the Minimal Viable Products of each of the pipelines of the Data Sanitization project. _This document is for documenting my approach and the progress of the sanitization pipeline, which will also help me highlight challenges faced during the project and how I implemented solutions for those._

# Project Overview
The Project is currently divided into 4 modules as follows:
1. **Data Preprocessor:** The data preprocessing module cleans and prepares raw text data for analysis and model training.
2. **Contamination Detector:** This module identifies portions of the training data that should not be there, such as leaked evaluation benchmarks or any data that overlaps with a reference set.
3. **Membership Inference Checker:** It checks the dataset for indications that certain data might be over-represented or memorized, using a neighborhood-based membership inference approach. Instead of _attacking_ a trained model, we preemptively analyze the training data to find risky segments.
4. **Sanitization Engine:** consolidates all flags from the contamination detector and membership inference checker, and then cleans the data accordingly. It ensures that by the end of the pipeline, the training data is sanitized – free of known contamination, duplicates, and high-risk memorization content.

My first approach is to develop an MVP for each of the modules starting with data preprocessor and then add complexity in the project.
