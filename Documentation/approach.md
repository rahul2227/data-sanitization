The Starting of the project worked with the Minimal Viable Products of each of the pipelines of the Data Sanitization project. _This document is for documenting my approach and the progress of the sanitization pipeline, which will also help me highlight challenges faced during the project and how I implemented solutions for those._

# Project Overview
The Project is currently divided into 4 modules as follows:
1. **Data Preprocessor:** The data preprocessing module cleans and prepares raw text data for analysis and model training.
2. **Contamination Detector:** This module identifies portions of the training data that should not be there, such as leaked evaluation benchmarks or any data that overlaps with a reference set.
3. **Membership Inference Checker:** It checks the dataset for indications that certain data might be over-represented or memorized, using a neighborhood-based membership inference approach. Instead of _attacking_ a trained model, we preemptively analyze the training data to find risky segments.
4. **Sanitization Engine:** consolidates all flags from the contamination detector and membership inference checker, and then cleans the data accordingly. It ensures that by the end of the pipeline, the training data is sanitized – free of known contamination, duplicates, and high-risk memorization content.

My first approach is to develop an MVP for each of the modules starting with data preprocessor and then add complexity in the project.

The working-preprocessed dataset is over 45GB so it is not stored over Git but only locally. 

# Data Preprocessor:
In this module, I have mostly focused on Data cleaning, Normalization and getting the data ready for the Further processing steps. 
Most of the explanations are given in the [Data Preprocessor file](../data_preprocessor.ipynb)

# Contamination Detector:
According to the seminar and project scope we have performed two main scopes.
1. **Reference Benchmark Comparison** 
2. **PaCoST-Inspired Confidence Testing**

With in this module implementation I have majorly tested the contamination strategy, because the preliminary step of data preprocessing was already done
this module tacked the contamination step and flagging of contaminated dataset. 

As of right now, this was done with a manually defined dataset(or just a small list of string), but I plan to replace that by a more extensive benchmark dataset.

More details on the module is present in the notebook [Contamination detection file](../contamination_detector.ipynb)

# Membership Inference Checker:
Initially we implemented this with coherence of Data Preprocessor module, this time we have removed redundant code and made improvements in computational management.

According the MIA, we wanted to check the risk level on the data, with the help of visualisation we were able to undertand better the data distributions and flags. Now I think we will need to implement a refined data_preprocessing module to make this notebook more aligned with project goals.
