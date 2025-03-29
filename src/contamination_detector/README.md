## Contamination Detector Module

**This module identifies portions of training data that should not be there such as leaked evaluation benchmarks or 
any data that overlaps with a reference set.**

### Module Functionality

#### **Reference Benchmark Comparison:** 

Here the goal is to maintain reference datasets(like benchmark datasets). Each training data segment is then 
compared with these references to check for overlap.
   - For our use case we are using pg19 as the reference dataset


#### **PaCoST inspired Statistical Testing:** 
For each candidate segment, we create a “counterpart” variant with similar distribution (shuffled version of the text that preserves topic/length in our case) and measure a language model’s confidence on both the original and the counterpart. 
The intuition from PaCoST is that if the model is significantly more confident (i.e., assigns higher probability or lower perplexity) on the exact original text than on a comparable alternative, the original may have appeared in training or is memorized.


#### **Flagging and reporting:** 
Any segment that either (a) directly matches a reference benchmark entry or (b) triggers the PaCoST-style confidence anomaly is flagged. 
The output is a list of segment IDs (or content snippets) with flags such as “Potential contamination: overlaps with XYZ benchmark” or “Model confidence too high (possible leak)”. These flags include the confidence scores and similarity metrics for transparency. 
All flagged instances move on to the sanitization stage.