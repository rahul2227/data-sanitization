## Membership Inference Checker
**This checks the dataset for indications that certain data might be over-represented or memorized, using a neighborhood-based membership inference approach. Instead of attacking a trained model, we preemptively analyze the training data to find risky segments.**

### Module Functionality

#### Embedding Generation
Each segment is encoded into a vector embedding using a small language model or encoder.
- In our case we are using "all-MiniLM-L6-v2"
	- This is a relatively small and robust model which suits our use-case
- Initially we used bert-uncased but switched from it as it was memory intensive

#### Neighborhood similarity Analysis

For each segment’s embedding, we compute its similarity with embeddings of all other segments. We use cosine similarity as the metric for closeness in this semantic space.

_Segments that have one or more very high-similarity neighbors are essentially duplicates or nearly identical in content_

#### Outlier detection for uniqueness
In addition to duplicates, we consider the opposite case – segments that are extremely unique compared to the rest of the data. If an item has no neighbors with any moderate similarity (i.e., it’s an outlier), it might be an anomaly.

#### Threshold based Alerts
two key thresholds – one for high similarity (to catch duplicates) and one for outlier scoring (to catch isolated points).

> *These are tuned on synthetic data*