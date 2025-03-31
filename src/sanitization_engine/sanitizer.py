import logging
import pandas as pd
from tqdm import tqdm

def aggregate_flags(df_contamination, df_membership):
    logging.info("Aggregating contamination and membership inference flags.")
    contamination_indices = set(df_contamination[df_contamination['contamination_flag']].index)
    membership_indices = set(df_membership[df_membership['membership_inference_flag']].index)

    combined_flags = contamination_indices.union(membership_indices)

    flag_reason = {}
    for idx in combined_flags:
        reasons = []
        if idx in contamination_indices:
            reasons.append("contamination")
        if idx in membership_indices:
            reasons.append("membership")
        flag_reason[idx] = ", ".join(reasons)

    return combined_flags, flag_reason

def sanitize_data(df, flagged_indices, flag_reason, action="remove"):
    log_entries = []

    for idx in tqdm(flagged_indices, desc="Sanitizing Data", dynamic_ncols=True):
        original_text = df.at[idx, 'segments']
        reason = flag_reason.get(idx, "unknown")

        if action == "remove":
            df.drop(idx, inplace=True)
            log_entries.append((idx, "removed", reason, original_text))

        elif action == "anonymize":
            df.at[idx, 'segments'] = "[REMOVED DUE TO PRIVACY RISK]"
            log_entries.append((idx, "anonymized", reason, original_text))

        elif action == "rewrite":
            df.at[idx, 'segments'] = original_text + " [REWRITTEN]"
            log_entries.append((idx, "rewritten", reason, original_text))

        else:
            logging.warning(f"Invalid action: {action}, skipping index {idx}")

    log_df = pd.DataFrame(log_entries, columns=["index", "action", "reason", "original_text"])
    return df.reset_index(drop=True), log_df