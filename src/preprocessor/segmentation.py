from nltk.tokenize import sent_tokenize
from tokenization import tokenize_text


def segment_text(text, mode='sentence', fixed_token_length=100):
    """
    Segment text into units based on mode.

    Args:
        text (str): Input text.
        mode (str): 'sentence' for sentence segmentation,
                    'fixed' for fixed token-length segments,
                    'none' to return the text as a single segment.
        fixed_token_length (int): Token count per segment (for 'fixed' mode).

    Returns:
        list: List of text segments.
    """
    if mode == 'sentence':
        return sent_tokenize(text)
    elif mode == 'fixed':
        tokens = tokenize_text(text)
        return [' '.join(tokens[i:i + fixed_token_length]) for i in range(0, len(tokens), fixed_token_length)]
    else:
        return [text]


def segment_dataframe(df, text_column='cleaned_text', mode='sentence'):
    """
    Apply segmentation to each entry in a DataFrame column and explode the result.

    Args:
        df (pd.DataFrame): Input DataFrame.
        text_column (str): Column containing the text to segment.
        mode (str): Segmentation mode.

    Returns:
        pd.DataFrame: DataFrame with a new column 'segments', exploded into one row per segment.
    """
    df['segments'] = df[text_column].apply(lambda x: segment_text(x, mode=mode))
    return df.explode('segments')