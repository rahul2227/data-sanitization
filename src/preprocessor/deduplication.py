import pandas as pd


def remove_duplicates(df, text_column='cleaned_text'):
    """
    Remove exact duplicate rows in the DataFrame based on the given text column.

    Args:
        df (pd.DataFrame): DataFrame containing text data.
        text_column (str): Column name to check for duplicates (default: 'cleaned_text').

    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    return df.drop_duplicates(subset=[text_column])