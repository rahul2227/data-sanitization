from transformers import AutoTokenizer

# DEFAULT_TOKENIZER_MODEL = 'bert-base-uncased'
DEFAULT_TOKENIZER_MODEL = 'distilgpt2'
# Initialize the fast tokenizer.
tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_MODEL, use_fast=True)


def tokenize_text(text):
    """
    Tokenize text using Hugging Face's fast tokenizer.

    Args:
        text (str): Input text.

    Returns:
        list: List of tokens.
    """
    return tokenizer.tokenize(text, truncation=True, max_length=1024) # changes made because of warning