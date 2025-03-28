import re
import unicodedata
import nltk
from nltk.corpus import stopwords

# Ensure required NLTK data is available.
nltk.download('stopwords', quiet=True)


def normalize_text(text, remove_stopwords=False, language="english"):
    """
    Normalize and clean text by:
      - Removing HTML tags.
      - Normalizing Unicode (NFC).
      - Converting to lowercase and stripping accents.
      - Removing non-UTF characters and extra whitespace.
      - Optionally removing stopwords.

    Args:
        text (str): The input text.
        remove_stopwords (bool): Whether to remove stopwords (default: False).
        language (str): Language for stopwords (default: "english").

    Returns:
        str: Cleaned text.
    """
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # Normalize Unicode
    text = unicodedata.normalize('NFC', text)
    # Lowercase conversion and accent stripping
    text = text.lower()
    text = ''.join(c for c in text if not unicodedata.combining(c))
    # Remove non-UTF characters and extra whitespace
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    if remove_stopwords:
        stop_words = set(stopwords.words(language))
        words = text.split()
        words = [word for word in words if word not in stop_words]
        text = " ".join(words)

    return text