import random
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_LM_MODEL_NAME = "distilgpt2"


def perturb_text(text):
    """
    Perturb the text by shuffling its words.

    Args:
        text (str): Input text.

    Returns:
        str: Perturbed text.
    """
    words = text.split()
    if len(words) <= 1:
        return text
    random.shuffle(words)
    return " ".join(words)


def load_language_model(model_name=DEFAULT_LM_MODEL_NAME):
    """
    Load a lightweight language model and its tokenizer.

    Args:
        model_name (str): Model name (default: distilgpt2).

    Returns:
        (model, tokenizer): The loaded model and tokenizer.
    """
    logging.info("Loading language model: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    if torch.mps.is_available():
        model.to("mps")
    return model, tokenizer


def compute_perplexity(text, model, tokenizer):
    """
    Compute perplexity for a given text using a language model.

    Args:
        text (str): Input text.
        model: Language model.
        tokenizer: Corresponding tokenizer.

    Returns:
        float: Computed perplexity or None on error.
    """
    try:
        encodings = tokenizer(text, return_tensors='pt')
        if torch.mps.is_available():
            encodings = {k: v.to("mps") for k, v in encodings.items()}
        with torch.no_grad():
            outputs = model(**encodings, labels=encodings["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss)
        return perplexity.item()
    except Exception as e:
        logging.error("Error computing perplexity: %s", e)
        return None