"""Tiny text utilities shared across SwitchMixBench."""

import re


def normalize_ws(text: str) -> str:
    """Collapse runs of whitespace into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def simple_tokenize(text: str):
    """Very lightweight tokenizer returning word and punctuation tokens."""
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
