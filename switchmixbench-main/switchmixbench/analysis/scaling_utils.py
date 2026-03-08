"""Utility helpers for scaling analyses in SwitchMixBench."""

from __future__ import annotations

from typing import Optional


_MODEL_SIZE_MILLIONS = {
    "distilbert-base-multilingual-cased": 134,
    "bert-base-multilingual-cased": 179,
    "xlm-roberta-base": 270,
    "xlm-roberta-large": 550,
    "google/flan-t5-small": 80,
    "google/flan-t5-base": 250,
}


def estimate_model_size(model_name: str) -> Optional[int]:
    """Return approximate parameter count (in millions) for known models.

    Parameters
    ----------
    model_name:
        HF model identifier (e.g. ``"xlm-roberta-base"``).

    Returns
    -------
    Optional[int]
        Parameter count in millions when the model is known, otherwise
        ``None``.
    """

    return _MODEL_SIZE_MILLIONS.get(model_name)

