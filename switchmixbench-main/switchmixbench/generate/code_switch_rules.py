"""Code-switching helpers for SwitchMixBench.

This module encodes a tiny, interpretable rule for FR↔EN mixing used by the
prototype dataset builder. It operates at the token level and is intentionally
simple enough to make it easy to reason about resulting perturbations.
"""

import random


def mix_sentences(fr: str, en: str, p_switch: float = 0.35, seed: int = 0):
    """Insert a short English span into a French sentence.

    Parameters
    ----------
    fr:
        Base French sentence that will receive the insertion.
    en:
        Parallel English sentence from which a token span is sampled.
    p_switch:
        Probability of actually performing a switch; otherwise ``fr`` is
        returned unchanged.
    seed:
        Integer seed controlling the sampling procedure.

    Returns
    -------
    str
        Either the original ``fr`` sentence or a mixed version with an
        English span inserted at a random interior position.
    """
    rng = random.Random(seed)

    fr_tokens = fr.split()
    en_tokens = en.split()

    if rng.random() > p_switch or len(fr_tokens) < 6 or len(en_tokens) < 6:
        return fr

    span_len = rng.randint(2, min(6, len(en_tokens) - 1))
    start_en = rng.randint(0, len(en_tokens) - span_len)
    en_span = en_tokens[start_en:start_en + span_len]

    insert_pos = rng.randint(1, len(fr_tokens) - 2)
    mixed = fr_tokens[:insert_pos] + en_span + fr_tokens[insert_pos:]

    return " ".join(mixed)
