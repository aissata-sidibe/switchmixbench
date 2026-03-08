"""Simple informal-noise injection utilities.

This module defines a small perturbation function used by the original
SwitchMixBench prototype to approximate casual user text. It is deliberately
minimal and operates directly on whitespace-delimited tokens, making it easy
to reason about in tests and analyses.
"""

import random


def inject_informal_noise(text: str, p: float = 0.15, seed: int = 0):
    """Inject light informal noise into a sentence.

    Parameters
    ----------
    text:
        Input string to perturb.
    p:
        Per-token probability of applying a local edit.
    seed:
        Integer seed controlling the internal RNG.

    Returns
    -------
    str
        A perturbed copy of ``text`` with small abbreviations, dropped
        negation tokens and occasional extra punctuation.
    """
    rng = random.Random(seed)
    tokens = text.split()
    out = []

    for t in tokens:
        if rng.random() < p:
            if t.lower() in ["ne", "pas"]:
                continue
            if t.lower() == "you":
                t = "u"
            if t.lower() == "are":
                t = "r"
        out.append(t)

    s = " ".join(out)

    if rng.random() < p:
        s = s + rng.choice(["!", "!!", "...", " :)"])
    return s
