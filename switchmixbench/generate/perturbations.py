from __future__ import annotations

import random
from typing import Any, Dict, Tuple


def mix_sentences_with_meta(
    base: str,
    insert: str,
    p_switch: float = 0.35,
    seed: int = 0,
) -> Tuple[str, Dict[str, Any]]:
    """
    Deterministic, minimal code-switching with metadata.

    Strategy (mirrors v0.1): take a short span from `insert` and insert into `base`.
    """
    rng = random.Random(seed)
    base_tokens = base.split()
    insert_tokens = insert.split()

    meta: Dict[str, Any] = {
        "p_switch": float(p_switch),
        "seed": int(seed),
        "applied": False,
        "span_len": 0,
        "insert_pos": None,
        "insert_lang_span": "",
    }

    if rng.random() > p_switch or len(base_tokens) < 6 or len(insert_tokens) < 6:
        return base, meta

    span_len = rng.randint(2, min(6, len(insert_tokens) - 1))
    start_insert = rng.randint(0, len(insert_tokens) - span_len)
    insert_span = insert_tokens[start_insert : start_insert + span_len]

    insert_pos = rng.randint(1, len(base_tokens) - 2)
    mixed = base_tokens[:insert_pos] + insert_span + base_tokens[insert_pos:]

    meta.update(
        {
            "applied": True,
            "span_len": int(span_len),
            "insert_pos": int(insert_pos),
            "insert_lang_span": " ".join(insert_span),
        }
    )
    return " ".join(mixed), meta


def inject_informal_noise_with_meta(
    text: str,
    p: float = 0.15,
    seed: int = 0,
) -> Tuple[str, Dict[str, Any]]:
    """
    Deterministic informal noise with metadata.

    Strategy (mirrors v0.1):
    - drop French negation tokens "ne"/"pas" sometimes
    - abbreviate "you"->"u", "are"->"r" sometimes
    - optionally add casual punctuation / emoticon
    """
    rng = random.Random(seed)
    tokens = text.split()
    out = []

    dropped = 0
    abbr = 0
    for t in tokens:
        if rng.random() < p:
            low = t.lower()
            if low in ["ne", "pas"]:
                dropped += 1
                continue
            if low == "you":
                t = "u"
                abbr += 1
            elif low == "are":
                t = "r"
                abbr += 1
        out.append(t)

    s = " ".join(out)

    punct_added = ""
    if rng.random() < p:
        punct_added = rng.choice(["!", "!!", "...", " :)"])
        s = s + punct_added

    meta: Dict[str, Any] = {
        "p": float(p),
        "seed": int(seed),
        "dropped_tokens": int(dropped),
        "abbreviations": int(abbr),
        "punct_added": punct_added,
    }
    return s, meta


def apply_switchmix_perturbation(
    base: str,
    insert: str,
    seed: int,
    p_switch: float = 0.35,
    noise_p: float = 0.15,
) -> Tuple[str, Dict[str, Any]]:
    mixed, cs_meta = mix_sentences_with_meta(base, insert, p_switch=p_switch, seed=seed)
    noisy, noise_meta = inject_informal_noise_with_meta(mixed, p=noise_p, seed=seed)
    meta: Dict[str, Any] = {
        "type": "switchmix+informal_noise",
        "code_switch": cs_meta,
        "noise": noise_meta,
    }
    return noisy, meta

