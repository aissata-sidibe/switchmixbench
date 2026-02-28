import random

def mix_sentences(fr: str, en: str, p_switch: float = 0.35, seed: int = 0):
    """
    Minimal FR<->EN code-switching:
    Take a short EN span and insert into a FR sentence.
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
