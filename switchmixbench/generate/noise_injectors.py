import random

def inject_informal_noise(text: str, p: float = 0.15, seed: int = 0):
    """
    Adds mild informal noise:
    - small abbreviation
    - dropped negation tokens sometimes
    - casual punctuation
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
