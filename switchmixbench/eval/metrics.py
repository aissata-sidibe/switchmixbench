import re

def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s

def accuracy(preds, labels):
    correct = 0
    for p, y in zip(preds, labels):
        if normalize(p) == normalize(y):
            correct += 1
    return correct / max(1, len(labels))

def exact_match(preds, labels):
    return accuracy(preds, labels)
