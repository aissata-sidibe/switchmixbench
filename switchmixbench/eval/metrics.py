import re
from collections import Counter

def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s

def tokenize(s: str):
    return normalize(s).split()

def accuracy(preds, labels):
    correct = 0
    for p, y in zip(preds, labels):
        if normalize(p) == normalize(y):
            correct += 1
    return correct / max(1, len(labels))

def exact_match(preds, labels):
    return accuracy(preds, labels)

def f1_score(pred: str, gold: str) -> float:
    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)

def mean_f1(preds, labels):
    scores = [f1_score(p, y) for p, y in zip(preds, labels)]
    return sum(scores) / max(1, len(scores))

