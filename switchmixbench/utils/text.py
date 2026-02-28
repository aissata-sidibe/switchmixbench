import re

def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def simple_tokenize(text: str):
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
