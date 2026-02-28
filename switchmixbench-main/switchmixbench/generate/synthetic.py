from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class NLIPair:
    pair_id: str
    premise_fr: str
    premise_en: str
    hypothesis_fr: str
    hypothesis_en: str
    label: str  # entailment | neutral | contradiction


@dataclass(frozen=True)
class QAPair:
    pair_id: str
    context_fr: str
    context_en: str
    question_fr: str
    question_en: str
    answer: str


NLI_LABELS = ["entailment", "neutral", "contradiction"]


def _choice(rng: random.Random, xs: List[str]) -> str:
    return xs[rng.randrange(0, len(xs))]


def generate_synthetic_nli(
    n: int,
    seed: int = 42,
    id_prefix: str = "syn_nli",
) -> List[NLIPair]:
    """
    Generate many NLI examples by templating.
    This is designed as a robust offline fallback when HF download is unavailable.
    """
    rng = random.Random(seed)

    names = ["Marie", "Paul", "Aïcha", "Ibrahim", "Sofia", "Lucas", "Fatou", "Noah"]
    cities = ["Paris", "Lyon", "Marseille", "Dakar", "Abidjan", "Montréal", "Bruxelles"]
    items = ["une voiture", "un livre", "un téléphone", "des pommes", "des bananes", "un vélo"]
    items_en = ["a car", "a book", "a phone", "apples", "bananas", "a bicycle"]
    hobbies = ["le sport", "la musique", "la lecture", "le cinéma", "la danse"]
    hobbies_en = ["sports", "music", "reading", "movies", "dancing"]

    out: List[NLIPair] = []

    for i in range(n):
        lab = _choice(rng, NLI_LABELS)
        name = _choice(rng, names)
        city = _choice(rng, cities)
        idx_item = rng.randrange(0, len(items))
        item_fr = items[idx_item]
        item_en = items_en[idx_item]
        idx_hobby = rng.randrange(0, len(hobbies))
        hobby_fr = hobbies[idx_hobby]
        hobby_en = hobbies_en[idx_hobby]

        if lab == "entailment":
            premise_fr = f"{name} a acheté {item_fr} hier."
            premise_en = f"{name} bought {item_en} yesterday."
            hypothesis_fr = f"{name} a acheté quelque chose."
            hypothesis_en = f"{name} bought something."
        elif lab == "contradiction":
            premise_fr = f"Il pleut aujourd'hui à {city}."
            premise_en = f"It is raining today in {city}."
            hypothesis_fr = f"Le temps est sec aujourd'hui à {city}."
            hypothesis_en = f"The weather is dry today in {city}."
        else:
            premise_fr = f"{name} aime {hobby_fr}."
            premise_en = f"{name} likes {hobby_en}."
            hypothesis_fr = f"{name} a acheté {item_fr}."
            hypothesis_en = f"{name} bought {item_en}."

        pair_id = f"{id_prefix}_{seed}_{i:06d}"
        out.append(
            NLIPair(
                pair_id=pair_id,
                premise_fr=premise_fr,
                premise_en=premise_en,
                hypothesis_fr=hypothesis_fr,
                hypothesis_en=hypothesis_en,
                label=lab,
            )
        )

    return out


def generate_synthetic_qa(
    n: int,
    seed: int = 42,
    id_prefix: str = "syn_qa",
) -> List[QAPair]:
    rng = random.Random(seed)

    names = ["Marie", "Paul", "Aïcha", "Ibrahim", "Sofia", "Lucas", "Fatou", "Noah"]
    places = ["au marché", "à la bibliothèque", "au parc", "au restaurant"]
    places_en = ["to the market", "to the library", "to the park", "to the restaurant"]
    fruits = [("des pommes", "apples"), ("des bananes", "bananas"), ("des oranges", "oranges")]

    out: List[QAPair] = []

    for i in range(n):
        name = _choice(rng, names)
        j = rng.randrange(0, len(places))
        place_fr, place_en = places[j], places_en[j]
        k = rng.randrange(0, len(fruits))
        fruit_fr, fruit_en = fruits[k]

        context_fr = f"{name} est allé {place_fr} ce matin pour acheter des fruits. Il a choisi {fruit_fr}."
        context_en = f"{name} went {place_en} this morning to buy fruit. He chose {fruit_en}."
        question_fr = f"Quels fruits {name} a-t-il achetés ?"
        question_en = f"What fruits did {name} buy?"
        answer = fruit_fr.replace("des ", "").strip()  # keep as short target string

        pair_id = f"{id_prefix}_{seed}_{i:06d}"
        out.append(
            QAPair(
                pair_id=pair_id,
                context_fr=context_fr,
                context_en=context_en,
                question_fr=question_fr,
                question_en=question_en,
                answer=answer,
            )
        )

    return out

