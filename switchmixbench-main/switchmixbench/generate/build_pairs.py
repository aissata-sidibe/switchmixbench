"""SwitchMix pair construction utilities.

This module contains a small helper dataclass and a single convenience function
used by the prototype pipeline to construct aligned clean vs. switchmixed
pairs from parallel French/English inputs. It intentionally stays lightweight
so that it can be reused by both data-building scripts and tests.
"""

from dataclasses import dataclass
from switchmixbench.generate.code_switch_rules import mix_sentences
from switchmixbench.generate.noise_injectors import inject_informal_noise


@dataclass
class Example:
    uid: str
    task: str
    split: str
    variant: str  # clean or switchmix
    text_a: str
    text_b: str
    label: str


def build_switchmix_pair(uid, task, split, fr_a, en_a, fr_b=None, en_b=None, label=""):
    """Build a clean / switchmix pair for a single underlying example.

    Parameters
    ----------
    uid:
        Unique identifier for the underlying semantic example.
    task:
        Task name (e.g. ``\"nli\"`` or ``\"qa\"``).
    split:
        Dataset split (e.g. ``\"train\"`` or ``\"test\"``).
    fr_a, en_a:
        Parallel French / English texts for the first field
        (premise or context depending on the task).
    fr_b, en_b:
        Optional parallel texts for the second field (hypothesis / question).
        When omitted, ``text_b`` in the clean example is left empty.
    label:
        Gold label or answer string, copied verbatim to both variants.

    Returns
    -------
    list[Example]
        A list of length two containing a clean example and its switchmixed
        counterpart. The order is always ``[clean, switchmix]``.

    Notes
    -----
    For NLI, ``text_a`` is the premise and ``text_b`` is the hypothesis.
    For QA,  ``text_a`` is the context and ``text_b`` is the question.
    """
    clean_a = fr_a
    clean_b = fr_b if fr_b is not None else ""

    mixed_a = inject_informal_noise(
        mix_sentences(fr_a, en_a, seed=hash(uid) % 10000),
        seed=hash(uid) % 10000
    )

    mixed_b = ""
    if fr_b is not None and en_b is not None:
        mixed_b = inject_informal_noise(
            mix_sentences(fr_b, en_b, seed=(hash(uid) + 7) % 10000),
            seed=(hash(uid) + 7) % 10000
        )

    clean = Example(uid=uid, task=task, split=split, variant="clean", text_a=clean_a, text_b=clean_b, label=label)
    switchmix = Example(uid=uid, task=task, split=split, variant="switchmix", text_a=mixed_a, text_b=mixed_b, label=label)

    return [clean, switchmix]
